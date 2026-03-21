"""
BaseFederate.py

Base class for HELICS federates in the Cosim_gym framework, managing communication and simulation steps.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""
import importlib
import math
import os
import sys
import pprint
from pathlib import Path
import helics as h
import numpy as np
from abc import ABC, abstractmethod
print(os.getcwd())
sys.path.append('src/')
from datetime import datetime, timedelta
from dataclasses import fields
from utils.config_dataclasses import FederateConfig, StartupSyncConfig
from models.model_catalog.ModelCatalog import ModelCatalog, ModelMetadata, InterfaceType
from models.model_catalog.RedisCatalog import RedisCatalog
from utils.influxdb_client import InfluxClient
pp = pprint.PrettyPrinter(indent=4)

# TODO make an abstract class for base federate and change the name of this one as simplefederate
class BaseFederate():
    """
    Base class for all HELICS federates.
    
    This class provides the basic structure and lifecycle methods
    that all federates must implement. Concrete federate classes
    should inherit from this class and override the specific methods.
    
    Attributes:
        name (str): The name of the federate
        helics_federate: The HELICS federate instance (initialized in subclasses)
    """
    
    def __init__(self, name: str, config: FederateConfig, logger=None, sim_id=None, federation_name=None):
        """
        Initialize the base federate.
        
        Args:
            name (str): The name of the federate
        """
        self.name = name
        self.config = config
        self.logger = logger
        self.federate = None
        self.meta_data = None
        self.simulation_id = sim_id
        self.federation_name = federation_name
        self.entities = [] # list with dict [{"entity_name": entity_object}]
        self.catalog = RedisCatalog(logger=self.logger)
        
        # connection handles
        self.pubs = []
        self.subs = []
        self.eps = []

        # data exchange structures
        self.inputs = {}
        self.outputs = {}
        self._last_input_meta = {}
        self._deferred_inputs = {}

        # storage buffer
        self.storage = {'inputs':{},
                        'outputs':{},
                        'params':{},
                        'time':[]}
        self.batch_size = self.config.memory_config.batch_size


        # co.simualtion time management
        self.start_time = datetime.fromisoformat(self.config.timing_configs.start_time)
        self.end_time = datetime.fromisoformat(self.config.timing_configs.end_time)
        self.stop_time = self.config.timing_configs.time_stop
        self.time_period = self.config.timing_configs.time_period
        self.real_period = self.config.timing_configs.real_period
        self.offset = self.config.timing_configs.time_offset
        # changing state variables for timing
        self.date_time = self.start_time  # keeps track of simulation evolution with datetime format (real wolrd time)
        self.time_granted = 0.0  # time logic for helics it is granted_time + time_period
        self.ts = 0  # counter of the number of steps done
        self.startup_sync = self.config.startup_sync or StartupSyncConfig()

        # rl specific variables only for normal federates
        if not self.config.rl_task:
            self.mode = self.config.rl_config.get('mode', 'test') #TODO should passs the mode from unique point ScenarioManager
            self.episode_length = self.config.rl_config.get('episode_length', None) if self.config.rl_config else None
            self.n_episodes = self.config.rl_config.get('n_episodes', None) if self.config.rl_config else None
            self.reset_type = self.config.rl_config.get('reset_type', None) if self.config.rl_config else None
            self.reset_length = self.config.rl_config.get('reset_period', None) if self.config.rl_config else None
            self.new_starting_point = 0
            self.rolling_window = self.config.rl_config.get('rolling_window', None) if self.config.rl_config else None
            self.episode_count = 0  
            self.reset_count = 0 

    def initialize(self):
        """
        Initialize the federate.
        
        This method should be overridden by concrete federate classes
        to implement specific initialization logic, including HELICS
        federate creation and configuration.
        """
        self.logger.info(f'federate {self.name} initialization')
        self.federate = self._register_federate()
        self.entities = self._register_entities()
        self.inputs = {mod['id']:{} for mod in self.entities}
        self.outputs = {mod['id']:{} for mod in self.entities}
        self._deferred_inputs = {mod['id']: {} for mod in self.entities}
        self.pubs, self.subs, self.eps = self._register_connections()
        self.initialize_storage()  # Set up storage structure based on entities and config

        self.logger.info(f'federate {self.name} initialized')

    def initialize_storage(self):
        '''
        Initialize the storage buffer based on the entities and their inputs/outputs.
        Storage is partitioned by mode (train / test) so that timeseries
        recorded during training are kept separate from testing.
        '''
        self.storage = {
            'train': self._create_storage_partition(),
            'test': self._create_storage_partition(),
        }

    def _create_storage_partition(self):
        """Create one empty storage partition (used for both train and test)."""
        partition = {
            'inputs': {},
            'outputs': {},
            'params': {},
            'time': [],
        }
        memory_items = self.config.memory_config.attrs

        if memory_items == 'all':
            for entity in self.entities:
                entity_id = entity['id']
                partition['inputs'][entity_id] = {var: [] for var in entity['object'].state.inputs.keys()}
                partition['outputs'][entity_id] = {var: [] for var in entity['object'].state.outputs.keys()}
        else:
            for entity in self.entities:
                entity_id = entity['id']
                partition['inputs'][entity_id] = {var: [] for var in entity['object'].state.inputs.keys() if var in memory_items}
                partition['outputs'][entity_id] = {var: [] for var in entity['object'].state.outputs.keys() if var in memory_items}
                partition['params'][entity_id] = {var: [] for var in entity['object'].state.parameters.keys() if var in memory_items}

        return partition

    def _register_federate(self):
        """
        Register the federate with HELICS.
        
        This method should be overridden by concrete federate classes
        to implement specific registration logic, including HELICS
        federate registration.
        """
        self.logger.info(f'federate {self.name} registration')
        self.logger.info(f'config: {pp.pformat(self.config)}')
        
        # create fedInfo *****
        fedInfo = h.helicsCreateFederateInfo()
        setattr(fedInfo, 'name', self.name)
        core_name = self.config.core_name if self.config.core_name else self.name+'_core'
        # self.core_type = self.config.core_type 
        self.logger.debug(f'CORE NAME: {core_name} - CORE TYPE: {self.config.core_type}')
        setattr(fedInfo, 'core_name', core_name)
        setattr(fedInfo, 'core_type', self.config.core_type)

        # setattr(fedInfo, 'broker_address', self.config.broker_address)
        
        # set broker informations  
        if self.config.broker_address:     
            h.helicsFederateInfoSetBroker(fedInfo, self.config.broker_address)
            self.logger.info(f"Using broker address: {self.config.broker_address}")

            address = self.config.broker_address.split(':')[0]
            port = self.config.broker_address.split(':')[1] 
            h.helicsFederateInfoSetBroker(fedInfo, address)
            h.helicsFederateInfoSetBrokerPort(fedInfo, int(port))
        else:
            self.logger.warning("No broker address specified in config! This works only for single federation with zmq. If nto need to specify a broker address.")
        
        # Set timing configurations if provided
        timing_configs = self.config.timing_configs
        if timing_configs:
            period = float(timing_configs.time_period) if timing_configs.time_period is not None else 1.0
            delta = (
                float(timing_configs.time_delta)
                if getattr(timing_configs, "time_delta", None) is not None
                else period
            )
            offset = float(timing_configs.time_offset) if timing_configs.time_offset is not None else 0.0
            h.helicsFederateInfoSetTimeProperty(fedInfo, h.helics_property_time_period, period)
            h.helicsFederateInfoSetTimeProperty(fedInfo, h.helics_property_time_offset, offset)
            h.helicsFederateInfoSetTimeProperty(fedInfo, h.helics_property_time_delta, delta)
            # h.helicsFederateInfoSetTimeProperty(fedInfo, h.helics_property_time_stoptime, timing_configs.time_stop )
            # h.helics_time_maxtime = timing_configs.time_stop 

            self.logger.debug(f'Setting federate timing configs: {pp.pformat(timing_configs)}')
        else:
            self.logger.warning('No timing_configs found in config, using default period of 1.0 seconds')
            setattr(fedInfo, 'period', 1.0)
        
        # create the federate and set timings and flags
        fed = h.helicsCreateCombinationFederate(self.name, fedInfo)

        # Route HELICS internal log messages into our Python logger.
        # helics-python uses cffi: the callback must be wrapped with ffi.callback
        # and kept alive (stored on self) to prevent garbage collection.
        def _helics_log_callback(loglevel, identifier, message, user_data):
            # HELICS levels: 0=no_print,1=error,2=warning,3=summary,4=connections,
            #                5=interfaces,6=timing,7=data,8=debug,9=trace
            # identifier and message arrive as cffi char* — decode them first
            try:
                id_str  = h.ffi.string(identifier).decode('utf-8', errors='replace') if identifier != h.ffi.NULL else ''
                msg_str = h.ffi.string(message).decode('utf-8', errors='replace')    if message    != h.ffi.NULL else ''
            except Exception:
                id_str, msg_str = '', ''
            if loglevel <= 1:
                self.logger.error(f"[HELICS] {id_str}: {msg_str}")
            elif loglevel == 2:
                self.logger.warning(f"[HELICS] {id_str}: {msg_str}")
            elif loglevel == 3:
                self.logger.info(f"[HELICS] {id_str}: {msg_str}")
            else:
                self.logger.debug(f"[HELICS] {id_str}: {msg_str}")

        # Wrap as cffi callback and store on self so it is not garbage-collected
        self._helics_log_cb = h.ffi.callback("void(int, char *, char *, void *)", _helics_log_callback)
        h.helicsFederateSetLoggingCallback(fed, self._helics_log_cb, h.ffi.NULL)
        #  TODO other timings properties make. a general time properties setter (both TIME_ and INT_ ) need to change namings in yaml
        
        # set the flags
        flags = self.config.flags
        if flags:
            for field in fields(flags):
                flag_name = field.name
                flag_value = getattr(flags, flag_name)
                flag_name = 'HELICS_FLAG_{}'.format(flag_name.upper())
                fed.flag[getattr(h,flag_name)] = flag_value
        else:
            self.logger.warning("Federate Flags in yaml config file are empty!\n")
 
        self.logger.debug(f"Federate class variables: \n{pp.pformat(fed.__dict__)}")

        return fed

    def _register_entities(self):
        model_configs = self.config.model_configs 
        self.logger.debug(f"Model configs: {pp.pformat(model_configs)}")
        model_configs.time_step = self.config.timing_configs.real_period # add the real period to the model configs
        model_configs.time_stop = self.config.timing_configs.time_stop # add the simulation stop time to model configs
        model_configs.start_time = self.config.timing_configs.start_time # add the simulation start time to model configs
        model_configs.end_time = self.config.timing_configs.end_time # add the simulation end time to model configs
        model_configs.real_period = self.config.timing_configs.real_period # add the real period to model configs
        model_configs.inputs , model_configs.outputs= self.input_output_names() 
        self.logger.debug(f"Updated Model configs: {pp.pformat(model_configs)}")
        
        # Info for Instantiation of model class 
        model_name_catalog = model_configs.instantiation.model_name
        model_metadata=self.catalog.get_model_metadata(model_name_catalog)
        script = model_metadata.module_path
        class_name = model_metadata.class_name
        self.logger.debug(f"Model metadata for '{model_name_catalog}': {pp.pformat(model_metadata)}")
        # inputs, outputs = self.input_output_names()

        
        entities = []
        for i in range(model_configs.instantiation.n_instances):
            # model_name = f"{self.name}.{prefix}-{i}"
            model_name = f'{self.name}.{i}'
            entity={}
            module = importlib.import_module(script)
            model_class = getattr(module, class_name)
            entity['id'] = model_name
            entity['object'] = model_class(model_name, model_metadata, model_configs, self.logger)
            entities.append(entity)

        self.logger.info(f"Registered entities: {pp.pformat(entities)}")
        return entities

    def input_output_names(self):
        ''' this small methods returns the list of inputs and outputs names for the models, it only consider pub/sub not endpoints for now'''
        pubs = [p.key for p in self.config.connections.publishes]
        subs = [s.key for s in self.config.connections.subscribes]
        # eps = self.config.get('endpoints', []) TODO understand how to manage eps for input output names
        return subs, pubs
    
    def _register_connections(self):
        pubs=[]
        subs=[]
        eps=[]

        for mod in self.entities:
            pub = self._register_pubs(mod)
            # self.logger.debug(f"Registered publications for {mod['id']}: {pp.pformat(pub)}")
            sub = self._register_subs(mod)
            # self.logger.debug(f"Registered subscriptions for {mod['id']}: {pp.pformat(sub)}")
            ep = self._register_eps(mod)
            # self.logger.debug(f"Registered endpoints for {mod['id']}: {pp.pformat(ep)}")

            pubs.extend(pub)
            subs.extend(sub)
            eps.extend(ep)
        
        self.logger.info(f"All registered publications: {pp.pformat(pubs)}")
        self.logger.info(f"All registered subscriptions: {pp.pformat(subs)}")
        self.logger.info(f"All registered endpoints: {pp.pformat(eps)}")

        return pubs, subs, eps 

    def _register_pubs(self, mod):
        pubs = []
        for pub_var in self.config.connections.publishes:
            pub = {}
            pub['entity_name'] = mod['id']
            pub['topic'] = mod['id'] + f"/{pub_var.key}"
            pub['pubid'] = self.federate.register_global_publication(pub['topic'], kind=pub_var.type, units=pub_var.units)
            pubs.append(pub)
        return pubs
    
    def _register_subs(self, mod):
        subs = []
        for sub_var in self.config.connections.subscribes:
            sub = {}
            sub['entity_name'] = mod['id']
            sub['topic'] = mod['id'] + f"/{sub_var.key}"
            sub['subid'] = self.federate.register_global_input(sub['topic'], kind=sub_var.type, units=sub_var.units)
            sub['causality'] = self._normalize_subscription_causality(getattr(sub_var, 'causality', None))
            
            assert sub_var.targets, f"Subscription '{sub_var.key}' has no target specified!"
            
            # case in which each model instance receive data from different topics (e.g. multiple model instances or different federates)
            if isinstance(sub_var.targets, dict):
                target_list = sub_var.targets.get(str(mod['id'].split('.')[-1]), [])
            else:
                target_list = sub_var.targets

            if len(target_list) > 1:
                # Get multi_input_handling value with fallback logic
                self.logger.debug(f'sub_var: {sub_var}')
                if hasattr(sub_var, 'multi_input_handling') and sub_var.multi_input_handling:
                    multi_input_handling_config = sub_var.multi_input_handling
                else:   
                    multi_input_handling_config = 'sum'
                    self.logger.warning(f"Multiple targets found for subscription '{sub_var.key}' but no multi_input_handling specified. Defaulting to SUM.")

                
                if isinstance(multi_input_handling_config, dict):
                    instance_id = str(mod['id'].split('.')[-1])
                    if instance_id not in multi_input_handling_config:
                        self.logger.warning(f"Multi input handling method for instance '{instance_id}' not specified. Defaulting to SUM.")
                        multi_input_handling = 'sum'
                    else:
                        multi_input_handling = multi_input_handling_config[instance_id]
                else:
                    multi_input_handling = multi_input_handling_config
                
                # Set the flag for multi input handling
                val = h.helicsGetOptionValue(multi_input_handling)
                h.helicsInputSetOption(sub['subid'], h.HELICS_HANDLE_OPTION_MULTI_INPUT_HANDLING_METHOD, val)

    
            self.logger.debug(f"Subscription '{sub['topic']}' targets: {target_list}")
            for target in target_list:
                # sub['subid'].add_target(target)
                self.logger.debug(f"Adding target '{target}' to subscription '{sub['topic']}'")
                h.helicsInputAddTarget(sub['subid'], target)

            subs.append(sub)
            self.logger.debug(f"class_variables from subid {sub['subid'].__dict__}")

        return subs
            
    def _register_eps(self, mod):
        return []  # TODO implement endpoint registration

    def run(self):
        """
        Run the main federate logic.
        
        This method should be overridden by concrete federate classes
        to implement the main simulation/computation logic of the federate.
        """
        h.helicsFederateEnterExecutingMode(self.federate)
        self.logger.info(f'Federate {self.name} entered executing mode. Start simulation at datetime: {self.start_time} - Stop time: {self.stop_time}')
        self.logger.info(f'Federate will perform a total number of steps: {self.stop_time}, with a realWolrd frequency of: {self.real_period} s and a simulation frequency of {self.time_period}')
        self.time_granted = 0.0
        self.ts = 0

        self._publish_init_state()  # publish initial state before starting the simulation loop, this is useful for the other federates to receive the initial conditions if needed
        self._enforce_startup_input_sync()
        # self.logger.info(f'📅 Starting simulation at datetime: {self.date_time}')
        # TODO: manage possible offsets for now no offset is used
        #  TODO: capire se devo richiedere al primo step 0 o cosa cambia

        # while self.time_granted < self.stop_time*self.time_period: TODO
        while self.ts < self.stop_time:

            # calculate time to request and automatically update ts
            # new_time = self.time_to_request()
            # self.logger.info(f'Requesting time {new_time}.')
            self.logger.info('=======================================================================================================================')
            # self._reset() # check if reset is needed at the beginning of the step to manage the reset of the federate in case of training, this is because usually after the reset i want to publish the new initial conditions to let other federates receive them and then request time advance to start the new episode with the new initial conditions
            # request time
            self.request_time_advance()
            # self.logger.info(f'Granted time {self.time_granted}, datetime: {self.date_time}')



            # get inputs
            self._apply_deferred_inputs()
            self._receive_inputs()

            # update models & do step & get outputs TODO manage multithreading for severla modle instances and intensive step methods
            if self.config.model_configs.instantiation.parallel_execution:
            
                self._step_models_parallel() #TODO implement parallel execution
                
            else:
                for entity in self.entities:
                    model = entity['object']
                    self.outputs[entity['id']] = model._step(self.ts, self.inputs[entity['id']]) 
                  
        
            # publish outputs
            self._publish_outputs()

            self._reset() # check if reset is needed at the end of the step to manage the reset of the federate in case of training, this is because usually after the reset i want to publish the new initial conditions to let other federates receive them and then request time advance to start the new episode with the new initial conditions

            
            
            # upate storage
            self.update_storage() # TODO i want the flushing to a be a routine that is not coded here but embeded beacuse this is the method that will be overridden by other types of federates
            
            # no longer need of reset here its above
            # if self.mode == 'train' and self.reset_length is not None and self.ts > 0 and self.ts % self.reset_length == 0:
            #     self.logger.info(f"Resetting federate {self.name} at step {self.ts} for training purposes.")
            #     self._reset() # TODO manage the reset of the federate in case of training, this should be called at the end of the step after the storage update and before the time request of the next step, this is because usually after the reset i want to publish the new initial conditions to let other federates receive them and then request time advance to start the new episode with the new initial conditions
            if self.mode == 'train' and self.episode_length is not None and self.ts > 0 and self.ts % self.episode_length == 0:
                self._track_episodes() #TODO track the episodes and manage the end of episode conditions, this should be called at the end of the step after the reset to update the episode count and manage the end of episode conditions like setting a flag in the outputs or something like that, this is for now a simple episode tracking based on the number of steps but in the future it could be more complex and depend on specific conditions on the outputs or inputs
            
            # flush storage at batch size using influxdb client too slow
            # if len(self.storage['time']) >= self.batch_size:
            #     self.flush_storage()

        # flush remaining storage at the end of the simulation
        self.store_local_file() # TODO store the local file with the storage data, this is for testing purpose and will be substituted by the flushing to the database8
               
    def store_local_file(self):
        '''Store each mode partition as a separate JSON file (train / test).
        Will be substituted by flushing to database in the future.'''
        import json
        scenario_name = self.simulation_id[:-16]
        sim_id = self.simulation_id[-15:]
        repo_root = Path(__file__).resolve().parents[2]
        base_dir = os.path.join(
            str(repo_root / "results"),
            scenario_name, sim_id, self.federation_name,
        )
        os.makedirs(base_dir, exist_ok=True)

        for mode_key, partition in self.storage.items():
            # Skip empty partitions (no timesteps recorded)
            if not partition['time']:
                continue
            filename = f"{self.name}_{mode_key}_storage.json"
            file_path = os.path.join(base_dir, filename)
            with open(file_path, 'w') as f:
                json.dump(partition, f, default=str, indent=4)
            self.logger.info(f"Storage data ({mode_key}) saved to {file_path}")
    
    def request_time_advance(self):
        '''
        Request time advance from HELICS and update time_granted. Thought to be overridenn for special federates that request time in different fashions
        '''
        time_request = self.time_granted + self.time_period
        self.ts += 1 
        self.logger.debug(f"Requesting time advance to {time_request} (current granted time: {self.time_granted})")
        time_granted = h.helicsFederateRequestTime(self.federate, time_request)
        self.date_time = self.date_time + timedelta(seconds=self.real_period)
        self.time_granted = time_granted
        self.logger.info(f'-------------------- Step {self.ts} of {self.stop_time} in MODE: {self.mode} -------------------------')
        self.logger.info(f'Current realworld datetime: {self.date_time} - Current granted time for HELICS: {self.time_granted}')
    
    def _read_subscription_value(self, subid):
        """Read the current HELICS input value based on declared type."""
        if subid.type == "double":
            return subid.double
        if subid.type == "integer":
            return subid.integer
        if subid.type == "complex":
            return subid.complex
        if subid.type == "string":
            return subid.string
        if subid.type == "vector":
            return subid.vector
        if subid.type == "complex vector":
            return subid.complex_vector
        if subid.type == "boolean":
            return subid.boolean
        self.logger.error(f"Unknown HELICS input type '{subid.type}' for subscription '{subid.name}'")
        return None

    def _normalize_subscription_causality(self, raw_value):
        causality = str(raw_value or "same_step").lower()
        if causality not in {"same_step", "next_step"}:
            self.logger.warning(
                f"Unknown subscription causality '{raw_value}' in federate '{self.name}'. "
                "Falling back to 'same_step'."
            )
            return "same_step"
        return causality

    def _required_input_names(self, entity_id):
        if self.startup_sync.required_inputs:
            return list(self.startup_sync.required_inputs)
        entity = next((e for e in self.entities if e['id'] == entity_id), None)
        if entity and hasattr(entity['object'], 'state') and hasattr(entity['object'].state, 'inputs'):
            return list(entity['object'].state.inputs.keys())
        return [sub['subid'].name.split('/')[-1] for sub in self.subs if sub['entity_name'] == entity_id]

    def _get_missing_required_inputs(self):
        missing = {}
        for entity in self.entities:
            entity_id = entity['id']
            required_inputs = self._required_input_names(entity_id)
            if not required_inputs:
                continue
            available_inputs = self.inputs.get(entity_id, {})
            missing_inputs = [var for var in required_inputs if var not in available_inputs]
            if missing_inputs:
                missing[entity_id] = missing_inputs
        return missing

    def _get_startup_input_meta(self, entity_id, var_name):
        return self._last_input_meta.get(entity_id, {}).get(var_name, {})

    def _is_numeric_value(self, value):
        return isinstance(value, (int, float, np.number, complex)) and not isinstance(value, bool)

    def _flatten_values(self, value):
        if isinstance(value, np.ndarray):
            return value.flatten().tolist()
        if isinstance(value, (list, tuple)):
            flat = []
            for item in value:
                flat.extend(self._flatten_values(item))
            return flat
        return [value]

    def _has_non_finite_numeric(self, value):
        for item in self._flatten_values(value):
            if not self._is_numeric_value(item):
                continue
            if isinstance(item, complex):
                if not math.isfinite(float(item.real)) or not math.isfinite(float(item.imag)):
                    return True
            else:
                try:
                    if not math.isfinite(float(item)):
                        return True
                except (TypeError, ValueError):
                    return True
        return False

    def _contains_invalid_numeric_sentinel(self, value):
        sentinels = self.startup_sync.invalid_numeric_sentinels or []
        if not sentinels:
            return False
        for item in self._flatten_values(value):
            if not self._is_numeric_value(item) or isinstance(item, complex):
                continue
            try:
                item_f = float(item)
            except (TypeError, ValueError):
                continue
            for sentinel in sentinels:
                try:
                    sentinel_f = float(sentinel)
                except (TypeError, ValueError):
                    continue
                if math.isclose(item_f, sentinel_f, rel_tol=0.0, abs_tol=max(1e-12, abs(sentinel_f) * 1e-12)):
                    return True
        return False

    def _get_invalid_required_inputs(self):
        invalid = {}
        for entity in self.entities:
            entity_id = entity['id']
            required_inputs = self._required_input_names(entity_id)
            if not required_inputs:
                continue

            available_inputs = self.inputs.get(entity_id, {})
            for var_name in required_inputs:
                if var_name not in available_inputs:
                    continue

                reasons = []
                meta = self._get_startup_input_meta(entity_id, var_name)
                if self.startup_sync.require_updated_inputs and not meta.get('updated', False):
                    reasons.append("not_updated")

                value = available_inputs.get(var_name)
                if self.startup_sync.require_finite_numeric and self._has_non_finite_numeric(value):
                    reasons.append("non_finite")

                if self._contains_invalid_numeric_sentinel(value):
                    reasons.append("invalid_sentinel")

                if reasons:
                    invalid.setdefault(entity_id, {})[var_name] = {
                        'value': value,
                        'reasons': reasons,
                    }
        return invalid

    def _apply_startup_sync_policy(self, policy, message):
        resolved_policy = (policy or "error").lower()
        if resolved_policy == "ignore":
            self.logger.info(message)
        elif resolved_policy == "warn":
            self.logger.warning(message)
        else:
            raise RuntimeError(message)

    def _enforce_startup_input_sync(self):
        """Ensure required inputs are available before entering the first model step."""
        if not self.startup_sync.enabled:
            return
        if not self.subs:
            return

        self._receive_inputs(force_read_all=self.startup_sync.force_read_all_subscriptions)
        missing = self._get_missing_required_inputs()
        invalid = self._get_invalid_required_inputs()
        if not missing and not invalid:
            self.logger.info(f"Startup input synchronization complete for federate {self.name}.")
            return

        if missing:
            missing_message = (
                f"Startup input synchronization found missing required inputs for federate {self.name}: {missing}. "
                f"Policy={self.startup_sync.missing_inputs_policy}"
            )
            self._apply_startup_sync_policy(self.startup_sync.missing_inputs_policy, missing_message)

        if invalid:
            invalid_message = (
                f"Startup input synchronization found invalid required inputs for federate {self.name}: {invalid}. "
                f"Policy={self.startup_sync.invalid_inputs_policy}"
            )
            self._apply_startup_sync_policy(self.startup_sync.invalid_inputs_policy, invalid_message)

    def _apply_deferred_inputs(self):
        """Promote inputs marked as next_step into the active input view."""
        for entity_id, staged in self._deferred_inputs.items():
            if not staged:
                continue
            for var_name, value in staged.items():
                self.inputs.setdefault(entity_id, {})[var_name] = value
        for entity_id in self._deferred_inputs.keys():
            self._deferred_inputs[entity_id] = {}

    def _clear_deferred_inputs(self):
        """Clear staged next_step inputs (used on episode reset boundaries)."""
        for entity_id in self._deferred_inputs.keys():
            self._deferred_inputs[entity_id] = {}

    def _receive_inputs(self, force_read_all=False):
        '''
        Method to receive inputs from subscriptions and endpoints. Thought to be overridden by special federates
        TODO endpoints
        '''

        call_meta = {}
        for sub in self.subs:
            mod = sub['entity_name']
            var_name = sub['subid'].name.split('/')[-1]
            causality = sub.get('causality', 'same_step')
            is_updated = bool(sub['subid'].is_updated())
            should_read = force_read_all or is_updated
            staged_value = None
            read_value = None
            if should_read:
                read_value = self._read_subscription_value(sub['subid'])
                if read_value is not None:
                    if not force_read_all and causality == "next_step":
                        self._deferred_inputs.setdefault(mod, {})[var_name] = read_value
                        staged_value = read_value
                    else:
                        self.inputs[mod][var_name] = read_value
            # During startup force-read, consider a successfully read value as usable
            # for synchronization checks even if HELICS does not mark it "updated".
            effective_updated = is_updated or (force_read_all and read_value is not None)
            call_meta.setdefault(mod, {})[var_name] = {
                'updated': effective_updated,
                'source_updated': is_updated,
                'causality': causality,
                'value': self.inputs.get(mod, {}).get(var_name),
                'staged_value': staged_value,
            }
        self._last_input_meta = call_meta
        
        self.logger.debug(f"Received inputs at time {self.time_granted}: {pp.pformat(self.inputs)}")
                
    def _publish_outputs(self):
        '''
        Method to publish outputs to publications and endpoints. Thought to be overridden by special federates
        '''
        
        for pub in self.pubs:
            data = self.outputs[pub['entity_name']].get(pub['topic'].split('/')[-1], None)
            # self.logger.debug(f" 3333 pub_entity_name:{pub['entity_name']} - pub_var_name: {pub['topic'].split('/')[-1]} - data to publish: {data}")
            if data is not None:
                pub['pubid'].publish(data)
                self.logger.debug(f"Published output on {pub['topic']}: {data}")

    def _publish_init_state(self):
        # TODO check that this is not fucking all the normal co-simulation
        # fill the self.outputs dict with initial condition from init state only the one for which there is a pub
        # self.logger.debug(f'OUTPUTS : {self.outputs}. PUBLISHES: {self.config.connections.publishes}')
        for entity in self.entities:
            model = entity['object']
            for var_name, var_value in model.init_state.outputs.items():
                self.outputs[entity['id']][var_name] = var_value
                self.logger.debug(f'Setting initial state for var: {var_name}, var_value: {var_value}')
            
        self._publish_outputs()

    def _step_models_parallel(self):
        """Execute model steps in parallel using threads."""
        import concurrent.futures

        
        max_workers = min(len(self.entities), self.config.model_configs.instantiation.max_paraller_workers or os.cpu_count())
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(entity['object'].step, self.time_granted, self.time_period)
                for entity in self.entities
            ]
            
            # Wait for all to complete and handle any exceptions
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Model step failed: {e}")
                    raise
    
    def update_storage(self):
        mode = getattr(self, 'mode', 'test') or 'test'  # default to 'test' if mode not set
        partition = self.storage.get(mode, self.storage['test'])

        partition['time'].append(self.date_time)
        for entity in self.entities:
            entity_id = entity['id']
            model = entity['object']
            for var_name in partition['inputs'].get(entity_id, {}).keys():
                partition['inputs'][entity_id][var_name].append(self.inputs[entity_id].get(var_name, None))
            for var_name in partition['outputs'].get(entity_id, {}).keys():
                partition['outputs'][entity_id][var_name].append(self.outputs[entity_id].get(var_name, None))
            for var_name in partition['params'].get(entity_id, {}).keys():
                partition['params'][entity_id][var_name].append(model.state.parameters.get(var_name, None))

    def flush_storage(self):
        # TODO implement the flushing of the storage to the database, this method can be called at the end of the simulation or during the simulation if the batch size is reached to avoid storing too much data in memory
        try:
            bucket = 'simulation_data'
            measurement = 'sim_ts'
            time_series_data = []
            
            for ts in self.storage['time']:
                for entity in self.entities:
                    entity_id = entity['id']
                    for var_name, values in self.storage['inputs'][entity_id].items():
                        if len(values) > 0:
                            time_series_data.append({
                                'measurement': measurement,
                                'tags': {
                                    'simulation_id': self.simulation_id,
                                    'federate': entity_id.split('.')[0],
                                    'model_instance': entity_id.split('.')[1],
                                    'type': 'input',
                                    'attribute': var_name
                                },
                                'time': ts,
                                'fields': {
                                    'value': values.pop(0)  # Get the first value and remove it from the list
                                }
                            })
                    for var_name, values in self.storage['outputs'][entity_id].items():
                        if len(values) > 0:
                            time_series_data.append({
                                'measurement': measurement,
                                'tags': {
                                    'simulation_id': self.simulation_id,
                                    'federate': entity_id.split('.')[0],
                                    'model_instance': entity_id.split('.')[1],
                                    'type': 'output',
                                    'attribute': var_name
                                },
                                'time': ts,
                                'fields': {
                                    'value': values.pop(0)  # Get the first value and remove it from the list
                                }
                            })
                    for var_name, values in self.storage['params'][entity_id].items():
                        if len(values) > 0:
                            time_series_data.append({
                                'measurement': measurement,
                                'tags': {
                                    'simulation_id': self.simulation_id,
                                    'federate': entity_id.split('.')[0],
                                    'model_instance': entity_id.split('.')[1],
                                    'type': 'param',
                                    'attribute': var_name
                                },
                                'time': ts,
                                'fields': {
                                    'value': values.pop(0)  # Get the first value and remove it from the list
                                }
                            })
            self.storage['time'] = []
            if len(time_series_data) > 0:
                # Log first and last timestamps being written
                first_time = time_series_data[0]['time']
                last_time = time_series_data[-1]['time']
                self.logger.info(f"💾 Flushing {len(time_series_data)} points - Time range: {first_time} to {last_time}")
                self.infl_client.write_time_series_batch(bucket,measurement, time_series_data)
            else:
                self.logger.warning("⚠️  No data to flush (storage empty)")
        
        except Exception as e:       
            self.logger.error(f"Failed to flush storage to InfluxDB: {e}")
    
    def _reset(self):
        # check if reset time has been reached
        if self.mode == 'train' and self.reset_length is not None and self.ts > 0 and self.ts % self.reset_length == 0:
            self.reset_count += 1
            if self.reset_type == 'full':
                self.logger.info(f"Performing full reset of federate {self.name}")
                # publish init_state
                self._publish_init_state()
                # reset init_state for every model
                for entity in self.entities:
                    model = entity['object']
                    model.reset(mode=self.reset_type)

            elif self.reset_type == 'soft':
                self.logger.info(f"Performing partial reset of federate {self.name}")
                return
            
            elif self.reset_type == 'rolling':
                if self.rolling_window is None:
                    self.logger.error("Rolling window size must be specified for rolling reset type")
                    raise AssertionError("Rolling window size must be specified for rolling reset type")
                self.logger.info(f"Performing rolling reset of federate {self.name}")
                self.new_starting_point += self.rolling_window 
                for entity in self.entities:
                    model = entity['object']
                    model.reset(mode=self.reset_type, ts= self.new_starting_point)
                return

            elif self.reset_type == 'random':
                self.logger.info(f"Performing random reset of federate {self.name}")
                #  TODO: must keep track of boundaries for random conditions
            else:
                self.logger.warning(f"Unknown reset type '{self.reset_type}' specified. No reset will be performed.")
        # TODO define the reset types!
        # depending on the type:
        # update the self.outputs dict with initial condition from init state
        # publish this initial condition to the publications so that other federates can receive it if needed
        # reset the model states to the initial conditions
        
        else:
            return
        
    def _track_episodes(self):
        self.episode_count += 1
        self.logger.debug("TRACKING EPISODE COUNT: {}".format(self.episode_count))
        
        if self.episode_count >= self.n_episodes:
            self.mode = 'test'  # Switch to test mode after reaching the specified number of episodes
            self.logger.info(f"Reached {self.n_episodes} episodes. Switching to test mode.")

    def finalize(self):
        """
        Finalize and cleanup the federate.
        
        This method should be overridden by concrete federate classes
        to implement proper cleanup and finalization logic, including
        HELICS federate finalization.
        """
        self.logger.info(f'federate {self.name} finalizing')
        status = h.helicsFederateDisconnect(self.federate)
        h.helicsFederateDestroy(self.federate)
        # Flush and close the InfluxDB client AFTER HELICS is done so that
        # all buffered async writes are delivered before the process exits.
        if hasattr(self, 'infl_client') and self.infl_client:
            self.logger.info('Flushing InfluxDB write buffer...')
            self.infl_client.close()
        self.logger.info("Federate finalized\n")

 # def time_to_request(self):
    #     '''
    #     Calculate the next time to request based on the federate's timing configuration.
    #     Thought to be overridden by special federate classes
    #     '''
    #     # TODO manage the offset is possible that there is no need to add it here if it has been inserted as time property
    #     self.ts += 1
    #     return self.time_granted + self.time_period


if __name__ == "__main__":
    import logging
    
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('test_federate')
    
    # Test configuration structure matching the BaseFederate expectations
    test_config = {
        "name": "test_federate",
        "core_name": "test_core",
        "core_type": "zmq",
        
        "timing_configs": {
            "time_period": 1.0,          # Time step in seconds
            "real_period": 1.0,          # Real time step passed to models
            "offset": 0.0,               # Time offset
            "time_stop": 10.0            # Simulation stop time
        },
        
        "flags": {
            "uninterruptible": True,     # HELICS flag example
            "wait_for_current_time_update": False
        },
        
        "model_configs": {
            "time_step": 1.0,            # Will be overwritten by real_period
            "instantiation": {
                "model_name": "example_model",
                "n_instances": 1,
                "prefix": "test_model"
            },
            "parameters": {
                "mass": 2.0,
                "stiffness": 10.0,
                "damping": 0.5
            },
            "init_state": {
                "position": 0.0,
                "velocity": 0.0,
                "acceleration": 0.0
            },
            "user_defined": {
                "solver": "euler",
                "integrator": "fixed-step"
            }
        },
        
        "connections": {
            "publishes": [
                {
                    "key": "position",
                    "type": "double",
                    "unit": "m",
                    "info": "Mass position output"
                },
                
            ],
            "subscribes": [
                {
                    "key": "force",
                    "type": "double",
                    "unit": "N",
                    "info": "External force input",
                    "targets": ["source_federate.source_model/force"]
                }
            ],
            "endpoints": []  # No endpoints for this simple test
        }
    }
    
    print("Testing BaseFederate with configuration:")
    pp.pprint(test_config)
    print("\n" + "="*50 + "\n")
    
    try:
        # Create and test the federate
        federate = BaseFederate("test_federate", test_config, logger)
        print(f"Created federate: {federate.name}")
        
        # Test initialization (this will try to register with HELICS)
        print("Attempting to initialize federate...")
        print("Note: This will fail without a HELICS broker running")
        print("To run with HELICS, start a broker first with:")
        print("  helics_broker --federates=1 --loglevel=warning")
        
        # Uncomment the following lines to actually test initialization
        federate.initialize()
        federate.run()
        print("Federate initialized successfully!")
        print(f"Entities created: {list(federate.entities.keys())}")
        federate.finalize()
        
    except Exception as e:
        print(f"Error during federate testing: {e}")
        print("This is expected if no HELICS broker is running")
    
    print("\nTest configuration structure is valid!")
