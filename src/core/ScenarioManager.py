"""
ScenarioManager.py

Orchestrates HELICS federations, managing the lifecycle of brokers and federate processes.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

import time
import subprocess
import signal
import os
from pathlib import Path
import shutil
import atexit
import threading
import logging
import json
import pprint
import socket
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Any
from utils.config_dataclasses import BrokerConfig, FederationConfig, FederateConfig, RLFederateConfig, FedTimingConfig, FedFlags, FedConnections, MemoryConfig, FedPublication, FedSubscription, StartupSyncConfig
from utils.config_reader import read_scenario_config
from utils.logging_config import FederationLogger
from utils.redis_client import RedisClient
from utils.ports import redis_port as default_redis_port, helics_port_range
from core.remote_executor import RemoteExecutor

pp = pprint.PrettyPrinter(indent=4)







class ScenarioManager:
    """
    Manages HELICS federation lifecycle including broker and federate processes.
    
    This class handles:
    - Starting and stopping HELICS broker
    - Managing federate subprocesses
    - Graceful shutdown and emergency cleanup
    - Process monitoring and error handling
    """

    # How long to let brokers exit on their own after the last federate is gone,
    # before cleanup terminates them (see _monitor_processes).
    BROKER_SHUTDOWN_GRACE_S = 30

    # How long a freshly spawned broker gets to start accepting connections
    # before startup is declared failed (see _wait_for_broker_listening).
    BROKER_STARTUP_TIMEOUT_S = 15

    # Broker hosts that mean "this machine only". A broker advertised on any other
    # address must bind all interfaces to be reachable (see _broker_binds_externally).
    LOOPBACK_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})

    def __init__(self, config_path):
        """
        Initialize the ScenarioManager with configuration.
        
        Args:
            config_path (str): Path to the scenario configuration file
            
        Raises:
            ValueError: If federation configuration is invalid
        """
        
        # READING comnfiguration and setting up internal data structures
        self.config = read_scenario_config(config_path)
        self.scenario_name = self.config.name


        # setting up logging system & metrics tracking
        self._logging_sys_setup()
        self._setup_metrics()
        self.simulation_id = self.logger_system.simulation_id # this id is useful for logs and results correlation, we set it here after the logging system is set up to ensure it's available for all logs and metadata storage

        # scenario timings will be initialized during scenario setup
        self.start_time = datetime.fromisoformat(self.config.start_time)
        self.end_time = datetime.fromisoformat(self.config.end_time)
        self.duration_time = None


        # Process management attributes
        self.broker_processes: List[subprocess.Popen] = []
        self.federate_processes: List[subprocess.Popen] = []

        # Remote execution: alias -> RemoteExecutor, populated by _setup_remote_execution
        # only when the scenario has ≥1 federate with a `host:` key. Empty dict = fully
        # local scenario, everything remote-related is a no-op.
        self.remote_executors: Dict[str, RemoteExecutor] = {}
        
        # Cleanup management
        self._cleanup_done = False
        self._cleanup_lock = threading.Lock()
        
        self._hierarchy_broker_config = None  # Set by _normalize_broker_and_core_configs when >1 federation
        
        # Redis client & key (will be initialized during scenario setup)
        self.redis_client = None
        self.redis_key = None
        self.redis_url = None
        self.dynamic_catalog_index_key = None

       
        # register various graceful and emergency cleanups
        atexit.register(self._emergency_cleanup)  # Register cleanup function to run on exit
        signal.signal(signal.SIGINT, self._signal_handler) # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler) # Register signal handlers for graceful shutdown
        
        # Mark initialization as complete
        self.metrics['initialization_end'] = datetime.now()
        initialization_duration = (self.metrics['initialization_end'] - self.metrics['initialization_start']).total_seconds()
        self.metrics['phase_durations']['initialization_duration'] = initialization_duration
        self.logger.info(f"ScenarioManager initialization completed in {initialization_duration:.3f} seconds")

    def _logging_sys_setup(self):
        # Initialize logging system    
        self.logger_system = FederationLogger(self.scenario_name)
        self.logger = self.logger_system.setup_manager_logger(self.config.log_level)
        # Log initialization
        self.logger.info(f"Initializing ScenarioManager for scenario: {self.scenario_name}")
        self.logger.info(f"Log directory: {self.logger_system.scenario_log_dir}")
    
    def _setup_metrics(self):
        self.metrics = {
            'initialization_start': datetime.now(),
            'initialization_end': None,
            'setup_start': None,
            'setup_end': None,
            'simulation_start': None,
            'simulation_end': None,
            'cleanup_start': None,
            'cleanup_end': None,
            'total_duration': None,
            'phase_durations': {
                'initialization_duration': None,
                'setup_duration': None,
                'simulation_duration': None,
                'cleanup_duration': None
            },
            'process_counts': {
                'brokers_started': 0,
                'federates_started': 0,
                'brokers_completed': 0,
                'federates_completed': 0,
                'brokers_failed': 0,
                'federates_failed': 0
            }
        }

    def _signal_handler(self, signum, frame):
        """
        Handle signals for graceful shutdown.
        
        Args:
            signum (int): Signal number received
            frame: Current stack frame (unused)
        """
        print(f"\nReceived signal {signum}. Shutting down scenario...")
        self._emergency_cleanup()
        exit(0)

    def _emergency_cleanup(self, success= False):
        """
        Emergency cleanup function - terminates all subprocesses.
        
        This function is called on exit, signal reception, or exceptions.
        It ensures all federate and broker processes are properly terminated
        and temporary files are cleaned up.
        
        Uses a lock to ensure cleanup runs only once.
        """
        with self._cleanup_lock:
            if self._cleanup_done:
                return
                
            self._cleanup_done = True
            self.metrics['cleanup_start'] = datetime.now()
            self.logger.info("Emergency cleanup: Terminating all subprocesses...")
        
        # Kill all federate processes
        for process in getattr(self, 'federate_processes', []):
            if process and process.poll() is None:  # Still running
                try:
                    if hasattr(process, 'pid'):
                        self.logger.info(f"Terminating federate process PID: {process.pid}")
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            self.logger.warning(f"Force killing federate process PID: {process.pid}")
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError, AttributeError):
                    pass
        
        # Kill all broker processes
        for process in getattr(self, 'broker_processes', []):
            if process and process.poll() is None:  # Still running
                try:
                    self.logger.info(f"Terminating broker process PID: {process.pid}")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"Force killing broker process PID: {process.pid}")
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass

        # Distributed SSH: sweep any lingering remote federates and close ssh
        # control masters. Wrapped internally so teardown never raises.
        self._cleanup_remote_execution()

        self.metrics['cleanup_end'] = datetime.now()
        
        # Calculate cleanup phase duration
        if 'cleanup_start' in self.metrics and self.metrics['cleanup_start']:
            cleanup_delta = (self.metrics['cleanup_end'] - self.metrics['cleanup_start']).total_seconds()
            self.metrics['phase_durations']['cleanup'] = cleanup_delta
        
        # Calculate total duration
        if self.metrics['simulation_start']:
            end_time = self.metrics['cleanup_end']
            self.metrics['total_duration'] = (end_time - self.metrics['simulation_start']).total_seconds()
        
        # Delete redis key for this simulation
        if self.redis_client and self.redis_key:
            self.redis_client.delete(self.redis_key)

        if success:
            self.logger.info("Scenario execution completed successfully.")
            print("Scenario execution completed successfully.")
            print("Cleanup completed")
        else:
            self.logger.warning("Scenario execution did not complete successfully.")
            print("Scenario execution did NOT complete successfully.")
            print("Emergency cleanup completed")


        self._log_execution_summary()
        
        paths = self.logger_system.get_log_paths()
        print("\n=== LOGS AVAILABLE ===")
        print(f"Log directory: {paths['scenario_log_dir']}")
        print(f"Manager logs: {paths['manager_logs']}")
        print(f"Broker logs: {paths['broker_logs']}")
        print(f"Federate logs: {paths['federate_logs']}")

    def _setup_redis_client(self):
        """Initialize Redis client for configuration distribution."""
        try:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = default_redis_port()
            redis_db = int(os.getenv('REDIS_DB', '0'))

            self.redis_client = RedisClient(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                logger=self.logger
            )
            self.logger.info(f"✓ Redis client initialized at {redis_host}:{redis_port}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis client: {e}")
            self.logger.error("Redis is required for configuration distribution. Please ensure Redis is running.")
            raise RuntimeError(f"Redis initialization failed: {e}")
        
    def start_scenario(self):
        """
        Run scenario with proper subprocess management.
        
        This method sets up the federation and monitors all processes until completion
        or until an error occurs. Cleanup is always performed regardless of outcome.
        """
        self.metrics['simulation_start'] = datetime.now()
        self._simulation_start_real_time = time.time()

        # Assigned before the try so the finally's _emergency_cleanup(success=...) is
        # always bound — e.g. a SIGINT during setup triggers the signal handler's
        # exit(0) (SystemExit, caught by neither except below), which still unwinds
        # through finally.
        success = False

        try:
            # setting up scenario (scenario config, federations, brokers federates)
            self._setup_scenario()

            # Monitor federation
            self._monitor_processes() # TODO: better understand the monitoring part

            # Distributed SSH: pull each remote federate's results + logs back to the
            # manager before the run is declared complete. No-op for local scenarios.
            # Runs before the exit-code verdict below on purpose: a failed run is
            # exactly when the remote logs are needed to diagnose it.
            self._collect_remote_results()

            self.metrics['simulation_end'] = datetime.now()
            duration = (self.metrics['simulation_end'] - self.metrics['simulation_start']).total_seconds()
            self.metrics['phase_durations']['simulation_duration'] = duration

            # _monitor_processes returning means every federate exited — not that any
            # of them succeeded. Without this check a run where all federates died on
            # startup still reported "completed successfully".
            failed = self.metrics['process_counts']['federates_failed']
            if failed:
                started = self.metrics['process_counts']['federates_started']
                raise RuntimeError(
                    f"{failed}/{started} federate(s) exited with a non-zero code — see the per-federate "
                    f"logs under {self.logger_system.scenario_log_dir}/federates/"
                )

            self.logger.info(f"Simulation completed in {duration:.3f} seconds")
            success = True

        except KeyboardInterrupt:
            self.logger.warning("\nKeyboard interrupt received. Shutting down...")
            success = False
        except Exception as e:
            self.logger.error(f"Scenario error: {e}")
            success = False
        finally:
            # Always cleanup (but only once due to lock mechanism)
            self.logger.info("Scenario execution finished. Performing cleanup...")
            self._emergency_cleanup(success=success)

    def _setup_scenario(self):
        """
        Set up the scenario by initializing all federations. 
        """
        self.metrics['setup_start'] = datetime.now()
        self.logger.info("Setting up scenario...")

        # check RL task presence
        if self.config.reinforcement_learning_config:
            self.logger.info("Reinforcement Learning task detected in scenario configuration")
            #  this opens to 4 training possibilities :
            # 2. Training Online 
            # 3. Training Offline 
            # 4. Training Offline + Training Online 
            # All of these possibilities could be folowed by a testing part

            _run = self.config.reinforcement_learning_config.run
            if _run.train is not None:
                if _run.mode == "online":
                    self.logger.info("Setting up Online TRAINING!")
                    # modify config file adding rl agent as a federate in a new federation
                    self._modify_config_for_online_training()

                # TODO: offline training logic
                elif _run.mode == "offline":
                    self.logger.info("Starting Offline TRAINING!")
                    # do offline learning
                    self._offline_learning()

                # TODO mixed training logic
                elif _run.mode == "mixed":
                    self.logger.info("Starting Mixed training loop (Offline + Online)...")
                    # do offline learning
                    self._offline_learning
                    # prepare config for online training
                    self._modify_config_for_online_training()

                else:
                    self.logger.warning(f"Unknown run mode specified: {_run.mode}. Proceeding without training.")

            # ******************+TESTING PART****************** TODO
            if _run.test is not None:
                self.logger.info("Setting up TESTING !")
                self._modify_config_for_testing() # TODO: this method should modify the config in a way that the trained agent is loaded and ready to be tested in the co-simulation execution, we can have different testing modalities here as well (e.g. deterministic, stochastic, with or without rendering, etc.)
                
            

            # run normal scenario setup with new config file
            self._setup_classic_scenario() # this will start the co-simulation with the new config that includes the RL agent as a federate, the RL agent will then start its online training loop and interact with the environment during the co-simulation execution

        # THIS is the setup section for normal co-simulation (plain no RL or other sovrastructures) 
        else:
            self.logger.info("No Reinforcement Learning task detected in scenario configuration, proceeding with standard setup")
            self._setup_classic_scenario()

        


        # metrics and logging
        self.metrics['setup_end'] = datetime.now()
        setup_duration = (self.metrics['setup_end'] - self.metrics['setup_start']).total_seconds()
        self.metrics['phase_durations']['setup_duration'] = setup_duration
        self.logger.info(f"Scenario setup completed in {setup_duration:.3f} seconds")
    
    def _setup_results_folder(self):
        repo_root = Path(__file__).resolve().parents[2]
        results_dir = str(repo_root / "results")
        sim_id = self.simulation_id[-15:]
        scenario_results_dir = os.path.join(results_dir, self.scenario_name, sim_id)
        os.makedirs(scenario_results_dir, exist_ok=True)
        # scenario_metadata = {} to be filled
        # TODO enrich medatada
        scenario_metadata = {
            "scenario_name": self.scenario_name,
            "simulation_id": self.simulation_id,
            "start_date": self.start_time.isoformat(),
            "end_date": self.end_time.isoformat(),
            "duration_seconds": self.duration_time,
        }
        json.dump(scenario_metadata, open(os.path.join(scenario_results_dir, 'metadata.json'), 'w'), indent=4)
        for federation_name, federation_conf in self.config.federations.items():
            federation_result_dir = os.path.join(scenario_results_dir, federation_name)
            os.makedirs(federation_result_dir, exist_ok=True)
        
    def _get_rl_period(self, controlled_models):
        periods=[]
        for mod in controlled_models:
            splitted = mod.split('.')
            period = self.config.federations[splitted[0]].federate_configs[splitted[1]].timing_configs.real_period 
            periods.append(int(period))
        
        rl_period = min(periods)
        return rl_period

    def _get_rl_pubsubs(self, rl_task):
        publications = []
        subscriptions = []

        # All observations (state + extra role) are subscribed; role filtering happens in RL_Federate.
        for obs, spec in rl_task.environment.observations.items():
            pubs_model = self.config.federations[obs.split('.')[0]].federate_configs[obs.split('.')[1]].connections.publishes
            for p in pubs_model:
                if p.key==obs.split('.')[-1]:
                    targets = [f'{obs.split(".")[1]}.{obs.split(".")[2]}/{obs.split(".")[3]}']
                    causality = self._normalize_subscription_causality(spec.causality)
                    subscriptions.append(
                        FedSubscription(
                            key=obs.split('.')[-1],
                            type=p.type,
                            units=p.units,
                            targets=targets,
                            causality=causality,
                        )
                    )

        for action in rl_task.environment.actions:
            subs_model = self.config.federations[action.split('.')[0]].federate_configs[action.split('.')[1]].connections.subscribes
            for s in subs_model:
                if s.key==action.split('.')[-1]:
                    publications.append(FedPublication(key=action.split('.')[-1], type=s.type, units=s.units))
                    # NOTABENE QUI STO MODIFICANDO LA SUBSCRIPTION DEL MODELLO CONTROLLATO NON DELL'AGENTE
                    s.targets = [f'rl_agent/{action.split(".")[-1]}']  # Assuming the RL agent will publish to a key like 'rl_agent/<action_key>'

        self.logger.debug(f"RL Agent Publications: {publications}")
        self.logger.debug(f"RL Agent Subscriptions: {subscriptions}")
        
        return publications, subscriptions
    
    def _get_rl_controlled_models(self):
        rl_task = self.config.reinforcement_learning_config
        controlled_models = [action.rsplit('.',2)[0] for action in rl_task.environment.actions]
        controlled_models = set(controlled_models)
        return controlled_models

    def _build_rl_reset_observation_defaults(self, rl_task):
        explicit_defaults = {
            obs: spec.reset_default
            for obs, spec in rl_task.environment.observations.items()
            if spec.reset_default is not None
        }
        reset_defaults = dict(explicit_defaults)
        all_obs = list(rl_task.environment.observations)

        for obs_key in all_obs:
            if obs_key in reset_defaults:
                continue
            parts = obs_key.split('.')
            if len(parts) < 4:
                self.logger.warning(f"Cannot derive reset default for malformed observation key '{obs_key}'.")
                reset_defaults[obs_key] = 0.0
                continue

            federation_name = parts[0]
            federate_name = parts[1]
            var_name = parts[-1]
            federation = self.config.federations.get(federation_name)
            if not federation:
                self.logger.warning(f"Cannot derive reset default: unknown federation '{federation_name}' for '{obs_key}'.")
                reset_defaults[obs_key] = 0.0
                continue

            federate_cfg = federation.federate_configs.get(federate_name)
            if not federate_cfg or not federate_cfg.model_configs:
                self.logger.warning(f"Cannot derive reset default: missing model config for '{obs_key}'.")
                reset_defaults[obs_key] = 0.0
                continue

            init_state = getattr(federate_cfg.model_configs, 'init_state', {}) or {}
            if var_name not in init_state:
                publishes = federate_cfg.connections.publishes if federate_cfg.connections else []
                pub_keys = {pub.key for pub in publishes}
                if var_name in pub_keys:
                    reset_defaults[obs_key] = 0.0
                    self.logger.warning(
                        f"No init_state value found for observed variable '{obs_key}'. "
                        "Using fallback reset default 0.0."
                    )
                    continue
                self.logger.warning(
                    f"No init_state/publication default found for observed variable '{obs_key}'. "
                    "Using fallback reset default 0.0."
                )
                reset_defaults[obs_key] = 0.0
                continue

            reset_defaults[obs_key] = init_state[var_name]

        return reset_defaults

    def _offline_learning(self):
        # TODO: to be implemented
        # instantiate the agent
        # run the offline loop from agent passing the datasource
        # include possibility to run federation as datasource..
        pass
    
    def _create_RL_federation(self):
        rl_task = self.config.reinforcement_learning_config
        self.logger.info("Creating federation for RL + federate configuration for RL agent...")
        
        controlled_models = self._get_rl_controlled_models()
        rl_period = self._get_rl_period(controlled_models)  # TODO: the period must be the one of controlled model if multiple controlled models now i'm taking the minimum correct?!?!?
        
        # TODO: the timings is automatically done at classic scenario setup but i have to chek how about possible federate offset and flags
        publications, subscriptions =self._get_rl_pubsubs(rl_task)

        broker_config = BrokerConfig(core_type=None, port=None, log_level=self.config.log_level, federates=1) # TODO: this is hardcoded, we should understand how to set this in a dynamic way based on the existing brokers and the expected communication needs of the RL agent
        
        fed_configs={
                                            'rl_agent': RLFederateConfig(
                                                type='rl',
                                                id='',
                                                name='rl_agent',
                                                timing_configs=FedTimingConfig(real_period=rl_period),
                                                connections=FedConnections(publishes=publications, subscribes=subscriptions),
                                                log_level=self.config.log_level,
                                                core_type=None,
                                                # rl_federation is created at runtime, after the
                                                # scenario validator that propagates memory_config
                                                # to federates — so inject it here (BaseFederate
                                                # init needs memory_config.batch_size).
                                                memory_config=self.config.memory_config,
                                            )
                                        }
        federation_conf= FederationConfig(broker_config=broker_config,
                                        federate_configs=fed_configs,
                                        name="rl_federation",
                                        )
        # All observations (state + extra role) are subscribed inputs the RL federate must wait
        # for at startup.
        rl_required_inputs = [obs.split('.')[-1] for obs in rl_task.environment.observations]
        federation_conf.federate_configs['rl_agent'].startup_sync = StartupSyncConfig(
            required_inputs=sorted(set(rl_required_inputs))
        )

        # add the knowledge of controlled model using the model name from model catalog to retrieve normalization boundaries for each attr
        real_controlled_models = {}
        for cm in rl_task.environment.actions:
            mod_name = self.config.federations[cm.split('.')[0]].federate_configs[cm.split('.')[1]].model_configs.instantiation.model_name
            real_controlled_models[cm]= mod_name
        federation_conf.federate_configs['rl_agent'].controlled_models = real_controlled_models

        # observed_models covers every observation key (state + extra) so catalog specs and
        # reset defaults resolve for all; additional_observed_models tracks the extra-role
        # subset (visible to reward/logging, excluded from the policy observation space).
        real_observed_models = {}
        real_add_observed_models = {}
        for cm, spec in rl_task.environment.observations.items():
            mod_name = self.config.federations[cm.split('.')[0]].federate_configs[cm.split('.')[1]].model_configs.instantiation.model_name
            real_observed_models[cm]= mod_name
            if spec.role == "extra":
                real_add_observed_models[cm] = mod_name
        federation_conf.federate_configs['rl_agent'].observed_models = real_observed_models
        if real_add_observed_models:
            federation_conf.federate_configs['rl_agent'].additional_observed_models = real_add_observed_models

        federation_conf.federate_configs['rl_agent'].reset_observation_defaults = self._build_rl_reset_observation_defaults(rl_task)
            

        # saving to self.config the new federation RL config
        self.config.federations['rl_federation'] = federation_conf 



    def _modify_config_for_testing(self):
        # TODO this method should modify the config in a way that the trained agent is loaded and ready to be tested in the co-simulation execution, we can have different testing modalities here as well (e.g. deterministic, stochastic, with or without rendering, etc.)
        
        self.logger.debug(f"Modifying scenario configuration for testing...\n initial start_time:{self.start_time} \n initial end_time: {self.end_time}")
        
        rl_task = self.config.reinforcement_learning_config
        controlled_models = self._get_rl_controlled_models()
        rl_period = self._get_rl_period(controlled_models)
        additional_period = rl_task.run.test.total_steps * rl_period

        has_online_training = (
            self.config.reinforcement_learning_config.run.train is not None
            and self.config.reinforcement_learning_config.run.mode != "offline"
        )

        if has_online_training:
            # Online/mixed: training already extended end_time, append test duration
            self.end_time = self.end_time + timedelta(seconds=additional_period)
            self.logger.debug(f"Online training detected, extending end_time by additional {additional_period} seconds for testing")
        else:
            # Offline or no training: create RL federation and set test-only duration from start
            self._create_RL_federation()
            self.end_time = self.start_time + timedelta(seconds=additional_period)
            self.logger.debug(f"Only test case! Setting end_time to {self.end_time.isoformat()} for testing")

        self.config.end_time = self.end_time.isoformat()

        self.logger.debug(f"Modified scenario configuration for testing...\n new start_time:{self.start_time} \n new end_time: {self.end_time}")


    def _modify_config_for_online_training(self):
        
        self.logger.debug(f"Modifying scenario configuration for online training\n initial start_time:{self.start_time} \n initial end_time: {self.end_time}")
        self._create_RL_federation()
        controlled_models = self._get_rl_controlled_models()
        rl_period = self._get_rl_period(controlled_models)
        # modifying simulation length to accomodate training duration TODO moving into setting timing vars
        rl_task = self.config.reinforcement_learning_config
        additional_period = rl_task.run.train.total_steps * rl_period
        self.end_time = self.start_time + timedelta(seconds=additional_period)
        self.config.end_time = self.end_time.isoformat() #probably redundant
        self.logger.debug(f"Modified scenario configuration for online training\n new start_time:{self.start_time} \n new end_time: {self.end_time}")

    def _has_remote_federates(self) -> bool:
        """True if any federate in the scenario sets `host:` (distributed SSH spawning)."""
        return any(
            getattr(fed, 'host', None)
            for federation in self.config.federations.values()
            for fed in federation.federate_configs.values()
        )

    def _setup_remote_execution(self):
        """Preflight-verify and deploy code to every remote machine used by ≥1 federate.

        Must run before any broker starts: a machine that fails preflight aborts the
        whole scenario before any local process (broker or federate) has been spawned.
        No-op when the scenario has no `host:`-tagged federates — `self.remote_executors`
        stays empty and every later remote-dispatch check short-circuits to the existing
        local code path, unchanged.
        """
        if not self._has_remote_federates():
            return

        deployment = self.config.deployment
        control_dir = str(self.logger_system.scenario_log_dir / 'ssh_control')
        project_root = str(Path(__file__).resolve().parents[2])

        used_aliases = {
            fed.host
            for federation in self.config.federations.values()
            for fed in federation.federate_configs.values()
            if getattr(fed, 'host', None)
        }

        try:
            for alias in used_aliases:
                machine_conf = deployment.machines[alias]
                executor = RemoteExecutor(
                    machine_alias=alias,
                    machine_conf=machine_conf,
                    manager_address=deployment.manager_address,
                    logger=self.logger,
                    control_dir=control_dir,
                )
                self.logger.info(f"Opening ssh control master for remote machine '{alias}' ({machine_conf.host})...")
                executor.open_master()
                self.remote_executors[alias] = executor

            scenario_log_dir_rel = str(self.logger_system.scenario_log_dir)
            for alias, executor in self.remote_executors.items():
                self.logger.info(f"Running preflight checks on remote machine '{alias}'...")
                executor.verify()
                self.logger.info(f"Deploying code to remote machine '{alias}'...")
                executor.deploy(project_root)

                # federate_launcher.py's FileHandler doesn't create parent dirs (matches
                # local behavior, where FederationLogger pre-creates them) — the remote
                # federates/ subdir must exist before any federate on this machine spawns.
                machine_conf = deployment.machines[alias]
                remote_federates_dir = os.path.join(machine_conf.workdir, scenario_log_dir_rel, 'federates')
                rc, _, err = executor.run(['mkdir', '-p', remote_federates_dir], timeout=10)
                if rc != 0:
                    raise RuntimeError(f"[{alias}] failed to create remote federate log dir: {err.strip()}")
        except Exception:
            # Any single machine's preflight/deploy failure aborts the whole scenario.
            # Close whatever control masters were already opened so we don't leak sockets.
            for executor in self.remote_executors.values():
                executor.close()
            self.remote_executors = {}
            raise

        self.logger.info(f"Remote execution ready: {len(self.remote_executors)} machine(s) verified and deployed.")

    def _setup_classic_scenario(self):
        # Set up of timings, Synchronization variables
        self._scenario_setup_timing_vars()
        # Initialize all federation and start the processes Spawn
        # This also automatically starts the co-simulation
        # 2 options - local or multi computer
        self._setup_results_folder()
        # Distributed SSH spawning: verify + deploy to remote machines before any
        # broker/federate process starts locally. No-op for fully local scenarios.
        self._setup_remote_execution()
        if self.config.multi_computer and self.config.multi_computer_config:
            self._setup_multi_computer_scenario() # TODO: multi computer must be implemented
        else:
            # Resolve broker/core/protocol settings for every federation and propagate
            # them to each federate's core, so the YAML can be minimal (defaults filled
            # in) or fully explicit (validated for consistency) without ever producing
            # a broken HELICS wiring (mismatched protocols, missing broker addresses,
            # clashing ports/core names, wrong federate counts, ...).
            self._normalize_broker_and_core_configs()
            # Automatically add a main (hierarchy) broker when there is more than one
            # federation, so federations can talk to each other (e.g. the RL case).
            # (for now with this method only 2 level hierarchy is supported)
            if self._hierarchy_broker_config is not None:
                self._start_local_hierarchy_broker(self._hierarchy_broker_config)
            # uploading config for all federates
            self._upload_config_on_redis()
            self._enrich_dynamic_catalog_metadata()
            self._assert_catalog_ready()
            for federation_name, federation in self.config.federations.items():
                self._setup_local_federation(federation_name, federation)
          
    def _upload_config_on_redis(self):
        
        self._setup_redis_client()
         # Push full scenario configuration:
        self.redis_key = f"cosim:config:{self.simulation_id}"
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = default_redis_port()
        redis_db = int(os.getenv('REDIS_DB', '0'))
        self.redis_url = os.getenv('REDIS_URL', f'redis://{redis_host}:{redis_port}/{redis_db}')
        # Kept for _redis_url_for: remote federates reach Redis at manager_address on the
        # same port/db the manager itself uses (Redis always runs on the manager machine).
        self._redis_port = redis_port
        self._redis_db = redis_db

        config_dict = self.config.model_dump()
        self.logger.info(f"Storing scenario configuration in Redis. config={pp.pformat(config_dict)}")
        if self.redis_client:
                # Store config in Redis with 1 hour expiration
                success = self.redis_client.set_json(self.redis_key, config_dict, expire_seconds=3600)
                if not success:
                    raise RuntimeError("Failed to store config in Redis!")
                
                self.logger.debug(f"Stored config in Redis at key: {self.redis_key}")
        else:
                raise RuntimeError("Redis client not initialized")

    def _catalog_override_key(self, federation_name, federate_name, instance_id):
        return f"cosim:catalog_override:{self.simulation_id}:{federation_name}:{federate_name}:{instance_id}"

    def _resolve_csv_path_for_base_reader(self, csv_path):
        if os.path.isabs(csv_path):
            return csv_path
        base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        return os.path.abspath(os.path.join(base_dir, csv_path))

    def _to_parameter_value(self, raw_value, instance_id):
        if isinstance(raw_value, list):
            if not raw_value:
                return None
            if instance_id < len(raw_value):
                return raw_value[instance_id]
            return raw_value[-1]
        return raw_value

    def _build_dynamic_base_csv_specs(self, federate_conf, instance_id):
        params = federate_conf.model_configs.parameters or {}
        csv_path = self._to_parameter_value(params.get("csv_path"), instance_id)
        skip_rows = self._to_parameter_value(params.get("skip_rows", 0), instance_id)
        if csv_path is None:
            raise ValueError(f"Missing csv_path for dynamic model in federate '{federate_conf.name}'")

        full_csv_path = self._resolve_csv_path_for_base_reader(str(csv_path))
        if not os.path.exists(full_csv_path):
            raise FileNotFoundError(f"CSV file not found for dynamic model: {full_csv_path}")

        output_cols = [p.key for p in federate_conf.connections.publishes]
        input_cols = [s.key for s in federate_conf.connections.subscribes]
        required_cols = sorted(set(output_cols + input_cols))
        if not required_cols:
            raise ValueError(
                f"Dynamic model '{federate_conf.model_configs.instantiation.model_name}' has no pub/sub columns in federate '{federate_conf.name}'"
            )

        df = pd.read_csv(full_csv_path, skiprows=int(skip_rows), usecols=required_cols)

        def _python_scalar(value):
            if hasattr(value, "item"):
                return value.item()
            return value

        def _spec_for_column(col_name):
            series = df[col_name]
            if pd.api.types.is_integer_dtype(series):
                ptype = "int"
            elif pd.api.types.is_numeric_dtype(series):
                ptype = "float"
            else:
                ptype = "string"

            min_value = _python_scalar(series.min()) if pd.api.types.is_numeric_dtype(series) else None
            max_value = _python_scalar(series.max()) if pd.api.types.is_numeric_dtype(series) else None
            default_value = _python_scalar(series.iloc[0]) if len(series) > 0 else None

            return {
                "type": ptype,
                "default_value": default_value,
                "description": f"Dynamic spec inferred from CSV column '{col_name}'",
                "unit": "-",
                "min_value": min_value,
                "max_value": max_value,
                "required": True,
                "tags": ["dynamic", "csv"],
            }

        outputs = {col: _spec_for_column(col) for col in output_cols}
        inputs = {col: _spec_for_column(col) for col in input_cols}
        return {
            "inputs": inputs,
            "outputs": outputs,
            "model_name": federate_conf.model_configs.instantiation.model_name,
            "source": {
                "csv_path": full_csv_path,
                "skip_rows": int(skip_rows),
            },
        }

    def _enrich_dynamic_catalog_metadata(self):
        """Populate scenario-scoped Redis overrides for models with dynamic IO metadata."""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")

        self.dynamic_catalog_index_key = f"cosim:catalog_override_index:{self.simulation_id}"
        override_index = []

        for federation_name, federation_conf in self.config.federations.items():
            for federate_name, federate_conf in federation_conf.federate_configs.items():
                model_name = federate_conf.model_configs.instantiation.model_name if federate_conf.model_configs else None
                #  TODO this is the discriminant for dynamic enrichment (when i do not know attr before because like in csv reader depends from specific files and i'm using a generalized model)
                #  need to use something different than model_name something like a category (dynamic_interfaces)
                if model_name != "base_csv_reader":
                    continue

                n_instances = federate_conf.model_configs.instantiation.n_instances
                for instance_id in range(n_instances):
                    payload = self._build_dynamic_base_csv_specs(federate_conf, instance_id)
                    override_key = self._catalog_override_key(federation_name, federate_name, instance_id)
                    success = self.redis_client.set_json(override_key, payload, expire_seconds=3600)
                    if not success:
                        raise RuntimeError(f"Failed to store dynamic metadata at Redis key: {override_key}")
                    override_index.append({
                        "key": override_key,
                        "model_name": model_name,
                        "federation": federation_name,
                        "federate": federate_name,
                        "instance": instance_id,
                    })
                    self.logger.info(f"Dynamic catalog override written: {override_key}")

        idx_ok = self.redis_client.set_json(self.dynamic_catalog_index_key, {"overrides": override_index}, expire_seconds=3600)
        if not idx_ok:
            raise RuntimeError(f"Failed to store dynamic catalog override index at key: {self.dynamic_catalog_index_key}")
        self.logger.info(f"Dynamic catalog enrichment completed with {len(override_index)} override(s)")

    def _assert_catalog_ready(self):
        """Fail fast if required IO specs for RL spaces are missing before federate startup."""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")

        index = self.redis_client.get_json(self.dynamic_catalog_index_key) if self.dynamic_catalog_index_key else {"overrides": []}
        override_keys = {item.get("key") for item in (index or {}).get("overrides", [])}

        for federation_name, federation_conf in self.config.federations.items():
            for federate_name, federate_conf in federation_conf.federate_configs.items():
                if not getattr(federate_conf, "observed_models", None) and not getattr(federate_conf, "controlled_models", None):
                    continue

                for attr_key, model_name in (federate_conf.observed_models or {}).items():
                    self._assert_attr_spec_available(attr_key, model_name, "outputs", override_keys)
                for attr_key, model_name in (federate_conf.controlled_models or {}).items():
                    self._assert_attr_spec_available(attr_key, model_name, "inputs", override_keys)

    def _assert_attr_spec_available(self, attr_key, model_name, io_section, override_keys):
        parts = attr_key.split(".")
        if len(parts) < 4:
            raise ValueError(f"Invalid attribute key format '{attr_key}'. Expected 'federation.federate.instance.attr'")
        federation_name, federate_name, instance_raw = parts[0], parts[1], parts[2]
        var_name = parts[-1]

        override_key = self._catalog_override_key(federation_name, federate_name, int(instance_raw))
        if override_key in override_keys:
            override_payload = self.redis_client.get_json(override_key) or {}
            if var_name in (override_payload.get(io_section, {}) or {}):
                return

        # fallback check on static catalog metadata uploaded by catalog_loader
        catalog_index = self.redis_client.get_json("catalog:index") or {}
        category_key = next((cat for cat, names in catalog_index.items() if model_name in names), None)
        if category_key is None:
            raise RuntimeError(
                f"Catalog readiness failed: model '{model_name}' not found in static catalog and no dynamic override for '{attr_key}'"
            )
        model_doc = self.redis_client.get_json(f"catalog:{category_key}:{model_name}") or {}
        if var_name not in (model_doc.get(io_section, {}) or {}):
            raise RuntimeError(
                f"Catalog readiness failed: missing '{io_section}.{var_name}' for model '{model_name}' required by '{attr_key}'"
            )

    def _setup_multi_computer_scenario(self):
        """
        Set up scenario for multi-computer execution.
        TODO: this is a placeholder for the multi-computer setup logic, which may involve SSH connections, remote command execution, and distributed configuration management.
        """
        self.logger.info("Setting up multi-computer scenario...")
   
    def _get_total_scenario_duration(self):
        duration_time = (self.end_time - self.start_time).total_seconds()
        return duration_time

    def _iter_federates(self):
        for federation_name, federation_conf in self.config.federations.items():
            for federate_name, federate_conf in federation_conf.federate_configs.items():
                yield federation_name, federate_name, federate_conf

    def _apply_startup_sync_defaults(self):
        default_sync = self.config.synchronization.default_startup_sync if self.config.synchronization else StartupSyncConfig()
        for _, _, federate_conf in self._iter_federates():
            if federate_conf.startup_sync is not None:
                continue
            federate_conf.startup_sync = StartupSyncConfig(
                enabled=default_sync.enabled,
                force_read_all_subscriptions=default_sync.force_read_all_subscriptions,
                require_updated_inputs=default_sync.require_updated_inputs,
                require_finite_numeric=default_sync.require_finite_numeric,
                invalid_numeric_sentinels=list(default_sync.invalid_numeric_sentinels) if default_sync.invalid_numeric_sentinels else None,
                missing_inputs_policy=default_sync.missing_inputs_policy,
                invalid_inputs_policy=default_sync.invalid_inputs_policy,
                required_inputs=list(default_sync.required_inputs) if default_sync.required_inputs else None,
            )

    def _flatten_targets(self, targets):
        if targets is None:
            return []
        if isinstance(targets, list):
            return [t for t in targets if t]
        if isinstance(targets, dict):
            flat = []
            for _, value in targets.items():
                if isinstance(value, list):
                    flat.extend([t for t in value if t])
                elif value:
                    flat.append(value)
            return flat
        return [targets]

    def _resolve_target_federate_node(self, current_federation_name, target, nodes, name_lookup):
        if not target:
            return None
        endpoint = str(target).split('/')[0]
        parts = endpoint.split('.')

        # Explicit federation-qualified target: federation.federate.instance
        if len(parts) >= 3 and parts[0] in self.config.federations:
            candidate = (parts[0], parts[1])
            if candidate in nodes:
                return candidate

        # Same-federation form: federate.instance
        if len(parts) >= 2:
            local_candidate = (current_federation_name, parts[0])
            if local_candidate in nodes:
                return local_candidate

            # Fallback: unique federate name across all federations
            matches = name_lookup.get(parts[0], [])
            if len(matches) == 1:
                return matches[0]

        # Cross-federation short form: federate
        if len(parts) == 1:
            matches = name_lookup.get(parts[0], [])
            if len(matches) == 1:
                return matches[0]

        return None

    def _normalize_subscription_causality(self, raw_value):
        default_causality = (
            self.config.synchronization.default_subscription_causality
            if self.config.synchronization
            else "same_step"
        )
        causality = (raw_value or default_causality or "same_step").lower()
        if causality not in {"same_step", "next_step"}:
            self.logger.warning(
                f"Unknown subscription causality '{raw_value}'. Falling back to '{default_causality}'."
            )
            causality = default_causality.lower()
            if causality not in {"same_step", "next_step"}:
                causality = "same_step"
        return causality

    def _build_federate_dependency_graph(self, include_next_step=True):
        nodes = {
            (federation_name, federate_name)
            for federation_name, federation_conf in self.config.federations.items()
            for federate_name in federation_conf.federate_configs.keys()
        }
        adjacency = {node: set() for node in nodes}
        indegree = {node: 0 for node in nodes}
        edge_causality = {}
        name_lookup = {}
        for node in nodes:
            name_lookup.setdefault(node[1], []).append(node)

        for federation_name, federation_conf in self.config.federations.items():
            for federate_name, federate_conf in federation_conf.federate_configs.items():
                consumer = (federation_name, federate_name)
                for sub in federate_conf.connections.subscribes:
                    causality = self._normalize_subscription_causality(getattr(sub, "causality", None))
                    if causality == "next_step" and not include_next_step:
                        continue
                    for target in self._flatten_targets(sub.targets):
                        producer = self._resolve_target_federate_node(
                            federation_name, target, nodes, name_lookup
                        )
                        if producer is None or producer == consumer:
                            continue
                        edge_causality.setdefault((producer, consumer), set()).add(causality)
                        if consumer not in adjacency[producer]:
                            adjacency[producer].add(consumer)
                            indegree[consumer] += 1

        return nodes, adjacency, indegree, edge_causality

    def _validate_causality_cycles(self):
        sync_cfg = self.config.synchronization if self.config.synchronization else None
        if not sync_cfg or not sync_cfg.validate_causality_cycles:
            return

        nodes, adjacency, _, _ = self._build_federate_dependency_graph(include_next_step=False)
        if not nodes:
            return

        sccs = self._compute_sccs(nodes, adjacency)
        problematic = [sorted(comp) for comp in sccs if len(comp) > 1]
        if not problematic:
            return

        message = (
            "Detected same_step dependency cycles that cannot be resolved with non-iterative HELICS time requests: "
            f"{problematic}. Mark at least one subscription in each cycle as causality='next_step' "
            "or switch to iterative HELICS execution for that loop."
        )
        raise RuntimeError(message)

    def _compute_sccs(self, nodes, adjacency):
        index = 0
        stack = []
        on_stack = set()
        indices = {}
        lowlinks = {}
        sccs = []

        def strongconnect(node):
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)

            for nxt in adjacency.get(node, []):
                if nxt not in indices:
                    strongconnect(nxt)
                    lowlinks[node] = min(lowlinks[node], lowlinks[nxt])
                elif nxt in on_stack:
                    lowlinks[node] = min(lowlinks[node], indices[nxt])

            if lowlinks[node] == indices[node]:
                component = []
                while True:
                    popped = stack.pop()
                    on_stack.remove(popped)
                    component.append(popped)
                    if popped == node:
                        break
                sccs.append(component)

        for node in sorted(nodes):
            if node not in indices:
                strongconnect(node)

        return sccs

    def _apply_auto_time_offsets(self):
        sync_cfg = self.config.synchronization.auto_offset if self.config.synchronization else None
        if not sync_cfg or not sync_cfg.enabled:
            self.logger.info("Auto time-offset sequencing disabled by scenario synchronization policy.")
            return

        nodes, adjacency, _, _ = self._build_federate_dependency_graph(include_next_step=False)
        if not nodes:
            return

        sccs = self._compute_sccs(nodes, adjacency)
        node_to_comp = {}
        for comp_idx, comp_nodes in enumerate(sccs):
            for node in comp_nodes:
                node_to_comp[node] = comp_idx

        comp_adjacency = {idx: set() for idx in range(len(sccs))}
        comp_indegree = {idx: 0 for idx in range(len(sccs))}
        for src_node, dst_nodes in adjacency.items():
            src_comp = node_to_comp[src_node]
            for dst_node in dst_nodes:
                dst_comp = node_to_comp[dst_node]
                if src_comp == dst_comp:
                    continue
                if dst_comp not in comp_adjacency[src_comp]:
                    comp_adjacency[src_comp].add(dst_comp)
                    comp_indegree[dst_comp] += 1

        comp_order = {}
        comp_span = {}
        for comp_idx, comp_nodes in enumerate(sccs):
            if len(comp_nodes) == 1:
                comp_order[comp_idx] = comp_nodes
                comp_span[comp_idx] = 1
                continue

            def _cycle_sort_key(node):
                fed_cfg = self.config.federations[node[0]].federate_configs[node[1]]
                timing = fed_cfg.timing_configs
                explicit = 0 if timing.time_offset_explicit else 1
                return (timing.time_offset, explicit, node[0], node[1])

            ordered_cycle = sorted(comp_nodes, key=_cycle_sort_key)
            comp_order[comp_idx] = ordered_cycle
            comp_span[comp_idx] = len(ordered_cycle)
            self.logger.warning(
                f"Detected synchronization cycle among federates: {ordered_cycle}. "
                "Applying deterministic in-cycle ordering to assign offsets."
            )

        queue = deque(sorted([idx for idx, deg in comp_indegree.items() if deg == 0]))
        comp_stage = {idx: 0 for idx in range(len(sccs))}
        processed = 0
        while queue:
            comp_idx = queue.popleft()
            processed += 1
            comp_end_stage = comp_stage[comp_idx] + comp_span[comp_idx] - 1
            for nxt_idx in sorted(comp_adjacency[comp_idx]):
                comp_stage[nxt_idx] = max(comp_stage[nxt_idx], comp_end_stage + 1)
                comp_indegree[nxt_idx] -= 1
                if comp_indegree[nxt_idx] == 0:
                    queue.append(nxt_idx)

        if processed != len(sccs):
            self.logger.warning("Unexpected component-graph cycle during auto offset sequencing. Offsets left unchanged.")
            return

        node_stage = {}
        for comp_idx, ordered_nodes in comp_order.items():
            base_stage = comp_stage[comp_idx]
            for idx, node in enumerate(ordered_nodes):
                node_stage[node] = base_stage + idx

        max_stage = max(node_stage.values()) if node_stage else 0
        offset_step = sync_cfg.offset_step
        if max_stage > 0 and offset_step * max_stage >= 1.0:
            adjusted = 0.9 / max_stage
            self.logger.warning(
                f"Auto offset step {offset_step} is too large for max stage {max_stage}. "
                f"Clamping step to {adjusted:.6f}."
            )
            offset_step = adjusted

        applied_offsets = {}
        for federation_name, federate_name, federate_conf in self._iter_federates():
            node = (federation_name, federate_name)
            stage = node_stage.get(node, 0)
            old_offset = federate_conf.timing_configs.time_offset
            has_explicit = federate_conf.timing_configs.time_offset_explicit

            if has_explicit and not sync_cfg.override_existing_offsets:
                applied_offsets[node] = old_offset
                continue

            new_offset = round(stage * offset_step, 10)
            federate_conf.timing_configs.time_offset = new_offset
            applied_offsets[node] = new_offset

        self.logger.info(f"Auto time-offset sequencing applied: {applied_offsets}")
    
    def _scenario_setup_timing_vars(self):
    
        
        # get the modified total scenario duration
        self.duration_time = self._get_total_scenario_duration()
        
        # Calculate total duration and number of steps
        freq_list = [fed.timing_configs.real_period for federation in self.config.federations.values() for _, fed in federation.federate_configs.items()]
        self.min_real_period = min(freq_list) if freq_list else 60 # default to 60s if not specified but should be specified

        # Apply synchronization defaults/policies before final timing assignment.
        self._apply_startup_sync_defaults()
        self._validate_causality_cycles()
        self._apply_auto_time_offsets()

        # set timing configs for all federates
        for federation_name, federation in self.config.federations.items():
            for _, federate in federation.federate_configs.items():
                federate.timing_configs.start_time = self.start_time.isoformat()
                federate.timing_configs.end_time = self.end_time.isoformat()
                # TODO: converting to int will only accept model frequency that are divisors of the minimum real period, we should add some error handling for this
                federate.timing_configs.time_period = int(federate.timing_configs.real_period / self.min_real_period) # convert minutes to seconds
                if federate.timing_configs.time_delta is None:
                    federate.timing_configs.time_delta = float(federate.timing_configs.time_period)
                n_steps = int(self.duration_time / federate.timing_configs.real_period)
                federate.timing_configs.time_stop = n_steps

    def _setup_local_federation(self, federation_name, federation_conf):
        """
        Set up the federation by starting broker and all federates.
        
        This method orchestrates the federation startup process:
        1. Starts the HELICS broker
        2. Creates and starts all configured federates
        """
        self.logger.info(f"Setting up federation: {federation_name}...")

        # Start broker for this federation (each federation has a broker)
        self._start_local_federation_broker(federation_conf.broker_config, federation_name)
        
        # Start all federates of one federation
        for federate_name, federate_config in federation_conf.federate_configs.items():
            self._create_federate(federate_name, federate_config, federation_name)
    

        self.logger.info("All federates started. Monitoring execution...")
    
    def _get_n_available_tcp_ports(self, n, exclude_ports=None):
        exclude_ports = set(exclude_ports or [])
        available_ports = []

        _helics_lo, _helics_hi = helics_port_range()
        for port in range(_helics_lo, _helics_hi):
            if len(available_ports) >= n:
                break
            if port in exclude_ports:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))  # Try to bind specific port
                    available_ports.append(port)
                except OSError:
                    pass  # Port is in use, skip

        return available_ports
    
    def _broker_binds_externally(self, broker_conf):
        """True if this broker must accept connections from other machines.

        Keyed on the advertised host rather than on `_has_remote_federates()` so an
        explicit LAN `broker_config.host` in the YAML is honored too. Loopback-advertised
        brokers keep binding loopback only — an all-local scenario must not start
        listening on every interface just because this code path grew a remote mode.
        """
        return bool(broker_conf.host) and broker_conf.host not in self.LOOPBACK_HOSTS

    def _port_is_free(self, port):
        """True if a broker could bind `port` on this machine right now.

        Probing by bind rather than by connecting keeps the check passive: it never
        opens a TCP connection to a broker's zmq socket, so it cannot be mistaken for
        a malformed peer.

        SO_REUSEADDR matters — it makes this probe agree with what the broker itself
        can do. A broker that just exited leaves its federate connections in TIME_WAIT
        for up to a minute; a plain bind() reports those ports as taken even though no
        process is listening, so back-to-back runs of the same scenario would fail on a
        port the broker would have bound fine. With SO_REUSEADDR, TIME_WAIT is ignored
        (as the broker ignores it) while a live listener still fails the bind — which is
        the only case worth reporting.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('', port))
                return True
            except OSError:
                return False

    def _broker_ports(self, broker_conf):
        """Every port a broker occupies: the advertised one, plus zmq's paired reply socket."""
        ports = [broker_conf.port]
        if broker_conf.core_type == 'zmq':
            ports.append(broker_conf.port + 1)
        return ports

    def _assert_broker_ports_free(self, broker_conf, label):
        """Fail before spawning a broker whose ports are already taken.

        Most often an orphaned broker from an earlier run: it keeps holding the port,
        the new broker dies on bind, and every federate then times out against nothing.
        Naming the real problem here beats debugging it from five federate timeouts.
        """
        taken = [str(p) for p in self._broker_ports(broker_conf) if not self._port_is_free(p)]
        if taken:
            raise RuntimeError(
                f"{label}: port(s) {', '.join(taken)} already in use — most likely an orphaned broker "
                f"from an earlier run. Find it with:  ps -eo pid,args | grep -F helics_broker"
            )

    def _wait_for_broker_listening(self, broker_process, broker_conf, label):
        """Block until the broker is really accepting connections, or raise.

        A brief sleep is not enough: a broker that cannot bind stays alive for several
        seconds before giving up, so a short check sees a live process and reports
        success — then every federate burns its full connection timeout against a broker
        that was already dead, and the only real error sits in the broker log. Waiting
        for the ports to actually be occupied surfaces the failure here instead.
        """
        ports = self._broker_ports(broker_conf)
        deadline = time.time() + self.BROKER_STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if broker_process.poll() is not None:
                _, stderr = broker_process.communicate()
                raise RuntimeError(
                    f"{label} exited during startup with code {broker_process.returncode}: "
                    f"{stderr.decode(errors='replace').strip() or '(no stderr output)'}"
                )
            if all(not self._port_is_free(p) for p in ports):
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"{label} did not start listening on port(s) {', '.join(str(p) for p in ports)} "
            f"within {self.BROKER_STARTUP_TIMEOUT_S}s"
        )

    def _normalize_broker_and_core_configs(self):
        """
        Resolve a fully-specified, internally-consistent broker/core/protocol
        wiring for the whole scenario, supporting two authoring styles:

          1. Explicit  - the YAML specifies broker_config (core_type/port/host/
             federates) and/or per-federate core_name/core_type/broker_address.
             Those values are kept and cross-checked for consistency.
          2. Minimal   - some or all of the above are omitted. Sensible defaults
             are filled in (protocol defaults to "tcp", ports/addresses are
             auto-assigned, federate counts are derived, core names are
             generated, broker addresses are derived from the federation broker).

        HELICS constraints this enforces:
          - Every broker and every federate core in the scenario must speak the
            same wire protocol (core_type), including the hierarchy broker that
            ties multiple federations together.
          - Each federate must point at its own federation's broker address.
          - Each federate's core_name must be globally unique.
          - A federation broker's `federates` count must match the number of
            federates that will actually connect to it.
          - When more than one federation exists, a top-level hierarchy broker
            is required so federations can see each other (e.g. the RL case).
        """
        federations = self.config.federations
        n_federations = len(federations)

        # ------------------------------------------------------------------
        # 1. Resolve a single scenario-wide protocol (core_type).
        #    HELICS brokers/cores cannot mix protocols, so rather than letting
        #    each federation disagree we pick one value (first explicit one
        #    found, defaulting to "tcp") and apply it everywhere.
        # ------------------------------------------------------------------
        explicit_core_types = {
            federation_conf.broker_config.core_type
            for federation_conf in federations.values()
            if federation_conf.broker_config and federation_conf.broker_config.core_type
        }
        if len(explicit_core_types) > 1:
            raise ValueError(
                f"Inconsistent broker core_type across federations: {sorted(explicit_core_types)}. "
                "HELICS requires every broker and federate core in a scenario to use the same protocol."
            )
        core_type = next(iter(explicit_core_types), 'tcp')
        self.logger.info(f"Resolved scenario-wide HELICS protocol (core_type): '{core_type}'")

        # Distributed scenarios on a non-single-socket protocol are a footgun worth naming
        # up front: with `zmq`/`tcp` every federate core binds its OWN inbound listener
        # (broker_port + 10 + n) and the broker dials back into it, so each remote must be
        # directly reachable from the manager on those ports. That fails outright when the
        # remotes sit behind NAT — the core advertises the private address it bound, the
        # broker's reply is unroutable, and the only symptom is every remote federate dying
        # with "core is unable to register and has timed out" long after a *successful* TCP
        # connect. The `_ss` (single-socket) variants carry all traffic over the core's own
        # outbound connection, need no inbound listener, and are NAT-proof.
        if self._has_remote_federates() and not core_type.endswith('_ss'):
            self.logger.warning(
                f"Scenario has remote federates but core_type is '{core_type}'. Each remote "
                f"federate core will bind its own port and the broker must reach it directly — "
                f"this cannot work if the remotes are behind NAT, and needs the core port range "
                f"open inbound on every remote otherwise. Prefer core_type: '{core_type}_ss' "
                f"(single socket, outbound-only) for distributed runs. "
                f"See docs/user_guide/distributed_deployment.md."
            )

        # ------------------------------------------------------------------
        # 2. Reserve user-specified ports and pre-allocate enough free ones for
        #    every broker that needs one (including the hierarchy broker).
        # ------------------------------------------------------------------
        reserved_ports = {
            federation_conf.broker_config.port
            for federation_conf in federations.values()
            if federation_conf.broker_config and federation_conf.broker_config.port
        }
        n_to_assign = sum(
            1 for federation_conf in federations.values()
            if not (federation_conf.broker_config and federation_conf.broker_config.port)
        )
        if n_federations > 1:
            n_to_assign += 1  # hierarchy (main) broker
        auto_ports = deque(self._get_n_available_tcp_ports(n_to_assign, exclude_ports=reserved_ports))

        def _next_free_port():
            try:
                port = auto_ports.popleft()
            except IndexError:
                raise RuntimeError("Could not find enough free local TCP ports to assign to HELICS brokers.")
            reserved_ports.add(port)
            return port

        # ------------------------------------------------------------------
        # 3. Normalize each federation's broker and propagate protocol, broker
        #    address and a unique core name down to every federate it owns.
        #    Brokers always run on the manager machine (placement rule: only
        #    federates go remote). When ≥1 federate has `host:`, the broker's
        #    default listen address becomes `deployment.manager_address`
        #    (a LAN-reachable IP) instead of loopback, so remote federate
        #    cores can dial in; an explicit YAML `broker_config.host` still
        #    wins either way. Advertising the LAN address alone is NOT
        #    enough to make the broker reachable: the zmq core still binds
        #    its receive sockets to 127.0.0.1 by default regardless of the
        #    advertised address, so `--local_interface=0.0.0.0` must be
        #    passed explicitly whenever the broker's host isn't loopback
        #    (see _broker_binds_externally).
        # ------------------------------------------------------------------
        default_broker_host = (
            self.config.deployment.manager_address
            if self.config.deployment and self._has_remote_federates()
            else '127.0.0.1'
        )
        seen_core_names: Dict[str, str] = {}
        for federation_name, federation_conf in federations.items():
            broker_conf = federation_conf.broker_config

            broker_conf.core_type = core_type
            broker_conf.host = broker_conf.host or default_broker_host
            if not broker_conf.port:
                broker_conf.port = _next_free_port()
            broker_conf.address = f'{broker_conf.host}:{broker_conf.port}'

            n_federates = len(federation_conf.federate_configs)
            if broker_conf.federates is None:
                broker_conf.federates = n_federates
            elif broker_conf.federates != n_federates:
                raise ValueError(
                    f"Federation '{federation_name}': broker_config.federates={broker_conf.federates} "
                    f"does not match the {n_federates} federate(s) configured under federate_configs."
                )

            for federate_name, federate_conf in federation_conf.federate_configs.items():
                qualified_name = f'{federation_name}.{federate_name}'

                if federate_conf.core_type and federate_conf.core_type != core_type:
                    self.logger.warning(
                        f"Federate '{qualified_name}' requested core_type '{federate_conf.core_type}' "
                        f"but the scenario protocol is '{core_type}'. Overriding it for consistency "
                        "(HELICS requires a single protocol for every broker/core in a scenario)."
                    )
                federate_conf.core_type = core_type

                # Every HELICS federate needs its OWN core, so core_name must be globally
                # unique across the scenario. Rather than require the YAML to get this right
                # (or fail on a wrong/duplicate value), assign a unique short human-readable
                # name automatically: keep the YAML value only if it is free, otherwise fall
                # back to the (already-unique-per-federation) federate name, then qualify with
                # the federation, then suffix. This never raises.
                requested = federate_conf.core_name
                candidates = []
                if requested:
                    candidates.append(requested)
                candidates.append(federate_name)
                candidates.append(f'{federation_name}_{federate_name}')
                core_name = next((c for c in candidates if c not in seen_core_names), None)
                if core_name is None:
                    base = f'{federation_name}_{federate_name}'
                    i = 2
                    while f'{base}_{i}' in seen_core_names:
                        i += 1
                    core_name = f'{base}_{i}'
                if requested and core_name != requested:
                    self.logger.info(
                        f"Federate '{qualified_name}': core_name '{requested}' already in use "
                        f"by '{seen_core_names[requested]}'; assigned unique core_name "
                        f"'{core_name}' instead."
                    )
                federate_conf.core_name = core_name
                seen_core_names[core_name] = qualified_name

                # Each federate connects to its own federation's broker.
                federate_conf.broker_address = broker_conf.address

        # ------------------------------------------------------------------
        # 4. Multiple federations need a hierarchy (main) broker above the
        #    per-federation brokers so they can see each other (only a 2-level
        #    hierarchy is supported for now).
        # ------------------------------------------------------------------
        self._hierarchy_broker_config = None
        if n_federations > 1:
            if self.config.multi_computer:
                # TODO: localhost addresses won't be reachable from remote brokers;
                # the hierarchy broker needs the host's public IP/hostname instead,
                # and host:port combinations need their own validation.
                self.logger.error("Broker hierarchy for multi-computer scenarios is not implemented yet.")
            else:
                main_port = _next_free_port()
                self._hierarchy_broker_config = BrokerConfig(
                    core_type=core_type,
                    port=main_port,
                    log_level=self.config.log_level,
                    host=default_broker_host,
                    address=f'{default_broker_host}:{main_port}',
                    sub_brokers=n_federations,
                )
                main_broker_address = f'{core_type}://{default_broker_host}:{main_port}'
                for federation_conf in federations.values():
                    federation_conf.broker_config.broker_address = main_broker_address

        self.logger.debug(
            "Normalized broker/core configuration: "
            f"protocol={core_type}, federations={ {name: fc.broker_config.address for name, fc in federations.items()} }, "
            f"hierarchy_broker={self._hierarchy_broker_config.address if self._hierarchy_broker_config else None}"
        )

    def _start_local_hierarchy_broker(self, broker_conf):
        """Start a hierarchy broker for multi-federation coordination."""
        self.logger.info("Starting local hierarchy broker...")
        try:


            broker_logger = self._broker_cmd_logger_set('main')
            broker_cmd = [
                'helics_broker',
                f'--sub_brokers={broker_conf.sub_brokers}',
                f'--port={broker_conf.port}',
                f'--loglevel={broker_conf.log_level.to_helics_level()}',
                f'--coreType={broker_conf.core_type}',
                '--name=main.broker'
            ]
            if self._broker_binds_externally(broker_conf):
                # The zmq core binds its receive sockets to 127.0.0.1 by default no
                # matter what host is advertised, so remote federates time out against
                # a socket that never accepted them. Bind all interfaces explicitly.
                broker_cmd.append('--local_interface=0.0.0.0')
            
            broker_logger.info(f"Broker command: {' '.join(broker_cmd)}")
            self.logger.info(f"Starting hierarchy broker cmd: {' '.join(broker_cmd)}")
            self._assert_broker_ports_free(broker_conf, 'Hierarchy broker')
            broker_log_file = self.logger_system.scenario_log_dir / "brokers" / "main_broker_process.log"
            
            # Create environment with log file path TODO: for multiple brokers we need to create multiple env vars
            env = os.environ.copy()
            env['BROKER_LOG_FILE'] = str(broker_log_file)
            env['BROKER_NAME'] = "main.broker"
            
            broker_process = subprocess.Popen(
                broker_cmd, 
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )

            # Registered before the readiness wait can raise, so a broker that spawned
            # but failed to bind is still tracked for cleanup rather than orphaned.
            self.broker_processes.append(broker_process)
            try:
                self._wait_for_broker_listening(broker_process, broker_conf, 'Hierarchy broker')
            except RuntimeError as e:
                self.logger.error(str(e))
                broker_logger.error(str(e))
                raise

            success_msg = f"Broker started and listening with PID: {broker_process.pid}"
            self.logger.info(success_msg)
            broker_logger.info(success_msg)
            self.metrics['process_counts']['brokers_started'] += 1
            self._start_broker_log_reader(broker_process, broker_logger)
                
            
        
        except Exception as e:
            self.logger.error(f"Exception during broker startup: {str(e)}")
            broker_logger.error(f"Exception during broker startup: {str(e)}")
            raise
    
    def _start_local_federation_broker(self, broker_conf, federation_name):
        """Start HELICS broker for a federation with logging support."""
        
        try:
            broker_logger = self._broker_cmd_logger_set(federation_name)
            
            broker_cmd = [
                'helics_broker',
                f'--federates={broker_conf.federates}',
                f'--port={broker_conf.port}',
                f'--loglevel={broker_conf.log_level.to_helics_level()}',
                f'--coreType={broker_conf.core_type}',
                f'--name={federation_name}.broker',
                f'--broker_address={broker_conf.broker_address}' if broker_conf.broker_address else ''
            ]
            if self._broker_binds_externally(broker_conf):
                # The zmq core binds its receive sockets to 127.0.0.1 by default no
                # matter what host is advertised, so remote federates time out against
                # a socket that never accepted them. Bind all interfaces explicitly.
                broker_cmd.append('--local_interface=0.0.0.0')
            
            broker_logger.info(f"Broker command: {' '.join(broker_cmd)}")
            self.logger.info(f"Starting broker for federation {federation_name} cmd: {' '.join(broker_cmd)}")
            self._assert_broker_ports_free(broker_conf, f"Broker for federation '{federation_name}'")

            # Create log file path for broker process
            broker_log_file = self.logger_system.scenario_log_dir / "brokers" / f"broker_{federation_name}_process.log"
            
            # Create environment with log file path TODO: for multiple brokers we need to create multiple env vars
            env = os.environ.copy()
            env['BROKER_LOG_FILE'] = str(broker_log_file)
            env['BROKER_NAME'] = f"{federation_name}.broker"
            
            broker_process = subprocess.Popen(
                broker_cmd, 
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            # Registered before the readiness wait can raise, so a broker that spawned
            # but failed to bind is still tracked for cleanup rather than orphaned.
            self.broker_processes.append(broker_process)
            try:
                self._wait_for_broker_listening(
                    broker_process, broker_conf, f"Broker for federation '{federation_name}'")
            except RuntimeError as e:
                self.logger.error(str(e))
                broker_logger.error(str(e))
                raise

            success_msg = f"Broker started and listening with PID: {broker_process.pid}"
            self.logger.info(success_msg)
            broker_logger.info(success_msg)
            self.metrics['process_counts']['brokers_started'] += 1
            self._start_broker_log_reader(broker_process, broker_logger)

        except Exception as e:
            self.logger.error(f"Exception during broker startup: {str(e)}")
            broker_logger.error(f"Exception during broker startup: {str(e)}")
            raise
    
    def _start_broker_log_reader(self, process: subprocess.Popen, broker_logger: logging.Logger):
        """Spawn daemon threads that drain broker stdout/stderr into broker_logger."""
        def _drain(stream, log_fn):
            try:
                for raw in stream:
                    line = raw.decode('utf-8', errors='replace').rstrip()
                    if line:
                        log_fn(f"[HELICS] {line}")
            except Exception:
                pass

        threading.Thread(target=_drain, args=(process.stdout, broker_logger.debug), daemon=True).start()
        threading.Thread(target=_drain, args=(process.stderr, broker_logger.warning), daemon=True).start()

    def _broker_cmd_logger_set(self, subsystem_name):
        broker_logger = self.logger_system.get_broker_logger(
            broker_name=f"{subsystem_name}_broker",
            federation_name=subsystem_name
        )
        broker_path = shutil.which('helics_broker')
        if not broker_path:
            error_msg = "helics_broker executable not found"
            self.logger.error(error_msg)
            broker_logger.error(error_msg)
            raise RuntimeError(f"{error_msg}. Please ensure HELICS is installed and in your PATH.")
        return broker_logger
    
    def _build_federate_args(self, federate_name, federate_config, federation_name, redis_url, log_file):
        """Build the federate_launcher.py CLI arg list, shared by the local and remote spawn paths.

        Args:
            redis_url: Redis URL this federate should use (see `_redis_url_for`).
            log_file: path (local or remote) for the federate's own file logger.

        Returns:
            List[str]: args only, WITHOUT the leading interpreter/script tokens — the
            local path prepends `['python', <local launcher path>]`, the remote path
            prepends the remote python invocation + `'src/core/federate_launcher.py'`.
        """
        return [
            '--name', federate_name,
            '--scenario_name', self.scenario_name,
            '--federation_name', federation_name,
            '--type', federate_config.type,
            '--simid', self.simulation_id,
            '--redis-url', redis_url,
            '--redis-key', self.redis_key,
            '--log-file', log_file,
            '--log-level', federate_config.log_level.value,
        ]

    def _redis_url_for(self, federate_config) -> str:
        """Redis URL a federate should use: manager_address for a remote federate (Redis always
        runs on the manager machine, alongside the manager itself), else the manager's own
        `self.redis_url` unchanged."""
        if getattr(federate_config, 'host', None):
            return f'redis://{self.config.deployment.manager_address}:{self._redis_port}/{self._redis_db}'
        return self.redis_url

    def _create_federate(self, federate_name, federate_config, federation_name):
        """Create and start one federate, dispatching to local Popen or remote ssh spawn.

        Both paths append their process handle to `self.federate_processes`, so
        `_monitor_processes`/`_emergency_cleanup` need no branching of their own.
        """
        host_alias = getattr(federate_config, 'host', None)
        if host_alias:
            return self._create_remote_federate(federate_name, federate_config, federation_name, host_alias)
        return self._create_local_federate(federate_name, federate_config, federation_name)

    def _create_local_federate(self, federate_name, federate_config, federation_name):
        """
        Create and start a federate subprocess with logging support.
        Uses Redis for configuration distribution.

        Args:
            federate_config: Configuration object for the federate

        Returns:
            subprocess.Popen: The created federate process
        """

        try:
            # Path to Federate class script depending on the type of federate
            federate_launcher = os.path.join(os.path.dirname(__file__), 'federate_launcher.py')

            # Create log file path for this federate (this will be used by the subprocess)
            federate_log_file = self.logger_system.scenario_log_dir / "federates" / f"federate_{federate_name}.log"

            args = self._build_federate_args(
                federate_name, federate_config, federation_name,
                redis_url=self.redis_url,
                log_file=str(federate_log_file),
            )
            cmd = ['python', federate_launcher] + args

            self.logger.info(f"Creating local federate: {federate_name} (type: {federate_config.type})")

            # Capture subprocess stdout/stderr to a file so uncaught exceptions / tracebacks
            # (which bypass the federate's own logger) are visible instead of being lost in an
            # undrained PIPE. Separate from the federate's logger file to avoid handler clashes.
            stdio_path = self.logger_system.scenario_log_dir / "federates" / f"federate_{federate_name}.stdio.log"
            stdio_file = open(stdio_path, 'wb')

            # Create subprocess in new process group for proper cleanup
            process = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,  # Create new process group
                stdout=stdio_file,
                stderr=subprocess.STDOUT
            )
            success_msg = f"Federate process started with PID: {process.pid}"
            self.logger.info(success_msg)

            self.federate_processes.append(process)
            self.metrics['process_counts']['federates_started'] += 1
            return process

        except Exception as e:
            error_msg = f"Exception during Local Federate Creation: {str(e)}"
            self.logger.error(error_msg)
            raise

    def _create_remote_federate(self, federate_name, federate_config, federation_name, host_alias):
        """Create and start a federate on a remote machine via its RemoteExecutor.

        Mirrors `_create_local_federate`'s log layout on the remote filesystem: since
        `logger_system.scenario_log_dir` is already relative to the project root
        (e.g. `logs/<scenario>/<run_timestamp>`), the identical relative path resolved
        against the machine's `workdir` gives the remote log location — `T5`'s rsync-back
        then merges it into the same local directory tree with no path translation needed.

        Returns:
            subprocess.Popen: the local ssh child that is this federate's process handle
            (see `RemoteExecutor.spawn`).
        """
        try:
            executor = self.remote_executors[host_alias]
            machine_conf = self.config.deployment.machines[host_alias]
            scenario_log_dir_rel = str(self.logger_system.scenario_log_dir)

            remote_log_file = os.path.join(
                machine_conf.workdir, scenario_log_dir_rel, 'federates', f'federate_{federate_name}.log'
            )
            remote_stdio_file = os.path.join(
                machine_conf.workdir, scenario_log_dir_rel, 'federates', f'federate_{federate_name}.stdio.log'
            )

            args = self._build_federate_args(
                federate_name, federate_config, federation_name,
                redis_url=self._redis_url_for(federate_config),
                log_file=remote_log_file,
            )
            launcher_args = ['src/core/federate_launcher.py'] + args

            self.logger.info(
                f"Creating remote federate: {federate_name} (type: {federate_config.type}) on '{host_alias}'"
            )
            process = executor.spawn(launcher_args, remote_stdio_file)
            success_msg = f"Remote federate process started (ssh child PID: {process.pid}) on '{host_alias}'"
            self.logger.info(success_msg)

            self.federate_processes.append(process)
            self.metrics['process_counts']['federates_started'] += 1
            return process

        except Exception as e:
            error_msg = f"Exception during Remote Federate Creation ('{host_alias}'): {str(e)}"
            self.logger.error(error_msg)
            raise








    def _monitor_processes(self):
        """Poll until every federate exits, then wait briefly for the brokers to follow.

        Only the federates are waited on unconditionally: they are the simulation, and a
        broker exits on its own once every federate has disconnected. Waiting on brokers
        with the same `while` hangs forever in the failure case — if the federates all die
        before registering (unreachable broker, bad config), the broker goes on waiting for
        federates that will never arrive and nothing ever exits. So once no federate is
        left, give the brokers a grace period to shut down cleanly and let
        `_emergency_cleanup` terminate whatever is still standing.
        """
        active_federates = list(self.federate_processes)
        active_brokers = list(self.broker_processes)
        self.logger.info(f"Monitoring {len(active_federates)} federates, {len(active_brokers)} brokers")

        while active_federates:
            time.sleep(1)
            active_federates = self._collect_completed(
                active_federates, 'Federate', 'federates_completed', 'federates_failed')
            active_brokers = self._collect_completed(
                active_brokers, 'Broker', 'brokers_completed', 'brokers_failed')

            # A HELICS federation is all-or-nothing: every declared federate must join for
            # it to form. Once one has died the survivors can never proceed — they block
            # inside HELICS waiting for a peer that will never arrive, exit on their own
            # never, and waiting on them hangs the run forever (which is how brokers and
            # federates got orphaned by hand-killed managers). Stop waiting and let cleanup
            # terminate them, so the run fails fast with its logs collected.
            failed = self.metrics['process_counts']['federates_failed']
            if failed and active_federates:
                self.logger.error(
                    f"{failed} federate(s) failed — the federation can no longer form. "
                    f"Abandoning {len(active_federates)} federate(s) still blocked in HELICS; "
                    "cleanup will terminate them."
                )
                break

        deadline = time.time() + self.BROKER_SHUTDOWN_GRACE_S
        while active_brokers and time.time() < deadline:
            time.sleep(1)
            active_brokers = self._collect_completed(
                active_brokers, 'Broker', 'brokers_completed', 'brokers_failed')

        if active_brokers:
            self.logger.warning(
                f"{len(active_brokers)} broker(s) still alive {self.BROKER_SHUTDOWN_GRACE_S}s after the last "
                "federate exited; cleanup will terminate them."
            )
        self.logger.info("Monitoring finished: no federates left running")
        self._log_execution_summary()

    def _collect_completed(self, processes, label, count_key, fail_key):
        """Remove finished processes, log result, update counters. Returns still-running list."""
        still_running = []
        for p in processes:
            if p.poll() is None:
                still_running.append(p)
            elif p.returncode == 0:
                self.logger.info(f"✓ {label} completed")
                self.metrics['process_counts'][count_key] += 1
            else:
                self.logger.error(f"✗ {label} failed with code {p.returncode}")
                self.metrics['process_counts'][fail_key] += 1
        return still_running

    def _collect_remote_results(self):
        """rsync results + logs back from every remote machine after the run.

        Called on the success path once `_monitor_processes` returns (all federate
        ssh children — hence the remote federates — have exited, so `sink: json` is
        fully written and `sink: parquet` is finalized). Every remote federate wrote
        distinct files under the *same* relative `results/<scenario>/<sim_id>/` and
        `logs/<scenario>/<run_timestamp>/` layout the manager uses, so merging each
        machine's tree into the manager's local dirs is collision-free (no `--delete`
        on collect — `RemoteExecutor.collect`).

        A collection failure is logged (ERROR + manual-rsync hint) but never raised:
        a run whose results are sitting valid on a remote disk must not be reported
        as failed just because the copy-back hiccuped.

        Results and logs are collected independently. A run whose federates died early
        never created a remote results dir, and sharing one try/except let that expected
        rsync failure skip the log collection that follows — stranding the remote logs on
        the remote box in exactly the case they are needed to explain the failure.
        """
        if not self.remote_executors:
            return

        sim_id = self.simulation_id[-15:]
        repo_root = Path(__file__).resolve().parents[2]
        scenario_log_dir_rel = str(self.logger_system.scenario_log_dir)

        local_results = str(repo_root / "results" / self.scenario_name / sim_id)
        local_logs = str(repo_root / scenario_log_dir_rel)

        for alias, executor in self.remote_executors.items():
            machine_conf = self.config.deployment.machines[alias]
            remote_results = os.path.join(machine_conf.workdir, "results", self.scenario_name, sim_id)
            remote_logs = os.path.join(machine_conf.workdir, scenario_log_dir_rel)
            self.logger.info(f"Collecting results + logs back from remote machine '{alias}'...")
            for what, remote_dir, local_dir in (
                ('results', remote_results, local_results),
                ('logs', remote_logs, local_logs),
            ):
                try:
                    executor.collect(remote_dir, local_dir)
                except Exception as e:
                    target = f"{machine_conf.user}@{machine_conf.host}" if machine_conf.user else machine_conf.host
                    self.logger.error(
                        f"[{alias}] failed to collect remote {what}: {e}. "
                        f"Manually retrieve with: rsync -az {target}:{remote_dir}/ {local_dir}/"
                    )

    def _cleanup_remote_execution(self):
        """Remote-machine teardown for `_emergency_cleanup`. Never raises.

        Belt-and-suspenders: the `-tt` pty in `RemoteExecutor.spawn` already
        propagates SIGHUP to the remote federate when the local ssh child is killed,
        but a `pkill -f <simulation_id>` guarantees no orphaned `federate_launcher`
        survives on a remote box (`simulation_id` is unique per run → the pattern
        only ever matches this run's federates). Then every ssh ControlMaster socket
        is closed. Each step is wrapped so an ssh failure during teardown can never
        turn a clean run into a crash, nor mask the original error on a failing one.
        """
        if not self.remote_executors:
            return

        for alias, executor in self.remote_executors.items():
            try:
                # rc=1 (no match) is fine — we ignore the return code.
                executor.run(['pkill', '-f', self.simulation_id], timeout=10)
            except Exception as e:
                self.logger.warning(f"[{alias}] remote federate sweep failed during cleanup: {e}")
            try:
                executor.close()
            except Exception as e:
                self.logger.warning(f"[{alias}] closing ssh control master failed during cleanup: {e}")

        self.remote_executors = {}

    def stop_federation(self):
        """
        Manually stop the federation.
        
        This method can be called to explicitly stop the federation
        and cleanup all resources.
        """
        print("Stopping federation...")
        self._emergency_cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self._emergency_cleanup()
        return False  # Don't suppress exceptions

    def debug_cleanup(self):
        """
        Explicit cleanup method for debugging sessions.
        Call this manually if needed during debugging.
        """
        print("Debug cleanup requested...")
        self._emergency_cleanup()
        
    def get_running_processes(self):
        """
        Get status of all managed processes for debugging.
        """
        status = {
            'brokers': [],
            'federates': []
        }
        
        # Check all broker processes
        for i, process in enumerate(self.broker_processes):
            status['brokers'].append({
                'index': i,
                'pid': process.pid if process.pid else None,
                'running': process.poll() is None
            })
        
        # Check all federate processes
        for i, process in enumerate(self.federate_processes):
            status['federates'].append({
                'index': i,
                'pid': process.pid if process.pid else None,
                'running': process.poll() is None
            })
        
        return status

    def stop_all_brokers(self):
        """
        Stop all broker processes specifically.
        Useful for debugging or partial shutdowns.
        """
        self.logger.info("Stopping all broker processes...")
        for i, process in enumerate(self.broker_processes):
            if process and process.poll() is None:
                try:
                    self.logger.info(f"Stopping broker {i} (PID: {process.pid})")
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                        self.logger.info(f"Broker {i} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        self.logger.warning(f"Force killing broker {i}")
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    self.logger.warning(f"Broker {i} already terminated")

    def _log_execution_summary(self):
        """Log phase timings and process counts, save to JSON."""
        if not hasattr(self, 'logger') or not self.logger:
            return
        self.logger.info("=" * 60)
        self.logger.info("EXECUTION SUMMARY")
        for key, duration in self.metrics['phase_durations'].items():
            if duration is not None:
                self.logger.info(f"  {key}: {duration:.3f}s")
        if self.metrics['total_duration']:
            self.logger.info(f"  total: {self.metrics['total_duration']:.3f}s")
        counts = self.metrics['process_counts']
        self.logger.info(
            f"  brokers {counts['brokers_completed']}/{counts['brokers_started']}, "
            f"federates {counts['federates_completed']}/{counts['federates_started']}"
        )
        if counts['federates_failed'] or counts['brokers_failed']:
            self.logger.error(
                f"  FAILED: {counts['federates_failed']} federate(s), {counts['brokers_failed']} broker(s)"
            )
        self.logger.info("=" * 60)
        try:
            metrics_file = self.logger_system.scenario_log_dir / "execution_metrics.json"
            serializable = {}
            for key, value in self.metrics.items():
                if isinstance(value, datetime):
                    serializable[key] = value.isoformat() if value else None
                elif isinstance(value, dict):
                    serializable[key] = {
                        k: (v.isoformat() if isinstance(v, datetime) else v)
                        for k, v in value.items()
                    }
                else:
                    serializable[key] = value
            with open(metrics_file, 'w') as f:
                json.dump(serializable, f, indent=2)
            self.logger.info(f"Metrics saved to: {metrics_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save metrics: {e}")

    def get_execution_metrics(self):
        """Return copy of current metrics dict."""
        return self.metrics.copy()







def main(scenario_name):
    """Entry point: run the named scenario (from src/scenarios/) end-to-end via ScenarioManager."""
    try:
        print("Scenario Manager - starting SETUP phase")
        with ScenarioManager(scenario_name) as manager:
           
            manager.start_scenario()

            # Debug: Check what's running
            status = manager.get_running_processes()
            print(f"\nCompleted! Final status: {len(status['brokers'])} brokers, {len(status['federates'])} federates")

            # Show final metrics
            metrics = manager.get_execution_metrics()
            if metrics.get('total_duration'):
                print(f"Federation completed in {metrics['total_duration']:.3f} seconds")

            # Show phase durations if available
            phase_durations = metrics.get('phase_durations', {})
            if phase_durations:
                print("\nPhase breakdown:")
                for phase, duration in phase_durations.items():
                    if duration:
                        print(f"   {phase}: {duration:.3f}s")

    except Exception as e:
        print(f"Error: {e}")
    # All processes (including multiple brokers) are cleaned up automatically
