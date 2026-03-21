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
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import List, Dict, Any
from utils.config_dataclasses import BrokerConfig, FederationConfig,FederateConfig,FedTimingConfig,FedFlags,FedConnections,MemoryConfig,FedPublication,FedSubscription,StartupSyncConfig
from utils.config_reader import read_scenario_config
from utils.logging_config import FederationLogger
from utils.influxdb_client import InfluxClient
from utils.redis_client import RedisClient

pp = pprint.PrettyPrinter(indent=4)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Progress bar import (optional)
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# HELICS import for broker queries
try:
    import helics as h
    HELICS_AVAILABLE = True
except ImportError:
    HELICS_AVAILABLE = False





class ScenarioManager:
    """
    Manages HELICS federation lifecycle including broker and federate processes.
    
    This class handles:
    - Starting and stopping HELICS broker
    - Managing federate subprocesses
    - Graceful shutdown and emergency cleanup
    - Process monitoring and error handling
    """
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


        # entities datatsructures TODO find an intelligent way to use these datastructures
        self.brokers = {}
        self.federations = {}  # store federations by name for easy access
        for federation_name, federation_conf in self.config.federations.items():
            self.federations[federation_name] = {
                'config': federation_conf,
                'broker_process': None,
                'federate_processes': []
            }
        self.rl_federates = {}

        
        # Process management attributes
        self.broker_processes: List[subprocess.Popen] = []  # Fixed: plural and proper type
        self.federate_processes: List[subprocess.Popen] = []
        self.temp_files: List[str] = []
        
        # Cleanup management
        self._cleanup_done = False
        self._cleanup_lock = threading.Lock()
        
        # HELICS query management
        self._broker_query_lock = threading.Lock()
        self._broker_federation_map = {}  # Maps broker process to federation name
        
        # InfluxDB client 
        # self.influx_client = self._set_influxdb_client()
        
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
        
        # Check psutil availability
        self._ensure_psutil_available()

    def _logging_sys_setup(self):
        # Initialize logging system    
        self.logger_system = FederationLogger(self.scenario_name)
        self.logger = self.logger_system.setup_manager_logger(self.config.log_level)
        # Log initialization
        self.logger.info(f"Initializing ScenarioManager for scenario: {self.scenario_name}")
        self.logger.info(f"Log directory: {self.logger_system.scenario_log_dir}")
    
    def _setup_metrics(self):
         # Query configuration (configurable parameters)
        self.query_config = {
            'enabled': True,          # Enable/disable progress bar monitoring
            'frequency_ms': 500,      # Query every 500ms by default
            'timeout_ms': 500,        # Query timeout: 500ms  
            'init_delay_s': 1.0,      # Initial delay before first query
            'adaptive': True,         # Enable adaptive frequency based on time_step
            'min_frequency_ms': 100,  # Minimum query interval (max 10 queries/sec)
            'max_frequency_ms': 2000  # Maximum query interval (min 0.5 queries/sec)
        }
        
        # Execution metrics tracking
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
            'memory_usage': {
                'initial': None,
                'peak': None,
                'final': None
            },
            'process_counts': {
                'brokers_started': 0,
                'federates_started': 0,
                'brokers_completed': 0,
                'federates_completed': 0
            }
        }
        
        # Record initial memory usage if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                self.metrics['memory_usage']['initial'] = {
                    'rss': process.memory_info().rss,
                    'vms': process.memory_info().vms,
                    'system_available': psutil.virtual_memory().available
                }
            except Exception as e:
                self.logger.warning(f"Failed to get initial memory usage: {e}")

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
        
        # Record final memory usage if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                self.metrics['memory_usage']['final'] = {
                    'rss': process.memory_info().rss,
                    'vms': process.memory_info().vms,
                    'system_available': psutil.virtual_memory().available
                }
            except Exception:
                pass
        
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
        
        # Clean up temporary files
        for temp_file in getattr(self, 'temp_files', []):
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except OSError:
                pass
        
        self.metrics['cleanup_end'] = datetime.now()
        
        # Calculate cleanup phase duration
        if 'cleanup_start' in self.metrics and self.metrics['cleanup_start']:
            cleanup_delta = (self.metrics['cleanup_end'] - self.metrics['cleanup_start']).total_seconds()
            self.metrics['phase_durations']['cleanup'] = cleanup_delta
        
        # Calculate total duration
        if self.metrics['simulation_start']:
            end_time = self.metrics['cleanup_end']
            self.metrics['total_duration'] = (end_time - self.metrics['simulation_start']).total_seconds()
        
        # Close InfluxDB connection
        # if self.influx_client:
            # self.influx_client.close()

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

    def _set_influxdb_client(self):
        '''old method no longer used influx was a huge bottleneck
         Initialize InfluxDB client for metrics storage.'''
        try:
            influx_client = InfluxClient(
                logger=self.logger,
                auto_start=True  # Will auto-start if not running
            )
            
            # Log health status
            health = influx_client.get_health_status()
            if health['overall_healthy']:
                self.logger.info("✓ InfluxDB is healthy and ready")
            else:
                self.logger.warning(f"InfluxDB health issues: {health}")
            
            self.config.influxdb = {}
            self.config.influxdb['url'] = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
            self.config.influxdb['token'] = os.getenv('INFLUXDB_TOKEN', 'mytoken123456')
            self.config.influxdb['org'] = os.getenv('INFLUXDB_ORG', 'myorg')
            self.config.influxdb['bucket'] = os.getenv('INFLUXDB_BUCKET', 'simulation_data')
            return influx_client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize InfluxDB client: {e}")
            influx_client = None
        
        return influx_client
    
    def _setup_redis_client(self):
        """Initialize Redis client for configuration distribution."""
        try:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
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
        
        try:
            # setting up scenario (scenario config, federations, brokers federates)
            self._setup_scenario()

            # Monitor federation
            self._monitor_processes() # TODO: better understand the monitoring part
            
            self.metrics['simulation_end'] = datetime.now()
            duration = (self.metrics['simulation_end'] - self.metrics['simulation_start']).total_seconds()
            self.metrics['phase_durations']['simulation_duration'] = duration
            self.logger.info(f"Simulation completed in {duration:.3f} seconds")
            
        except KeyboardInterrupt:
            self.logger.warning("\nKeyboard interrupt received. Shutting down...")
        except Exception as e:
            self.logger.error(f"Scenario error: {e}")
        finally:
            # Always cleanup (but only once due to lock mechanism)
            self.logger.info("Scenario execution finished. Performing cleanup...")
            self._emergency_cleanup(success=True)

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

            if self.config.reinforcement_learning_config.training.mode == "online":
                self.logger.info("Setting up Online TRAINING!")
                # modify config file adding rl agent as a federate in a new federation
                self._modify_config_for_online_training() 

            
            # TODO: offline training logic
            elif self.config.reinforcement_learning_config.training.mode == "offline":
                self.logger.info("Starting Offline TRAINING!")
                # do offline learning
                self._offline_learning()
            
            # TODO mixed training logic
            elif self.config.reinforcement_learning_config.training.mode == "mixed":
                self.logger.info("Starting Mixed training loop (Offline + Online)...")
                # do offline learning
                self._offline_learning
                # prepare config for online training
                self._modify_config_for_online_training()
            
            else:
                self.logger.warning(f"Unknown training mode specified: {self.config.reinforcement_learning_config.training.mode}. Proceeding without training.")
            
            # ******************+TESTING PART****************** TODO
            if self.config.reinforcement_learning_config.test:
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

    def _resolve_observation_causality(self, causality_list, index):
        if causality_list and index < len(causality_list) and causality_list[index]:
            return self._normalize_subscription_causality(causality_list[index])
        return "same_step"
    
    def _get_rl_pubsubs(self, rl_task):
        publications = []
        subscriptions = []
        obs_causality = rl_task.agent.env.observation_causality
        add_obs_causality = rl_task.agent.env.additional_observation_causality
        if obs_causality and len(obs_causality) != len(rl_task.agent.env.observations):
            self.logger.warning(
                "Length mismatch for RL observation causality list: "
                f"{len(obs_causality)} entries for {len(rl_task.agent.env.observations)} observations. "
                "Missing entries will default to 'same_step'."
            )
        if (
            rl_task.agent.env.additional_observations
            and add_obs_causality
            and len(add_obs_causality) != len(rl_task.agent.env.additional_observations)
        ):
            self.logger.warning(
                "Length mismatch for RL additional_observation_causality list: "
                f"{len(add_obs_causality)} entries for {len(rl_task.agent.env.additional_observations)} additional observations. "
                "Missing entries will default to 'same_step'."
            )

        for obs_idx, obs in enumerate(rl_task.agent.env.observations):
            pubs_model = self.config.federations[obs.split('.')[0]].federate_configs[obs.split('.')[1]].connections.publishes
            for p in pubs_model:
                if p.key==obs.split('.')[-1]:
                    targets = [f'{obs.split(".")[1]}.{obs.split(".")[2]}/{obs.split(".")[3]}']
                    causality = self._resolve_observation_causality(obs_causality, obs_idx)
                    subscriptions.append(
                        FedSubscription(
                            key=obs.split('.')[-1],
                            type=p.type,
                            units=p.units,
                            targets=targets,
                            causality=causality,
                        )
                    )
        # TODO: check if additional works (never tried)
        if rl_task.agent.env.additional_observations:
            for obs_idx, obs in enumerate(rl_task.agent.env.additional_observations):
                pubs_model = self.config.federations[obs.split('.')[0]].federate_configs[obs.split('.')[1]].connections.publishes
                for p in pubs_model:
                    if p.key==obs.split('.')[-1]:
                        targets = [f'{obs.split(".")[1]}.{obs.split(".")[2]}/{obs.split(".")[3]}']
                        causality = self._resolve_observation_causality(add_obs_causality, obs_idx)
                        subscriptions.append(
                            FedSubscription(
                                key=obs.split('.')[-1],
                                type=p.type,
                                units=p.units,
                                targets=targets,
                                causality=causality,
                            )
                        )
        
        for action in rl_task.agent.env.actions:
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
        controlled_models = [action.rsplit('.',2)[0] for action in rl_task.agent.env.actions]
        controlled_models = set(controlled_models)
        return controlled_models

    def _build_rl_reset_observation_defaults(self, rl_task):
        explicit_defaults = getattr(rl_task.agent.env, "reset_observation_defaults", None) or {}
        reset_defaults = dict(explicit_defaults)
        all_obs = list(rl_task.agent.env.observations)
        if rl_task.agent.env.additional_observations:
            all_obs.extend(list(rl_task.agent.env.additional_observations))

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

    # def _get_rl_training_total_duration(self, rl_task, rl_period):
    #     total_number_of_trainig_steps = rl_task.training.n_episodes * rl_task.training.episode_length
    #     rl_task.training.total_steps = total_number_of_trainig_steps
    #     return total_number_of_trainig_steps * rl_period

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
                                            'rl_agent': FederateConfig(
                                                type='rl',
                                                id='',
                                                name='rl_agent',
                                                model_configs=None, 
                                                timing_configs=FedTimingConfig(real_period=rl_period), 
                                                flags=FedFlags(),
                                                connections=FedConnections(publishes=publications, subscribes=subscriptions, endpoints=[]), 
                                                memory_config=None, # TODO: this is hardcoded, we should understand how to set this in a dynamic way based on the expected memory usage of the RL agent
                                                log_level=self.config.log_level,
                                                core_name=None,
                                                core_type=None,
                                                broker_address=None
                                            )
                                        }
        federation_conf= FederationConfig(broker_config=broker_config,
                                        federate_configs=fed_configs,
                                        name="rl_federation",
                                        )
        rl_required_inputs = [obs.split('.')[-1] for obs in rl_task.agent.env.observations]
        if rl_task.agent.env.additional_observations:
            rl_required_inputs.extend([obs.split('.')[-1] for obs in rl_task.agent.env.additional_observations])
        federation_conf.federate_configs['rl_agent'].startup_sync = StartupSyncConfig(
            required_inputs=sorted(set(rl_required_inputs))
        )
        
        # add the knowledge of controlled model using the model name from model catalog to retrieve normalization boundaries for each attr
        real_controlled_models = {}
        for cm in rl_task.agent.env.actions:
            mod_name = self.config.federations[cm.split('.')[0]].federate_configs[cm.split('.')[1]].model_configs.instantiation.model_name
            real_controlled_models[cm]= mod_name
        federation_conf.federate_configs['rl_agent'].controlled_models = real_controlled_models

        real_observed_models = {} 
        for cm in rl_task.agent.env.observations:
            mod_name = self.config.federations[cm.split('.')[0]].federate_configs[cm.split('.')[1]].model_configs.instantiation.model_name
            real_observed_models[cm]= mod_name
        federation_conf.federate_configs['rl_agent'].observed_models = real_observed_models

        if rl_task.agent.env.additional_observations:
            real_add_observed_models = {}
            for cm in rl_task.agent.env.additional_observations:
                mod_name = self.config.federations[cm.split('.')[0]].federate_configs[cm.split('.')[1]].model_configs.instantiation.model_name
                real_add_observed_models[cm]= mod_name
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
        additional_period = rl_task.test.total_steps * rl_period

        has_online_training = (
            self.config.reinforcement_learning_config.training
            and self.config.reinforcement_learning_config.training.mode != "offline"
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
        additional_period = rl_task.training.total_steps * rl_period
        self.end_time = self.start_time + timedelta(seconds=additional_period)
        self.config.end_time = self.end_time.isoformat() #probably redundant
        self.logger.debug(f"Modified scenario configuration for online training\n new start_time:{self.start_time} \n new end_time: {self.end_time}")

    def _setup_classic_scenario(self):
        # Set up of timings, Synchronization variables
        self._scenario_setup_timing_vars()
        # Initialize all federation and start the processes Spawn
        # This also automatically starts the co-simulation
        # 2 options - local or multi computer
        self._setup_results_folder()
        if self.config.multi_computer and self.config.multi_computer_config:
            self._setup_multi_computer_scenario() # TODO: multi computer must be implemented
        else:
            # Automatically add a main broker when it spots multifederations (for now with this method only 2 level hierarchy is supported)
            if len(self.config.federations) > 1:
                main_broker = self._modify_config_for_broker_hierarchy()
                self._start_local_hierarchy_broker(main_broker) # TODO: must be a generic call to fix broker inconsistencies in any case also single federation
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
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        redis_db = int(os.getenv('REDIS_DB', '0'))
        self.redis_url = os.getenv('REDIS_URL', f'redis://{redis_host}:{redis_port}/{redis_db}')

        config_dict = asdict(self.config)
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
   
    def _store_simulation_metadata(self):
        """Store simulation metadata in InfluxDB. OLD method no longer used"""
        
        try:
            tags = {
                'simulation_id': self.simulation_id,
            }
            
            # todo add information of possible training periods
            fields = {
                'start_time': int(self.start_time.timestamp() * 1_000_000_000),  # Store as nanoseconds
                'end_time': int(self.end_time.timestamp() * 1_000_000_000),  # Store as nanoseconds
            }
            
            # Log the metadata times for debugging
            self.logger.info(f"📊 Storing metadata - Start: {self.start_time}, End: {self.end_time}")
            self.logger.debug(f"   Start timestamp (ns): {fields['start_time']}")
            self.logger.debug(f"   End timestamp (ns): {fields['end_time']}")
            
            bucket = self.config.influxdb['bucket']
            measurement = 'sim_metadata'
            
            success = self.influx_client.write_metadata(bucket, measurement, tags, fields)
            
            if success:
                self.logger.info("Simulation metadata stored in InfluxDB")
            else:
                self.logger.warning("Failed to store simulation metadata")
                
        except Exception as e:
            self.logger.error(f"Error storing simulation metadata: {e}")
    
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
            self._create_local_federate(federate_name, federate_config, federation_name)
    

        self.logger.info("All federates started. Monitoring execution...")
    
    def _get_n_available_tcp_ports(self, n, exclude_ports=None):
        exclude_ports = set(exclude_ports or [])
        available_ports = []

        for port in range(20000, 30000 ):
            if len(available_ports) >= n - len(exclude_ports):
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
    
    def _modify_config_for_broker_hierarchy(self):
        ''' This method create and retunr config for the main broker
        and modifies the existing federation broker adding broker address of the main one'''

        list_specified_ports = [fed_conf.broker_config.port for fed_conf in self.config.federations.values() if fed_conf.broker_config.port] 
        available_ports = self._get_n_available_tcp_ports(len(self.config.federations)+1, exclude_ports=list_specified_ports)

        core_type = 'tcp' # TODO This is hardcoded IT is not possible to have different core types between cores and brokers!!
        main_broker_port = available_ports.pop() 
        main_broker_config = BrokerConfig(core_type=core_type, port=main_broker_port, 
                                          log_level=self.config.log_level, sub_brokers=len(self.config.federations))
        # Update each federation's broker config to point to the main broker
        for federation_name, federation_conf in self.config.federations.items():
            
            if not self.config.multi_computer :
                # updatting federation broker config
                broker_port = federation_conf.broker_config.port if federation_conf.broker_config.port else available_ports.pop()                 
                federation_conf.broker_config.broker_address = f'tcp://127.0.0.1:{main_broker_port}'
                federation_conf.broker_config.core_type = core_type
                federation_conf.broker_config.port = broker_port
                federation_conf.broker_config.address = f'127.0.0.1:{broker_port}'
                # adding broker address to federate configs
                for federate_name, federate_conf in federation_conf.federate_configs.items():
                    federate_conf.broker_address = federation_conf.broker_config.address
                    federate_conf.core_type = core_type
            else:
                # TODO: i'm using localhost so this broker will not be reachable by remote brokers
                # need to add the publich ip address instead of loaclhost
                # need to add the assertion but for combination of ip:port
                self.logger.error("MULTI-COMPUTER SCENARIO WITH BROKER HIERARCHY NOT IMPLEMENTED YET")

        return main_broker_config

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
            
            broker_logger.info(f"Broker command: {' '.join(broker_cmd)}")
            self.logger.info(f"Starting hierarchy broker cmd: {' '.join(broker_cmd)}")
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

            time.sleep(0.5)  # increased from 0.5 – give main broker time to fully init before sub-brokers register
            if broker_process.poll() is not None:
                stdout, stderr = broker_process.communicate()
                error_msg = f"Broker failed to start: {stderr.decode()}"
                self.logger.error(error_msg)
                broker_logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                success_msg = f"Broker started successfully with PID: {broker_process.pid}"
                self.logger.info(success_msg)
                broker_logger.info(success_msg)
                self.broker_processes.append(broker_process)
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
            
            broker_logger.info(f"Broker command: {' '.join(broker_cmd)}")
            self.logger.info(f"Starting broker for federation {federation_name} cmd: {' '.join(broker_cmd)}")

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
            time.sleep(0.5)

            # waiting depending on the reachability of the broker
            # deadline = time.time() + 10  # 10s timeout
            # while time.time() < deadline:
            #     try:
            #         if broker_conf.host:
            #             with socket.create_connection((broker_conf.host, broker_conf.port), timeout=1):
            #                 break  # port is open, broker is ready
            #         else:
            #             with socket.create_connection(('127.0.0.1', broker_conf.port), timeout=1):
            #                 break  # port is open, broker is ready
            #     except OSError:
            #         time.sleep(0.1)
            # else:
                # raise RuntimeError(f"Broker on port {broker_conf.port} never became ready")

            if broker_process.poll() is not None:
                stdout, stderr = broker_process.communicate()
                error_msg = f"Broker failed to start: {stderr.decode()}"
                self.logger.error(error_msg)
                broker_logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                if broker_process.pid:
                    success_msg = f"Broker started successfully with PID: {broker_process.pid}"
                    self.logger.info(success_msg)
                    broker_logger.info(success_msg)
                    self.broker_processes.append(broker_process)
                    self.metrics['process_counts']['brokers_started'] += 1
                    
                    # Store broker-federation mapping for HELICS queries
                    # Get expected time stop from federation config
                    # TODO: rivedere come funziona il monitoring perche questo expected time credo serva a quello
                    expected_time_stop = 3600.0  
                    for federation_name_key, federation_config in self.config.federations.items():
                        if federation_name_key == federation_name and federation_config.federate_configs:
                            expected_time_stop = self.duration_time
                            break
                    
                    self._broker_federation_map[broker_process] = {
                        'federation_name': federation_name,
                        'port': broker_conf.port,
                        'federates': broker_conf.federates,
                        'expected_time_stop': expected_time_stop
                    }
                    self._start_broker_log_reader(broker_process, broker_logger)
                else:
                    error_msg = "Failed to start broker process"
                    self.logger.error(error_msg)
                    broker_logger.error(error_msg)
                    raise RuntimeError(error_msg)
                    
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
           
            cmd = [
                'python', federate_launcher,
                '--name', federate_name,
                '--scenario_name', self.scenario_name,
                '--federation_name', federation_name,
                '--type', federate_config.type,
                '--simid', self.simulation_id,
                '--redis-url', self.redis_url,
                '--redis-key', self.redis_key,
                '--log-file', str(federate_log_file),
                '--log-level', federate_config.log_level.value
            ]


            self.logger.info(f"Creating local federate: {federate_name} (type: {federate_config.type})")

            # Create subprocess in new process group for proper cleanup
            process = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,  # Create new process group
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
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
    







    def get_federation_current_time(self, broker_process):
        """
        Query HELICS broker for current federation simulation time.
        This is non-blocking and doesn't interfere with simulation.
        
        Args:
            broker_process: The broker subprocess to query
            
        Returns:
            float: Current simulation time or None if query fails
        """
        if not HELICS_AVAILABLE:
            return None
            
        try:
            broker_info = self._broker_federation_map.get(broker_process)
            if not broker_info:
                self.logger.debug("No broker info found for process")
                return None
            
            # Method 1: Try querying broker for global time
            try:
                query = h.helicsCreateQuery("broker", "time")
                h.helicsQuerySetTarget(query, f"broker_{broker_info['federation_name']}")
                result = h.helicsQueryExecute(query, self.query_config['timeout_ms'] / 1000.0)
                h.helicsQueryFree(query)
                
                if result and result.strip():
                    time_val = float(result)
                    self.logger.debug(f"HELICS query returned time: {time_val}")
                    return time_val
            except Exception as e:
                self.logger.debug(f"Broker time query failed: {e}")
            
            # Method 2: Try querying for federation status
            try:
                query = h.helicsCreateQuery("broker", "global_time")
                result = h.helicsQueryExecute(query, self.query_config['timeout_ms'] / 1000.0)
                h.helicsQueryFree(query)
                
                if result and result.strip():
                    time_val = float(result)
                    self.logger.debug(f"Global time query returned: {time_val}")
                    return time_val
            except Exception as e:
                self.logger.debug(f"Global time query failed: {e}")
            
            # Method 3: Fallback - estimate based on elapsed time and expected time step
            # This is not ideal but provides visual feedback
            if hasattr(self, '_simulation_start_real_time'):
                elapsed_real = time.time() - self._simulation_start_real_time
                # Rough estimate: assume simulation runs in real-time initially
                estimated_sim_time = min(elapsed_real * 60, broker_info.get('expected_time_stop', 3600))
                self.logger.debug(f"Using time estimation: {estimated_sim_time:.1f}s")
                return estimated_sim_time
            
            return None
            
        except Exception as e:
            federation_name = broker_info.get('federation_name', 'unknown') if broker_info else 'unknown'
            self.logger.debug(f"HELICS query failed for federation {federation_name}: {e}")
            return None

    def get_all_federation_parameters(self):
        """
        Extract timing parameters for all federations with broker mapping.
        
        Returns:
            dict: Federation parameters keyed by federation name
        """
        federation_params = {}
        
        for federation_name, federation_config in self.config.federations.items():
            
            # Find corresponding broker process
            broker_process = self.find_broker_for_federation(federation_name)
            
            if broker_process and federation_config.federate_configs:
                # Get time_stop from first federate (they should all have same stop time)
                time_stop = federation_config.federate_configs[0].timing_configs.time_stop
                
                federation_params[federation_name] = {
                    'time_stop': time_stop,
                    'broker_process': broker_process,
                    'federation_config': federation_config
                }
        
        return federation_params

    def find_broker_for_federation(self, federation_name):
        """
        Find broker process for a specific federation.
        
        Args:
            federation_name: Name of the federation
            
        Returns:
            subprocess.Popen: Broker process or None if not found
        """
        for broker_process, broker_info in self._broker_federation_map.items():
            if broker_info['federation_name'] == federation_name:
                return broker_process
        
        # Fallback: if only one federation, return first broker
        if len(self.config.federations) == 1 and self.broker_processes:
            return self.broker_processes[0]
            
        return None

    def _can_use_helics_queries(self):
        """
        Check if HELICS queries can be used for progress tracking.
        
        Returns:
            bool: True if HELICS queries are available, enabled, and feasible
        """
        return (self.query_config['enabled'] and
                HELICS_AVAILABLE and 
                self.broker_processes and 
                self._broker_federation_map)

    def get_adaptive_query_frequency(self):
        """
        Calculate adaptive query frequency based on simulation time steps.
        
        Returns:
            float: Query interval in seconds
        """
        if not self.query_config['adaptive']:
            return self.query_config['frequency_ms'] / 1000.0
        
        # Get minimum time step across all federates
        min_time_step = float('inf')
        
        for federation_name, federation_config in self.config.federations.items():
            for _, federate_config in federation_config.federate_configs.items():
                time_period = federate_config.timing_configs.time_period
                min_time_step = min(min_time_step, time_period)
        
        if min_time_step == float('inf'):
            # Fallback to default if no time steps found
            return self.query_config['frequency_ms'] / 1000.0
        
        # Adaptive logic: Query at most 10 times per smallest time step
        # This ensures we capture progress without overwhelming the broker
        adaptive_frequency_s = min_time_step / 10.0
        
        # Apply min/max bounds
        min_freq_s = self.query_config['min_frequency_ms'] / 1000.0
        max_freq_s = self.query_config['max_frequency_ms'] / 1000.0
        
        adaptive_frequency_s = max(min_freq_s, min(max_freq_s, adaptive_frequency_s))
        
        self.logger.debug(f"Adaptive query frequency: {adaptive_frequency_s:.3f}s (based on min_time_step: {min_time_step}s)")
        
        return adaptive_frequency_s

    def configure_query_frequency(self, enabled=None, frequency_ms=None, adaptive=None, timeout_ms=None):
        """
        Configure HELICS query parameters.
        
        Args:
            enabled (bool, optional): Enable/disable progress bar monitoring
            frequency_ms (int, optional): Query frequency in milliseconds
            adaptive (bool, optional): Enable adaptive frequency based on time steps
            timeout_ms (int, optional): Query timeout in milliseconds
        """
        if enabled is not None:
            self.query_config['enabled'] = enabled
            
        if frequency_ms is not None:
            self.query_config['frequency_ms'] = max(self.query_config['min_frequency_ms'], 
                                                   min(self.query_config['max_frequency_ms'], frequency_ms))
            
        if adaptive is not None:
            self.query_config['adaptive'] = adaptive
            
        if timeout_ms is not None:
            self.query_config['timeout_ms'] = max(100, min(5000, timeout_ms))  # 100ms to 5s range
            
        self.logger.info(f"Query config updated: enabled={self.query_config['enabled']}, "
                        f"frequency={self.query_config['frequency_ms']}ms, "
                        f"adaptive={self.query_config['adaptive']}, timeout={self.query_config['timeout_ms']}ms")

    def enable_progress_bar(self, enabled=True):
        """
        Quick method to enable/disable progress bar monitoring.
        
        Args:
            enabled (bool): True to enable progress bar, False to disable
        """
        self.query_config['enabled'] = enabled
        if enabled:
            self.logger.info("Progress bar monitoring ENABLED")
        else:
            self.logger.info("Progress bar monitoring DISABLED - using minimal overhead monitoring")

    def disable_progress_bar(self):
        """
        Quick method to disable progress bar monitoring.
        """
        self.enable_progress_bar(False)

    def _monitor_with_federation_progress(self):
        """
        Monitor federations with HELICS query-based progress tracking.
        Shows simulation time progress for each federation.
        """
        sim_params = self.get_all_federation_parameters()
        
        if not sim_params:
            self.logger.warning("No federation parameters found, falling back to process monitoring")
            return self._monitor_processes_fallback()
        
        self.logger.info(f"Starting HELICS query-based monitoring for {len(sim_params)} federation(s)")
        
        # Try to use rich progress bars if available
        try:
            from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
            from rich.live import Live
            return self._monitor_with_rich_progress(sim_params)
        except ImportError:
            # Fallback to tqdm
            return self._monitor_with_tqdm_federation_progress(sim_params)

    def _monitor_with_rich_progress(self, sim_params):
        """Monitor with rich multi-federation progress bars."""
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
        from rich.console import Console
        
        console = Console()
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed:.1f}s/{task.total:.1f}s)"),
            TimeElapsedColumn(),
            console=console,
            expand=True
        ) as progress:
            
            # Create progress bars for each federation
            federation_bars = {}
            for federation_name, params in sim_params.items():
                task_id = progress.add_task(
                    f"Federation {federation_name}", 
                    total=params['time_stop']
                )
                federation_bars[federation_name] = {
                    'task_id': task_id,
                    'broker_process': params['broker_process'],
                    'last_time': 0.0,
                    'time_stop': params['time_stop']
                }
            
            # Monitor all processes
            active_federate_processes = list(self.federate_processes)
            active_broker_processes = list(self.broker_processes)
            
            # Track peak memory usage during monitoring
            peak_memory = self.metrics['memory_usage']['initial'] if self.metrics['memory_usage']['initial'] else {'rss': 0, 'vms': 0}
            
            # Give brokers time to initialize before querying
            time.sleep(self.query_config['init_delay_s'])
            
            # Calculate adaptive query frequency
            query_interval = self.get_adaptive_query_frequency()
            self.logger.info(f"Using query interval: {query_interval:.3f}s ({1/query_interval:.1f} queries/sec)")
            
            while active_federate_processes or active_broker_processes:
                time.sleep(query_interval)
                
                # Update peak memory usage if psutil is available
                if PSUTIL_AVAILABLE:
                    try:
                        process = psutil.Process()
                        current_memory = process.memory_info()
                        if current_memory.rss > peak_memory['rss']:
                            peak_memory = {
                                'rss': current_memory.rss,
                                'vms': current_memory.vms,
                                'system_available': psutil.virtual_memory().available
                            }
                            self.metrics['memory_usage']['peak'] = peak_memory
                    except Exception:
                        pass
                
                # Update federation progress using HELICS queries
                with self._broker_query_lock:
                    for federation_name, bar_info in federation_bars.items():
                        try:
                            current_time = self.get_federation_current_time(bar_info['broker_process'])
                            
                            if current_time is not None and current_time > bar_info['last_time']:
                                # Update progress bar
                                progress.update(
                                    bar_info['task_id'], 
                                    completed=current_time,
                                    description=f"Federation {federation_name} (T={current_time:.1f}s)"
                                )
                                bar_info['last_time'] = current_time
                                
                                # Check if federation is complete
                                if current_time >= bar_info['time_stop']:
                                    progress.update(
                                        bar_info['task_id'],
                                        description=f"Federation {federation_name} - COMPLETE ✅"
                                    )
                        except Exception as e:
                            self.logger.debug(f"Query failed for {federation_name}: {e}")
                
                # Check for completed processes (reuse existing logic)
                active_federate_processes, active_broker_processes = self._update_process_lists(
                    active_federate_processes, active_broker_processes
                )
            
            # Mark all federations as complete
            for federation_name, bar_info in federation_bars.items():
                progress.update(
                    bar_info['task_id'],
                    completed=bar_info['time_stop'],
                    description=f"Federation {federation_name} - COMPLETE ✅"
                )

        self.logger.info("All processes completed!")
        self._log_execution_summary()

    def _monitor_with_tqdm_federation_progress(self, sim_params):
        """Fallback monitoring with tqdm progress bars."""
        if len(sim_params) == 1:
            # Single federation - use single progress bar
            federation_name = list(sim_params.keys())[0]
            params = sim_params[federation_name]
            
            with tqdm(
                total=params['time_stop'], 
                desc=f"Federation {federation_name}", 
                unit="s",
                bar_format='{l_bar}{bar}| {n:.1f}s/{total:.1f}s [{elapsed}, {rate_fmt}{postfix}]'
            ) as pbar:
                
                active_processes = list(self.federate_processes) + list(self.broker_processes)
                last_time = 0.0
                
                # Give brokers time to initialize
                time.sleep(self.query_config['init_delay_s'])
                
                # Use adaptive query frequency
                query_interval = self.get_adaptive_query_frequency()
                self.logger.info(f"Using query interval: {query_interval:.3f}s")
                
                while active_processes:
                    time.sleep(query_interval)
                    
                    # Query current simulation time
                    current_time = self.get_federation_current_time(params['broker_process'])
                    
                    if current_time is not None and current_time > last_time:
                        pbar.update(current_time - last_time)
                        pbar.set_postfix(sim_time=f"{current_time:.1f}s")
                        last_time = current_time
                    
                    # Update process lists
                    active_processes = self._update_process_lists_simple(active_processes)
                
                pbar.update(params['time_stop'] - last_time)  # Complete the bar
        else:
            # Multiple federations - fall back to process-based monitoring
            self.logger.warning("Multiple federations detected, falling back to process monitoring")
            return self._monitor_processes_fallback()

        self.logger.info("All processes completed!")
        self._log_execution_summary()

    def _monitor_processes_minimal(self):
        """
        Minimal overhead monitoring without progress bars.
        Only logs basic status updates without any visual progress indicators.
        """
        active_federate_processes = list(self.federate_processes)
        active_broker_processes = list(self.broker_processes)
        self.logger.info(f"Monitoring {len(active_federate_processes)} federates and {len(active_broker_processes)} brokers (minimal mode)")

        # Track peak memory usage during monitoring
        peak_memory = self.metrics['memory_usage']['initial'] if self.metrics['memory_usage']['initial'] else {'rss': 0, 'vms': 0}
        
        while active_federate_processes or active_broker_processes:
            time.sleep(1)  # Standard monitoring interval
            
            # Update peak memory usage if psutil is available
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    current_memory = process.memory_info()
                    if current_memory.rss > peak_memory['rss']:
                        peak_memory = {
                            'rss': current_memory.rss,
                            'vms': current_memory.vms,
                            'system_available': psutil.virtual_memory().available
                        }
                        self.metrics['memory_usage']['peak'] = peak_memory
                except Exception:
                    pass
            
            # Check federate processes (without progress bar updates)
            completed_federates = []
            for i, process in enumerate(active_federate_processes):
                if process.poll() is not None:  # Process finished
                    if process.returncode == 0:
                        self.logger.info("✓ Federate completed successfully")
                        self.metrics['process_counts']['federates_completed'] += 1
                    else:
                        self.logger.error(f"✗ Federate failed with code {process.returncode}")
                    completed_federates.append(i)
            
            # Remove completed federate processes
            for i in reversed(completed_federates):
                active_federate_processes.pop(i)
            
            # Check broker processes
            completed_brokers = []
            for i, process in enumerate(active_broker_processes):
                if process.poll() is not None:  # Process finished
                    if process.returncode == 0:
                        self.logger.info("✓ Broker completed successfully")
                        self.metrics['process_counts']['brokers_completed'] += 1
                    else:
                        self.logger.error(f"✗ Broker failed with code {process.returncode}")
                    completed_brokers.append(i)
            
            # Remove completed broker processes
            for i in reversed(completed_brokers):
                active_broker_processes.pop(i)
            
            # Periodic status update (less frequent than with progress bar)
            total_active = len(active_federate_processes) + len(active_broker_processes)
            if total_active > 0:
                # Only log status every few iterations to reduce overhead
                if len(active_federate_processes) + len(active_broker_processes) != total_active:
                    self.logger.info(f"Still running: {len(active_federate_processes)} federates, {len(active_broker_processes)} brokers")

        self.logger.info("All processes completed!")
        self._log_execution_summary()

    def _update_process_lists(self, active_federate_processes, active_broker_processes):
        """
        Update process lists by removing completed processes.
        Returns updated lists of active processes.
        """
        # Check federate processes
        completed_federates = []
        for i, process in enumerate(active_federate_processes):
            if process.poll() is not None:  # Process finished
                if process.returncode == 0:
                    self.logger.info("✓ Federate completed successfully")
                    self.metrics['process_counts']['federates_completed'] += 1
                else:
                    self.logger.error(f"✗ Federate failed with code {process.returncode}")
                completed_federates.append(i)
        
        # Remove completed federate processes
        for i in reversed(completed_federates):
            active_federate_processes.pop(i)
        
        # Check broker processes  
        completed_brokers = []
        for i, process in enumerate(active_broker_processes):
            if process.poll() is not None:  # Process finished
                if process.returncode == 0:
                    self.logger.info("✓ Broker completed successfully")
                    self.metrics['process_counts']['brokers_completed'] += 1
                else:
                    self.logger.error(f"✗ Broker failed with code {process.returncode}")
                completed_brokers.append(i)
        
        # Remove completed broker processes
        for i in reversed(completed_brokers):
            active_broker_processes.pop(i)
        
        return active_federate_processes, active_broker_processes

    def _update_process_lists_simple(self, active_processes):
        """
        Simple version that updates a single list of all processes.
        Returns updated list of active processes.
        """
        completed_processes = []
        for i, process in enumerate(active_processes):
            if process.poll() is not None:  # Process finished
                if process.returncode == 0:
                    self.logger.info("✓ Process completed successfully")
                else:
                    self.logger.error(f"✗ Process failed with code {process.returncode}")
                completed_processes.append(i)
        
        # Remove completed processes
        for i in reversed(completed_processes):
            active_processes.pop(i)
        
        return active_processes

    def _monitor_processes(self):
        """
        Monitor all running federate and broker processes.
        
        This method first tries HELICS query-based monitoring for simulation progress,
        then falls back to process-based monitoring if queries are not available.
        """
        # Check if progress monitoring is enabled
        if not self.query_config['enabled']:
            self.logger.info("Progress bar monitoring disabled - using minimal overhead monitoring")
            return self._monitor_processes_minimal()
        
        # Try HELICS query-based monitoring first
        if self._can_use_helics_queries():
            self.logger.info("Using HELICS query-based progress monitoring")
            return self._monitor_with_federation_progress()
        else:
            self.logger.info("HELICS queries not available, using process-based monitoring")
            return self._monitor_processes_fallback()

    def _monitor_processes_fallback(self):
        """
        Fallback process-based monitoring with progress bar.
        
        This method continuously checks the status of all processes,
        reports when they complete, and maintains lists of active processes.
        """
        active_federate_processes = list(self.federate_processes)
        active_broker_processes = list(self.broker_processes)
        self.logger.info(f"Monitoring {len(active_federate_processes)} federates and {len(active_broker_processes)} brokers")

        # Track peak memory usage during monitoring
        peak_memory = self.metrics['memory_usage']['initial'] if self.metrics['memory_usage']['initial'] else {'rss': 0, 'vms': 0}
        
        # Initialize progress bar if tqdm is available and enabled
        total_processes = len(active_federate_processes) + len(active_broker_processes)
        pbar = None
        
        if TQDM_AVAILABLE and total_processes > 0 and self.query_config['enabled']:
            pbar = tqdm(
                total=total_processes, 
                desc="Federation Progress", 
                unit="process",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} processes [{elapsed}, {rate_fmt}{postfix}]'
            )
        
        while active_federate_processes or active_broker_processes:
            time.sleep(1)
            
            # Update peak memory usage if psutil is available
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    current_memory = process.memory_info()
                    if current_memory.rss > peak_memory['rss']:
                        peak_memory = {
                            'rss': current_memory.rss,
                            'vms': current_memory.vms,
                            'system_available': psutil.virtual_memory().available
                        }
                        self.metrics['memory_usage']['peak'] = peak_memory
                except Exception:
                    pass  # Ignore memory monitoring errors
            
            # Check federate processes
            completed_federates = []
            for i, process in enumerate(active_federate_processes):
                if process.poll() is not None:  # Process finished
                    if process.returncode == 0:
                        self.logger.info("✓ Federate completed successfully")
                        self.metrics['process_counts']['federates_completed'] += 1
                        if pbar:
                            pbar.set_description("Federation Progress - Federate completed")
                    else:
                        self.logger.error(f"✗ Federate failed with code {process.returncode}")
                        if pbar:
                            pbar.set_description("Federation Progress - Federate failed")
                    completed_federates.append(i)
                    if pbar:
                        pbar.update(1)
            
            # Remove completed federate processes
            for i in reversed(completed_federates):
                active_federate_processes.pop(i)
            
            # Check broker processes
            completed_brokers = []
            for i, process in enumerate(active_broker_processes):
                if process.poll() is not None:  # Process finished
                    if process.returncode == 0:
                        self.logger.info("✓ Broker completed successfully")
                        self.metrics['process_counts']['brokers_completed'] += 1
                        if pbar:
                            pbar.set_description("Federation Progress - Broker completed")
                    else:
                        self.logger.error(f"✗ Broker failed with code {process.returncode}")
                        if pbar:
                            pbar.set_description("Federation Progress - Broker failed")
                    completed_brokers.append(i)
                    if pbar:
                        pbar.update(1)
            
            # Remove completed broker processes
            for i in reversed(completed_brokers):
                active_broker_processes.pop(i)
            
            # Status update
            total_active = len(active_federate_processes) + len(active_broker_processes)
            if total_active > 0:
                self.logger.info(f"Still running: {len(active_federate_processes)} federates, {len(active_broker_processes)} brokers")
                if pbar:
                    # Update progress bar with memory info if available
                    postfix_info = {}
                    if PSUTIL_AVAILABLE:
                        try:
                            current_memory = self.get_current_memory_usage()
                            if current_memory:
                                postfix_info['memory'] = f"{current_memory['rss'] / 1024**2:.1f}MB"
                        except Exception:
                            pass
                    pbar.set_description(f"Running: {len(active_federate_processes)} federates, {len(active_broker_processes)} brokers")
                    if postfix_info:
                        pbar.set_postfix(postfix_info)

        # Close progress bar if it was created
        if pbar:
            pbar.set_description("✅ All processes completed!")
            pbar.close()
            
        self.logger.info("All processes completed!")
        self._log_execution_summary()

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
        """
        Log comprehensive execution summary with timing and memory metrics.
        """
        if not hasattr(self, 'logger') or not self.logger:
            return
            
        self.logger.info("\n" + "="*60)
        self.logger.info("FEDERATION EXECUTION SUMMARY")
        self.logger.info("="*60)
        
        # Timing metrics
        if self.metrics['initialization_start'] and self.metrics['initialization_end']:
            init_duration = (self.metrics['initialization_end'] - self.metrics['initialization_start']).total_seconds()
            self.logger.info(f"Initialization time: {init_duration:.3f} seconds")
        
        # Phase durations from delta time tracking
        phase_durations = self.metrics.get('phase_durations', {})
        
        if 'initialization' in phase_durations:
            self.logger.info(f"Initialization time: {phase_durations['initialization']:.3f} seconds")
            
        if 'setup' in phase_durations:
            self.logger.info(f"Setup time: {phase_durations['setup']:.3f} seconds")
        
        if 'simulation' in phase_durations:
            self.logger.info(f"Simulation time: {phase_durations['simulation']:.3f} seconds")
        
        if 'cleanup' in phase_durations:
            self.logger.info(f"Cleanup time: {phase_durations['cleanup']:.3f} seconds")
        
        if self.metrics['total_duration']:
            self.logger.info(f"Total execution time: {self.metrics['total_duration']:.3f} seconds")
        
        # Process metrics
        self.logger.info(f"Brokers started: {self.metrics['process_counts']['brokers_started']}")
        self.logger.info(f"Brokers completed: {self.metrics['process_counts']['brokers_completed']}")
        self.logger.info(f"Federates started: {self.metrics['process_counts']['federates_started']}")
        self.logger.info(f"Federates completed: {self.metrics['process_counts']['federates_completed']}")
        
        # Memory metrics (if available)
        if PSUTIL_AVAILABLE and self.metrics['memory_usage']['initial']:
            def format_bytes(bytes_val):
                """Convert bytes to human readable format."""
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if bytes_val < 1024.0:
                        return f"{bytes_val:.2f} {unit}"
                    bytes_val /= 1024.0
                return f"{bytes_val:.2f} TB"
            
            initial_mem = self.metrics['memory_usage']['initial']
            self.logger.info(f"Initial memory (RSS): {format_bytes(initial_mem['rss'])}")
            
            if self.metrics['memory_usage']['peak']:
                peak_mem = self.metrics['memory_usage']['peak']
                self.logger.info(f"Peak memory (RSS): {format_bytes(peak_mem['rss'])}")
                memory_increase = peak_mem['rss'] - initial_mem['rss']
                self.logger.info(f"Memory increase: {format_bytes(memory_increase)}")
            
            if self.metrics['memory_usage']['final']:
                final_mem = self.metrics['memory_usage']['final']
                self.logger.info(f"Final memory (RSS): {format_bytes(final_mem['rss'])}")
        
        self.logger.info("="*60)
        
        # Save metrics to JSON file for programmatic access
        try:
            metrics_file = self.logger_system.scenario_log_dir / "execution_metrics.json"
            with open(metrics_file, 'w') as f:
                # Convert datetime objects to strings for JSON serialization
                serializable_metrics = {}
                for key, value in self.metrics.items():
                    if isinstance(value, datetime):
                        serializable_metrics[key] = value.isoformat() if value else None
                    elif isinstance(value, dict):
                        serializable_metrics[key] = {}
                        for subkey, subvalue in value.items():
                            if isinstance(subvalue, datetime):
                                serializable_metrics[key][subkey] = subvalue.isoformat() if subvalue else None
                            else:
                                serializable_metrics[key][subkey] = subvalue
                    else:
                        serializable_metrics[key] = value
                
                json.dump(serializable_metrics, f, indent=2)
            self.logger.info(f"Execution metrics saved to: {metrics_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save metrics to JSON: {e}")

    def get_execution_metrics(self):
        """
        Get current execution metrics.
        
        Returns:
            dict: Current metrics dictionary
        """
        return self.metrics.copy()

    def get_current_memory_usage(self):
        """
        Get current memory usage information.
        
        Returns:
            dict: Current memory usage or None if psutil not available
        """
        if not PSUTIL_AVAILABLE:
            return None
            
        try:
            process = psutil.Process()
            return {
                'rss': process.memory_info().rss,
                'vms': process.memory_info().vms,
                'system_available': psutil.virtual_memory().available,
                'system_used_percent': psutil.virtual_memory().percent
            }
        except Exception as e:
            self.logger.warning(f"Failed to get current memory usage: {e}")
            return None

    def _ensure_psutil_available(self):
        """
        Check if psutil is available and provide installation instructions if not.
        """
        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available - memory monitoring will be limited")
            self.logger.info("To enable full memory monitoring, install psutil:")
            self.logger.info("  conda install psutil")
            self.logger.info("  or pip install psutil")







def main(scenario_name, enable_progress_bar=True):
    try:
        print("Scenario Manager - starting SETUP phase")
        with ScenarioManager(scenario_name) as manager:
            # Configure progress bar monitoring
            if not enable_progress_bar:
                manager.disable_progress_bar()
                print("Progress bar monitoring disabled for maximum performance")
            else:
                print("Progress bar monitoring enabled")
            
            # Example: Additional query configuration (optional)
            # manager.configure_query_frequency(frequency_ms=200, adaptive=True, timeout_ms=1000)
            
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
            
            manager.debug_cleanup()

    except Exception as e:
        print(f"Error: {e}")
    # All processes (including multiple brokers) are cleaned up automatically
