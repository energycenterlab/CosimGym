"""
config_reader.py

Parses and validates YAML configuration files into structured dataclasses for the simulation.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

import yaml
from typing import Dict, Any, Optional
import os
import inspect
from dataclasses import fields
from .config_dataclasses import (
    LogLevel,
    FedTimingConfig,
    StartupSyncConfig,
    AutoOffsetConfig,
    SynchronizationConfig,
    FedFlags,
    FedEndpoint,
    FedPublication,
    FedSubscription,
    FedConnections,
    ModelInstantiationConfig,
    ModelConfig,
    MemoryConfig,
    InfluxDBConfig,
    FederateConfig,
    BrokerConfig,
    FederationConfig,
    MultiComputerConfig,
    ScenarioConfig,
    RLHyperparametersConfig,
    RLExplorationConfig,
    RLReplayBufferConfig,
    RLOfflineTrainingConfig,
    RLEarlyStoppingConfig,
    RLTrainingConfig,
    RLEnvironmentConfig,
    RLAgentConfig,
    RLCheckpointingConfig,
    RLLoggingConfig,
    RLTestConfig,
    ReinforcementLearningConfig,
)


# TODO:  this must be refactor following the exact new schema from the dataclasses
#  by creating a method for each datclass and following the hierarchic structure
#  TODO: investigate if its possible to automatically translate dicts into dataclassec with embedded validation using pydantic
#  TODO: once refactored need to update anmd refactor places in which thse methods are utilized (federation manager, RL federate, etc)
def create_dataclass_from_dict(dataclass_type, data_dict: Dict[str, Any]):
    """
    Automatically instantiate a dataclass from a dictionary.
    
    This function handles both required and optional parameters and validates
    that all required fields are present in the input dictionary.
    
    Args:
        dataclass_type: The dataclass type to instantiate
        data_dict (Dict[str, Any]): Dictionary containing the data
        
    Returns:
        Instance of the dataclass
        
    Raises:
        ValueError: If required fields are missing from the dictionary
    """
    # Get all fields from the dataclass
    dataclass_fields = fields(dataclass_type)
    
    # Separate required and optional fields
    required_fields = []
    optional_fields = []
    
    for field in dataclass_fields:
        if field.default == field.default_factory == inspect.Parameter.empty:
            required_fields.append(field.name)
        else:
            optional_fields.append(field.name)
    
    # Check if all required fields are present in the dictionary
    missing_fields = [field for field in required_fields if field not in data_dict]
    if missing_fields:
        raise ValueError(f"Missing required fields for {dataclass_type.__name__}: {missing_fields}")
    
    # Build the arguments dictionary with only the fields that exist in the dataclass
    # and are present in the input dictionary
    valid_fields = {field.name for field in dataclass_fields}
    kwargs = {key: value for key, value in data_dict.items() if key in valid_fields}
    
    return dataclass_type(**kwargs)


def read_yaml(file_path: str) -> dict:
    """
    Basic YAML file reader returning raw dictionary.
    
    Args:
        file_path (str): Path to the YAML file
        
    Returns:
        dict: Parsed YAML content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config




def reconstruct_federate_config_from_dict(config_dict: Dict[str, Any], rl_task = None, rl_config = None) -> FederateConfig:
    """
    Reconstruct a FederateConfig dataclass from a dictionary (e.g., from Redis).
    
    This function properly handles nested dataclasses that were serialized with asdict().
    When configuration is stored in Redis using asdict(), all nested dataclasses become
    dictionaries. This function reconstructs the complete dataclass hierarchy.
    
    Args:
        config_dict: Dictionary containing federate configuration
        
    Returns:
        FederateConfig: Reconstructed federate configuration object
        
    Example:
        >>> config_dict = redis_client.get_json_path(key, path)
        >>> federate_config = reconstruct_federate_config_from_dict(config_dict)
    """
    from .config_dataclasses import LogLevel
    
    # Reconstruct timing config
    timing_config = FedTimingConfig(**config_dict['timing_configs'])

    # Reconstruct startup synchronization policy (optional)
    startup_sync_data = config_dict.get('startup_sync')
    startup_sync = StartupSyncConfig(**startup_sync_data) if startup_sync_data else None
    
    # Reconstruct flags
    flags = FedFlags(**config_dict['flags'])
    
    # Reconstruct connections
    connections_data = config_dict['connections']
    endpoints = [FedEndpoint(**ep) for ep in connections_data.get('endpoints', [])]
    subscribes = [FedSubscription(**sub) for sub in connections_data.get('subscribes', [])]
    publishes = [FedPublication(**pub) for pub in connections_data.get('publishes', [])]
    connections = FedConnections(endpoints=endpoints, subscribes=subscribes, publishes=publishes)
    
    # Reconstruct memory config
    memory_config = MemoryConfig(**config_dict['memory_config']) if config_dict['memory_config'] else MemoryConfig()
    
    # Reconstruct log level enum
    log_level = LogLevel(config_dict.get('log_level', 'INFO'))



    if rl_task:
            # Reconstruct RL_task complete TODO
        rl_task = parse_Rl_configs(rl_task)
        
        fed_conf = FederateConfig(
            name=config_dict['name'],
            type=config_dict['type'],
            id=config_dict['id'],
            core_name=config_dict.get('core_name', None),
            core_type=config_dict.get('core_type', None),
            broker_address=config_dict.get('broker_address', None),
            timing_configs=timing_config,
            flags=flags,
            connections=connections,
            model_configs=None,
            memory_config=memory_config,
            log_level=log_level,
            startup_sync=startup_sync,
            reset_observation_defaults=config_dict.get('reset_observation_defaults'),
        )
        # new dataclass attributes only for RL federate
        fed_conf.rl_task = rl_task
        fed_conf.controlled_models = config_dict['controlled_models']  # dcitionary {full att id:model_name} e.g. {"federation_1.spring_federate.0.force": "spring_mass_damper"}
        fed_conf.observed_models = config_dict['observed_models']
        fed_conf.additional_observed_models = config_dict.get('additional_observed_models', {})
        return fed_conf

        
    else:   # Reconstruct model config normal federates
         # Reconstruct model config
        model_data = config_dict['model_configs']
        instantiation = ModelInstantiationConfig(**model_data['instantiation'])
        model_config = ModelConfig(
            instantiation=instantiation,
            init_state=model_data.get('init_state', {}),
            parameters=model_data.get('parameters', {}),
            inputs=model_data.get('inputs', {}),
            outputs=model_data.get('outputs', {}),
            user_defined=model_data.get('user_defined', {})
        )

        # Reconstruct complete FederateConfig
        return FederateConfig(
            name=config_dict['name'],
            type=config_dict['type'],
            id=config_dict['id'],
            core_name=config_dict.get('core_name', None),
            core_type=config_dict.get('core_type', None),
            broker_address=config_dict.get('broker_address', None),
            timing_configs=timing_config,
            flags=flags,
            connections=connections,
            model_configs=model_config,
            memory_config=memory_config,
            log_level=log_level,
            startup_sync=startup_sync,
            rl_config=rl_config )
        


def read_scenario_config(file_path: str) -> ScenarioConfig:
    """
    Read and parse a scenario configuration file.
    
    This function reads a YAML scenario configuration and returns a fully
    instantiated ScenarioConfig object with all nested federation configurations.
    
    Args:
        file_path (str): Path to the scenario configuration file.
                        Can be relative to scenarios/ directory or absolute path.
        
    Returns:
        ScenarioConfig: Fully instantiated scenario configuration object
        
    Raises:
        FileNotFoundError: If the configuration file is not found
        yaml.YAMLError: If the YAML file is malformed
        KeyError: If required configuration keys are missing
        ValueError: If configuration values are invalid
    """
    # Handle relative file paths by prepending scenarios directory
    if '/' not in file_path and '\\' not in file_path:
        scenarios_dir = os.path.join(os.path.dirname(__file__), '..', 'scenarios')
        file_path = os.path.join(scenarios_dir, file_path + '.yaml')
    elif file_path.startswith('scenarios/') or file_path.startswith('scenarios\\'):
        scenarios_dir = os.path.join(os.path.dirname(__file__), '..')
        file_path = os.path.join(scenarios_dir, file_path + '.yaml')
    
    try:
        raw_config = read_yaml(file_path)
        synchronization_config = _parse_synchronization_config(raw_config.get('synchronization', {}))

        scenario_conf = ScenarioConfig(
            name=raw_config['scenario_name'],
            start_time=raw_config['start_time'],
            end_time=raw_config['end_time'],
            federations={
                name: read_federation_config(name, config, MemoryConfig(**raw_config.get('memory_config', {}))) 
                for name, config in raw_config['federations'].items()
            },      
            memory_config=MemoryConfig(**raw_config.get('memory_config', {})),
            influxdb_config=InfluxDBConfig(**raw_config.get('influxdb_config', {})),
            reinforcement_learning_config=parse_Rl_configs(raw_config.get('reinforcement_learning_config', {})),
            synchronization=synchronization_config,
            log_level=LogLevel(raw_config.get('log_level', 'INFO'))
        )
        return scenario_conf

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")
    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e}")
    except Exception as e:
        raise ValueError(f"Error creating federation configuration: {e}")


def _parse_synchronization_config(sync_dict: Dict[str, Any]) -> SynchronizationConfig:
    """Parse scenario-level synchronization section."""
    if not sync_dict:
        return SynchronizationConfig()

    auto_offset_cfg = create_dataclass_from_dict(
        AutoOffsetConfig,
        sync_dict.get('auto_offset', {})
    )
    startup_default_cfg = create_dataclass_from_dict(
        StartupSyncConfig,
        sync_dict.get('default_startup_sync', {})
    )
    return SynchronizationConfig(
        auto_offset=auto_offset_cfg,
        default_startup_sync=startup_default_cfg,
        default_subscription_causality=sync_dict.get('default_subscription_causality', 'same_step'),
        validate_causality_cycles=sync_dict.get('validate_causality_cycles', True),
    )

def parse_Rl_configs(rl_config_dict: Dict[str, Any]) -> Optional[ReinforcementLearningConfig]:
    if not rl_config_dict:
        return None

    def _none_like_to_none(value):
        if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
            return None
        return value

    # --- agent (contains two nested dataclasses) ---
    agent_dict = rl_config_dict.get('agent', {})
    env_dict = agent_dict.pop('env', {})
    env_config = RLEnvironmentConfig(**env_dict) if env_dict else None
    hp_dict = agent_dict.pop('hyperparameters', None)
    hp_config = RLHyperparametersConfig(**hp_dict) if hp_dict else None
    agent_config = RLAgentConfig(
        **agent_dict,
        env=env_config,
        hyperparameters=hp_config,
    )

    # --- training (contains up to four nested dataclasses) ---
    training_dict = dict(rl_config_dict.get('training', {}))
    exploration_dict    = training_dict.pop('exploration', None)
    replay_buffer_dict  = training_dict.pop('replay_buffer', None)
    offline_config_dict = training_dict.pop('offline_config', None)
    early_stopping_dict = training_dict.pop('early_stopping', None)
    training_config = RLTrainingConfig(
        **training_dict,
        exploration=RLExplorationConfig(**exploration_dict)       if exploration_dict    else None,
        replay_buffer=RLReplayBufferConfig(**replay_buffer_dict)  if replay_buffer_dict  else None,
        offline_config=RLOfflineTrainingConfig(**offline_config_dict) if offline_config_dict else None,
        early_stopping=RLEarlyStoppingConfig(**early_stopping_dict)   if early_stopping_dict else None,
    )

    # --- flat nested configs (all fields are primitives) ---
    checkpointing_dict = rl_config_dict.get('checkpointing', None)
    logging_dict       = rl_config_dict.get('logging', None)
    test_dict          = rl_config_dict.get('test', None)
    if isinstance(test_dict, dict):
        test_dict = dict(test_dict)
        test_dict['checkpoint_path'] = _none_like_to_none(test_dict.get('checkpoint_path'))
    checkpointing_config = RLCheckpointingConfig(**checkpointing_dict) if checkpointing_dict else None
    logging_config       = RLLoggingConfig(**logging_dict)             if logging_dict       else None
    test_config          = RLTestConfig(**test_dict)                   if test_dict          else None

    return ReinforcementLearningConfig(
        agent=agent_config,
        training=training_config,
        checkpointing=checkpointing_config,
        logging=logging_config,
        test=test_config,
        seed=rl_config_dict.get('seed', None),
    )


def read_federation_config(name: str, federation_config: dict, memory_config: MemoryConfig) -> FederationConfig:
    """
    Read and parse a federation configuration from dictionary.
    
    This function takes a federation configuration dictionary and returns
    a fully instantiated FederationConfig object with all nested configurations.
    
    Args:
        name (str): Name of the federation
        federation_config (dict): Dictionary containing federation configuration
        
    Returns:
        FederationConfig: Fully instantiated federation configuration object
        
    Raises:
        ValueError: If configuration values are invalid or validation fails
    """
    raw_config = federation_config
    federation_name = name
    
    # Parse broker configuration
    broker_data = raw_config['broker_config']
    broker_config = BrokerConfig(
        core_type=broker_data['core_type'],
        port=broker_data['port'],
        federates=broker_data['federates'],
        log_level=LogLevel(broker_data.get('log_level', 'INFO'))
    )
    
    # Parse federate configurations
    federate_configs = {}
    for name, fed_data in raw_config['federate_configs'].items():
        id_prefix = f'{federation_name}_{name}'
        federate_config = _parse_federate_config(name,fed_data, id_prefix, memory_config)
        federate_configs[name] = federate_config

    # Create and return the complete FederationConfig
    federation_config_obj = FederationConfig(
        name=str(name),
        broker_config=broker_config,
        federate_configs=federate_configs
    )
    
    if not validate_federation_config(federation_config_obj):
        raise ValueError("Invalid federation configuration")
    
    return federation_config_obj


def _parse_federate_config(name, fed_data: Dict[str, Any], id_prefix: str, memory_config: MemoryConfig) -> FederateConfig:
    """
    Parse a single federate configuration from raw dictionary data.
    
    This function takes raw federate configuration data and converts it into
    a fully instantiated FederateConfig object with all nested configurations.
    
    Args:
        fed_data (Dict[str, Any]): Raw federate configuration data
        id_prefix (str): Prefix for generating unique federate IDs
        memory_config (MemoryConfig): Memory configuration to be associated with the federate
        
    Returns:
        FederateConfig: Parsed federate configuration object
        
    Note:
        Connections and model instances parsing structure needs further design.
    """
    # Parse timing configurations
    timing_data = fed_data['timing_configs']
    timing_config = create_dataclass_from_dict(FedTimingConfig, timing_data)
    timing_config.time_offset_explicit = 'time_offset' in timing_data

    # Optional federate startup synchronization override
    startup_sync_data = fed_data.get('startup_sync')
    startup_sync = create_dataclass_from_dict(StartupSyncConfig, startup_sync_data) if startup_sync_data else None
    
    # Parse flags
    flags_data = fed_data['flags']
    flags_config = create_dataclass_from_dict(FedFlags, flags_data)

    # Parse connections (structure needs further design)
    connections_data = fed_data['connections']
    
    # Parse endpoints
    endpoints = []
    for endpoint_data in connections_data.get('endpoints', []):
        endpoint = FedEndpoint(
            key=endpoint_data['key'],
            name=endpoint_data['name']
        )
        endpoints.append(endpoint)
    
    # Parse subscriptions
    subscribes = []
    for sub_data in connections_data.get('subscribes', []):
        subscription = FedSubscription(
            key=sub_data['key'],
            type=sub_data['type'],
            units=sub_data['units'],
            targets=sub_data.get('targets'),
            causality=sub_data.get('causality', 'same_step'),
            multi_input_handling=sub_data.get('multi_input_handling')
        )
        subscribes.append(subscription)
    
    # Parse publications
    publishes = []
    for pub_data in connections_data.get('publishes', []):
        publication = FedPublication(
            key=pub_data['key'],
            type=pub_data['type'],
            units=pub_data['units']
        )
        publishes.append(publication)
    
    connections = FedConnections(
        endpoints=endpoints,
        subscribes=subscribes,
        publishes=publishes
    )
    
    # Parse model configurations
    model_data = fed_data['model_configs']
    
    # Parse instantiation config
    inst_data = model_data['instantiation']
    instantiation = ModelInstantiationConfig(
        model_name=inst_data['model_name'],
        n_instances=inst_data['n_instances'],
        prefix=inst_data.get('prefix', 'model'),
        parallel_execution=inst_data.get('parallel_execution', False)
    )
    
    model_config = ModelConfig(
        instantiation=instantiation,
        init_state=model_data.get('init_state', {}),
        parameters=model_data.get('parameters', {}),
        inputs=model_data.get('inputs', {}),
        outputs=model_data.get('outputs', {}),
        user_defined=model_data.get('user_defined', {})
    )
    
    # Create and return the complete FederateConfig
    # TODO: quell'id che tanto non uso da nessuna parte va corretto per ora ripete il nome due volte...
    federate_config = FederateConfig(
        name=name,
        type=fed_data['type'],
        id=id_prefix,
        log_level=LogLevel(fed_data.get('log_level', 'INFO')),
        timing_configs=timing_config,
        flags=flags_config,
        connections=connections,
        model_configs=model_config,
        memory_config=memory_config,
        startup_sync=startup_sync,
    )
    
    return federate_config


def validate_federation_config(config: FederationConfig) -> bool:
    """
    Validate a FederationConfig object for consistency and completeness.
    
    This function performs comprehensive validation of federation configuration
    including federate count consistency, unique identifiers, and model instances.
    
    Args:
        config (FederationConfig): The federation configuration to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValueError: If validation fails with specific error message
    """
    # Check if broker federate count matches actual federate count
    if config.broker_config.federates != len(config.federate_configs.keys()):
        raise ValueError(
            f"Broker expects {config.broker_config.federates} federates, "
            f"but {len(config.federate_configs)} are configured"
        )
    
    # Check for unique federate IDs
    federate_ids = [fed.id for _,fed in config.federate_configs.items()]
    if len(federate_ids) != len(set(federate_ids)):
        raise ValueError("Federate IDs must be unique")
    
    # Check for unique federate names
    federate_names = [fed.name for _,fed in config.federate_configs.items()]
    if len(federate_names) != len(set(federate_names)):
        raise ValueError("Federate names must be unique")
    
    # Validate that each federate has at least one model instance
    for fed_name, fed in config.federate_configs.items():
        if fed.model_configs.instantiation.n_instances < 1:
            raise ValueError(f"Federate {fed.name} must have at least one model instance")
    
    return True


# Main execution for testing purposes
if __name__ == "__main__":
    """
    Test configuration reading functionality.
    
    This section is executed when the module is run directly and is used
    for testing the configuration reader with example files.
    """
    conf_test = 'federation_config_exampleFULL.yaml'
    config_obj = read_federation_config(conf_test)
    val = validate_federation_config(config_obj)
    print(f"Configuration validation result: {val}")
