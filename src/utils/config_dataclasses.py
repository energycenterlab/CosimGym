
"""
Configuration Data Classes

This module defines all data classes used for configuration management
in the COSIM Gym framework. These classes represent the structure of
HELICS federations, federates, and their configurations.

Author: Pietro Rando Mazzarino
Date: 2025
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any, Tuple
from enum import Enum

class LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR    = "ERROR"
    WARNING  = "WARNING"
    INFO     = "INFO"
    DEBUG    = "DEBUG"
    NOTSET  = "NOTSET"

    def to_logging_level(self) -> int:
        """Return the corresponding logging level constant."""
        # equivalent to logging.DEBUG, logging.INFO, etc.
        return getattr(logging, self.value)
    
    def as_string(self) -> str:
        """Return the log level as a plain string: 'INFO', 'DEBUG', etc."""
        return self.value
    def as_lc_str(self) -> str:
        """Return the log level as a lowercase string: 'info', 'debug', etc."""
        return self.value.lower()

    def to_helics_level(self) -> str:
        """Return the equivalent HELICS broker log level string.
        HELICS does not have 'info'; the nearest equivalent is 'summary'.
        """
        _map = {
            'DEBUG':    'debug',
            'INFO':     'summary',
            'WARNING':  'warning',
            'ERROR':    'error',
            'CRITICAL': 'error',
            'NOTSET':   'no_print',
        }
        return _map.get(self.value, 'summary')



@dataclass
class FedTimingConfig:
    """
    Configuration for federate timing parameters.
    
    This class defines timing-related settings for HELICS federates,
    including simulation time periods, timeouts, and synchronization options.
    
    Attributes:
        time_period (int): Base time period for the federate
        time_stop (float): Simulation stop time
        start_time (str): Simulation start time
        timeout (int): Timeout value in seconds (default: 30)
        real_period (Optional[int]): Real-time period, defaults to time_period
        time_offset (float): Time offset for the federate (default: 0.0)
        int_max_iterations (int): Maximum iterations (default: 10000)
    """
    real_period: int   # This is the only required field the others will be updated accordingly depending on the scenario configs
    time_period: Optional[int] = None
    time_delta: Optional[float] = None
    time_stop: Optional[float] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    timeout: Optional[int] = 30
    time_offset: Optional[float] = 0.0
    int_max_iterations: Optional[int] = 10000
    # Filled by parser: True if `time_offset` was explicitly set in YAML.
    time_offset_explicit: bool = False
    
    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


@dataclass
class StartupSyncConfig:
    """
    Per-federate startup synchronization policy.

    Attributes:
        enabled: Enable startup input synchronization before first model step.
        force_read_all_subscriptions: Read current values even when inputs are not flagged as updated.
        missing_inputs_policy: One of "error", "warn", "ignore".
        required_inputs: Optional explicit list of required input variable names.
    """
    enabled: bool = True
    force_read_all_subscriptions: bool = True
    require_updated_inputs: bool = True
    require_finite_numeric: bool = True
    invalid_numeric_sentinels: Optional[List[float]] = field(default_factory=lambda: [-1e49])
    missing_inputs_policy: str = "warn"
    invalid_inputs_policy: str = "warn"
    required_inputs: Optional[List[str]] = None

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


@dataclass
class AutoOffsetConfig:
    """
    Scenario-wide automatic time-offset sequencing policy.

    Attributes:
        enabled: Enable dependency-based auto offset assignment.
        offset_step: Offset increment per dependency stage (HELICS time units).
        override_existing_offsets: If True, overwrite explicit time offsets.
    """
    enabled: bool = True
    offset_step: float = 0.1
    override_existing_offsets: bool = False

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


@dataclass
class SynchronizationConfig:
    """
    Scenario-level synchronization policy.
    """
    auto_offset: AutoOffsetConfig = field(default_factory=AutoOffsetConfig)
    default_startup_sync: StartupSyncConfig = field(default_factory=StartupSyncConfig)
    default_subscription_causality: str = "same_step"
    validate_causality_cycles: bool = True

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"
    
@dataclass
class FedFlags:
    """
    Configuration flags for HELICS federate behavior.
    
    This class defines boolean flags that control various aspects of
    federate behavior, error handling, and execution modes.
    
    All attributes default to False unless otherwise specified.
    """
    terminate_on_error: bool = True
    debugging: bool = False
    realtime: bool = False
    uninterruptible: bool = False
    observer: bool = False
    strict_config_checking: bool = False
    source_only: bool = False
    only_transmit_on_change: bool = False
    only_update_on_change: bool = False
    wait_for_current_time_update: bool = False
    restrictive_time_policy: bool = False
    rollback: bool = False
    forward_compute: bool = False
    event_triggered: bool = False
    single_thread_federate: bool = False
    ignore_time_mismatch_warnings: bool = False
    force_logging_flush: bool = False
    dumplog: bool = False
    slow_responding: bool = False

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"

@dataclass
class FedEndpoint:
    """
    Configuration for a HELICS federate endpoint.
    
    Endpoints are used for point-to-point communication between federates.
    
    Attributes:
        key (str): Unique identifier for the endpoint
        name (str): Human-readable name for the endpoint
    
    Note:
        Endpoint design and implementation is still under development.
    """
    key: str 
    name: str 

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"

@dataclass
class FedPublication:
    """
    Configuration for a HELICS federate publication.
    
    Publications are used to publish data that can be subscribed to by other federates.
    
    Attributes:
        key (str): Unique identifier for the publication
        type (str): Data type of the publication
        units (str): Units of measurement for the published data
    """
    key: str
    type: str
    units: str

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"

@dataclass
class FedSubscription:
    """
    Configuration for a HELICS federate subscription.
    
    Subscriptions are used to receive data published by other federates.
    
    Attributes:
        key (str): Unique identifier for the subscription
        type (str): Data type of the subscription
        units (str): Units of measurement for the subscribed data
        targets (Optional[List[str]]): Target publications to subscribe to
        multi_input_handling (Optional[str]): Method for handling multiple inputs
    """
    key: str
    type: str
    units: str
    targets: Optional[Union[ List[Any], Dict[str, Any]]] = None
    causality: str = "same_step"
    multi_input_handling: Optional[Union[ str, Dict[str, Any]]] = None

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"

@dataclass
class FedConnections:
    """
    Container for all federate connection configurations.
    
    This class groups together all types of federate connections including
    endpoints, subscriptions, and publications.
    
    Attributes:
        endpoints (List[FedEndpoint]): List of endpoint configurations
        subscribes (List[FedSubscription]): List of subscription configurations
        publishes (List[FedPublication]): List of publication configurations
    """
    endpoints: List[FedEndpoint]
    subscribes: List[FedSubscription]
    publishes: List[FedPublication]

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"

@dataclass
class ModelInstantiationConfig:
    """
    Configuration for model instantiation within a federate.
    
    This class defines how models are instantiated and executed within federates.
    
    Attributes:
        class_name (str): Name of the model class to instantiate
        model_script (str): Path to the script containing the model
        n_instances (int): Number of model instances to create
    """
    model_name: str
    prefix: str
    n_instances: int = 1
    parallel_execution: bool = False
    max_paraller_workers: Optional[int] = None  # Only relevant if parallel_execution is True

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"

@dataclass
class ModelConfig:
    """
    Complete configuration for federate models.
    
    This class contains all model-related configuration including
    initialization state, parameters, inputs, outputs, and instantiation settings.
    
    Attributes:
        init_state (Dict[str, List[float]]): Initial state values
        parameters (Dict[str, List[float]]): Model parameters
        inputs (Dict[str, List[str]]): Input configuration
        outputs (Dict[str, List[str]]): Output configuration
        instantiation (ModelInstantiationConfig): Instantiation configuration
    """
    init_state: Dict[str, List[float]]
    parameters: Dict[str, List[float]]
    inputs: Dict[str, List[str]]
    outputs: Dict[str, List[str]]
    instantiation: ModelInstantiationConfig
    user_defined: Optional[Dict[str, any]] = field(default_factory=dict)  # For any extra model-specific config
    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"
    
@dataclass
class MemoryConfig:
    """
    Configuration for data storage and persistence.
    
    This class defines how simulation data is stored, buffered, and written
    to InfluxDB during execution.
    
    Attributes:
        batch_size (int): Number of time steps to buffer before writing to InfluxDB
        storage_interval (int): Store data every N steps (default: 1 = every step)
        attrs (List[str]): List of attributes to store. 
                          Use ['all'] to store everything (inputs, outputs, params)
                          Or specify categories: ['inputs', 'outputs', 'params']
                          Or specific attributes: ['position', 'velocity', etc.]
        use_numpy_storage (bool): Use NumPy arrays for storage (faster for large datasets)
        enabled (bool): Enable/disable storage completely
    """
    batch_size: int = 100
    attrs: List[str] = field(default_factory=lambda: ['all'])
    
    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"
    
@dataclass
class InfluxDBConfig:
    """
    Configuration for InfluxDB connection and settings.
    
    Attributes:
        url (str): InfluxDB server URL
        token (str): Authentication token
        org (str): Organization name
        bucket (str): Bucket name for storing data
        auto_start (bool): Automatically start InfluxDB if not running
        health_check_timeout (int): Timeout for health checks in seconds
    """
    url: str = "http://localhost:8086"
    token: str = "mytoken123456"
    org: str = "myorg"
    bucket: str = "cosim_data"
    auto_start: bool = True
    health_check_timeout: int = 30

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"

@dataclass
class FederateConfig:
    """
    Complete configuration for a HELICS federate.
    
    This class contains all configuration needed to create and run a federate,
    including identification, timing, flags, connections, and model settings.
    
    Attributes:
        name (str): Name of the federate
        type (str): Type/class of the federate
        id (int): Unique identifier for the federate
        log_level (int): Logging level for the federate
        timing_configs (FedTimingConfig): Timing configuration
        flags (FedFlags): Behavior flags
        connections (FedConnections): Connection configurations
        model_configs (ModelConfig): Model configuration
    """
    name: str
    type: str
    id: int
    timing_configs: FedTimingConfig
    flags: FedFlags
    connections: FedConnections
    model_configs: ModelConfig
    memory_config : MemoryConfig
    log_level: LogLevel = LogLevel.INFO
    core_name: Optional[str] = None  # e.g., "zmq", "tcp", etc.
    core_type: Optional[str] = "zmq"  # e.g., "zmq", "tcp", etc.
    broker_address: Optional[str] = None  # Address of the broker to connect to (e.g., "tcp://localhost:23404")
    rl_config: Optional[Dict] = None  # Number of steps between episode resets (for RL federates)
    mode: Optional[str] = 'test'  # "training" or "test"    
    startup_sync: Optional[StartupSyncConfig] = None
    # only for RL federates, TODO: create a child class RL_FederateConfig that inherits from this and adds these fields, this is just for testing now
    controlled_models: Optional[Dict[str, str]] = None   # {full_attr_id: model_name}
    observed_models: Optional[Dict[str, str]] = None
    additional_observed_models: Optional[Dict[str, str]] = None
    reset_observation_defaults: Optional[Dict[str, Any]] = None
    rl_task: Optional[Any] = None
    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


# TODO: this class is still under development and may be modified significantly. It is currently used only for RL federates, but it may be necessary to add some of these fields to the general FederateConfig if they are relevant for non-RL federates as well (e.g., controlled_models and observed_models could be useful for non-RL federates that still need to specify which models they control and observe for data storage purposes, etc.)
@dataclass
class AgentConfig:
    """
    Configuration for a reinforcement learning agent.
    
    This class defines the settings for an RL agent, including the model it controls,
    the attributes it observes, and the details of the RL task it is designed to solve.
    
    Attributes:
        model_configs (ModelConfig): Configuration for the models used by the agent"""
    instantiation: ModelInstantiationConfig
    observations: Dict[str, str]  # {full_attr_id: model_name}
    actions: Dict[str, str]  # {full_attr_id: model_name}
    additonal_observations: Optional[Dict[str, str]] = None  # {full_attr_id: model_name}

# TODO add this new dataclass in the generla workflow!
@dataclass
class RLfederateConfig(FederateConfig):
    """
    Configuration for a reinforcement learning federate.
    
    This class extends the base FederateConfig with additional settings specific
    to RL federates, such as controlled and observed models, and RL task details.
    
    Attributes:
        controlled_models (Dict[str, str]): Mapping of controlled attributes to model names
        observed_models (Dict[str, str]): Mapping of observed attributes to model names
        additional_observed_models (Dict[str, str]): Additional observed attributes
        rl_task (Any): Details of the RL task (e.g., reward function, action space, etc.)
    """
    model_configs: AgentConfig
    controlled_models: Dict[str, str] = field(default_factory=dict)   # {full_attr_id: model_name}
    observed_models: Dict[str, str] = field(default_factory=dict)
    additional_observed_models: Dict[str, str] = field(default_factory=dict)
    rl_task: Optional[Any] = None

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"

@dataclass
class BrokerConfig:
    """
    Configuration for the HELICS broker.
    
    The broker coordinates communication between federates in a federation.
    
    Attributes:
        core_type (str): Type of broker core (e.g., 'zmq', 'tcp')
        port (int): Port number for broker communication
        federates (int): Number of federates expected to connect
        log_level (LogLevel): Logging level for the broker
    """
    core_type: str
    port: int
    federates: Optional[int] = None# This is optional only for main broker TODO should be autocalculated on the number of federates
    log_level: LogLevel = LogLevel.INFO
    host:Optional[str] = None  # Hostname or IP address for the broker (default: localhost)
    address: Optional[str] = None  # Full address for THIS broker (e.g., "tcp://localhost:23404")
    broker_address: Optional[str] = None  # e.g., "localhost" or IP address for remote broker
    sub_brokers:Optional[int]= None

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"

@dataclass
class FederationConfig:
    """
    Complete configuration for a HELICS federation.
    
    A federation consists of a broker and multiple federates working together.
    
    Attributes:
        broker_config (BrokerConfig): Broker configuration
        federate_configs (List[FederateConfig]): List of federate configurations
        name (str): Name of the federation
    """
    broker_config: BrokerConfig
    federate_configs: Dict[str, FederateConfig]
    name: str

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"

@dataclass
class MultiComputerConfig:
    """
    TODO: this class is still under development and may be modified significantly.
    Configuration for multi-computer execution of a scenario.
    
    This class defines the """
    ssh_user: str
    ssh_key_path: str
    hostnames: List[str]
    # Additional fields can be added as needed (e.g., port forwarding settings, etc.)

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


@dataclass
class ScenarioConfig:
    """
    Complete configuration for a simulation scenario.
    
    A scenario can contain multiple federations and defines the overall
    simulation parameters.
    
    Attributes:
        name (str): Name of the scenario
        federations (List[FederationConfig]): List of federation configurations
        start_time (str): Scenario start time
        end_time (str): Scenario end time
        
    Note:
        It may be necessary to add a parent broker configuration for scenarios
        with multiple federations, depending on HELICS requirements.
    """
    name: str
    federations: Dict[str, FederationConfig]
    start_time: str
    end_time: str
    memory_config: MemoryConfig
    influxdb_config: InfluxDBConfig
    reinforcement_learning_config: Optional['ReinforcementLearningConfig'] = None
    synchronization: SynchronizationConfig = field(default_factory=SynchronizationConfig)
    log_level: LogLevel = LogLevel.INFO
    multi_computer: bool = False  # Flag to indicate if the scenario is designed for multi-computer execution
    multi_computer_config: Optional[MultiComputerConfig] = None  # Additional configuration for multi-computer setups (e.g., SSH credentials, hostnames, etc.)
    # TODO: Consider adding parent broker configuration for multi-federation scenarios

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"




# ==============================================================================
# REINFORCEMENT LEARNING CONFIGURATION
# ==============================================================================

@dataclass
class RLHyperparametersConfig:
    """
    Core hyperparameters for deep reinforcement learning algorithms.
    
    This class defines common hyperparameters shared across different RL libraries
    (Stable-Baselines3, RLlib, etc.). Algorithm-specific parameters can be passed
    through the algorithm_kwargs field.
    
    Attributes:
        # Core learning parameters
        learning_rate (float): Learning rate for the optimizer
        gamma (float): Discount factor for future rewards [0.0-1.0]
        batch_size (int): Mini-batch size for training
        
        # Network architecture
        net_arch (Optional[List[int]]): Hidden layer sizes (e.g., [256, 256])
        activation_fn (str): Activation function ("relu", "tanh", "elu", "gelu")
        
        # Optimization
        optimizer (str): Optimizer type ("adam", "rmsprop", "sgd")
        gradient_clip (Optional[float]): Max gradient norm for clipping
        
        # Policy gradient specific (PPO, A2C, etc.)
        n_epochs (Optional[int]): Number of epochs for on-policy algorithms
        ent_coef (Optional[float]): Entropy coefficient for exploration
        vf_coef (Optional[float]): Value function coefficient
        gae_lambda (Optional[float]): GAE lambda parameter [0.0-1.0]
        clip_range (Optional[float]): PPO clipping parameter
        normalize_advantages (bool): Normalize advantages
        
        # Q-learning specific (DQN, SAC, TD3, etc.)
        target_update_interval (Optional[int]): Steps between target network updates
        tau (Optional[float]): Soft update coefficient for target network [0.0-1.0]
        
        # Advanced
        use_sde (bool): Use State-Dependent Exploration (for SAC)
        algorithm_kwargs (Dict[str, Any]): Additional algorithm-specific parameters
    """
    # Core parameters
    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 64
    
    # Network architecture
    net_arch: Optional[List[int]] = None
    activation_fn: str = "relu"
    
    # Optimization
    optimizer: str = "adam"
    gradient_clip: Optional[float] = None
    
    # Policy gradient specific
    n_epochs: Optional[int] = None
    ent_coef: Optional[float] = None
    vf_coef: Optional[float] = None
    gae_lambda: Optional[float] = None
    clip_range: Optional[float] = None
    normalize_advantages: bool = True
    
    # Q-learning specific
    target_update_interval: Optional[int] = None
    tau: Optional[float] = None
    
    # Advanced
    use_sde: bool = False
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


@dataclass
class RLExplorationConfig:
    """
    Exploration strategy configuration for reinforcement learning.
    
    Supports multiple exploration strategies compatible with various RL libraries.
    
    Attributes:
        strategy (str): Exploration strategy type
            - "epsilon_greedy": Epsilon-greedy exploration (DQN)
            - "gaussian_noise": Gaussian noise for continuous actions (DDPG, TD3)
            - "ornstein_uhlenbeck": OU noise for continuous actions
            - "entropy": Entropy-based exploration (policy gradient methods)
        
        # Epsilon-greedy parameters
        initial_epsilon (float): Initial exploration rate
        final_epsilon (float): Final exploration rate
        epsilon_decay_steps (int): Steps to decay epsilon from initial to final
        
        # Gaussian/OU noise parameters
        noise_std (float): Standard deviation of exploration noise
        noise_std_decay (float): Decay factor for noise std
        noise_std_min (float): Minimum noise std
        
        # OU noise specific
        ou_theta (float): Mean reversion rate
        ou_sigma (float): Volatility
    """
    strategy: str = "epsilon_greedy"
    
    # Epsilon-greedy
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05
    epsilon_decay_steps: int = 100000
    
    # Gaussian/OU noise
    noise_std: float = 0.1
    noise_std_decay: float = 0.9999
    noise_std_min: float = 0.01
    
    # OU noise
    ou_theta: float = 0.15
    ou_sigma: float = 0.2

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


@dataclass
class RLReplayBufferConfig:
    """
    Replay buffer configuration for off-policy RL algorithms.
    
    Supports both uniform and prioritized experience replay, compatible
    with Stable-Baselines3, RLlib, and other libraries.
    
    Attributes:
        buffer_size (int): Maximum number of transitions to store
        prioritized (bool): Use Prioritized Experience Replay (PER)
        alpha (float): Prioritization exponent for PER [0.0-1.0]
        beta (float): Initial importance sampling exponent [0.0-1.0]
        beta_annealing_steps (int): Steps to anneal beta to 1.0
        n_step (int): Number of steps for n-step returns (1 = standard TD)
        prefill_steps (int): Random transitions to collect before training
    """
    buffer_size: int = 1000000
    prioritized: bool = False
    alpha: float = 0.6
    beta: float = 0.4
    beta_annealing_steps: int = 100000
    n_step: int = 1
    prefill_steps: int = 0

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


@dataclass
class RLOfflineTrainingConfig:
    """
    Configuration for offline reinforcement learning training.
    
    Attributes:
        dataset_path (str): Path to offline dataset (HDF5, pickle, parquet)
        dataset_type (str): Dataset format ("hdf5", "pickle", "parquet", "d4rl")
        n_epochs (int): Number of training epochs
        validation_split (float): Fraction of data for validation [0.0-1.0]
        shuffle (bool): Shuffle dataset during training
        normalize_observations (bool): Normalize observations from dataset
        normalize_rewards (bool): Normalize rewards from dataset
    """
    dataset_path: str = ""
    dataset_type: str = "pickle"
    n_epochs: int = 100
    validation_split: float = 0.1
    shuffle: bool = True
    normalize_observations: bool = True
    normalize_rewards: bool = False

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


@dataclass
class RLEarlyStoppingConfig:
    """
    Early stopping configuration for training.
    TODO: probably this will not be possible with co-simulation
    
    Attributes:
        enabled (bool): Enable early stopping
        metric (str): Metric to monitor ("episode_reward", "success_rate", "loss")
        patience (int): Episodes/evaluations without improvement before stopping
        min_delta (float): Minimum change to qualify as improvement
        mode (str): "max" for maximizing metric, "min" for minimizing
    """
    enabled: bool = False
    metric: str = "episode_reward"
    patience: int = 100
    min_delta: float = 0.01
    mode: str = "max"

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


@dataclass
class RLTrainingConfig:
    """
    Comprehensive configuration for reinforcement learning training.
    
    This class defines all training-related settings for deep RL agents,
    designed to be compatible with multiple RL libraries (Stable-Baselines3,
    RLlib, etc.) through a shared taxonomy of parameters.
    
    Attributes:
        # Training mode
        mode (str): Training mode - "online", "offline", or "mixed"
        online (bool): Enable online training (deprecated, use mode)
        offline (bool): Enable offline training (deprecated, use mode)
        
        # Training duration
        total_timesteps (Optional[int]): Total timesteps to train (overrides n_episodes)
        n_episodes (int): Number of training episodes (if total_timesteps not set)
        max_steps_per_episode (int): Maximum steps per episode
        
        # Online training schedule
        warmup_steps (int): Random action steps before training starts
        train_frequency (int): Steps between training updates
        gradient_steps (int): Gradient updates per training step
        
        # Evaluation during training
        eval_frequency (int): Steps between evaluations (0 = no evaluation)
        n_eval_episodes (int): Number of episodes per evaluation
        eval_deterministic (bool): Use deterministic policy for evaluation
        
        # Sub-configurations
        hyperparameters (RLHyperparametersConfig): Core RL hyperparameters
        exploration (Optional[RLExplorationConfig]): Exploration strategy
        replay_buffer (Optional[RLReplayBufferConfig]): Replay buffer config (off-policy)
        offline_config (Optional[RLOfflineTrainingConfig]): Offline training settings
        early_stopping (Optional[RLEarlyStoppingConfig]): Early stopping criteria
        
        # Logging
        log_interval (int): Steps between logging updates
        verbose (int): Verbosity level (0=none, 1=info, 2=debug)
    """
    # Training mode
    mode: str = "online"  # "online", "offline", "mixed"
    
    # Training duration These established the lenght of 2 loops dipendent with the simulation
    episode_length: int = 100 # Max steps per episode
    n_episodes: int = 100 
    reset_mode: str = 'full' # "full" (reset_env from starting point), "soft"(only applydone flag but do not reset env), "rolling" (reset env but keep the same starting point for the next episode) ,random (reset env and randomize the starting point for the next episode) 
    rolling_window: Optional[int] = None # Only relevant if reset_mode is "rolling", number of timestep to shift the data window at reset 
    reset_period: Optional[int] = None # decoupled from episode_length; defaults to episode_length if not set
    total_steps: Optional[int] = None # do not use is set in scenariomanager

    # Online training schedule This established independet loop from simulation
    warmup_steps: int = 0
    train_frequency: int = 1
    gradient_steps: int = 1
    
    # Evaluation during training
    eval_frequency: int = 10000
    n_eval_episodes: int = 10
    eval_deterministic: bool = True
    
    # Sub-configurations
    exploration: Optional[RLExplorationConfig] = None
    replay_buffer: Optional[RLReplayBufferConfig] = None
    offline_config: Optional[RLOfflineTrainingConfig] = None
    early_stopping: Optional[RLEarlyStoppingConfig] = None
    
    # Logging
    log_interval: int = 100
    verbose: int = 1

    def __post_init__(self):
        if self.reset_period is None:
            self.reset_period = self.episode_length

        if self.total_steps is None:
            self.total_steps = self.n_episodes * self.episode_length

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"

@dataclass
class RLEnvironmentConfig:
    observations: Union[List[Any], Dict[str, Any]]
    actions: Union[List[Any], Dict[str, Any]]
    action_spaces_type: List[str]  # list of action space types corresponding to the actions (only discrete, box)
    action_bins: Optional[List[int]] = None  # list of action space bins for discretization (only for discrete action spaces, int for uniform bins, list of floats for custom bin edges)
    action_boundaries: Optional[List[Tuple[float, float]]] = None  # list of action space boundaries for normalization (only for continuous action spaces)
    additional_observations: Optional[Union[List[Any], Dict[str, Any]]] = None
    observation_causality: Optional[List[str]] = None  # per-observation causality: "same_step" or "next_step"
    additional_observation_causality: Optional[List[str]] = None
    reset_observation_defaults: Optional[Dict[str, Any]] = None
    force_reset_observation_defaults: bool = False
    action_space_remapping: Optional[List[Tuple]] = None  # list of action space remapping corresponding to the actions (for discrete only)
    include_prev_obs: Optional[List[int]] = None  # list of observation keys to include from the previous time step (for partially observable environments)
    # TODO: refine and expand



@dataclass
class RLAgentConfig:
    """
    Configuration for reinforcement learning agent specification.
    
    This class defines which RL agent implementation to use and how to
    instantiate it.
    
    Attributes:
        module (str): Python module path containing the agent class
        class_name (str): Name of the agent class to instantiate
        algorithm (str): RL algorithm type (e.g., "PPO", "DQN", "SAC", "TD3", "A2C")
        library (str): RL library to use ("sb3", "rllib", "custom")
    """
    model_name: str
    env: RLEnvironmentConfig
    algorithm: Optional[str] = None
    library: Optional[str] = None
    hyperparameters: Optional[RLHyperparametersConfig] = None
    reward_function: Optional[str] = None  # dotted import path, e.g. "models.model_catalog.RL_agents.reward_functions.spring_reward"

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"



@dataclass
class RLCheckpointingConfig:
    """
    Configuration for model checkpointing during training.
    
    Attributes:
        enabled (bool): Enable checkpoint saving
        directory (str): Directory to save checkpoints
        save_frequency (int): Save checkpoint every N steps
        save_best (bool): Save best model based on evaluation metric
        best_metric (str): Metric to track for best model ("episode_reward", "success_rate")
        best_mode (str): "max" or "min" for best metric
        keep_last_n (int): Keep only last N checkpoints (0 = keep all)
        save_replay_buffer (bool): Save replay buffer with checkpoint
    """
    enabled: bool = True
    directory: str = "src/models/model_catalog/RL_agents/checkpoints"
    save_frequency: int = 10000
    save_best: bool = True
    best_metric: str = "episode_reward"
    best_mode: str = "max"
    keep_last_n: int = 5
    save_replay_buffer: bool = False
    single_best_checkpoint: str = None  

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"
    def __post_init__(self):
        if self.single_best_checkpoint is None:
            return

        # Avoid duplicating the checkpoint directory when configs are
        # round-tripped through Redis and reconstructed multiple times.
        if os.path.isabs(self.single_best_checkpoint):
            return

        norm_directory = os.path.normpath(self.directory)
        norm_checkpoint = os.path.normpath(self.single_best_checkpoint)
        if norm_checkpoint == norm_directory or norm_checkpoint.startswith(norm_directory + os.sep):
            return

        self.single_best_checkpoint = os.path.join(self.directory, self.single_best_checkpoint)


@dataclass
class RLLoggingConfig:
    """
    Configuration for training logging and monitoring.
    
    Supports multiple backends for compatibility with different workflows.
    
    Attributes:
        backend (str): Logging backend ("tensorboard", "wandb", "mlflow", "csv", "none")
        log_dir (str): Directory for log files
        experiment_name (Optional[str]): Experiment name (auto-generated if None)
        project_name (str): Project name for wandb/mlflow
        tags (List[str]): Tags for experiment tracking
        log_gradients (bool): Log gradient histograms
        log_weights (bool): Log network weights
        
        # WandB specific
        wandb_entity (Optional[str]): WandB username/team
        wandb_mode (str): "online", "offline", or "disabled"
    """
    backend: str = "tensorboard"
    log_dir: str = "logs"
    experiment_name: Optional[str] = None
    project_name: str = "cosim_gym"
    tags: List[str] = field(default_factory=list)
    log_gradients: bool = False
    log_weights: bool = False
    
    # WandB specific
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


@dataclass
class RLTestConfig:
    """

    Configuration for reinforcement learning testing/evaluation phase.
    
    This class defines parameters for testing trained RL agents, including
    model loading, deterministic evaluation, and trajectory saving.
    
    Attributes:
        enabled (bool): Enable test mode (default: False)
        checkpoint_path (Optional[str]): Path to trained model checkpoint to load
        n_episodes (int): Number of test episodes to run (default: 100)
        deterministic (bool): Use deterministic policy during testing (default: True)
        render (bool): Render environment during testing (default: False)
        save_trajectories (bool): Save episode trajectories for analysis (default: False)
        trajectories_path (Optional[str]): Path to save trajectories (default: "results/test_trajectories.pkl")
    """
    total_steps: Optional[int] 

    enabled: bool = False
    checkpoint_path: Optional[str] = None # if None will use best checkpoint from model training or untrained agent
    n_episodes: Optional[int] = None
    episode_length: Optional[int] = None
    deterministic: bool = True
    render: bool = False
    save_trajectories: bool = False
    trajectories_path: Optional[str] = "results/test_trajectories.pkl"

    

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"


@dataclass
class ReinforcementLearningConfig:
    """
    Complete configuration for reinforcement learning
    """
    agent: RLAgentConfig
    training: RLTrainingConfig = None
    checkpointing: Optional[RLCheckpointingConfig] = None
    logging: Optional[RLLoggingConfig] = None
    test: Optional[RLTestConfig] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.training is None and self.test is None:
            raise ValueError(
                "At least one of 'training' or 'test' must be provided in ReinforcementLearningConfig."
            )

    def __repr__(self):
        classname = self.__class__.__name__
        return f"{classname}({self.__dict__})"
