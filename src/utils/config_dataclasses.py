"""
Configuration Models

Pydantic v2 models for COSIM Gym scenario, federation, and federate configuration.
All models use extra='ignore' so YAML keys without a corresponding field are silently
dropped, preserving the permissive behaviour of the original dataclass approach.
"""

import logging
import os
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ==============================================================================
# ENUM
# ==============================================================================

class LogLevel(str, Enum):
    CRITICAL = "CRITICAL"
    ERROR    = "ERROR"
    WARNING  = "WARNING"
    INFO     = "INFO"
    DEBUG    = "DEBUG"
    NOTSET   = "NOTSET"

    def to_logging_level(self) -> int:
        return getattr(logging, self.value)

    def as_string(self) -> str:
        return self.value

    def as_lc_str(self) -> str:
        return self.value.lower()

    def to_helics_level(self) -> str:
        _map = {
            'DEBUG':    'debug',
            'INFO':     'summary',
            'WARNING':  'warning',
            'ERROR':    'error',
            'CRITICAL': 'error',
            'NOTSET':   'no_print',
        }
        return _map.get(self.value, 'summary')


# ==============================================================================
# TIMING / SYNC
# ==============================================================================

class FedTimingConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    real_period: int
    time_period: Optional[int] = None
    time_delta: Optional[float] = None
    time_stop: Optional[float] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    timeout: Optional[int] = 30
    time_offset: Optional[float] = 0.0
    int_max_iterations: Optional[int] = 10000
    time_offset_explicit: bool = False

    @model_validator(mode='before')
    @classmethod
    def _mark_explicit_offset(cls, data: Any) -> Any:
        if isinstance(data, dict) and 'time_offset' in data and 'time_offset_explicit' not in data:
            data = dict(data)
            data['time_offset_explicit'] = True
        return data


class StartupSyncConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    enabled: bool = True
    force_read_all_subscriptions: bool = True
    require_updated_inputs: bool = True
    require_finite_numeric: bool = True
    invalid_numeric_sentinels: Optional[List[float]] = Field(default_factory=lambda: [-1e49])
    missing_inputs_policy: str = "warn"
    invalid_inputs_policy: str = "warn"
    required_inputs: Optional[List[str]] = None


class AutoOffsetConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    enabled: bool = True
    offset_step: float = 0.1
    override_existing_offsets: bool = False


class SynchronizationConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    auto_offset: AutoOffsetConfig = Field(default_factory=AutoOffsetConfig)
    default_startup_sync: StartupSyncConfig = Field(default_factory=StartupSyncConfig)
    default_subscription_causality: str = "same_step"
    validate_causality_cycles: bool = True


# ==============================================================================
# FEDERATE FLAGS
# ==============================================================================

class FedFlags(BaseModel):
    model_config = ConfigDict(extra='ignore')

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


# ==============================================================================
# CONNECTIONS
# ==============================================================================

class FedEndpoint(BaseModel):
    model_config = ConfigDict(extra='ignore')

    key: str
    name: str


class FedPublication(BaseModel):
    model_config = ConfigDict(extra='ignore')

    key: str
    type: str
    units: str


class FedSubscription(BaseModel):
    model_config = ConfigDict(extra='ignore')

    key: str
    type: str
    units: str
    targets: Optional[Union[List[Any], Dict[str, Any]]] = None
    causality: str = "same_step"
    multi_input_handling: Optional[Union[str, Dict[str, Any]]] = None


class FedConnections(BaseModel):
    model_config = ConfigDict(extra='ignore')

    endpoints: List[FedEndpoint] = Field(default_factory=list)
    subscribes: List[FedSubscription] = Field(default_factory=list)
    publishes: List[FedPublication] = Field(default_factory=list)


# ==============================================================================
# MODEL CONFIG
# All dict-valued fields default to {} because YAML often omits them.
# Values typed as Any to allow both scalars and lists (e.g. init_state: {x: 0.0}).
# ==============================================================================

class ModelInstantiationConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    model_name: str
    prefix: str = 'model'
    n_instances: int = 1
    parallel_execution: bool = False
    max_paraller_workers: Optional[int] = None


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    instantiation: ModelInstantiationConfig
    init_state: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    user_defined: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # Injected at runtime by BaseFederate from the federate timing_configs so
    # the model receives its time grid (see BaseFederate._setup_models).
    time_step: Optional[float] = None
    time_stop: Optional[float] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    real_period: Optional[float] = None


# ==============================================================================
# MEMORY CONFIG
# attrs can be the literal string "all" or a list of attribute names.
# ==============================================================================

class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    batch_size: int = 100
    attrs: Union[Literal['all'], List[str]] = Field(default_factory=lambda: ['all'])


# ==============================================================================
# RL CONFIGURATIONS
# Defined before FederateConfig so _FederateConfigBase can reference
# ReinforcementLearningConfig without forward references.
# ==============================================================================

class RLHyperparametersConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    learning_rate: float = 0.0003
    gamma: float = 0.99
    batch_size: int = 64
    net_arch: Optional[List[int]] = None
    activation_fn: str = "relu"
    optimizer: str = "adam"
    gradient_clip: Optional[float] = None
    n_epochs: Optional[int] = None
    ent_coef: Optional[float] = None
    vf_coef: Optional[float] = None
    gae_lambda: Optional[float] = None
    clip_range: Optional[float] = None
    normalize_advantages: bool = True
    target_update_interval: Optional[int] = None
    tau: Optional[float] = None
    use_sde: bool = False
    algorithm_kwargs: Dict[str, Any] = Field(default_factory=dict)


class RLExplorationConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    strategy: str = "epsilon_greedy"
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.05
    epsilon_decay_steps: int = 100000
    noise_std: float = 0.1
    noise_std_decay: float = 0.9999
    noise_std_min: float = 0.01
    ou_theta: float = 0.15
    ou_sigma: float = 0.2


class RLReplayBufferConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    buffer_size: int = 1000000
    prioritized: bool = False
    alpha: float = 0.6
    beta: float = 0.4
    beta_annealing_steps: int = 100000
    n_step: int = 1
    prefill_steps: int = 0


class RLOfflineTrainingConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    dataset_path: str = ""
    dataset_type: str = "pickle"
    n_epochs: int = 100
    validation_split: float = 0.1
    shuffle: bool = True
    normalize_observations: bool = True
    normalize_rewards: bool = False


class RLEarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    enabled: bool = False
    metric: str = "episode_reward"
    patience: int = 100
    min_delta: float = 0.01
    mode: str = "max"


class RLTrainingConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    mode: str = "online"
    episode_length: int = 100
    n_episodes: int = 100
    reset_mode: str = 'full'
    rolling_window: Optional[int] = None
    reset_period: Optional[int] = None
    total_steps: Optional[int] = None
    warmup_steps: int = 0
    train_frequency: int = 1
    gradient_steps: int = 1
    eval_frequency: int = 10000
    n_eval_episodes: int = 10
    eval_deterministic: bool = True
    exploration: Optional[RLExplorationConfig] = None
    replay_buffer: Optional[RLReplayBufferConfig] = None
    offline_config: Optional[RLOfflineTrainingConfig] = None
    early_stopping: Optional[RLEarlyStoppingConfig] = None
    log_interval: int = 100
    verbose: int = 1

    @model_validator(mode='after')
    def _set_derived_fields(self) -> 'RLTrainingConfig':
        if self.reset_period is None:
            self.reset_period = self.episode_length
        if self.total_steps is None:
            self.total_steps = self.n_episodes * self.episode_length
        return self


class RLEnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    observations: Union[List[Any], Dict[str, Any]]
    actions: Union[List[Any], Dict[str, Any]]
    action_spaces_type: List[str]
    action_bins: Optional[List[int]] = None
    action_boundaries: Optional[List[Tuple[float, float]]] = None
    additional_observations: Optional[Union[List[Any], Dict[str, Any]]] = None
    observation_causality: Optional[List[str]] = None
    additional_observation_causality: Optional[List[str]] = None
    reset_observation_defaults: Optional[Dict[str, Any]] = None
    force_reset_observation_defaults: bool = False
    action_space_remapping: Optional[List[Tuple]] = None
    include_prev_obs: Optional[List[int]] = None


class RLAgentConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    model_name: str
    env: RLEnvironmentConfig
    algorithm: Optional[str] = None
    library: Optional[str] = None
    hyperparameters: Optional[RLHyperparametersConfig] = None
    reward_function: Optional[str] = None


class RLCheckpointingConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    enabled: bool = True
    directory: str = "src/models/model_catalog/RL_agents/checkpoints"
    save_frequency: int = 10000
    save_best: bool = True
    best_metric: str = "episode_reward"
    best_mode: str = "max"
    keep_last_n: int = 5
    save_replay_buffer: bool = False
    single_best_checkpoint: Optional[str] = None

    @model_validator(mode='after')
    def _build_checkpoint_path(self) -> 'RLCheckpointingConfig':
        if self.single_best_checkpoint is not None and not os.path.isabs(self.single_best_checkpoint):
            norm_dir = os.path.normpath(self.directory)
            norm_cp = os.path.normpath(self.single_best_checkpoint)
            if not (norm_cp == norm_dir or norm_cp.startswith(norm_dir + os.sep)):
                self.single_best_checkpoint = os.path.join(self.directory, self.single_best_checkpoint)
        return self


class RLLoggingConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    backend: str = "tensorboard"
    log_dir: str = "logs"
    experiment_name: Optional[str] = None
    project_name: str = "cosim_gym"
    tags: List[str] = Field(default_factory=list)
    log_gradients: bool = False
    log_weights: bool = False
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"


class RLTestConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    total_steps: Optional[int] = None
    enabled: bool = False
    checkpoint_path: Optional[str] = None
    n_episodes: Optional[int] = None
    episode_length: Optional[int] = None
    deterministic: bool = True
    render: bool = False
    save_trajectories: bool = False
    trajectories_path: Optional[str] = "results/test_trajectories.pkl"

    @field_validator('checkpoint_path', mode='before')
    @classmethod
    def _normalize_none_like(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().lower() in {'none', 'null', ''}:
            return None
        return v


class ReinforcementLearningConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    agent: RLAgentConfig
    training: Optional[RLTrainingConfig] = None
    checkpointing: Optional[RLCheckpointingConfig] = None
    logging: Optional[RLLoggingConfig] = None
    test: Optional[RLTestConfig] = None
    seed: Optional[int] = None

    @model_validator(mode='after')
    def _validate_training_or_test(self) -> 'ReinforcementLearningConfig':
        if self.training is None and self.test is None:
            raise ValueError("At least one of 'training' or 'test' must be provided.")
        return self


# ==============================================================================
# FEDERATE CONFIGS — discriminated union on `type`
# ==============================================================================

class _FederateConfigBase(BaseModel):
    model_config = ConfigDict(extra='ignore')

    name: str
    id: str
    timing_configs: FedTimingConfig
    flags: FedFlags = Field(default_factory=FedFlags)
    connections: FedConnections = Field(default_factory=FedConnections)
    log_level: LogLevel = LogLevel.INFO
    core_name: Optional[str] = None
    core_type: Optional[str] = "zmq"
    broker_address: Optional[str] = None
    rl_config: Optional[Dict] = None
    mode: Optional[str] = 'test'
    startup_sync: Optional[StartupSyncConfig] = None
    reset_observation_defaults: Optional[Dict[str, Any]] = None
    rl_task: Optional[ReinforcementLearningConfig] = None


class BaseFederateConfig(_FederateConfigBase):
    type: Literal["base"]
    model_configs: ModelConfig
    memory_config: MemoryConfig


class RLFederateConfig(_FederateConfigBase):
    type: Literal["rl"]
    model_configs: Optional[ModelConfig] = None
    memory_config: Optional[MemoryConfig] = None
    controlled_models: Optional[Dict[str, str]] = None
    observed_models: Optional[Dict[str, str]] = None
    additional_observed_models: Optional[Dict[str, str]] = None


FederateConfig = Annotated[
    Union[BaseFederateConfig, RLFederateConfig],
    Field(discriminator='type')
]


# ==============================================================================
# BROKER AND FEDERATION
# ==============================================================================

class BrokerConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    core_type: Optional[str] = None
    port: Optional[int] = None
    federates: Optional[int] = None
    log_level: LogLevel = LogLevel.INFO
    host: Optional[str] = None
    address: Optional[str] = None
    broker_address: Optional[str] = None
    sub_brokers: Optional[int] = None


class FederationConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    name: str
    broker_config: BrokerConfig = Field(default_factory=BrokerConfig)
    federate_configs: Dict[str, FederateConfig]

    @model_validator(mode='before')
    @classmethod
    def _inject_federate_ids(cls, data: Any) -> Any:
        """Inject name and id into each federate dict from the dict key + federation name."""
        if not isinstance(data, dict):
            return data
        fed_name_prefix = data.get('name', '')
        fed_configs = data.get('federate_configs', {})
        if not isinstance(fed_configs, dict):
            return data
        data = dict(data)
        new_fed_configs: Dict[str, Any] = {}
        for fed_name, fdata in fed_configs.items():
            if isinstance(fdata, dict):
                fdata = dict(fdata)
                fdata.setdefault('name', fed_name)
                fdata.setdefault('id', f'{fed_name_prefix}_{fed_name}')
            new_fed_configs[fed_name] = fdata
        data['federate_configs'] = new_fed_configs
        return data

    @model_validator(mode='after')
    def _validate(self) -> 'FederationConfig':
        if self.broker_config.federates is not None:
            if self.broker_config.federates != len(self.federate_configs):
                raise ValueError(
                    f"Broker expects {self.broker_config.federates} federates, "
                    f"but {len(self.federate_configs)} are configured"
                )
        ids = [f.id for f in self.federate_configs.values()]
        if len(ids) != len(set(ids)):
            raise ValueError("Federate IDs must be unique")
        names = [f.name for f in self.federate_configs.values()]
        if len(names) != len(set(names)):
            raise ValueError("Federate names must be unique")
        for fed in self.federate_configs.values():
            if isinstance(fed, BaseFederateConfig) and fed.model_configs.instantiation.n_instances < 1:
                raise ValueError(f"Federate {fed.name} must have at least one model instance")
        return self


# ==============================================================================
# TOP-LEVEL SCENARIO CONFIG
# ==============================================================================

class MultiComputerConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    ssh_user: str
    ssh_key_path: str
    hostnames: List[str]


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    name: str
    federations: Dict[str, FederationConfig]
    start_time: str
    end_time: str
    memory_config: MemoryConfig
    reinforcement_learning_config: Optional[ReinforcementLearningConfig] = None
    synchronization: SynchronizationConfig = Field(default_factory=SynchronizationConfig)
    log_level: LogLevel = LogLevel.INFO
    multi_computer: bool = False
    multi_computer_config: Optional[MultiComputerConfig] = None

    @model_validator(mode='before')
    @classmethod
    def _prepare_federations(cls, data: Any) -> Any:
        """
        Inject federation name into each federation dict and propagate
        scenario-level memory_config into every federate that lacks one.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        memory_config = data.get('memory_config', {})
        feds = data.get('federations', {})
        if not isinstance(feds, dict):
            return data
        new_feds: Dict[str, Any] = {}
        for fed_name, fed_data in feds.items():
            if isinstance(fed_data, dict):
                fed_data = dict(fed_data)
                fed_data.setdefault('name', fed_name)
                # Propagate memory_config to each federate dict that doesn't have one
                fed_configs = fed_data.get('federate_configs', {})
                if isinstance(fed_configs, dict):
                    new_fed_configs: Dict[str, Any] = {}
                    for fname, fdata in fed_configs.items():
                        if isinstance(fdata, dict):
                            fdata = dict(fdata)
                            fdata.setdefault('memory_config', memory_config)
                        new_fed_configs[fname] = fdata
                    fed_data['federate_configs'] = new_fed_configs
            new_feds[fed_name] = fed_data
        data['federations'] = new_feds
        return data
