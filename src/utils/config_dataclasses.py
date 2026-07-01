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
    # Realtime pacing tolerance (only meaningful with flags.realtime: true). None = HELICS default.
    rt_lag: Optional[float] = None
    rt_lead: Optional[float] = None

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
    user_defined: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # All fields below are injected at runtime by BaseFederate._register_entities()
    # and must never appear in scenario YAML files.
    # inputs/outputs: derived from connections.subscribes/publishes key lists.
    # time_*: propagated from the federate timing_configs block.
    inputs: Optional[List[str]] = None
    outputs: Optional[List[str]] = None
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
# STREAMING CONFIG
# Outbound MQTT mirror, opt-in on any federate type (base/rl/interface).
# ==============================================================================

class StreamingConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')

    stream: bool = False
    stream_topic_prefix: Optional[str] = None
    every_n_ticks: int = 1


# ==============================================================================
# RL CONFIGURATIONS
# Defined before FederateConfig so _FederateConfigBase can reference
# ReinforcementLearningConfig without forward references.
# ==============================================================================

# Four orthogonal axes:
#   (a) environment  — the MDP (observations/actions/reward/reset)
#   (b) agent        — the solver (library/algorithm/hyperparameters)
#   (c) run          — what to execute (train/eval/test)
#   (d) experiment   — orthogonal infra (name/checkpoint/logging/offline)
# All RL models use extra='forbid'.


# ---- (a) ENVIRONMENT — the MDP. Framework-agnostic. ----

class ObservationSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    causality: Literal["same_step", "next_step"] = "same_step"
    history: int = 0                                   # frame-stack depth (0 = current only)
    reset_default: Optional[float] = None
    role: Literal["state", "extra"] = "state"          # extra = visible to reward/log, not policy
    space: Optional[str] = None                         # override; else derived from catalog
    bounds: Optional[Tuple[float, float]] = None


class ActionSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    space: Literal["box", "discrete", "multidiscrete", "multibinary"] = "box"
    bounds: Optional[Tuple[float, float]] = None
    # Number of discrete levels when discretizing a continuous variable. Validated at
    # runtime in RL_Federate._prepare_act_dict where catalog type/bounds are known.
    bins: Optional[int] = None


class ResetConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    mode: Literal["full", "rolling", "none"] = "full"
    period: Optional[int] = None                        # defaults to run.train.episode_length
    rolling_window: Optional[int] = None
    force_defaults: bool = False

    @model_validator(mode='after')
    def _check_rolling(self) -> 'ResetConfig':
        if self.mode == "rolling" and self.rolling_window is None:
            raise ValueError("reset.mode 'rolling' requires 'rolling_window'")
        return self


class EnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    observations: Dict[str, ObservationSpec]
    actions: Dict[str, ActionSpec]
    reward: Optional[str] = None
    termination: Optional[str] = None
    reset: ResetConfig = Field(default_factory=ResetConfig)

    @field_validator('observations', 'actions', mode='before')
    @classmethod
    def _coerce_null_specs(cls, v: Any) -> Any:
        # Assisted shorthand: `federation.fed.0.var:` (null value) == default spec.
        if isinstance(v, dict):
            return {k: ({} if spec is None else spec) for k, spec in v.items()}
        return v

    @model_validator(mode='after')
    def _non_empty(self) -> 'EnvironmentConfig':
        if not self.observations:
            raise ValueError("environment.observations must define at least one observation")
        if not self.actions:
            raise ValueError("environment.actions must define at least one action")
        return self


# ---- (b) AGENT — the solver. ----

class Hyperparameters(BaseModel):
    # All Optional → unset fields omitted so backend applies its own default.
    model_config = ConfigDict(extra='forbid')

    learning_rate: Optional[float] = None
    gamma: Optional[float] = None
    batch_size: Optional[int] = None
    net_arch: Optional[List[int]] = None
    train_frequency: Optional[int] = None
    gradient_steps: Optional[int] = None

    def as_kwargs(self) -> Dict[str, Any]:
        """Only explicitly-set fields, for forwarding to a backend constructor."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    model_name: str                                     # catalog key → concrete agent class
    backend: Optional[str] = None                       # informational now; adapter dispatch later
    algorithm: Optional[str] = None
    policy: Optional[str] = None
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    params: Dict[str, Any] = Field(default_factory=dict)  # backend-specific escape hatch


# ---- (c) RUN — what to execute. Single source of truth for length. ----

class PhaseConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    episodes: int
    episode_length: int
    deterministic: bool = False
    checkpoint: Optional[str] = None

    @field_validator('checkpoint', mode='before')
    @classmethod
    def _normalize_none_like(cls, v: Any) -> Any:
        if isinstance(v, str) and v.strip().lower() in {'none', 'null', ''}:
            return None
        return v

    @property
    def total_steps(self) -> int:
        return self.episodes * self.episode_length


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    every_steps: Optional[int] = None
    episodes: int = 10
    deterministic: bool = True


class RunConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    mode: Literal["online", "offline", "mixed"] = "online"
    train: Optional[PhaseConfig] = None
    eval: Optional[EvalConfig] = None
    test: Optional[PhaseConfig] = None

    @model_validator(mode='after')
    def _validate_phases(self) -> 'RunConfig':
        if self.train is None and self.test is None:
            raise ValueError("run must define at least one of 'train' or 'test'")
        if self.train is None and self.test is not None and self.test.checkpoint is None:
            raise ValueError("test-only run (no train phase) requires run.test.checkpoint")
        return self


# ---- (d) EXPERIMENT — orthogonal infrastructure. ----

class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    dir: str = "src/models/model_catalog/RL_agents/checkpoints"
    best: Optional[str] = None

    @property
    def best_path(self) -> Optional[str]:
        """Resolve `best` against `dir` unless already absolute or already under `dir`."""
        if self.best is None:
            return None
        if os.path.isabs(self.best):
            return self.best
        norm_dir = os.path.normpath(self.dir)
        norm_best = os.path.normpath(self.best)
        if norm_best == norm_dir or norm_best.startswith(norm_dir + os.sep):
            return self.best
        return os.path.join(self.dir, self.best)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: Optional[str] = None
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: Optional[Dict[str, Any]] = None
    offline: Optional[Dict[str, Any]] = None             # only when run.mode in {offline, mixed}


# ---- ROOT ----

class ReinforcementLearningConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    seed: Optional[int] = None
    environment: EnvironmentConfig
    agent: AgentConfig
    run: RunConfig
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)


# ==============================================================================
# INTERFACE FEDERATE CONFIG — digital-twin bidirectional bridge (type: interface)
# extra='forbid' like the RL axes: typos in this block must fail loudly.
# ==============================================================================

class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str                                   # catalog key, e.g. mqtt_adapter
    params: Dict[str, Any] = Field(default_factory=dict)


class StreamSpec(BaseModel):
    """co-sim -> external: subscribe in HELICS, publish to the adapter."""
    model_config = ConfigDict(extra='forbid')

    helics_key: str
    topic: str
    type: str = "double"
    units: str = ""
    every_n_ticks: int = 1


class BridgeSpec(BaseModel):
    """external -> co-sim: adapter inbound, publish onto a HELICS key."""
    model_config = ConfigDict(extra='forbid')

    helics_key: str
    topic: str
    bounds: Optional[Tuple[float, float]] = None
    scope: Literal["input", "output", "param"] = "input"
    mode: Literal["replace", "passthrough"] = "replace"


class InterfaceConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    adapter: AdapterConfig
    streams: List[StreamSpec] = Field(default_factory=list)
    bridges: List[BridgeSpec] = Field(default_factory=list)


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
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)


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


class InterfaceFederateConfig(_FederateConfigBase):
    type: Literal["interface"]
    interface_config: Optional[InterfaceConfig] = None
    # ScenarioManager._enrich_dynamic_catalog_metadata reads .model_configs on every
    # federate type generically (RLFederateConfig already declares it Optional=None).
    model_configs: Optional[ModelConfig] = None
    # BaseFederate.__init__ reads config.memory_config.batch_size unconditionally;
    # the interface federate keeps empty storage (see InterfaceFederate.update_storage),
    # so this only needs to exist, not do anything.
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)


FederateConfig = Annotated[
    Union[BaseFederateConfig, RLFederateConfig, InterfaceFederateConfig],
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
