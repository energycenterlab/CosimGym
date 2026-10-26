"""
base_model.py

Primary base class and logging adapters for all physical and behavioral models.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""
import logging
import pprint
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Set
from .model_catalog.ModelCatalog import ModelCatalog, ModelMetadata, InterfaceType
from datetime import timedelta, datetime

pp = pprint.PrettyPrinter(indent=4)


class ModelLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds model prefix to logs."""
    
    def __init__(self, logger, model_name):
        super().__init__(logger, {})
        self.model_name = model_name
    
    def process(self, msg, kwargs):
        # Add prefix to the message itself
        # TODO change it in something more nice modle name is not correct must be instance name
        return f"🔧 Inside MODEL: {self.model_name} - {msg}", kwargs
    


@dataclass
class State:
    """A model instance's mutable runtime state: parameters, current inputs/outputs, and simulation time."""

    parameters: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    # Absolute clock: the federation's tick count and the datetime it maps to
    # (``start_time + ts * period``). Set by the model's ``_step``, monotonic, never
    # moved by a reset. Results and logs use it.
    time: Optional[datetime] = None
    ts: Optional[int] = 0
    # Relative clock: the simulated moment the model represents. Moves with resets
    # and horizon restarts; date-dependent models read ``rel_time``. Both are
    # managed by ``BaseModel``; see the "Clocks" note there.
    rel_ts: Optional[int] = 0
    rel_time: Optional[datetime] = None


    def __repr__(self) -> str:
        import pprint
        cls_name = self.__class__.__name__
        pp = pprint.PrettyPrinter(indent=2, compact=False, width=80)
        param_str = f"parameters=\n{pp.pformat(self.parameters)}"
        inputs_str = f"inputs=\n{pp.pformat(self.inputs)}"
        outputs_str = f"outputs=\n{pp.pformat(self.outputs)}"
        time_str = f"time={self.time!r}"
        ts_str = f"ts={self.ts!r}"
        rel_ts_str = f"rel_ts={self.rel_ts!r}, rel_time={self.rel_time!r}"
        return (f"{cls_name}(\n"
                f"  {param_str},\n"
                f"  {inputs_str},\n"
                f"  {outputs_str},\n"
                f"  {time_str},\n"
                f"  {ts_str},\n"
                f"  {rel_ts_str}\n"
                f")")




class BaseModel(ABC):
    """
    Base model class for HELICS co-simulation framework.
    
    This abstract base class defines the standardized interface that all models
    must implement to ensure consistent integration with federate classes.
    
    All derived models must implement the mandatory methods: initialize, step, and finalize.
    All models must also define the mandatory class variables: state, inputs, outputs, 
    parameters, and init_state.
    """
    def __init__(self, name, catalog_metadata, user_config, logger):
        """
        Initialize the base model with mandatory class variables.
        
        These variables must be defined by all derived classes to ensure
        consistent interface with federate classes.
        """

        # utility attributes
        self.name = name
        self.mod_num = int(name.split('.')[-1])  
        self.logger = ModelLoggerAdapter(logger, name)
        self.config = user_config
        self.user_defined_configs = user_config.user_defined or {}
        self.metadata = catalog_metadata
        self.logger.debug(f"(0) - Model: {self.name} Constructed!\n metadata from catalog: {pp.pformat(self.metadata)}\n user config: {pp.pformat(self.config)}")


        # Mandatory class variables for all models
        self.ts = 0  # time step, will be set in _set_attrs
        # self.time_stop = None # Maximum simulation time for time-dependent models
        self.start_time = None  # Start time for time-dependent models
        self.real_period = None  # Real time period for time-dependent models
        self.date_time = None

        # model_state
        self.state = State()
        self.init_state = State()

        # Some models cannot run for an unlimited span of simulated time: an
        # EnergyPlus FMU stops at the end of its RunPeriod, a CSV at its last row.
        # Such a model declares ``max_sim_time`` (and, if its calendar does not start
        # at the scenario start, ``sim_start_date``) in its catalog entry and is
        # restarted at its origin whenever it reaches the horizon, RL or not.
        # ``max_sim_time is None`` (the default) leaves that inert. A scenario can
        # override both under the model's ``user_defined`` block, which is mostly
        # useful to exercise a restart in a short run.
        _scenario_limits = (user_config.user_defined or {}) if user_config else {}
        self.max_sim_time = _scenario_limits.get(
            'max_sim_time', getattr(catalog_metadata, 'max_sim_time', None))
        self.sim_start_date = _scenario_limits.get(
            'sim_start_date', getattr(catalog_metadata, 'sim_start_date', None))
        # Ticks the current run of the model has to cover (until the next reset or
        # the end of the scenario). Set by the federate: here for the first run, by
        # ``_reset`` for the following ones. None = unknown.
        self.span_ticks = getattr(user_config, 'initial_span_ticks', None)
        # Instantiate the model:
        self._instantiate()
        self.logger.debug(f"(1) - Model '{self.name}' Instantiated with state: {self.state} and init_state: {self.init_state}")
        
        self.initialize()
        self.logger.debug(f"(2) - Model '{self.name}' Initialized with state: {self.state} and init_state: {self.init_state}")

    def _get_defaults(self, interface_type: InterfaceType) -> Dict[str, Any]:
        """Get default values from catalog"""
        if self.metadata:
            return self.metadata.get_defaults(interface_type)
        return {}

    def _resolve_parameter_value(self, param_name: str, user_value: Any, default_value: Any) -> Any:
        """Resolve parameter value based on instance number. and return warining in case of not explicitly given params and in case of uncorrect uses of list for multiple model instances"""
        if user_value is None:
            self.logger.warning(f'Model Parameter: "{param_name}" not provided, using default from catalog')
            return default_value
            
        if isinstance(user_value, list):
            if len(user_value) <= self.mod_num:
                self.logger.warning(
                    f"Parameter '{param_name}' list too short for instance {self.mod_num}, using default"
                )
                return default_value
            return user_value[self.mod_num]
        else:
            return user_value

    def _instantiate(self, inp_list=None, out_list=None) -> None:
        """Set model interfaces using catalog metadata."""
        # TODO the resolve value (for when we have list of attrs for different model instances must be implemented in the federate to avoid passing huge lists to model base class)

        self.init_state.ts = 0
        self.init_state.rel_ts = 0
        self.init_state.rel_time = datetime.fromisoformat(self.config.start_time)
        self.init_state.time = datetime.fromisoformat(self.config.start_time)  # Initialize time in state
        # self.time_stop = self.config.time_stop  # Maximum simulation time for time-dependent models
        self.start_time = datetime.fromisoformat(self.config.start_time)  # Start time for time-dependent models
        self.real_period = self.config.real_period  # Real time period for time-dependent models
        # self.date_time = self.start_time
        self._origin_s = self._origin_seconds()
        self._epoch = self._horizon_epoch(0)
        self._span_end_rel = self.span_ticks  # relative tick the current run must reach (None = unknown)

        # Get defaults from catalog
        default_parameters = self._get_defaults(InterfaceType.PARAMETER)
        default_inputs = self._get_defaults(InterfaceType.INPUT)
        default_outputs = self._get_defaults(InterfaceType.OUTPUT)
        
        # Use provided lists or fall back to catalog defaults
        user_inputs = self.config.inputs or []
        user_outputs = self.config.outputs or []
        user_parameters = self.config.parameters or {}
        user_init_state = self.config.init_state or {}

        # override defualt attrs with user provided ones

        # TODO add the logic of required paramters Validate required parameters are available
        # missing_required_params = required_parameters - set(default_parameters.keys())
        # if missing_required_params:
        #     raise ValueError(f"Missing required parameters: {missing_required_params}")
        
        # Build parameters dictionary
        self.parameters = {}
        for param_name, default_value in default_parameters.items():
            user_value = user_parameters.get(param_name)
            resolved_value = self._resolve_parameter_value(param_name, user_value, default_value)
            self.init_state.parameters[param_name] = resolved_value
        
        # Initialize inputs and outputs
        self.inputs = {}
        for inp in user_inputs:
            if inp in default_inputs.keys():
                default_value = default_inputs.get(inp)
                user_value = user_init_state.get(inp, None)
                resolved_value = self._resolve_parameter_value(inp, user_value, default_value)
                self.init_state.inputs[inp] = resolved_value
            else:
                self.logger.warning(f"Input '{inp}' not defined in catalog, Not initialized")
        
        self.outputs = {}
        for out in user_outputs:
            if out in default_outputs.keys():
                default_value = default_outputs.get(out)
                user_value = user_init_state.get(out, None)
                resolved_value = self._resolve_parameter_value(out, user_value, default_value)
                self.init_state.outputs[out] = resolved_value
            else:
                self.logger.warning(f"Output '{out}' not defined in catalog, Not initialized")
        
        
        # Initialize current state
        self.state =copy.deepcopy( self.init_state)
      
    # ------------------------------------------------------------------
    # Clocks
    # ------------------------------------------------------------------
    # Absolute: ``state.ts`` / ``state.time``. The federation's own count of ticks;
    # monotonic, never moved by a reset. Results and logs use it.
    # Relative: ``state.rel_ts`` / ``state.rel_time``. The simulated moment the model
    # represents. It moves with an RL reset (rolling jumps it, full rewinds it) and
    # otherwise keeps going: a horizon restart does NOT fold it back (after December
    # comes the next January). Models that depend on the date use ``rel_time``.
    # A model with a horizon (``max_sim_time``) maps that clock onto its own window
    # [origin, origin + max_sim_time) with ``horizon_pos_s``; its backend (FMU start
    # time, CSV row) uses that position, and ``_step`` restarts the model each time
    # the clock crosses into a new window.
    # Convention, same as ``state.time``: during a step both name the *end* of that
    # step (``start + (rel_ts + 1) * period``), so with no reset ``rel_time == time``.

    def _origin_seconds(self) -> float:
        """Seconds from ``start_time`` to the model's origin: ``sim_start_date``
        (``MM-DD`` in the year of ``start_time``, or ``YYYY-MM-DD``), else the
        scenario start itself."""
        date = self.sim_start_date
        if not date:
            return 0.0
        date = str(date)
        if len(date) <= 5:
            date = f"{self.start_time.year}-{date}"
        return (datetime.fromisoformat(date) - self.start_time).total_seconds()

    def _horizon_epoch(self, rel_ts: int) -> int:
        """Index of the horizon window *rel_ts* falls in (0 without a horizon)."""
        if not self.max_sim_time:
            return 0
        return int((rel_ts * self.real_period - self._origin_s) // self.max_sim_time)

    def horizon_pos_s(self, rel_ts: Optional[int] = None) -> float:
        """Seconds into the model's horizon window at the start of tick *rel_ts*
        (default: the current one). The clock folded onto [0, max_sim_time); the
        plain ``rel_ts * real_period`` for a model without a horizon."""
        rel_ts = self.state.rel_ts if rel_ts is None else rel_ts
        if not self.max_sim_time:
            return rel_ts * self.real_period
        return (rel_ts * self.real_period - self._origin_s) % self.max_sim_time

    def _step(self, ts, inputs):
        """Internal step: advance the clocks, restart at the horizon, then call the model's step."""
        self.logger.debug(f"Model '{self.name}' stepping at absolute ts={ts} and relative ts={self.state.rel_ts}")
        self.state.ts = ts
        self.state.time = self.start_time + timedelta(seconds=ts * self.real_period)
        if self._horizon_epoch(self.state.rel_ts) != self._epoch:
            self.logger.warning(f"Model '{self.name}' reached its {self.max_sim_time}s simulation horizon: restarting at its origin.")
            left = None if self._span_end_rel is None else max(1, self._span_end_rel - self.state.rel_ts)
            self._reset('horizon', rel_ts_target=self.state.rel_ts, span_ticks=left)
        self.state.rel_time = self.start_time + timedelta(seconds=(self.state.rel_ts + 1) * self.real_period)
        self._set_inputs(inputs)
        self.step()
        out = self._get_outputs()  # Update outputs after stepping
        self.state.rel_ts += 1
        return out

    def _set_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Set input values for the model.
        
        Args:
            inputs: Dictionary of input values from other federates
        """
        self.logger.debug(f"Setting inputs: {inputs}")
        self.state.inputs.update(inputs)
    
    def _get_outputs(self) -> Dict[str, Any]:
        """
        Get output values from the model.

        Returns:
            Dictionary containing the current model outputs
        """
        self.logger.debug(f"Getting outputs: {self.outputs}")
        return self.state.outputs

    def set_parameter(self, name: str, value: Any) -> None:
        """
        Override a live parameter value, clipped to the catalog's declared
        min/max bounds. Unknown parameter names are ignored (logged).
        Used by the digital-twin interface federate's PARAMETER override (M4).
        """
        spec = self.metadata.parameters.get(name) if self.metadata else None
        if spec is None:
            self.logger.warning(f"set_parameter: unknown parameter '{name}', ignoring")
            return
        if spec.min_value is not None and value < spec.min_value:
            value = spec.min_value
        if spec.max_value is not None and value > spec.max_value:
            value = spec.max_value
        self.state.parameters[name] = value


    def _reset(self, mode='full', rel_ts_target=None, span_ticks=None) -> None:
        """
        Put the model back at its initial state, at relative tick *rel_ts_target*.

        ``full`` restarts at relative tick 0 (the scenario start), ``rolling`` at the
        tick the federate computed for this episode (it can lie beyond the model's
        horizon: the backend folds it with ``horizon_pos_s``), ``horizon`` keeps the
        relative clock where it is and only restarts the model. The absolute clock
        (``ts`` / ``time``) is the federation's and is never touched.
        ``span_ticks`` is how many ticks this run of the model will have to cover
        before the next restart (``None`` = until the end of the scenario); the
        federate computes it, models that need a bound (an FMU) read it.
        ``none`` never gets here: the federate does not call a model for it.
        """
        ts, time = self.state.ts, self.state.time
        self.state = copy.deepcopy(self.init_state)
        self.state.ts, self.state.time = ts, time

        target = 0 if rel_ts_target is None else int(rel_ts_target)
        self.state.rel_ts = target
        self._epoch = self._horizon_epoch(target)
        self.state.rel_time = self.start_time + timedelta(seconds=target * self.real_period)
        self.span_ticks = span_ticks
        self._span_end_rel = None if span_ticks is None else target + span_ticks
        self.reset(mode)

    def reset(self, mode: str = 'full') -> None:
        """
        Hook for models that keep state outside ``self.state`` (a solver, an
        external runtime, a cursor into data). No-op by default.

        Called after the generic restart in ``_reset`` has already put ``self.state``
        back to a copy of ``init_state`` and set the relative clock, so a plain
        Python model has nothing to do. ``mode`` is ``full``, ``rolling`` or
        ``horizon``.
        """
        return


    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the model with given parameters.
        
        This method is called once at the beginning of the simulation to set up
        the model's initial state and configure any necessary parameters.
    
        """
        pass
    
    @abstractmethod
    def step(self) -> None:
        """
        Execute one simulation step.
        This method must be overridden by the specific model to implement specific model step behaviour
        it finds inputs in the state and MUST update the outputs in the state, 
        the time state is updated in the base class before calling step method so it can be used by the model without worrying about time management
        
        """
        pass
    
    @abstractmethod  
    def finalize(self) -> None:
        """
        Finalize the model and clean up resources.
        
        This method is called once at the end of the simulation to perform
        any necessary cleanup operations, save final results, or close resources.
        
        Raises:
            NotImplementedError: If not implemented by derived class
        """
    pass    


