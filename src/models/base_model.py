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
    # Simulated datetime the model is at. It follows the model clock, so a reset
    # or a horizon restart moves it back along with everything else: every model
    # in a federation stays at the same simulated moment. Results and logs use the
    # federate's own monotonic clock, not this one.
    time: Optional[datetime] = None # Simulated datetime the model is at
    ts: Optional[int] = 0 #absolute timestep of the model in the simulation, it is updated by the federate before calling the step method of the model
    rel_ts: Optional[int] = 0 #relative timestep of the model in the simulation, it is internally managed by the model and it is updated by the model itself in the step method, it is used to manage the internal state of the model and to manage the local clock of the model


    def __repr__(self) -> str:
        import pprint
        cls_name = self.__class__.__name__
        pp = pprint.PrettyPrinter(indent=2, compact=False, width=80)
        param_str = f"parameters=\n{pp.pformat(self.parameters)}"
        inputs_str = f"inputs=\n{pp.pformat(self.inputs)}"
        outputs_str = f"outputs=\n{pp.pformat(self.outputs)}"
        time_str = f"time={self.time!r}"
        ts_str = f"ts={self.ts!r}"
        rel_ts_str = f"rel_ts={self.rel_ts!r}"
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

        # ---- model-local clock -------------------------------------------------
        # Some models cannot run for an unlimited span of simulated time: an
        # EnergyPlus FMU stops at the end of its RunPeriod. Such a model declares
        # ``max_sim_time`` in its catalog entry and is restarted automatically
        # whenever that horizon is reached, which makes its own clock a sawtooth
        # while federation time stays monotonic.
        #   local_ts(ts) = ts - ts_shift
        # ``max_sim_time is None`` (the default for every model) leaves all of this
        # inert: local time is federation time and no restart is ever triggered.
        # Declared in the catalog entry, because it is a property of the model
        # itself; a scenario can override it under the model's ``user_defined``
        # block, which is mostly useful to exercise a restart in a short run.
        _scenario_limits = (user_config.user_defined or {}) if user_config else {}
        self.max_sim_time = _scenario_limits.get(
            'max_sim_time', getattr(catalog_metadata, 'max_sim_time', None))
        self.sim_start_date = _scenario_limits.get(
            'sim_start_date', getattr(catalog_metadata, 'sim_start_date', None))
        # Offset between federation ticks and the model's own ticks. A reset or a
        # horizon restart moves it; with neither, the two are the same thing.
        self.ts_shift = 0
        self.epoch_index = 0        # how many restarts have happened so far
        # Episode-reset policy of the owning federate, injected at runtime. Only
        # 'rolling' can move a model backwards in time.
        self.reset_mode = getattr(user_config, 'reset_mode', None)
        self.rolling_window = getattr(user_config, 'rolling_window', None)
        self.n_episodes = getattr(user_config, 'n_episodes', None)
        self.episode_length = getattr(user_config, 'episode_length', None)
        self.reset_period = getattr(user_config, 'reset_period', None)

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
        self.init_state.time = datetime.fromisoformat(self.config.start_time)  # Initialize time in state
        # self.time_stop = self.config.time_stop  # Maximum simulation time for time-dependent models
        self.start_time = datetime.fromisoformat(self.config.start_time)  # Start time for time-dependent models
        self.real_period = self.config.real_period  # Real time period for time-dependent models
        # self.date_time = self.start_time

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
    # Model-local clock
    # ------------------------------------------------------------------

    # def local_ts(self, ts=None) -> int:
    #     """The model's own tick for federation tick *ts*.

    #     The two are identical until something moves the model back in time - an
    #     episode reset, or the restart of a model that has reached its simulation
    #     horizon. The federate moves every model it owns by the same amount at the
    #     same tick, so they all stay at the same simulated moment.
    #     """
    #     ts = self.state.ts if ts is None else ts
    #     ts = 0 if ts is None else ts
    #     return max(0, ts - self.ts_shift)

    # def local_time(self, ts=None) -> float:
    #     """The model's own simulated time in seconds, counted from its first step."""
    #     return max(0, self.local_ts(ts) - 1) * self.real_period

    # def horizon_ts(self):
    #     """ OK
    #     The declared simulation horizon expressed in steps, or None if unbounded."""
    #     if not self.max_sim_time or not self.real_period:
    #         return None
    #     return int(self.max_sim_time // self.real_period)

    # def reposition(self, target_ts: int, at_ts=None, reason: str = 'reset') -> None:
    #     """Make the model behave as if federation tick *at_ts* were tick *target_ts*.

    #     The single primitive behind every restart: an episode reset asks for tick
    #     1, a rolling reset for the episode's start tick, and a model that has run
    #     out of run period asks for tick 1 as well. ``_reposition_backend`` does
    #     whatever the model wraps - nothing for a plain Python model, a restart and
    #     replay for an FMU slave, a cursor move for a CSV reader.

    #     --> the only case in which i would like to do a reposition of the fmu is if horizon reached
    #     """
    #     target_ts = max(0, int(target_ts))
    #     horizon = self.horizon_ts()
    #     if horizon and target_ts > horizon:
    #         wrapped = ((target_ts - 1) % horizon) + 1
    #         self.logger.info(
    #             f"Restart target tick {target_ts} is past the {horizon}-step horizon, "
    #             f"wrapping to tick {wrapped}"
    #         )
    #         target_ts = wrapped
        

    #     self._reposition_backend(target_ts, reason=reason)

    #     at_ts = (self.state.ts or 0) + 1 if at_ts is None else at_ts
    #     self.ts_shift = at_ts - target_ts
    #     self.epoch_index += 1
    #     self.logger.debug(
    #         f"Model clock moved ({reason}): tick {at_ts} now counts as tick {target_ts} "
    #         f"(shift={self.ts_shift}, epoch={self.epoch_index})"
    #     )

    # def _reposition_backend(self, target_ts: int, reason: str = 'reset') -> None:
    #     """Bring whatever the model wraps to its own tick *target_ts*.

    #     No-op by default: a plain Python model is fully described by its state, so
    #     moving its clock needs nothing more. Overridden by models backed by an
    #     external runtime that must be restarted (BaseFMUModel) or by a cursor into
    #     data (BaseCSVReader).
    #     """
    #     return

    # def _enforce_sim_horizon(self) -> None:
    #     """Restart the model when the step about to run would pass its horizon.

    #     Inert unless the model declares ``max_sim_time``. This is the model's own
    #     limit, so it applies whatever the RL reset policy is, and with no RL at
    #     all. The federate normally restarts every model together before this can
    #     fire; the guard is the backstop that keeps any single model from being
    #     stepped past a limit it cannot honour.
    #     """
    #     horizon = self.horizon_ts()
    #     if not horizon:
    #         return
    #     if self.local_ts() <= horizon:
    #         return
    #     self.logger.info(
    #         f"Model '{self.name}' reached its {self.max_sim_time}s simulation horizon "
    #         f"({horizon} steps) at tick {self.state.ts}; restarting "
    #         f"(epoch {self.epoch_index + 1})"
    #     )
    #     self.reposition(1, at_ts=self.state.ts, reason='horizon reached')

    def _step(self, ts, inputs):
        """Internal step method to update time state and call user-defined step."""
        self.logger.debug(f"Model '{self.name}' stepping at absolute ts={ts} and relative ts={self.state.rel_ts}")
        #self._update_time_state(ts)
        #self._enforce_sim_horizon()
        #self._update_time_state(ts)   # a restart just moved the clock
        self.local_time(ts, mode='base')
        self._set_inputs(inputs)
        self.step()
        out = self._get_outputs()  # Update outputs after stepping
        self.state.rel_ts += 1  # Increment relative timestep after step
        return out

    def local_time(self, ts=None, mode='base') -> float:

        #this executed periodically every step
        if mode == 'base': # when local_time is called in classic step method and not during specific reset
            #Enforcing the horizon for those models that have a max simtime
            if self.max_sim_time and self.state.rel_ts*self.real_period >= self.max_sim_time: #ensure that max_sim_time is only for model with a horizon cap
                self.state.rel_ts = 0
                self.reset(mode='horizon_limit') #NB this call reset NOT _reset never call _reset in local_time() for recursion problems
                self.logger.warning(f"Model '{self.name}' exceeded its simulation horizon of {self.max_sim_time} steps.")
            self.state.ts = ts
            self.state.time = self.start_time + timedelta(seconds=self.state.ts * self.real_period) #decisione self.state.time  è sempre il tempo assoluto al pari dey tick del federate
            return

        #this elif blocks are executed only once in a while when reset is explicitally requested for RL application
        elif mode == 'reset full':
            self.state.ts = 0
            self.state.rel_ts = 0
            self.state.time = self.start_time
            return
        
        elif mode == 'reset rolling':
            self.state.rel_ts = self.state.ts - self.episode_length + self.rolling_window #new_relative ts
            self.state.time = self.start_time + timedelta(seconds=self.state.rel_ts * self.real_period)
            return

        else:
            self.state.ts = ts
            self.state.time = self.start_time + timedelta(seconds=self.state.ts * self.real_period)
            return
        


    # def _update_time_state(self, time_step: int) -> None:
    #     """ OK
    #     Update time-related state variables.

    #     The datetime follows the *model* clock, so a reset or a horizon restart
    #     rewinds it along with everything else: a schedule model and the building
    #     it feeds are always at the same simulated moment, episode after episode.
    #     With nothing ever restarted, the model clock is the federation clock and
    #     this is the plain elapsed time it has always been.
    #     """
    #     self.state.ts = time_step
    #     self.state.time = self.start_time + timedelta(
    #         seconds=self.local_ts(time_step) * self.real_period)

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


    def _reset(self, mode='full', ts= None, time=None) -> None:
        """
        Reset the model to its initial state.
        
        This method can be used to restart the model simulation from the
        initial conditions defined in init_state.
        NB. only reset interfaces in stateful models must be overridden to modify internals
        """
        if mode in ['full', 'rolling', 'horizon_limit']:
            self.state = copy.deepcopy(self.init_state)

        self.local_time(mode=f'reset {mode}')
        self.reset(mode)

    # def _reset_clock(self, mode: str, target_ts: Optional[int], current_ts: int) -> None:
    #     """Realign the model-local clock to the start point this reset asks for.

    #     `full` restarts the model at local time 0; `rolling` moves it to the
    #     absolute start point the federate computed; `none`/`soft` leave the clock
    #     alone. Inert for models with no horizon and no reposition backend.
    #     """
    #     if mode in ('none', 'soft'):
    #         return
    #     # `full` restarts the model at its first step; `rolling` starts it at the
    #     # absolute tick the federate computed for this episode.
    #     target = int(target_ts) if (mode == 'rolling' and target_ts is not None) else 1
    #     self.reposition(target, at_ts=current_ts + 1, reason=f"{mode} reset")
    @abstractmethod
    def reset(self, mode: str = 'full', ts=None, time=None) -> None:    
        """
        Reset the model to its initial state.
        
        This method can be used to restart the model simulation from the
        initial conditions defined in init_state. the timing aspects are already generally
        dealt with in this base classe and also a copy of the initial state is given
        
        Raises:
            NotImplementedError: If not implemented by derived class
        """
        pass

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


