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
    parameters: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    time: Optional[datetime] = None
    ts: Optional[int] = None

    def __repr__(self) -> str:
        import pprint
        cls_name = self.__class__.__name__
        pp = pprint.PrettyPrinter(indent=2, compact=False, width=80)
        param_str = f"parameters=\n{pp.pformat(self.parameters)}"
        inputs_str = f"inputs=\n{pp.pformat(self.inputs)}"
        outputs_str = f"outputs=\n{pp.pformat(self.outputs)}"
        time_str = f"time={self.time!r}"
        ts_str = f"ts={self.ts!r}"
        return (f"{cls_name}(\n"
                f"  {param_str},\n"
                f"  {inputs_str},\n"
                f"  {outputs_str},\n"
                f"  {time_str},\n"
                f"  {ts_str}\n"
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
    # Model identifier for catalog lookup - to be set by subclasses
    # MODEL_NAME: Optional[str] = None

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
        self.ts = None  # time step, will be set in _set_attrs
        # self.time_stop = None # Maximum simulation time for time-dependent models
        self.start_time = None  # Start time for time-dependent models
        self.real_period = None  # Real time period for time-dependent models
        self.date_time = None

        # model_state
        self.state = State()
        self.init_state = State()
        
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
      
    def _step(self, ts, inputs):
        """Internal step method to update time state and call user-defined step."""
        self._update_time_state(ts)
        self._set_inputs(inputs)
        self.step()
        out = self._get_outputs()  # Update outputs after stepping
        return out

    def _update_time_state(self, time_step: int) -> None:
        """Update time-related state variables."""
        self.state.ts = time_step
        self.state.time = self.start_time + timedelta(seconds=self.state.ts * self.real_period)
    
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
        
    def reset(self, mode='full', ts= None, time=None) -> None:
        """
        Reset the model to its initial state.
        
        This method can be used to restart the model simulation from the
        initial conditions defined in init_state.
        NB. only reset interfaces in stateful models must be overridden to modify internals
        """
        self.state = copy.deepcopy(self.init_state)
        if ts is not None:
            self.state.ts = ts
        if time is not None:
            self.state.time = time
     
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


