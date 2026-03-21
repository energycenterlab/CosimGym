"""
base_FMU_model.py

Support for Functional Mock-up Units (FMU) integration within the model framework.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-05

"""
# TODO: debug and finish
import os
import random
import time
import pprint as pp
import shutil
from .base_model import BaseModel
from fmpy import read_model_description, extract, dump
from fmpy.fmi1 import FMU1Slave
from fmpy.fmi2 import FMU2Slave
pp = pp.PrettyPrinter(indent=4)


class BaseFMUModel(BaseModel):
    def __init__(self, name, metadata, config, logger):
        super().__init__(name, metadata, config, logger)
        self.fmu = None  # Placeholder for the FMU instance
        self.model_description = None
        self.unzip_dir = None
        self.fmiVersion = None

        # variables dictionaries to store value references and types for inputs, outputs, and parameters
        self.vars={}  
        self.in_vars = {}
        self.ou_vars = {}
        self.params_vars = {} 

    
    def initialize(self, parameters = None):
        self.logger.debug(f"Initializing FMU model: {self.name} with metadata: {self.metadata} and config: {self.config}")
        
        fmu_path = self.metadata.get('fmu_path')  # TODO check how to get this information
        
        # Unpack FMU and extract model description
        self._unpack_fmu(fmu_path)

        # instantiate fmu
        self._instantiate_fmu()

        # set initial state, inputs, outputs, and parameters in both interface and fmu
        # set the initial state and the state and the inputs outputs and parameters
        # we have to do it now beacuse it depends on the FMU extraction (its is general)
        self._set_initial_state()

        # setup experiment (for FMI 2.0)
        self._setup_experiment()

        # Initilization Mode
        self._enter_initialization_mode()
        self._set_attributes_during_initialization()
        self._exit_initialization_mode()

        # there was a time.sleep here lets check if needed!TODO
        self.logger.info(f"FMU model {self.name} initialized successfully.")


    def _get_vars_from_fmu(self):
        for v in self.model_description.modelVariables:
            self.vars[v.name] = (v.valueReference, v.type, v.causality, v.variability)
            if v.causality == 'parameter':
                self.params_vars[v.name] = (v.valueReference, v.type)
            elif v.causality == 'input':
                self.in_vars[v.name] = (v.valueReference, v.type)
            elif v.causality == 'output':
                self.ou_vars[v.name] = (v.valueReference, v.type)
        
        self.logger.debug(f"Estrapolated vars from FMU: paramters:\n {pp.pformat(self.params_vars)} \n inputs: {pp.pformat(self.in_vars)} \n outputs: {pp.pformat(self.ou_vars)}")
    
    def _unpack_fmu(self, fmu_path):
        # Unpack the FMU file and prepare it for simulation
        self.model_description = read_model_description(fmu_path,
                                                        validate=True) 
        self._get_vars_from_fmu()
        self.unzipdir = extract(fmu_path)
        self.fmiVersion = self.model_description.fmiVersion
        self.logger.debug(f'FMU unpacked! info:\n {pp.pformat(dump(fmu_path))}')
    
    def _instantiate_fmu(self):
        # create slaves wrt FMI version
        if self.fmiVersion == '1.0':
            self.fmu = FMU1Slave(guid=self.model_description.guid, unzipDirectory=self.unzipdir,
                                           modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                                           instanceName=self.model_name)
            self.fmu.instantiate()
        elif self.fmiVersion == '2.0':
            self.fmu = FMU2Slave(guid=self.model_description.guid, unzipDirectory=self.unzipdir,
                                           modelIdentifier=self.model_description.coSimulation.modelIdentifier,
                                           instanceName=self.model_name)
        else:
            self.logger.error(f"Unsupported FMU version: {self.fmiVersion}")
            raise Exception('The FMU-CS version is not supported. Check the FMU version.')
        
        # initialize
        self.fmu.instantiate(loggingOn=self.config.get('logging_level', 7)) # TODO this does not work must get the log level and bring it to number ad the numbering verison in the dataclass
    
    def _set_initial_state(self):
        # TODO set all the datatstructures self.inputs, self.outputs, self.params, self.state, self.init_state
        pass
    
    def _setup_experiment(self): 
        start_time = self.config.get('start_time', 0) #TODO what to put and how to get this value
        end_time = self.config.get('end_time', 100) #TODO what to put and how to get this value
        if self.fmiVersion == '2.0':
            # Setup experiment: set the independent variable time
            self.fmu.setupExperiment(startTime=start_time, stopTime=end_time)
        else:
            self.logger.error('Experiment setup for FMI 1.0 version not Coded!')
            # raise Exception('fmi 1.0 NOT SUPPORTED')

    def _enter_initialization_mode(self):
        if self.fmiVersion == '2.0':
            self.fmu.enterInitializationMode()
        else:
            self.logger.error('Initialization mode for FMI 1.0 version not Coded!')
            # raise Exception('fmi 1.0 NOT SUPPORTED')
    
    def _set_attributes_during_initialization(self):
        #TODO overriding of all attributes that can be changed here 
        pass
    
    def _exit_initialization_mode(self):
        if self.fmiVersion == '2.0':
            status = self.fmu.exitInitializationMode()
            assert status == 0
        else:
            self.logger.error('Exit initialization mode for FMI 1.0 version not Coded!')
            status = self.fmu.initialize(tStart=0, stopTime=self.end_time)
            # raise Exception('fmi 1.0 NOT SUPPORTED')

    def _set_var(self, value_ref,value):
        '''generic method to set any attribute of an fmu given its valueref and value'''
        if isinstance(value, float):
            self.fmu.setReal([value_ref], [value])
        elif isinstance(value, int):
            self.fmu.setInteger([value_ref], [value])
        elif isinstance(value, str):
            self.fmu.setString([value_ref], [value])
        elif isinstance(value, bool):
            self.fmu.setBoolean([value_ref], [value])
        else:
            self.logger.error(f"{type(value)} variable type not supported - SET VAR -")
    
    def _get_vars(self, value_ref, tp):
        '''generic method to get any attribute of an fmu given its valueref and type'''

        if tp == 'Real':
            var = self.fmu.getReal([value_ref])[0]
        elif tp == 'Integer':
            var = self.fmu.getInteger([value_ref])[0]
        elif tp == 'String':
            var = self.fmu.getString([value_ref])[0]
        elif tp == 'Boolean':
            var = self.fmu.getBoolean([value_ref])[0]
        else:
            self.logger.error(f"type of variable {tp} not recognized! - GET VAR -")
            var = None
        return var

    def step(self):
        self.logger.debug(f"Stepping FMU model: {self.name}")
        # Perform a simulation step and return outputs
        self._inputs_to_fmu()  # Set inputs in FMU

        # TODO: the timing of fmu
        self.fmu.doStep(currentCommunicationPoint=fmu_time, communicationStepSize=self.real_period)

        self._outputs_from_fmu()  # Get outputs from FMU
        pass

    def reset(self):
        super().reset()
        self.logger.debug(f"Resetting FMU model: {self.name}")
        # finalize the fmu
        # restart the fmu with the same initialization procedure but faster if possible

    def finalize(self):
        self.logger.info(f"Finalizing FMU model: {self.name}")
        self.fmu.terminate()
        self.fmu.freeInstance()
        try:
            shutil.rmtree(self.unzipdir)
        except PermissionError as e:
            self.logger.error(f"Folder could not be removed. {e}")
        pass