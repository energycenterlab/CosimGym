"""
Configuration Validation Module

Validates federate and model configurations to ensure they meet framework requirements.
"""

from typing import Dict, Any, List, Union
import logging

# TODO: use validation at federation manager before starting the processes
# TODO: there is a validation method also inside config_reader join them or decide which one to use


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    pass

class ConfigValidator:
    """Validates configuration dictionaries for federates and models"""
    
    @staticmethod
    def validate_model_parameters(parameters: Dict[str, Any], num_instances: int, model_name: str = ""):
        """
        Validate model parameters dictionary.
        
        Args:
            parameters: Dictionary of parameter name -> value/list
            num_instances: Number of model instances expected
            model_name: Name of model for error reporting
            
        Raises:
            ConfigValidationError: If validation fails
        """
        prefix = f"Model '{model_name}': " if model_name else ""
        
        for param_name, param_value in parameters.items():
            if isinstance(param_value, list):
                if len(param_value) != num_instances:
                    raise ConfigValidationError(
                        f"{prefix}Parameter '{param_name}' list length ({len(param_value)}) "
                        f"doesn't match number of instances ({num_instances})"
                    )
            # Single values are OK - they'll be replicated across instances
            elif not isinstance(param_value, (int, float, str, bool)):
                raise ConfigValidationError(
                    f"{prefix}Parameter '{param_name}' must be a single value or list, "
                    f"got {type(param_value).__name__}"
                )
    
    @staticmethod
    def validate_federate_config(config: Dict[str, Any]) -> None:
        """
        Validate complete federate configuration.
        
        Args:
            config: Federate configuration dictionary
            
        Raises:
            ConfigValidationError: If validation fails
        """
        # Validate required top-level fields
        required_fields = ['name', 'type']  # Add other required fields
        for field in required_fields:
            if field not in config:
                raise ConfigValidationError(f"Missing required field: '{field}'")
        
        # Validate model configurations if present
        if 'models' in config:
            models_config = config['models']
            for model_name, model_config in models_config.items():
                if 'instances' in model_config and 'parameters' in model_config:
                    num_instances = model_config['instances']
                    parameters = model_config['parameters']
                    ConfigValidator.validate_model_parameters(
                        parameters, num_instances, model_name
                    )