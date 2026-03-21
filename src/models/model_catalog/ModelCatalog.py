"""
ModelCatalog.py

Centralized registry for model metadata, parameters, and interfaces for validation and discovery.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""

from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json
from pathlib import Path

class ParameterType(Enum):
    """Parameter data types for validation and catalog."""
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"

class InterfaceType(Enum):
    """Types of model interfaces."""
    INPUT = "input"
    OUTPUT = "output"
    PARAMETER = "parameter"
    STATE = "state"


@dataclass
class ParameterSpec:
    """Specification for a model parameter, input, or output."""
    name: str
    type: ParameterType
    default_value: Any
    description: str = ""
    unit: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    required: bool = False
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'type': self.type.value,
            'default_value': self.default_value,
            'description': self.description,
            'unit': self.unit,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'required': self.required,
            'tags': self.tags
        }



@dataclass
class ModelMetadata:
    """Complete metadata specification for a model."""
    name: str
    class_name: str
    module_path: str
    version: str
    description: str
    author: str = ""
    domain: str = ""  # e.g., "electrical", "mechanical", "thermal"
    category: str = ""  # e.g., "component", "system", "controller"
    tags: List[str] = field(default_factory=list)
    parameters: Dict[str, ParameterSpec] = field(default_factory=dict)
    inputs: Dict[str, ParameterSpec] = field(default_factory=dict)
    outputs: Dict[str, ParameterSpec] = field(default_factory=dict)
    states: Dict[str, ParameterSpec] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    time_step: int = 1.0  # e.g., time step, solver options
    min_time_step: Optional[float] = 0.0
    max_time_step: Optional[float] = float('inf')

    def get_defaults(self, interface_type: InterfaceType) -> Dict[str, Any]:
        """Get default values for specified interface type."""
        interface_map = {
            InterfaceType.PARAMETER: self.parameters,
            InterfaceType.INPUT: self.inputs,
            InterfaceType.OUTPUT: self.outputs,
            InterfaceType.STATE: self.states
        }
        return {name: spec.default_value for name, spec in interface_map[interface_type].items()}
    
    def get_required(self, interface_type: InterfaceType) -> Set[str]:
        """Get required parameters for specified interface type."""
        interface_map = {
            InterfaceType.PARAMETER: self.parameters,
            InterfaceType.INPUT: self.inputs,
            InterfaceType.OUTPUT: self.outputs,
            InterfaceType.STATE: self.states
        }
        return {name for name, spec in interface_map[interface_type].items() if spec.required}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'class_name': self.class_name,
            'module_path': self.module_path,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'domain': self.domain,
            'category': self.category,
            'tags': self.tags,
            'parameters': {name: spec.to_dict() for name, spec in self.parameters.items()},
            'inputs': {name: spec.to_dict() for name, spec in self.inputs.items()},
            'outputs': {name: spec.to_dict() for name, spec in self.outputs.items()},
            'states': {name: spec.to_dict() for name, spec in self.states.items()},
            'dependencies': self.dependencies,
            'time_step': self.time_step,
            'min_time_step': self.min_time_step,
            'max_time_step': self.max_time_step
        }

class ModelCatalog:
    """Centralized catalog of model metadata and specifications."""
    
    def __init__(self, catalog_path: Optional[str] = None):
        self.models: Dict[str, ModelMetadata] = {}
        self.catalog_path = catalog_path or "src/models/model_catalog/"
        self._load_catalog()
    
    def _load_catalog(self) -> None:
        """Load model catalog from YAML files."""
        catalog_dir = Path(self.catalog_path)
        if not catalog_dir.exists():
            raise FileNotFoundError(f"Catalog path {self.catalog_path} does not exist")
        
        for catalog_file in catalog_dir.glob("*.yaml"):
            try:
                with open(catalog_file, 'r') as f:
                    catalog_data = yaml.safe_load(f)
                
                for model_name, model_data in catalog_data.get('models', {}).items():
                    self.models[model_name] = self._parse_model_metadata(model_name, model_data)
            
            except Exception as e:
                print(f"Error loading catalog file {catalog_file}: {e}")
    
    def _parse_model_metadata(self, model_name: str, data: Dict[str, Any]) -> ModelMetadata:
        """Parse model metadata from dictionary."""
        
        def parse_parameter_specs(specs_data: Dict[str, Any]) -> Dict[str, ParameterSpec]:
            specs = {}
            for name, spec_data in specs_data.items():
                specs[name] = ParameterSpec(
                    name=name,
                    type=ParameterType(spec_data.get('type', 'float')),
                    default_value=spec_data['default_value'],
                    description=spec_data.get('description', ''),
                    unit=spec_data.get('unit', ''),
                    min_value=spec_data.get('min_value'),
                    max_value=spec_data.get('max_value'),
                    required=spec_data.get('required', False),
                    tags=spec_data.get('tags', [])
                )
            return specs
        
        return ModelMetadata(
            name=model_name,
            class_name=data['class_name'],
            module_path=data['module_path'],
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            author=data.get('author', ''),
            domain=data.get('domain', ''),
            category=data.get('category', ''),
            tags=data.get('tags', []),
            time_step=data.get('time_step', 1.0),
            min_time_step=data.get('min_time_step', 0.0),
            max_time_step=data.get('max_time_step', float('inf')),
            parameters=parse_parameter_specs(data.get('parameters', {})),
            inputs=parse_parameter_specs(data.get('inputs', {})),
            outputs=parse_parameter_specs(data.get('outputs', {})),
            states=parse_parameter_specs(data.get('states', {})),
            dependencies=data.get('dependencies', [])
        )
    
    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model."""
        return self.models.get(model_name, None)
    
    def register_model(self, metadata: ModelMetadata) -> None:
        """Register a new model in the catalog."""
        self.models[metadata.name] = metadata
    
    def search_models(self, domain: str = None, category: str = None, tags: List[str] = None) -> List[ModelMetadata]:
        """Search models by domain, category, or tags."""
        results = []
        for model in self.models.values():
            if domain and model.domain != domain:
                continue
            if category and model.category != category:
                continue
            if tags and not any(tag in model.tags for tag in tags):
                continue
            results.append(model)
        return results
    
    def export_to_json(self, output_path: str) -> None:
        """Export catalog to JSON for external systems."""
        catalog_data = {
            'models': {name: model.to_dict() for name, model in self.models.items()}
        }
        with open(output_path, 'w') as f:
            json.dump(catalog_data, f, indent=2)
    
    def get_model_graph_data(self) -> Dict[str, Any]:
        """Export data in format suitable for knowledge graph creation."""
        nodes = []
        edges = []
        
        for model_name, model in self.models.items():
            # Model node
            nodes.append({
                'id': model_name,
                'type': 'model',
                'properties': {
                    'class_name': model.class_name,
                    'domain': model.domain,
                    'category': model.category,
                    'description': model.description,
                    'version': model.version
                }
            })
            
            # Parameter/Interface nodes and relationships
            for interface_type in [InterfaceType.PARAMETER, InterfaceType.INPUT, InterfaceType.OUTPUT]:
                interface_dict = getattr(model, interface_type.value + 's')
                for param_name, param_spec in interface_dict.items():
                    param_id = f"{model_name}_{param_name}"
                    nodes.append({
                        'id': param_id,
                        'type': interface_type.value,
                        'properties': param_spec.to_dict()
                    })
                    edges.append({
                        'from': model_name,
                        'to': param_id,
                        'type': f'has_{interface_type.value}'
                    })
        
        return {'nodes': nodes, 'edges': edges}



if __name__ == "__main__":
    # Example usage
    catalog = ModelCatalog()
    print(f"Loaded {len(catalog.models)} models in catalog.")
    meta = catalog.get_model_metadata("example_model")
    catalog.export_to_json("model_catalog.json")