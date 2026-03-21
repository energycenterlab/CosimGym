# Custom Models & Catalog

CosimGym treats simulated environments as plug-and-play modules. To achieve this, it enforces an interface standard and manages deployments through a dynamic catalog.

## Extending BaseModel

Any physical subsystem, controller, or data feed in CosimGym must extend the `BaseModel` class at `src/models/base_model.py`.

A simplified lifecycle is defined by three strict overrides:

### `initialize(self)`
This is called once before the simulation clock starts ticking. Use it to ingest `self.parameters`, establish data arrays, and initialize physical matrices.

### `step(self)`
This method ticks every HELICS synchronous time step.
- Internal simulation state is advanced.
- All incoming values from the `self.inputs` dictionary should be read.
- The resulting values must be pushed into the `self.outputs` dictionary.

### `finalize(self)`
Triggered at simulation end for graceful teardown or plotting generation.

---

## 2. Registering in the Catalog

To allow the `ScenarioManager` to spin up your model by simply referencing string names in the YAML, your model must be registered in the **Model Catalog**.

1. Place your new python class (e.g. `src/models/physical_models/custom_hvac.py`).
2. Open `src/models/model_catalog/catalog.yaml`.
3. Add a mapping specifying the metadata of your model:

```yaml
Custom_HVAC:
  metadata:
    class_name: "CustomHVACModel"
    module_path: "src.models.physical_models.custom_hvac"
    description: "A custom HVAC simulation module"
  parameters:
    max_cooling_power:
      type: "float"
      default_value: 50.0
      units: "kW"
  inputs:
    ambient_temp:
      type: "float"
      units: "C"
  outputs:
    power_consumed:
      type: "float"
      units: "kW"
```

## How the Catalog is distributed

1. At simulation launch, `catalog_loader.py` reads `catalog.yaml`.
2. It constructs a `ModelCatalog` instance and drops it into **Redis**.
3. When the `ScenarioManager` spawns a disconnected `Federate` in a totally separate OS process, the federate queries Redis for the specific python module and parameters, instantiating your custom object without dealing with fragile local pathing dependencies.