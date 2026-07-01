# Custom Models & Catalog

CosimGym treats simulated environments as plug-and-play modules. To achieve this, it enforces a single interface standard (`BaseModel`) and resolves models at runtime through a dynamic **Model Catalog**.

## 1. Extending `BaseModel`

Every physical subsystem, controller, or data feed must extend `BaseModel`, defined in `src/models/base_model.py`. You implement three lifecycle methods. You do **not** override `__init__` — the base class builds the model's `state` from the catalog metadata and your scenario config for you.

All data lives on `self.state` (a `State` dataclass):

| Attribute | Meaning |
|---|---|
| `self.state.parameters` | Static parameters (from catalog defaults, overridden by the scenario `parameters` block) |
| `self.state.inputs` | Latest values received from subscriptions (filled by the federate before each `step()`) |
| `self.state.outputs` | Values your model produces — published to HELICS after `step()` |
| `self.state.time` | Current simulation `datetime` (maintained by the base class) |
| `self.state.ts` | Current integer time step |

### `initialize(self)`
Called once before the simulation clock starts. Use it to set up internal arrays, matrices, or solver state. Parameters are already available in `self.state.parameters`.

### `step(self)`
Called every HELICS time step. The base class has already updated the time state and copied incoming values into `self.state.inputs`. Your job:

1. Read inputs from `self.state.inputs`.
2. Advance your internal state.
3. Write results into `self.state.outputs`.

### `finalize(self)`
Called once at simulation end for cleanup (close files, free resources).

### Minimal example

```python
# src/models/model_catalog/physical_models/gain.py
from models.base_model import BaseModel


class Gain(BaseModel):
    """Multiplies its input by a constant gain parameter."""

    def initialize(self) -> None:
        self.k = self.state.parameters['gain']

    def step(self) -> None:
        x = self.state.inputs.get('x', 0.0)
        self.state.outputs['y'] = self.k * x

    def finalize(self) -> None:
        pass
```

---

## 2. Registering in the Catalog

For `ScenarioManager` to instantiate your model from a string name in the YAML, register it in `src/models/model_catalog/catalog.yaml`. Every entry lives under the top-level `models:` key. Use `model_template.yaml` in the same folder as a starting point.

```yaml
models:
  # ... existing entries ...

  gain:                                                    # ← model_name referenced in scenario YAML
    class_name: Gain                                       # Python class name
    module_path: models.model_catalog.physical_models.gain # import path (NB: no "src." prefix)
    version: 1.0.0
    description: Multiplies its input by a constant gain.
    author: Your Name
    domain: testing
    category: physical_model
    time_step: 60                                          # nominal step (seconds)
    max_time_step: 3600
    min_time_step: 1
    user_defined: {}
    parameters:
      gain:
        type: float
        default_value: 2.0       # NB: default_value, and unit (singular)
        unit: '-'
        description: Multiplication factor
        required: false
    inputs:
      x:
        type: float
        default_value: 0.0
        unit: '-'
        description: Input signal
        required: true
    outputs:
      y:
        type: float
        default_value: 0.0
        unit: '-'
        description: Scaled output signal
        required: true
```

> **Catalog vs. connections.** The `parameters` / `inputs` / `outputs` blocks in the catalog declare the model's interface schema and **default values**. The actual HELICS wiring (which variable subscribes to which publisher) is defined separately in the scenario's `connections.publishes` / `connections.subscribes`. The `key` of each publish/subscribe must match an `outputs` / `inputs` name here. See [Federate Configuration](scenario_configuration/federate.md).

---

## 3. How the Catalog is distributed

1. At setup, `catalog_loader.py` reads `catalog.yaml` and loads every entry into **Redis**.
2. When `ScenarioManager` spawns each federate in its own OS process, that federate queries Redis (via `RedisCatalog`) for the model's `module_path`, `class_name`, and interface defaults.
3. The federate dynamically imports the class and instantiates it — no fragile local path dependencies between processes.

Reload the catalog after editing `catalog.yaml`:

```bash
python src/models/model_catalog/catalog_loader.py
```

(The Docker / Makefile setups run this step for you.)

---

## 4. Interface Adapters (transport, not physics)

An **interface federate** (`type: interface`, see [Federate Configuration](scenario_configuration/federate.md#type-interface-interface-federate-digital-twin-bridge)) instantiates a **transport adapter** instead of a `BaseModel` — but through the identical catalog mechanism: `category: interface_adapter` entries in `catalog.yaml` (mapped to their own Redis key prefix in `catalog_loader.py`'s `CATEGORY_MAP`), dynamic-imported by class name/module path.

```yaml
models:
  mqtt_adapter:
    class_name: MqttAdapter
    module_path: adapters.mqtt_adapter
    category: interface_adapter
    parameters:
      client_id: { type: str, default_value: cosim_dt }
      host: { type: str, default_value: localhost }
      port: { type: int, default_value: 11883 }
      qos: { type: int, default_value: 0 }
```

To add a new transport (e.g. Kafka, Modbus, OPC-UA), implement `InterfaceAdapter` (`src/adapters/base_adapter.py`: `connect`, `publish`, `subscribe`, `latest`, `close`) and register it the same way. See [Digital-Twin Interfaces & Live Streaming](digital_twin_interfaces.md) for the adapter's role in both the `streaming` mirror and the interface federate.
