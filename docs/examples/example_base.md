# Example 1: Base Simulation (Case 0)

This scenario demonstrates the simplest CosimGym layout: one federation wiring a physics model (`spring_mass_damper`) to a signal generator (`inputs4spring`).

## Scenario Source
Found at `src/scenarios/simple_test.yaml`.

## Core Takeaways

### 1. Topology
A single federation `federation_1` runs a `zmq` broker and hosts two `base` federates:

- `input_federate` → runs `inputs4spring`, publishes `force` and `disturbance`.
- `spring_federate` → runs `spring_mass_damper`, subscribes to those signals and publishes `position`, `velocity`, `acceleration`.

Both federates instantiate **2 model instances** (`n_instances: 2`), so two independent springs run in parallel.

### 2. Model instantiation
`model_configs` is a single object with an `instantiation` block (not a list of models):

```yaml
model_configs:
  instantiation:
    model_name: "spring_mass_damper"
    n_instances: 2
    prefix: "spring"
  parameters:
    mass:      [5, 5]          # one value per instance
    damping:   [2.0, 3.0]
    stiffness: [10.0, 20]
  init_state:
    position: [0.0, 100.0]
    velocity: 0.0
    force:    [10.0, 5.0]
```

### 3. Publish / Subscribe wiring
The spring subscribes its `force` input to the input federate's `force` publication, per instance. The `targets` map is keyed by instance index:

```yaml
subscribes:
  - key: "force"
    type: "double"
    units: "N"
    targets:
      '0': [input_federate.0/force]   # <federate_name>.<instance>/<pub_key>
      '1': [input_federate.1/force]
```

## Execution
Set the scenario in the entry-point script and run it (no `--scenario` flag exists):

```python
# src/test_script.py
main('simple_test')
```

```bash
conda activate cosim_gym
python src/test_script.py        # or: make run
```

Once complete, open the dashboard (`make run-dashboard`) and select the run. You will see the recorded `position`, `velocity`, and `acceleration` traces for each spring instance responding to the driving `force`.
