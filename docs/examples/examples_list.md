# Provided Examples List

The CosimGym repository comes with a series of pre-configured tests (scenarios) to demonstrate the transition from simple physics simulations to comprehensive multi-federation Reinforcement Learning workloads.

These are defined inside `src/scenarios/`.

## Base Co-Simulation (Physics & Math)

- **`simple_test` (Case 0):** A fundamental single-federation test connecting a spring-mass-damper system to an autonomous input signal generator. Excellent for debugging that your HELICS installation is functional.
- **`simple_test_multifederations`:** Expands Case 0 to split the spring and the damper inputs across a hierarchical multi-broker network.
- **`bui_hp_test_base` (Case 1):** A building thermal zone integrated with a Heatpump module and CSV weather drivers, regulated by a classical explicit PID controller.
- **`pv_batt_test_base` (Case 4):** A micro-grid emulation consisting of Photovoltaic panels, a battery storage model, a static electrical load, and weather inputs controlled by a rule-based (RB) management system.

## Reinforcement Learning Training

- **`simple_DQN_test` / `simple_SACsb3_test`:** Implements discrete and continuous action algorithms over the simple spring system to demonstrate Gym wrapping bare mechanics.
- **`bui_hp_DQN` / `bui_hp_SAC` (Cases 2 & 3):** Replaces the PID controller from Case 1 with stable-baselines3 agents learning optimal thermal setpoints based on ambient conditions. Includes "rolling reset" variations.
- **`pv_batt_DQN` / `pv_batt_SAC` (Cases 5 & 6):** Replaces the Rule-Based manager from Case 4 with an algorithm learning deep policies to balance grid stability and storage degradation constraints. 

All cases can be executed via:
```bash
python src/test_script.py --scenario <filename_without_yaml>
```