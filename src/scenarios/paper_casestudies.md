| Scenario | Configuration | Main purpose | Done |
|---|---|---|
| **S0 – Baseline** | Weather + Building + Heat Pump + PID | Reference scenario used as the common starting point and performance baseline. | yes |
| **S1 – New model types** | Baseline + new models | The new m odels are (PV, BEMS, Battery ) Test the capability to integrate additional heterogeneous model types into the co-simulation. | x |
| **S2 – Scale instances** | S1 + increased number of model instances | Evaluate scalability when increasing the number of simulated components/federates. | x |
| **S3 – Distributed execution** | S2 + distributed deployment | Evaluate execution of the scaled scenario across multiple machines/nodes. |
| **S4.a – RL integration** | Baseline + RL controller | DQN + full reset | x |
| **S4.b – RL integration** | Baseline + RL controller | SAC + full reset  | x |
| **S4.c – RL integration** | Baseline + RL controller | DQN + rolling reset | x |
| **S4.d – RL integration** | Baseline + RL controller | SAC + rolling reset | x |
| **S5 – RL + model swap** | S2.a + model replacement | Test whether models can be exchanged while maintaining the RL/co-simulation workflow. | x |
| **S6 – RL + model swap + DT capabilities** | S2.b + Digital Twin functionalities | Demonstrate the complete workflow including RL, model replacement, and Digital Twin capabilities. | x |