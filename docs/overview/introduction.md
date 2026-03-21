# Introduction

**CosimGym** is a powerful orchestration framework that seamlessly bridges the gap between complex co-simulation environments and Reinforcement Learning (RL).

### The Challenge

Modern engineering systems—such as power grids, building energy systems, and robotic swarms—often rely on multiple interacting subsystems that are best modeled with specialized tools. **Co-simulation** (powered by the [HELICS](https://helics.org/) middleware) enables these heterogeneous models to run simultaneously and exchange data at every time step, providing a unified system-level perspective.

However, adding **Reinforcement Learning** to such environments presents a new, significant complexity: wiring up a co-simulation middleware with an RL framework (like Gymnasium) requires extensive and error-prone networking and synchronization boilerplate.

### The Solution

By natively integrating **Gymnasium**, CosimGym translates complex publish/subscribe data exchanges into the standard `reset()` and `step()` paradigm. This allows RL agents to directly interact with, learn from, and control realistic, physics-based simulations effortlessly. 

### Why Choose CosimGym?

- **Zero Boilerplate:** Declare your entire simulation architecture—including broker topology, Python models, and RL parameters—purely via YAML.
- **Reproducibility:** A centralized model catalog ensures that models run reliably anywhere.
- **Flexibility:** Run standard experiments (without RL), live online training, or evaluate pre-trained agents within the same engine.

<div style="text-align: center;">
<img src="../images/overviewCosimGym.png" alt="Overview" width="800">
</div>