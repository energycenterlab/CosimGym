"""
test_script_rl.py

Script for running and debugging RL-based scenarios in the Cosim_gym framework.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""
import os 
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from core.ScenarioManager import main
import time


# EXAMPLES base for debugging and development of new scenarios
# main('simple_test_rlagent')
# main('simple_DQN_test')
# main('simple_SACsb3_test')

# Example 1 — RL control of the BUI0 EnergyPlus FMU zone set-point. Same declarative MDP,
# two algorithms (SB3): continuous (SAC) vs discretized (DQN). Run from the project ROOT.
#main('bui0_setpoint_SAC')
# main('bui0_setpoint_DQN')
main('bui0_heatingpower_DQN')

# Example 2 — RLlib PPO on spring-mass-damper (standalone RLModule, no ray workers).
# main('simple_rllib_test')

# OSMSES26 - working examples. NOTE: RL injects an rl_federation, so these become 2-federation
# scenarios needing a hierarchy broker. The zmq hierarchy broker is broken, so these use
# core_type: tcp (verified: train + test run end-to-end, both full and rolling reset).
# main('bui_hp_DQN')
# main ('bui_hp_SAC')
#main('bui_hp_SAC_rollingreset')
# main('bui_hp_DQN_rollingreset')
# main('pv_batt_SAC')

# scenarios = ['bui_hp_SAC_rollingreset', 'bui_hp_DQN_rollingreset', 'pv_batt_SAC']

# for scenario in scenarios:
    # main(scenario)
    # time.sleep(5)