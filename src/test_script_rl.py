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
# main('simple_test_rlagent', enable_progress_bar=False)
# main('simple_DQN_test', enable_progress_bar=False)


# OSMSES26 - working examples
# main('bui_hp_DQN', enable_progress_bar=False)
# main ('bui_hp_SAC', enable_progress_bar=False)
#main('bui_hp_SAC_rollingreset', enable_progress_bar=False)
# main('bui_hp_DQN_rollingreset', enable_progress_bar=False)
main('pv_batt_SAC', enable_progress_bar=False)

# scenarios = ['bui_hp_SAC_rollingreset', 'bui_hp_DQN_rollingreset', 'pv_batt_SAC']

# for scenario in scenarios:
    # main(scenario, enable_progress_bar=False)
    # time.sleep(5)