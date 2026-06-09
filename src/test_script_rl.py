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


# OSMSES26 - working examples
# main('bui_hp_DQN')
# main ('bui_hp_SAC')
#main('bui_hp_SAC_rollingreset')
# main('bui_hp_DQN_rollingreset')
main('pv_batt_SAC')

# scenarios = ['bui_hp_SAC_rollingreset', 'bui_hp_DQN_rollingreset', 'pv_batt_SAC']

# for scenario in scenarios:
    # main(scenario)
    # time.sleep(5)