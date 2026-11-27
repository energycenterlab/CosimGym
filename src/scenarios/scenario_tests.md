# TESTING SCENARIOS

## bui_hp_test_base
weather -- PID -- heatpump -- building
Simple models
### tested features
- simple single federation simulation with sequential data flow + feedback loop to next step from building to PID
- single machine
- memory list , json default
- DEBUG logger
- both offset expressed per fed and synchronization object
### Working:
- YES
- n_ticks: 86400
- time : 9.58 s 

## bui0_fmu_test
specific feeder --- fmu from frassinetto Eplus
Simple models 
### tested features
- FMU

### Working
- YES 
- n_ticks: 144
- time 2.54 s

## dh_district_jan_base
 buildings rc instances --- DH network pandapipes
 more articolated scenario
### tested features
- json storage (no `sink` key → default `json`)
- no dashboard adaptation
- zmq brokers
- 1 federate multiple modle instances (building_federate) 
- streaming for live dashboard (with - without dashboard)
## Working
- YES
- n_ticks: 2976
- time: 62.593 s

## fmu_feedthrough_test
feeder test for FMU3

### tested features

### Working
- YES


## multi_building_grid_test
multiple building rc connected to a grid model
### tested features
tested paralle execution in federate model instance. no convenient for this case
### Working
- YES

## benchmark_parallel_par
demo test with high computational model in several parallel processes under the same federate
### tested features
parallelization of modle instances where it is convenient because the model step method used is high computation demand
### Working
- YES

## combo_parallel_rl_rollingreset
bui_hp rolling-reset DQN scenario whose weather federate runs 2 instances on 2 `parallel_execution` workers.
### tested features
A rolling reset reaches every worker (ack on each pipe) and the run completes.
### Working
- YES (federate log: `rolling reset #1/#2` at relative ticks 10/20)

## bui0_setpoint_DQN_rolling
BUI0 EnergyPlus FMU + DQN with `reset.mode: rolling`, window = 1 day (144 x 600 s), 4 train episodes.
### tested features
E+ restart at midnight on each rolling reset; one shared `fmu_output/` per run.
### Working
- YES on eclab-cloud1 with E+ 23.1.0 on `PATH` (KNOWN_ISSUES §14): 4 episodes, one shared `fmu_output/`, last slave starts 3 Jan (start day advances).

## cs_s2_reset_rolling
Case study S2 with `rolling_window: 15` (900 s = weather `real_period`).
### Working
- YES (suite-shortened copy; the original 10 is refused by the divisibility check)
