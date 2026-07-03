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
- parquet based storage
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
