from pathlib import Path
import shutil

import pandas as pd

from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave


# ============================================================
# CONFIGURATION
# ============================================================

FMU_PATH = Path("BUI0_modified.fmu")


START_TIME = 0.0          # [s] from Jan 1, 00:00
DURATION_DAYS = 364*2       # EnergyPlus: preferably whole days
STEP_SIZE = 600.0         # [s]

DURATION = DURATION_DAYS * 86400.0
STOP_TIME = START_TIME + DURATION


# ============================================================
# LOAD FMU
# ============================================================

model_description = read_model_description(FMU_PATH)
for var in model_description.modelVariables:
    print(var.name)

vr = {
    variable.name: variable.valueReference
    for variable in model_description.modelVariables
}

unzipdir = extract(FMU_PATH)

fmu = FMU2Slave(
    guid=model_description.guid,
    unzipDirectory=unzipdir,
    modelIdentifier=model_description.coSimulation.modelIdentifier,
    instanceName="BUI0",
)


# ============================================================
# INPUT / OUTPUT VARIABLES
# ============================================================

input_vrs = [
    vr["PeopleNumber"],
    vr["LightsWatt"],
    vr["EEquipWatt"],
    vr["OthEquRadWatt"],
    vr["OthEquFCWatt"],
    vr["ZoneSetPoint"],
]

input_values = [
    0.0,     # PeopleNumber
    0.0,     # LightsWatt
    0.0,     # EEquipWatt
    0.0,     # OthEquRadWatt
    0.0,     # OthEquFCWatt
    20.0,    # ZoneSetPoint [°C]
]

output_vrs = [
    vr["TBuilding"],
    vr["HeatingLoadTarget"],
    vr["Site Outdoor Air Drybulb Temperature"]
]


# ============================================================
# INITIALIZE
# ============================================================

fmu.instantiate()

fmu.setupExperiment(
    startTime=START_TIME,
    stopTime=STOP_TIME,
)

fmu.enterInitializationMode()

fmu.setReal(input_vrs, input_values)

fmu.exitInitializationMode()


# ============================================================
# SIMULATION
# ============================================================

time = START_TIME
results = []

while time < STOP_TIME:

    # Set inputs
    fmu.setReal(input_vrs, input_values)

    # Advance FMU
    fmu.doStep(
        currentCommunicationPoint=time,
        communicationStepSize=STEP_SIZE,
    )

    time += STEP_SIZE

    # Read outputs
    temperature, heating_load = fmu.getReal(output_vrs)

    results.append({
        "time_s": time,
        "TBuilding_C": temperature,
        "HeatingLoadTarget_W": heating_load,
    })


# ============================================================
# TERMINATE
# ============================================================

fmu.terminate()
fmu.freeInstance()

shutil.rmtree(unzipdir, ignore_errors=True)


# ============================================================
# RESULTS
# ============================================================

df = pd.DataFrame(results)

print(df)

df.to_csv("results.csv", index=False)