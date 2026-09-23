from datetime import datetime
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave


FMU_PATH = Path("BUI0.fmu")

# The IDF inside this FMU uses 6 EnergyPlus timesteps/hour = 600 s.
STEP_SIZE = 600.0

# How long to simulate from each requested start date.
SIMULATION_HOURS = 9600

# Compare two dates with very different weather conditions.
TEST_DATES = [
    ("winter", 1, 15),   # 15 January
    ("summer", 7, 15),   # 15 July
]


def seconds_from_jan1(month, day):
    """Convert a month/day to seconds since Jan 1.

    The year is arbitrary; only the day-of-year offset is used.
    """
    year = 2021  # non-leap year
    start_of_year = datetime(year, 1, 1)
    requested_date = datetime(year, month, day)
    return (requested_date - start_of_year).total_seconds()


def run_fmu(label, month, day):
    start_time = seconds_from_jan1(month, day)
    stop_time = start_time + SIMULATION_HOURS * 3600.0

    print()
    print("=" * 70)
    print(f"Test: {label}")
    print(f"Requested date: {day:02d}/{month:02d}")
    print(f"FMI start_time: {start_time:.0f} s")
    print(f"FMI stop_time : {stop_time:.0f} s")
    print("=" * 70)

    model_description = read_model_description(FMU_PATH)

    if model_description.fmiVersion != "2.0":
        raise RuntimeError(
            f"Expected FMI 2.0, found FMI {model_description.fmiVersion}"
        )

    if model_description.coSimulation is None:
        raise RuntimeError("This FMU is not an FMI 2.0 Co-Simulation FMU.")

    # Map variable names to FMI value references
    vr = {
        variable.name: variable.valueReference
        for variable in model_description.modelVariables
    }

    required = [
        "PeopleNumber",
        "LightsWatt",
        "EEquipWatt",
        "OthEquRadWatt",
        "OthEquFCWatt",
        "ZoneSetPoint",
        "TBuilding",
        "HeatingLoadTarget",
    ]

    missing = [name for name in required if name not in vr]
    if missing:
        raise RuntimeError(f"Missing expected FMU variables: {missing}")

    unzipdir = extract(FMU_PATH)

    fmu = FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=unzipdir,
        modelIdentifier=model_description.coSimulation.modelIdentifier,
        instanceName=f"BUI0_{label}",
    )

    try:
        # ---------------------------------------------------------------
        # THIS IS THE IMPORTANT PART OF THE TEST:
        # we explicitly tell the FMU which time of year to start from.
        # ---------------------------------------------------------------
        fmu.instantiate()
        fmu.setupExperiment(
            startTime=start_time,
            stopTime=stop_time,
        )

        fmu.enterInitializationMode()

        # Use identical inputs for both winter and summer tests.
        input_vrs = [
            vr["PeopleNumber"],
            vr["LightsWatt"],
            vr["EEquipWatt"],
            vr["OthEquRadWatt"],
            vr["OthEquFCWatt"],
            vr["ZoneSetPoint"],
        ]

        input_values = [
            0.0,   # PeopleNumber
            0.0,   # LightsWatt
            0.0,   # EEquipWatt
            0.0,   # OthEquRadWatt
            0.0,   # OthEquFCWatt
            20.0,  # ZoneSetPoint [degC]
        ]

        fmu.setReal(input_vrs, input_values)
        fmu.exitInitializationMode()

        output_vrs = [
            vr["TBuilding"],
            vr["HeatingLoadTarget"],
        ]

        time = start_time
        rows = []

        while time < stop_time - 1e-9:
            # Keep inputs constant throughout the experiment.
            fmu.setReal(input_vrs, input_values)

            # FMPy raises an exception if the FMI call reports an error.
            fmu.doStep(
                currentCommunicationPoint=time,
                communicationStepSize=STEP_SIZE,
            )

            time += STEP_SIZE

            t_building, heating_load = fmu.getReal(output_vrs)

            elapsed_hours = (time - start_time) / 3600.0

            rows.append(
                {
                    "case": label,
                    "month": month,
                    "day": day,
                    "fmi_time_s": time,
                    "elapsed_hours": elapsed_hours,
                    "TBuilding_C": t_building,
                    "HeatingLoadTarget_W": heating_load,
                }
            )

        return pd.DataFrame(rows)

    finally:
        try:
            fmu.terminate()
        except Exception:
            pass

        try:
            fmu.freeInstance()
        except Exception:
            pass

        shutil.rmtree(unzipdir, ignore_errors=True)


def main():
    if not FMU_PATH.exists():
        raise FileNotFoundError(
            f"{FMU_PATH} not found. Put this script next to BUI0.fmu "
            "or change FMU_PATH."
        )

    results = []

    for label, month, day in TEST_DATES:
        df = run_fmu(label, month, day)
        results.append(df)

        print(
            f"{label:>8}: "
            f"TBuilding first={df['TBuilding_C'].iloc[0]:.2f} °C, "
            f"mean={df['TBuilding_C'].mean():.2f} °C | "
            f"HeatingLoadTarget first={df['HeatingLoadTarget_W'].iloc[0]:.1f} W, "
            f"mean={df['HeatingLoadTarget_W'].mean():.1f} W"
        )

    result = pd.concat(results, ignore_index=True)
    result.to_csv("fmu_start_date_test.csv", index=False)

    # Temperature plot
    plt.figure(figsize=(9, 5))
    for label in result["case"].unique():
        d = result[result["case"] == label]
        plt.plot(
            d["elapsed_hours"],
            d["TBuilding_C"],
            label=label,
        )
    plt.xlabel("Hours after requested start")
    plt.ylabel("TBuilding [°C]")
    plt.title("EnergyPlus FMU start-date test: zone temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fmu_start_date_temperature.png", dpi=150)
    plt.show()

    # Heating-load plot
    plt.figure(figsize=(9, 5))
    for label in result["case"].unique():
        d = result[result["case"] == label]
        plt.plot(
            d["elapsed_hours"],
            d["HeatingLoadTarget_W"],
            label=label,
        )
    plt.xlabel("Hours after requested start")
    plt.ylabel("HeatingLoadTarget [W]")
    plt.title("EnergyPlus FMU start-date test: heating demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fmu_start_date_heating.png", dpi=150)
    plt.show()

    print()
    print("Saved:")
    print("  fmu_start_date_test.csv")
    print("  fmu_start_date_temperature.png")
    print("  fmu_start_date_heating.png")
    print()
    print("Interpretation:")
    print(
        "If the January and July runs are clearly different, especially "
        "HeatingLoadTarget, the FMU is responding to the requested FMI "
        "start_time rather than always beginning from January 1."
    )


if __name__ == "__main__":
    main()
