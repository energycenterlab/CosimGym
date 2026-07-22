"""s5_external_feeder — stand-in "real zone sensor" for the S5 digital-twin scenario.

Run alongside `cs_s5_dt.yaml`: publishes plausible indoor zone temperatures to the
MQTT topic the interface federate's `scope: input`, `mode: replace` bridge consumes
(`cosim/cs_s5_dt/sensor/T_indoor`, HELICS key `building_federate.0/T_indoor`). The
heat pump's downstream PID controller then regulates against this EXTERNAL value
instead of the simulated 1R1C building — the paper's config-only sim-to-real (BK4)
claim. Mirrors src/scenarios/bk4_demo_external_sensor.py.

Usage:
    python scripts/paper_case_study/s5_external_feeder.py [--host localhost] [--port 11883] [--duration 60] [--period 0.5]
"""
import argparse, json, math, time
from datetime import datetime

import paho.mqtt.client as mqtt

TOPIC = "cosim/cs_s5_dt/sensor/T_indoor"
KEY = "building_federate.0/T_indoor"


def zone_temp(tick: float) -> float:
    """Plausible zone temperature: ~19.8 degC mean, slow +/-0.6 degC drift."""
    return 19.8 + 0.6 * math.sin(tick * 0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=11883)
    ap.add_argument("--duration", type=int, default=60, help="seconds to run")
    ap.add_argument("--period", type=float, default=0.5, help="seconds between publishes")
    a = ap.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="s5_external_feeder")
    client.connect(a.host, a.port)
    client.loop_start()
    print(f"[s5_external_feeder] publishing {KEY} -> {TOPIC} at {a.host}:{a.port} for {a.duration}s")
    t0 = time.time()
    tick = 0
    try:
        while time.time() - t0 < a.duration:
            val = zone_temp(tick)
            client.publish(TOPIC, json.dumps({
                "sim_id": "cs_s5_dt", "key": KEY, "value": val,
                "sim_time": tick, "wall_time": datetime.now().isoformat(),
            }))
            print(f"tick={tick} T_indoor={val:.3f}")
            tick += 1
            time.sleep(a.period)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
