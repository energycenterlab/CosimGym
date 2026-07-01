"""
bk4_demo_external_sensor.py — stand-in "real hardware" for the BK4 digital-twin demo
(digitaltwin_interfaces plan, M5).

Run alongside `m5_bk4_demo_b_digital_twin.yaml`: publishes a sinusoidal "sensor" force and a
randomized disturbance onto the two MQTT topics input_federate's interface bridges subscribe to
(`scope: input`, `mode: replace`), once per second. This is the external process the BK4 pattern
bridges into the co-simulation in place of the `inputs4spring` physics model used in
`m5_bk4_demo_a_full_sim.yaml` — nothing here is CosimGym-specific, it is meant to stand in for a
real sensor/actuator process talking plain MQTT.

Usage:
    python src/scenarios/bk4_demo_external_sensor.py [--host localhost] [--port 11883] [--duration 20]
"""
import argparse
import json
import math
import random
import time
from datetime import datetime

import paho.mqtt.client as mqtt

FORCE_TOPIC = "cosim/bk4_demo/sensor/force"
DISTURBANCE_TOPIC = "cosim/bk4_demo/sensor/disturbance"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=11883)
    parser.add_argument("--duration", type=int, default=20, help="seconds to run")
    parser.add_argument("--period", type=float, default=1.0, help="seconds between publishes")
    args = parser.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="bk4_demo_external_sensor")
    client.connect(args.host, args.port)
    client.loop_start()

    print(f"Publishing external sensor values to {args.host}:{args.port} for {args.duration}s "
          f"(Ctrl+C to stop early)...")
    t0 = time.time()
    tick = 0
    try:
        while time.time() - t0 < args.duration:
            force = 15.0 * math.sin(tick * 0.3)
            disturbance = random.uniform(-2.0, 2.0)
            wall_time = datetime.now().isoformat()

            client.publish(FORCE_TOPIC, json.dumps({
                "sim_id": "bk4_demo", "key": "input_federate.0/force",
                "value": force, "sim_time": tick, "wall_time": wall_time,
            }))
            client.publish(DISTURBANCE_TOPIC, json.dumps({
                "sim_id": "bk4_demo", "key": "input_federate.0/disturbance",
                "value": disturbance, "sim_time": tick, "wall_time": wall_time,
            }))
            print(f"tick={tick} force={force:.2f} disturbance={disturbance:.2f}")

            tick += 1
            time.sleep(args.period)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
