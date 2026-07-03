# Digital-Twin Interfaces & Live Streaming

CosimGym federates normally exchange data only over HELICS and write results once, at the end of
a run. This page covers two opt-in, MQTT-backed mechanisms that externalize simulation data
*while it runs*, and the BK4 "config-only sim-to-real" pattern they enable.

Both mechanisms share one transport foundation: a background-thread MQTT client
(`src/adapters/mqtt_adapter.py`) talking to a Mosquitto broker (`docker-compose.yaml`, port
`11883` on the host). Publishing only enqueues onto a bounded, drop-oldest queue — the sim thread
never blocks on network I/O. Everything here is **off by default**; a scenario with no
`streaming`/`interface_config` block behaves exactly as it did before this feature existed.

## (A) `stream` — outbound telemetry mirror

Add `streaming: { stream: true }` to any `base` or `rl` federate's config to mirror its
inputs/outputs to MQTT each step, alongside normal HELICS traffic:

```yaml
federate_configs:
  spring_federate:
    type: base
    streaming:
      stream: true
      # stream_topic_prefix: cosim/${sim_id}/spring   # default: cosim/<sim_id>/<federate_name>
      # every_n_ticks: 1
```

Each message is published to `<prefix>/<inputs|outputs>/<entity_id>/<var_name>` as JSON:
`{sim_id, key, value, sim_time, wall_time}`. Use this for a **live dashboard** or any external
observer — it changes nothing about the co-simulation itself.

## (B) Interface federate — bidirectional external bridge

An interface federate (`type: interface`) has no physics model. Instead of stepping a model, it
relays its wired HELICS connections to and from the external world via an adapter
(`interface_config.adapter`, resolved from the same model catalog used for physics models).

```yaml
federate_configs:
  dt_bridge:
    type: interface
    timing_configs: { real_period: 1, rt_lag: 1.0, rt_lead: 1.0 }   # wall-clock pacing
    flags: { realtime: true }
    interface_config:
      adapter:
        name: mqtt_adapter
        params: { host: localhost, port: 11883, qos: 0, client_id: cosim_dt }
      streams:          # co-sim -> external: subscribe in HELICS, publish to MQTT
        - helics_key: plant.spring_federate.0/position
          topic: cosim/${sim_id}/spring/position
          every_n_ticks: 1
      bridges:          # external -> co-sim, or co-sim -> registry override
        - helics_key: plant.spring_federate.0/force
          topic: cosim/${sim_id}/sensor/force
          bounds: [-10, 10]
          scope: input        # input | output | param
          mode: replace        # replace external value | passthrough (real source + override)
```

`scope` picks how a bridge attaches to the target:

- **`input`** — the interface federate registers a normal HELICS global publication at
  `helics_key`. `mode: replace` publishes only once an external value has arrived; `mode:
  passthrough` (requires `source_key`) relays a real HELICS source until an external value shows
  up, then follows it. This is real-sensor-in-the-loop.
- **`output` / `param`** — the target already computes this value itself, so there is no HELICS
  representation to register. Instead the bridge writes the bounds-clipped external value into a
  Redis-backed `OverrideRegistry` (`src/core/override_registry.py`), keyed by
  `(scope, sim_id, federation, federate, entity, var)`. Any `base`/`rl` federate opts in with
  `override_enabled: true`; it substitutes the override in `_publish_outputs()` (output) or via
  `BaseModel.set_parameter()` (param, bounds-clipped against the catalog's `min`/`max`). Clearing
  the external value (no message on the bridge's topic) restores the federate's own computed
  behavior on the next step — no separate "disable" mechanism needed.

## The BK4 pattern: config-only sim-to-real

Because HELICS treats a physics-model federate and an interface federate identically — both just
register global publications/subscriptions under the same key names — swapping simulated hardware
for a real one is a change to **one federate's block only**: everything that subscribes to it is
untouched.

`src/scenarios/m5_bk4_demo_a_full_sim.yaml` and `m5_bk4_demo_b_digital_twin.yaml` are an identical
pair except for `input_federate`:

- **(a) full sim:** `input_federate` is `type: base`, running the `inputs4spring` model
  (constant force + randomized disturbance).
- **(b) digital twin:** `input_federate` is `type: interface` with two `scope: input` bridges
  registered at the *same* global publication keys (`input_federate.0/force`,
  `input_federate.0/disturbance`) the model federate would have used. `spring_federate`'s YAML —
  the "consumer" — is byte-identical between the two files.

Run (b), then feed it external "sensor" values with the demo actuator script:

```bash
python src/scenarios/bk4_demo_external_sensor.py   # publishes sinusoidal force + disturbance over MQTT
```

Watch both runs live with the live dashboard (below) to see (a)'s internally-generated values vs.
(b)'s externally-driven ones land on the same HELICS keys.

## Live dashboard

`src/dashboard/live_dashboard.py` is the "Live" page of the Streamlit dashboard (`dashboard_app.py`
is the "Results" page, the post-run historical explorer — both are served by the same app via
`st.navigation`). The Live page subscribes to `cosim/#` and shows the latest value per topic plus a
rolling chart, refreshed on a timer:

```bash
./src/dashboard/run_dashboard.sh    # http://localhost:8052, then switch to the "Live" page
```

It works with both mechanisms above — a `stream: true` federate's telemetry and an interface
federate's `streams`/`bridges` topics all show up as soon as they're published, no run needs to
finish first.
