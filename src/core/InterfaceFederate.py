"""
InterfaceFederate.py

Model-less HELICS federate whose "model" is a network bridge instead of
physics — the BK4 digital-twin building block (config docs/paper/
BREAKTHROUGH_INNOVATIONS.md:79-92). Wired into HELICS with normal pub/sub
like any model federate, but instead of stepping a model it relays its
connections to/from an external adapter (MQTT first).

M2 added outbound (co-sim -> external): `interface_config.streams` declares
which HELICS keys to subscribe to and relay out to the adapter's topic.
M3 adds inbound INPUT injection (external -> co-sim): `interface_config.bridges`
entries with `scope: input` register a HELICS publication that mirrors the
adapter's latest external value (bounds-clipped), for `mode: replace` (only the
external value, once one has arrived) or `mode: passthrough` (falls back to a
real HELICS source until an external value arrives).
M4 adds `scope: output`/`param` bridges: these have no HELICS representation
(the target federate already computes that value/parameter itself), so instead
this federate writes the bounds-clipped external value into the shared
`core.override_registry.OverrideRegistry` (Redis-backed), keyed at the target
(federation, federate, entity, var) parsed from `bridge.helics_key`. The target
federate (any `BaseFederate` with `config.override_enabled: true`) substitutes
it in `_publish_outputs()` (output) or `_apply_param_overrides()` (param).
`mode` is moot for these scopes — absence of an override already means "use
the computed value", so passthrough and replace behave identically.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-07-01
"""
import importlib
from datetime import datetime

import helics as h

from core.BaseFederate import BaseFederate
from core.override_registry import OverrideRegistry, parse_target
from models.model_catalog.ModelCatalog import InterfaceType


class InterfaceFederate(BaseFederate):
    """A federate whose "model" is a transport adapter instead of physics."""

    def _register_entities(self):
        # No physics model to instantiate — resolve the transport adapter instead,
        # via the same catalog dynamic-import mechanism used for physics models.
        self._override_registry = None
        self._override_bridges = []
        interface_config = self.config.interface_config
        if not interface_config:
            self._adapter = None
            return []

        adapter_name = interface_config.adapter.name
        adapter_meta = self.catalog.get_model_metadata(adapter_name)
        if adapter_meta is None:
            raise ValueError(f"Interface adapter '{adapter_name}' not found in catalog")

        module = importlib.import_module(adapter_meta.module_path)
        adapter_class = getattr(module, adapter_meta.class_name)
        params = {**adapter_meta.get_defaults(InterfaceType.PARAMETER), **interface_config.adapter.params}
        self._adapter = adapter_class(logger=self.logger, **params)
        self._adapter.connect()
        return []

    def initialize(self):
        self.logger.info(f'interface federate {self.name} initialization')
        self.federate = self._register_federate()
        self.entities = self._register_entities()
        self.inputs = {self.name: {}}
        self.outputs = {self.name: {}}
        self._deferred_inputs = {self.name: {}}
        self.pubs, self.subs, self.eps = self._register_connections()
        self.storage = {
            'train': self._create_storage_partition(),
            'test': self._create_storage_partition(),
        }
        self.logger.info(f'interface federate {self.name} initialized')

    def _register_connections(self):
        """Build HELICS pubs/subs from `interface_config`: `streams` (co-sim -> external,
        subscribe) and `bridges` with scope 'input' (external -> co-sim, publish)."""
        subs = []
        pubs = []
        interface_config = self.config.interface_config
        if not interface_config:
            return pubs, subs, []

        for i, stream in enumerate(interface_config.streams):
            topic_name = f"{self.name}/stream_{i}"
            subid = self.federate.register_global_input(topic_name, kind=stream.type, units=stream.units)
            h.helicsInputAddTarget(subid, stream.helics_key)
            subs.append({
                'entity_name': self.name,
                'topic': topic_name,
                'subid': subid,
                'causality': 'same_step',
                'stream_spec': stream,
            })
            self.logger.debug(f"Interface federate {self.name}: relaying '{stream.helics_key}' -> '{stream.topic}'")

        inbound_topics = []
        for i, bridge in enumerate(interface_config.bridges):
            if bridge.scope != "input":
                # output/param scopes have no HELICS registration — routed via the
                # override registry instead, handled in _publish_outputs().
                self._override_bridges.append(bridge)
                inbound_topics.append(bridge.topic)
                self.logger.debug(
                    f"Interface federate {self.name}: bridging '{bridge.topic}' -> "
                    f"override target '{bridge.helics_key}' (scope={bridge.scope})"
                )
                continue

            pubid = self.federate.register_global_publication(bridge.helics_key, kind=bridge.type, units=bridge.units)
            entry = {
                'entity_name': self.name,
                'topic': bridge.helics_key,
                'pubid': pubid,
                'bridge_spec': bridge,
            }
            if bridge.mode == "passthrough":
                source_topic = f"{self.name}/bridge_source_{i}"
                source_subid = self.federate.register_global_input(source_topic, kind=bridge.type, units=bridge.units)
                h.helicsInputAddTarget(source_subid, bridge.source_key)
                entry['source_subid'] = source_subid
            pubs.append(entry)
            inbound_topics.append(bridge.topic)
            self.logger.debug(
                f"Interface federate {self.name}: bridging '{bridge.topic}' -> '{bridge.helics_key}' "
                f"(mode={bridge.mode})"
            )

        if inbound_topics and self._adapter is not None:
            self._adapter.subscribe(inbound_topics)

        return pubs, subs, []

    def _receive_inputs(self, force_read_all=False):
        super()._receive_inputs(force_read_all=force_read_all)
        if self._adapter is None:
            return

        wall_time = datetime.now().isoformat()
        for sub in self.subs:
            stream = sub.get('stream_spec')
            if stream is None:
                continue
            every_n_ticks = max(1, stream.every_n_ticks)
            if self.ts % every_n_ticks != 0:
                continue
            var_name = sub['subid'].name.split('/')[-1]
            value = self.inputs.get(sub['entity_name'], {}).get(var_name)
            if value is None:
                continue
            self._adapter.publish(stream.topic, {
                'sim_id': self.simulation_id,
                'key': stream.helics_key,
                'value': value,
                'sim_time': self.time_granted,
                'wall_time': wall_time,
            })

    def _publish_outputs(self):
        """Publish each 'input'-scope bridge's current value: the external adapter
        value if one has arrived (bounds-clipped), else the real source in
        `mode: passthrough`, else nothing yet (`mode: replace` with no value)."""
        for pub in self.pubs:
            bridge = pub.get('bridge_spec')
            if bridge is None:
                continue

            value = None
            external = self._adapter.latest(bridge.topic) if self._adapter is not None else None
            if external is not None:
                value = external.get('value')
            elif bridge.mode == "passthrough":
                value = self._read_subscription_value(pub['source_subid'])

            if value is None:
                continue
            if bridge.bounds is not None:
                lo, hi = bridge.bounds
                value = max(lo, min(hi, value))

            pub['pubid'].publish(value)
            self.logger.debug(f"Interface federate {self.name}: published {value} onto '{bridge.helics_key}'")

        self._publish_override_bridges()

    def _publish_override_bridges(self):
        """`scope: output`/`param` bridges: write (or clear) the shared override
        registry so the target federate substitutes/restores each step."""
        if not self._override_bridges:
            return
        if self._override_registry is None:
            self._override_registry = OverrideRegistry(logger=self.logger)

        for bridge in self._override_bridges:
            federation, federate, entity, var = parse_target(bridge.helics_key, self.federation_name)
            external = self._adapter.latest(bridge.topic) if self._adapter is not None else None

            if external is None:
                self._override_registry.clear_override(
                    bridge.scope, self.simulation_id, federation, federate, entity, var
                )
                continue

            value = external.get('value')
            if value is None:
                continue
            if bridge.bounds is not None:
                lo, hi = bridge.bounds
                value = max(lo, min(hi, value))

            self._override_registry.set_override(
                bridge.scope, self.simulation_id, federation, federate, entity, var, value
            )
            self.logger.debug(
                f"Interface federate {self.name}: {bridge.scope} override {value} -> "
                f"{federation}.{federate}.{entity}/{var}"
            )

    def update_storage(self):
        # No entities/model state to record. Left empty (rather than inherited)
        # so store_local_file() stays a no-op for this federate.
        pass

    def finalize(self):
        if getattr(self, '_override_registry', None) is not None:
            for bridge in self._override_bridges:
                federation, federate, entity, var = parse_target(bridge.helics_key, self.federation_name)
                self._override_registry.clear_override(
                    bridge.scope, self.simulation_id, federation, federate, entity, var
                )
        if getattr(self, '_adapter', None) is not None:
            self._adapter.close()
        super().finalize()
