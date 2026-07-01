"""
InterfaceFederate.py

Model-less HELICS federate whose "model" is a network bridge instead of
physics — the BK4 digital-twin building block (config docs/paper/
BREAKTHROUGH_INNOVATIONS.md:79-92). Wired into HELICS with normal pub/sub
like any model federate, but instead of stepping a model it relays its
connections to/from an external adapter (MQTT first).

M2 scope: outbound only (co-sim -> external). `interface_config.streams`
declares which HELICS keys to subscribe to and relay out to the adapter's
topic; `interface_config.bridges` (external -> co-sim) lands in M3/M4.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-07-01
"""
import importlib
from datetime import datetime

import helics as h

from core.BaseFederate import BaseFederate
from models.model_catalog.ModelCatalog import InterfaceType


class InterfaceFederate(BaseFederate):
    """A federate whose "model" is a transport adapter instead of physics."""

    def _register_entities(self):
        # No physics model to instantiate — resolve the transport adapter instead,
        # via the same catalog dynamic-import mechanism used for physics models.
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
        """Build HELICS subscriptions from `interface_config.streams` (co-sim -> external)."""
        subs = []
        interface_config = self.config.interface_config
        if not interface_config:
            return [], subs, []

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

        return [], subs, []

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

    def update_storage(self):
        # No entities/model state to record. Left empty (rather than inherited)
        # so store_local_file() stays a no-op for this federate.
        pass

    def finalize(self):
        if getattr(self, '_adapter', None) is not None:
            self._adapter.close()
        super().finalize()
