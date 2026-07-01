"""
InterfaceFederate.py

Model-less HELICS federate whose "model" is a network bridge instead of
physics — the BK4 digital-twin building block (config docs/paper/
BREAKTHROUGH_INNOVATIONS.md:79-92). Wired into HELICS with normal pub/sub
like any model federate, but instead of stepping a model it relays its
connections to/from an external adapter (MQTT first).

M0 scope: the shell boots with no entities (so it also has no HELICS pubs/subs
to register — those come from `interface_config` in M2/M3) and shuts down
cleanly. Streaming/bridging logic lands in M2/M3/M4.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-07-01
"""
from core.BaseFederate import BaseFederate


class InterfaceFederate(BaseFederate):
    """A federate whose "model" is a transport adapter instead of physics."""

    def _register_entities(self):
        # No physics model to instantiate — nothing to relay yet in M0.
        return []

    def update_storage(self):
        # No entities/model state to record. Left empty (rather than inherited)
        # so store_local_file() stays a no-op for this federate.
        pass
