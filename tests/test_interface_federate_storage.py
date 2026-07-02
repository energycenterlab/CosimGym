"""
test_interface_federate_storage.py — unit tests for InterfaceFederate's
json/parquet result recording (streams -> inputs, input-scope bridges ->
outputs, output/param-scope bridges -> outputs/params).

Builds a bare InterfaceFederate via __new__ + manual attribute assignment,
so these tests exercise update_storage()/_create_storage_partition() in
isolation, without needing real HELICS/MQTT/Redis.

Run: pytest tests/test_interface_federate_storage.py -v
"""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.InterfaceFederate import InterfaceFederate
from utils.config_dataclasses import StreamSpec, BridgeSpec


def _make_federate(sink='json'):
    fed = InterfaceFederate.__new__(InterfaceFederate)
    fed.name = 'dt_bridge'
    fed.mode = 'test'
    fed.ts = 3
    fed.date_time = '2024-01-01T00:00:00'
    fed.config = SimpleNamespace(memory_config=SimpleNamespace(sink=sink, batch_size=10))

    stream = StreamSpec(helics_key='src_federate.0/temperature', topic='cosim/x/temperature')
    input_bridge = BridgeSpec(helics_key='dt_bridge_out/force', topic='cosim/x/force', scope='input')
    output_bridge = BridgeSpec(helics_key='target_federate.0/velocity', topic='cosim/x/velocity', scope='output')
    param_bridge = BridgeSpec(helics_key='target_federate.0/damping', topic='cosim/x/damping', scope='param')

    fed.subs = [{'stream_spec': stream, 'subid': SimpleNamespace(name='dt_bridge/stream_0')}]
    fed.pubs = [{'bridge_spec': input_bridge}]
    fed._override_bridges = [output_bridge, param_bridge]
    fed._last_override_values = {}

    fed.storage = {
        'train': fed._create_storage_partition(),
        'test': fed._create_storage_partition(),
    }
    return fed, stream, input_bridge, output_bridge, param_bridge


class TestCreateStoragePartition:

    def test_partition_keys_match_streams_and_bridges(self):
        fed, *_ = _make_federate()
        partition = fed.storage['test']
        entity_id = 'dt_bridge.0'
        assert set(partition['inputs'][entity_id].keys()) == {'temperature'}
        assert set(partition['outputs'][entity_id].keys()) == {'force', 'velocity'}
        assert set(partition['params'][entity_id].keys()) == {'damping'}
        assert partition['time'] == []


class TestUpdateStorageJson:

    def test_values_recorded_into_json_partition(self):
        fed, stream, input_bridge, output_bridge, param_bridge = _make_federate(sink='json')
        fed.subs[0]['last_value'] = 21.5
        fed.pubs[0]['last_value'] = 12.0
        fed._last_override_values[output_bridge.helics_key] = 3.3
        fed._last_override_values[param_bridge.helics_key] = 0.1

        fed.update_storage()

        entity_id = 'dt_bridge.0'
        partition = fed.storage['test']
        assert partition['time'] == ['2024-01-01T00:00:00']
        assert partition['inputs'][entity_id]['temperature'] == [21.5]
        assert partition['outputs'][entity_id]['force'] == [12.0]
        assert partition['outputs'][entity_id]['velocity'] == [3.3]
        assert partition['params'][entity_id]['damping'] == [0.1]

    def test_none_last_value_recorded_as_none(self):
        fed, *_ = _make_federate(sink='json')
        # no last_value set anywhere -> everything defaults to None this tick
        fed.update_storage()
        entity_id = 'dt_bridge.0'
        partition = fed.storage['test']
        assert partition['inputs'][entity_id]['temperature'] == [None]
        assert partition['outputs'][entity_id]['force'] == [None]

    def test_multiple_ticks_append(self):
        fed, stream, input_bridge, *_ = _make_federate(sink='json')
        fed.subs[0]['last_value'] = 1.0
        fed.update_storage()
        fed.subs[0]['last_value'] = 2.0
        fed.update_storage()
        entity_id = 'dt_bridge.0'
        assert fed.storage['test']['inputs'][entity_id]['temperature'] == [1.0, 2.0]


class TestUpdateStorageParquet:

    def test_json_partition_not_grown_for_parquet_sink(self):
        fed, *_ = _make_federate(sink='parquet')
        fed._enqueue_async_storage_row = lambda row: None  # avoid spinning up a real writer thread
        fed.subs[0]['last_value'] = 21.5
        fed.update_storage()
        entity_id = 'dt_bridge.0'
        assert fed.storage['test']['inputs'][entity_id]['temperature'] == []
        assert fed.storage['test']['time'] == []

    def test_row_enqueued_with_expected_shape(self):
        fed, stream, input_bridge, output_bridge, param_bridge = _make_federate(sink='parquet')
        captured = []
        fed._enqueue_async_storage_row = captured.append
        fed.subs[0]['last_value'] = 21.5
        fed.pubs[0]['last_value'] = 12.0
        fed._last_override_values[output_bridge.helics_key] = 3.3
        fed._last_override_values[param_bridge.helics_key] = 0.1

        fed.update_storage()

        assert len(captured) == 1
        row = captured[0]
        entity_id = 'dt_bridge.0'
        assert row['mode'] == 'test'
        assert row['ts'] == 3
        assert row['time'] == '2024-01-01T00:00:00'
        assert row['inputs'][entity_id]['temperature'] == 21.5
        assert row['outputs'][entity_id]['force'] == 12.0
        assert row['outputs'][entity_id]['velocity'] == 3.3
        assert row['params'][entity_id]['damping'] == 0.1
