"""
test_scenario_manager_remote.py — _has_remote_federates / _setup_remote_execution behavior.

Builds a minimal ScenarioManager instance via object.__new__ (bypassing __init__, which
would otherwise require a real Redis connection / full logging setup) and stubs
core.ScenarioManager.RemoteExecutor so no real ssh connection is ever attempted.

Run: pytest tests/test_scenario_manager_remote.py -v
"""

import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ScenarioManager import ScenarioManager
from utils.config_dataclasses import ScenarioConfig

# `core/__init__.py` does `from .ScenarioManager import ScenarioManager`, which rebinds
# the `core.ScenarioManager` package attribute to the class, shadowing the submodule.
# Go through sys.modules directly to get the actual module object to monkeypatch.
scenario_manager_module = sys.modules['core.ScenarioManager']


def _scenario_config(deployment=None, host=None):
    fed = {
        "type": "base",
        "timing_configs": {"real_period": 1},
        "model_configs": {"instantiation": {"model_name": "x"}},
        "memory_config": {},
    }
    if host:
        fed["host"] = host
    raw = {
        "name": "t",
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T01:00:00",
        "memory_config": {},
        "federations": {"fed1": {"federate_configs": {"f1": fed}}},
    }
    if deployment is not None:
        raw["deployment"] = deployment
    return ScenarioConfig.model_validate(raw)


def _fake_manager(config, tmp_path) -> ScenarioManager:
    mgr = object.__new__(ScenarioManager)
    mgr.config = config
    mgr.logger = logging.getLogger('test_scenario_manager_remote')
    mgr.logger_system = SimpleNamespace(scenario_log_dir=Path(tmp_path))
    mgr.remote_executors = {}
    return mgr


def _deployment():
    return {
        "manager_address": "192.168.1.10",
        "machines": {"gpu_box": {"host": "192.168.1.42", "workdir": "/home/rando/rt"}},
    }


class TestNoRemoteFederates:

    def test_has_remote_federates_false(self, tmp_path):
        mgr = _fake_manager(_scenario_config(), tmp_path)
        assert mgr._has_remote_federates() is False

    def test_setup_remote_execution_is_noop(self, tmp_path, monkeypatch):
        mgr = _fake_manager(_scenario_config(), tmp_path)
        monkeypatch.setattr(
            scenario_manager_module,
            'RemoteExecutor',
            MagicMock(side_effect=AssertionError("RemoteExecutor must not be constructed for a local-only scenario")),
        )
        mgr._setup_remote_execution()
        assert mgr.remote_executors == {}


class TestRemoteFederates:

    def test_has_remote_federates_true(self, tmp_path):
        cfg = _scenario_config(deployment=_deployment(), host="gpu_box")
        mgr = _fake_manager(cfg, tmp_path)
        assert mgr._has_remote_federates() is True

    def test_verify_and_deploy_called_for_each_machine(self, tmp_path, monkeypatch):
        cfg = _scenario_config(deployment=_deployment(), host="gpu_box")
        mgr = _fake_manager(cfg, tmp_path)

        fake_executor = MagicMock()
        fake_executor.run.return_value = (0, '', '')
        monkeypatch.setattr(scenario_manager_module, 'RemoteExecutor', MagicMock(return_value=fake_executor))

        mgr._setup_remote_execution()

        fake_executor.open_master.assert_called_once()
        fake_executor.verify.assert_called_once()
        fake_executor.deploy.assert_called_once()
        fake_executor.run.assert_called_once()
        assert mgr.remote_executors == {"gpu_box": fake_executor}

    def test_preflight_failure_aborts_and_closes_masters(self, tmp_path, monkeypatch):
        cfg = _scenario_config(deployment=_deployment(), host="gpu_box")
        mgr = _fake_manager(cfg, tmp_path)

        fake_executor = MagicMock()
        fake_executor.verify.side_effect = RuntimeError("preflight failed")
        monkeypatch.setattr(scenario_manager_module, 'RemoteExecutor', MagicMock(return_value=fake_executor))

        with pytest.raises(RuntimeError, match="preflight failed"):
            mgr._setup_remote_execution()

        fake_executor.close.assert_called_once()
        assert mgr.remote_executors == {}


class TestCollectionAndCleanup:

    def _remote_manager(self, tmp_path):
        cfg = _scenario_config(deployment=_deployment(), host="gpu_box")
        mgr = _fake_manager(cfg, tmp_path)
        mgr.simulation_id = "scenario_20240101_120000_abcd"
        mgr.scenario_name = "t"
        return mgr

    def test_collect_noop_when_local(self, tmp_path):
        mgr = _fake_manager(_scenario_config(), tmp_path)
        mgr.simulation_id = "x"
        mgr.scenario_name = "t"
        mgr._collect_remote_results()  # remote_executors empty → nothing happens

    def test_collect_rsyncs_results_and_logs(self, tmp_path):
        mgr = self._remote_manager(tmp_path)
        fake_executor = MagicMock()
        mgr.remote_executors = {"gpu_box": fake_executor}

        mgr._collect_remote_results()

        # Two collect calls per machine: results dir + logs dir.
        assert fake_executor.collect.call_count == 2
        remote_args = [c.args[0] for c in fake_executor.collect.call_args_list]
        sim_id = mgr.simulation_id[-15:]
        assert any(f"/home/rando/rt/results/t/{sim_id}" == p for p in remote_args)
        assert any(str(tmp_path) == p for p in remote_args)  # logs = scenario_log_dir_rel

    def test_collect_failure_does_not_raise(self, tmp_path):
        mgr = self._remote_manager(tmp_path)
        fake_executor = MagicMock()
        fake_executor.collect.side_effect = RuntimeError("rsync boom")
        mgr.remote_executors = {"gpu_box": fake_executor}

        mgr._collect_remote_results()  # must swallow the error

    def test_cleanup_sweeps_and_closes(self, tmp_path):
        mgr = self._remote_manager(tmp_path)
        fake_executor = MagicMock()
        fake_executor.run.return_value = (0, '', '')
        mgr.remote_executors = {"gpu_box": fake_executor}

        mgr._cleanup_remote_execution()

        # pkill pattern is the full unique simulation_id.
        pkill_call = fake_executor.run.call_args
        assert pkill_call.args[0] == ['pkill', '-f', mgr.simulation_id]
        fake_executor.close.assert_called_once()
        assert mgr.remote_executors == {}

    def test_cleanup_never_raises_on_ssh_failure(self, tmp_path):
        mgr = self._remote_manager(tmp_path)
        fake_executor = MagicMock()
        fake_executor.run.side_effect = RuntimeError("ssh dead")
        fake_executor.close.side_effect = RuntimeError("close dead")
        mgr.remote_executors = {"gpu_box": fake_executor}

        mgr._cleanup_remote_execution()  # both wrapped → no raise
        assert mgr.remote_executors == {}
