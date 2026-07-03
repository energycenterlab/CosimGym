"""
test_deployment_config.py — validate deployment/host schema for distributed SSH federate spawning.

Run: pytest tests/test_deployment_config.py -v
"""

import os
import sys
import copy

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config_dataclasses import ScenarioConfig, MachineConfig, DeploymentConfig


def _base_scenario(federate_extra=None, deployment=None):
    fed = {
        "type": "base",
        "timing_configs": {"real_period": 1},
        "model_configs": {"instantiation": {"model_name": "x"}},
        "memory_config": {},
    }
    if federate_extra:
        fed.update(federate_extra)
    scenario = {
        "name": "t",
        "start_time": "2024-01-01T00:00:00",
        "end_time": "2024-01-01T01:00:00",
        "memory_config": {},
        "federations": {
            "fed1": {
                "federate_configs": {
                    "f1": fed,
                }
            }
        },
    }
    if deployment is not None:
        scenario["deployment"] = deployment
    return scenario


class TestNoDeploymentUnchanged:

    def test_scenario_without_deployment_parses(self):
        cfg = ScenarioConfig.model_validate(_base_scenario())
        assert cfg.deployment is None
        assert cfg.federations["fed1"].federate_configs["f1"].host is None


class TestDeploymentSchema:

    def test_machine_config_defaults(self):
        m = MachineConfig.model_validate({"host": "1.2.3.4", "workdir": "/x"})
        assert m.ssh_port == 22
        assert m.conda_env == "cosim_gym"
        assert m.user is None
        assert m.python is None

    def test_full_valid_deployment_scenario(self):
        deployment = {
            "manager_address": "192.168.1.10",
            "machines": {
                "gpu_box": {"host": "192.168.1.42", "workdir": "/home/rando/cosimgym_rt"},
            },
        }
        cfg = ScenarioConfig.model_validate(
            _base_scenario(federate_extra={"host": "gpu_box"}, deployment=deployment)
        )
        fed = cfg.federations["fed1"].federate_configs["f1"]
        assert fed.host == "gpu_box"
        assert cfg.deployment.manager_address == "192.168.1.10"
        assert cfg.deployment.machines["gpu_box"].workdir == "/home/rando/cosimgym_rt"


class TestDeploymentValidation:

    def test_host_without_deployment_block_rejected(self):
        with pytest.raises(Exception):
            ScenarioConfig.model_validate(_base_scenario(federate_extra={"host": "gpu_box"}))

    def test_host_unknown_alias_rejected(self):
        deployment = {
            "manager_address": "192.168.1.10",
            "machines": {"gpu_box": {"host": "192.168.1.42", "workdir": "/x"}},
        }
        with pytest.raises(Exception):
            ScenarioConfig.model_validate(
                _base_scenario(federate_extra={"host": "nonexistent"}, deployment=deployment)
            )

    def test_host_without_manager_address_rejected(self):
        deployment = {
            "machines": {"gpu_box": {"host": "192.168.1.42", "workdir": "/x"}},
        }
        with pytest.raises(Exception):
            ScenarioConfig.model_validate(
                _base_scenario(federate_extra={"host": "gpu_box"}, deployment=deployment)
            )

    def test_host_on_rl_federate_rejected(self):
        deployment = {
            "manager_address": "192.168.1.10",
            "machines": {"gpu_box": {"host": "192.168.1.42", "workdir": "/x"}},
        }
        scenario = _base_scenario(deployment=deployment)
        scenario["federations"]["fed1"]["federate_configs"]["f1"] = {
            "type": "rl",
            "timing_configs": {"real_period": 1},
            "host": "gpu_box",
        }
        with pytest.raises(Exception):
            ScenarioConfig.model_validate(scenario)

    def test_deployment_without_any_host_is_fine(self):
        deployment = {
            "manager_address": "192.168.1.10",
            "machines": {"gpu_box": {"host": "192.168.1.42", "workdir": "/x"}},
        }
        cfg = ScenarioConfig.model_validate(_base_scenario(deployment=deployment))
        assert cfg.deployment.manager_address == "192.168.1.10"
