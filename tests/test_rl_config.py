"""
test_rl_config.py — validate RL config schema, extra='forbid', and scenario parsing.

Run: pytest tests/test_rl_config.py -v
"""

import os
import sys
import glob
import copy

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config_dataclasses import (
    ScenarioConfig,
    ReinforcementLearningConfig,
    ObservationSpec,
    ActionSpec,
    EnvironmentConfig,
    AgentConfig,
    RunConfig,
    PhaseConfig,
    ExperimentConfig,
    CheckpointConfig,
    Hyperparameters,
    ResetConfig,
    MemoryConfig,
    StreamingConfig,
    InterfaceConfig,
    InterfaceFederateConfig,
    StreamSpec,
    FedTimingConfig,
    BridgeSpec,
    BaseFederateConfig,
)
from utils.config_reader import read_scenario_config

SCENARIOS_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'scenarios')


def _scenario_yamls():
    """Yield (path, name) for every .yaml in src/scenarios/."""
    for path in sorted(glob.glob(os.path.join(SCENARIOS_DIR, '*.yaml'))):
        yield path, os.path.basename(path)


def _rl_scenario_yamls():
    """Yield (path, name) for RL scenarios only."""
    for path, name in _scenario_yamls():
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError:
            continue
        if raw and raw.get('reinforcement_learning_config'):
            yield path, name


# ── Parse gate: every scenario validates under ScenarioConfig ──


class TestScenarioParsing:

    @pytest.mark.parametrize("path,name", list(_scenario_yamls()), ids=lambda x: x if isinstance(x, str) else "")
    def test_scenario_parses(self, path, name):
        if name == 'Adelaide_test.yaml':
            pytest.skip('known broken YAML indentation')
        read_scenario_config(path)

    @pytest.mark.parametrize("path,name", list(_rl_scenario_yamls()), ids=lambda x: x if isinstance(x, str) else "")
    def test_rl_scenario_has_valid_rl_config(self, path, name):
        cfg = read_scenario_config(path)
        assert cfg.reinforcement_learning_config is not None
        rl = cfg.reinforcement_learning_config
        assert len(rl.environment.observations) > 0
        assert len(rl.environment.actions) > 0
        assert rl.agent.model_name is not None


# ── extra='forbid' rejects unknown keys ──


class TestExtraForbid:

    def test_observation_spec_rejects_unknown(self):
        with pytest.raises(Exception):
            ObservationSpec.model_validate({"causality": "next_step", "bogus_key": 42})

    def test_action_spec_rejects_unknown(self):
        with pytest.raises(Exception):
            ActionSpec.model_validate({"space": "box", "bogus_key": 42})

    def test_agent_config_rejects_unknown(self):
        with pytest.raises(Exception):
            AgentConfig.model_validate({"model_name": "x", "unknown_field": True})

    def test_hyperparameters_rejects_unknown(self):
        with pytest.raises(Exception):
            Hyperparameters.model_validate({"learning_rate": 0.001, "foo": "bar"})

    def test_run_config_rejects_unknown(self):
        with pytest.raises(Exception):
            RunConfig.model_validate({"train": {"episodes": 1, "episode_length": 10}, "extra_key": 1})

    def test_full_rl_config_rejects_typo(self):
        valid = {
            "environment": {
                "observations": {"a.b.0.x": None},
                "actions": {"a.b.0.y": None},
            },
            "agent": {"model_name": "test"},
            "run": {"train": {"episodes": 1, "episode_length": 10}},
        }
        ReinforcementLearningConfig.model_validate(valid)
        bad = copy.deepcopy(valid)
        bad["typo_field"] = True
        with pytest.raises(Exception):
            ReinforcementLearningConfig.model_validate(bad)


# ── Structural validators ──


class TestValidators:

    def test_phase_total_steps(self):
        p = PhaseConfig(episodes=10, episode_length=100)
        assert p.total_steps == 1000

    def test_test_only_without_checkpoint_rejected(self):
        with pytest.raises(Exception):
            RunConfig.model_validate({
                "test": {"episodes": 1, "episode_length": 10},
            })

    def test_test_only_with_checkpoint_ok(self):
        cfg = RunConfig.model_validate({
            "test": {"episodes": 1, "episode_length": 10, "checkpoint": "/path/to/ckpt"},
        })
        assert cfg.test.checkpoint == "/path/to/ckpt"

    def test_checkpoint_best_path_resolved(self):
        ckpt = CheckpointConfig(dir="checkpoints", best="best.pth")
        assert ckpt.best_path == "checkpoints/best.pth"

    def test_checkpoint_best_path_absolute(self):
        ckpt = CheckpointConfig(dir="checkpoints", best="/abs/best.pth")
        assert ckpt.best_path == "/abs/best.pth"

    def test_environment_observations_nonempty(self):
        with pytest.raises(Exception):
            EnvironmentConfig.model_validate({
                "observations": {},
                "actions": {"a.b.0.x": None},
            })

    def test_environment_actions_nonempty(self):
        with pytest.raises(Exception):
            EnvironmentConfig.model_validate({
                "observations": {"a.b.0.x": None},
                "actions": {},
            })

    def test_null_obs_spec_coerced(self):
        env = EnvironmentConfig.model_validate({
            "observations": {"a.b.0.x": None},
            "actions": {"a.b.0.y": None},
        })
        assert isinstance(env.observations["a.b.0.x"], ObservationSpec)

    def test_null_action_spec_coerced(self):
        env = EnvironmentConfig.model_validate({
            "observations": {"a.b.0.x": None},
            "actions": {"a.b.0.y": None},
        })
        assert isinstance(env.actions["a.b.0.y"], ActionSpec)

    def test_reset_config_defaults(self):
        r = ResetConfig()
        assert r.mode == "full"
        assert r.force_defaults is False

    def test_hyperparameters_as_kwargs_omits_none(self):
        hp = Hyperparameters(learning_rate=0.001, gamma=None)
        kw = hp.as_kwargs()
        assert "learning_rate" in kw
        assert "gamma" not in kw


# ── nonblocking_storage plan (S0): MemoryConfig.sink ──


class TestMemoryConfigSink:

    def test_sink_defaults_to_json(self):
        cfg = MemoryConfig()
        assert cfg.sink == "json"

    def test_sink_explicit_parquet(self):
        cfg = MemoryConfig.model_validate({"sink": "parquet"})
        assert cfg.sink == "parquet"

    def test_sink_explicit_none(self):
        cfg = MemoryConfig.model_validate({"sink": "none"})
        assert cfg.sink == "none"

    def test_sink_invalid_value_rejected(self):
        with pytest.raises(Exception):
            MemoryConfig.model_validate({"sink": "csv"})


# ── digitaltwin_interfaces plan (M0): StreamingConfig / InterfaceConfig ──


class TestStreamingAndInterfaceConfig:

    def test_streaming_config_defaults_opt_out(self):
        cfg = StreamingConfig()
        assert cfg.stream is False

    def test_streaming_config_rejects_typo_silently(self):
        # StreamingConfig follows the base federate's extra='ignore' convention,
        # unlike the RL axes' extra='forbid'.
        cfg = StreamingConfig.model_validate({"stream": True, "bogus_key": 1})
        assert cfg.stream is True

    def test_interface_config_requires_adapter(self):
        with pytest.raises(Exception):
            InterfaceConfig.model_validate({"streams": [], "bridges": []})

    def test_interface_config_rejects_unknown_key(self):
        valid = {"adapter": {"name": "mqtt_adapter", "params": {}}}
        InterfaceConfig.model_validate(valid)
        bad = copy.deepcopy(valid)
        bad["typo_field"] = True
        with pytest.raises(Exception):
            InterfaceConfig.model_validate(bad)

    def test_interface_federate_config_empty_interface_config_ok(self):
        cfg = InterfaceFederateConfig.model_validate({
            "name": "dt_bridge",
            "id": "fed_dt_bridge",
            "type": "interface",
            "timing_configs": {"real_period": 1},
        })
        assert cfg.interface_config is None
        assert cfg.model_configs is None

    def test_interface_federate_config_type_discriminates(self):
        cfg = ScenarioConfig.model_validate({
            "name": "s",
            "start_time": "2024-01-01T00:00:00",
            "end_time": "2024-01-01T00:01:00",
            "memory_config": {},
            "federations": {
                "f": {
                    "federate_configs": {
                        "dt_bridge": {
                            "type": "interface",
                            "timing_configs": {"real_period": 1},
                        }
                    }
                }
            },
        })
        fed = cfg.federations["f"].federate_configs["dt_bridge"]
        assert isinstance(fed, InterfaceFederateConfig)


# ── digitaltwin_interfaces plan (M2): StreamSpec type/units, rt_lag/rt_lead ──


class TestInterfaceOutboundConfig:

    def test_stream_spec_defaults(self):
        s = StreamSpec.model_validate({"helics_key": "fed.0/x", "topic": "cosim/x"})
        assert s.type == "double"
        assert s.units == ""
        assert s.every_n_ticks == 1

    def test_timing_config_rt_lag_lead_default_none(self):
        cfg = FedTimingConfig(real_period=1)
        assert cfg.rt_lag is None
        assert cfg.rt_lead is None

    def test_timing_config_rt_lag_lead_explicit(self):
        cfg = FedTimingConfig(real_period=1, rt_lag=1.0, rt_lead=2.0)
        assert cfg.rt_lag == 1.0
        assert cfg.rt_lead == 2.0


# ── digitaltwin_interfaces plan (M3): BridgeSpec passthrough/source_key ──


class TestInterfaceInboundConfig:

    def test_bridge_spec_replace_defaults(self):
        b = BridgeSpec.model_validate({"helics_key": "dt/force", "topic": "cosim/x"})
        assert b.mode == "replace"
        assert b.scope == "input"
        assert b.source_key is None

    def test_bridge_spec_passthrough_requires_source_key(self):
        with pytest.raises(Exception):
            BridgeSpec.model_validate({
                "helics_key": "dt/force", "topic": "cosim/x", "mode": "passthrough",
            })

    def test_bridge_spec_passthrough_with_source_key_ok(self):
        b = BridgeSpec.model_validate({
            "helics_key": "dt/force", "topic": "cosim/x",
            "mode": "passthrough", "source_key": "input_federate.0/force",
        })
        assert b.source_key == "input_federate.0/force"


# ── digitaltwin_interfaces plan (M4): override_enabled, output/param bridges, registry parsing ──


class TestInterfaceOverrideConfig:

    def test_override_enabled_default_false(self):
        cfg = BaseFederateConfig.model_validate({
            "name": "f", "id": "fed_f", "type": "base",
            "timing_configs": {"real_period": 1},
            "model_configs": {"instantiation": {"model_name": "spring_mass_damper"}},
            "memory_config": {},
        })
        assert cfg.override_enabled is False

    def test_bridge_spec_output_scope_passthrough_no_source_key_needed(self):
        # Only scope 'input' requires source_key for passthrough (M4 relaxation).
        b = BridgeSpec.model_validate({
            "helics_key": "spring_federate.0/velocity", "topic": "cosim/x",
            "scope": "output", "mode": "passthrough",
        })
        assert b.source_key is None

    def test_bridge_spec_param_scope_ok(self):
        b = BridgeSpec.model_validate({
            "helics_key": "spring_federate.0/damping", "topic": "cosim/x",
            "scope": "param", "bounds": [0.0, 10.0],
        })
        assert b.scope == "param"
        assert b.bounds == (0.0, 10.0)

    def test_parse_target_same_federation_reconstructs_entity_id(self):
        from core.override_registry import parse_target
        federation, federate, entity, var = parse_target("spring_federate.0/velocity", "federation_1")
        assert federation == "federation_1"
        assert federate == "spring_federate"
        assert entity == "spring_federate.0"  # NOT "0" — matches BaseFederate's entity id convention
        assert var == "velocity"

    def test_parse_target_cross_federation(self):
        from core.override_registry import parse_target
        federation, federate, entity, var = parse_target("plant.spring_federate.0/velocity", "ignored")
        assert federation == "plant"
        assert federate == "spring_federate"
        assert entity == "spring_federate.0"
        assert var == "velocity"
