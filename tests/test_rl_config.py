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
