"""
config_reader.py

Reads YAML scenario configuration files and validates them into Pydantic models.
"""

import os
import yaml
from pydantic import ValidationError
from .config_dataclasses import ScenarioConfig


def read_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def read_scenario_config(file_path: str) -> ScenarioConfig:
    """
    Read and validate a scenario YAML file into a ScenarioConfig model.

    Accepts a bare name (resolved relative to src/scenarios/), a path starting
    with scenarios/, or an absolute path. The .yaml extension is added when missing.
    """
    if '/' not in file_path and '\\' not in file_path:
        scenarios_dir = os.path.join(os.path.dirname(__file__), '..', 'scenarios')
        file_path = os.path.join(scenarios_dir, file_path + '.yaml')
    elif file_path.startswith('scenarios/') or file_path.startswith('scenarios\\'):
        scenarios_dir = os.path.join(os.path.dirname(__file__), '..')
        file_path = os.path.join(scenarios_dir, file_path + '.yaml')

    try:
        raw = read_yaml(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    try:
        return ScenarioConfig.model_validate(raw)
    except ValidationError as e:
        raise ValueError(f"Invalid scenario config '{file_path}':\n{e}")
