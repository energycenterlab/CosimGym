"""
test_ports.py — centralized port resolution (src/utils/ports.py).

Functions read os.environ live, so we drive them with monkeypatched env vars.
Covers: defaults, COSIM_* override, legacy alias, precedence, derived values.

Run: pytest tests/test_ports.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import ports


def _clear(monkeypatch, *names):
    for n in names:
        monkeypatch.delenv(n, raising=False)


class TestDefaults:

    def test_redis_default(self, monkeypatch):
        _clear(monkeypatch, 'COSIM_REDIS_PORT', 'REDIS_PORT')
        assert ports.redis_port() == 6379

    def test_mqtt_default(self, monkeypatch):
        _clear(monkeypatch, 'COSIM_MQTT_PORT', 'MQTT_PORT')
        assert ports.mqtt_port() == 11883

    def test_minio_defaults(self, monkeypatch):
        _clear(monkeypatch, 'COSIM_MINIO_PORT', 'MINIO_PORT',
               'COSIM_MINIO_CONSOLE_PORT', 'COSIM_MINIO_HOST', 'MINIO_HOST')
        assert ports.minio_port() == 9000
        assert ports.minio_console_port() == 9101
        assert ports.minio_endpoint() == 'http://localhost:9000'

    def test_helics_range_default(self, monkeypatch):
        _clear(monkeypatch, 'COSIM_HELICS_PORT_MIN', 'COSIM_HELICS_PORT_MAX')
        assert ports.helics_port_range() == (20000, 30000)


class TestOverrides:

    def test_cosim_var_override(self, monkeypatch):
        _clear(monkeypatch, 'REDIS_PORT')
        monkeypatch.setenv('COSIM_REDIS_PORT', '6380')
        assert ports.redis_port() == 6380

    def test_legacy_alias_still_honored(self, monkeypatch):
        _clear(monkeypatch, 'COSIM_REDIS_PORT')
        monkeypatch.setenv('REDIS_PORT', '6390')
        assert ports.redis_port() == 6390
        _clear(monkeypatch, 'COSIM_MQTT_PORT')
        monkeypatch.setenv('MQTT_PORT', '12000')
        assert ports.mqtt_port() == 12000

    def test_cosim_wins_over_legacy(self, monkeypatch):
        monkeypatch.setenv('COSIM_REDIS_PORT', '6400')
        monkeypatch.setenv('REDIS_PORT', '6500')
        assert ports.redis_port() == 6400

    def test_minio_endpoint_reflects_overrides(self, monkeypatch):
        _clear(monkeypatch, 'MINIO_HOST')
        monkeypatch.setenv('COSIM_MINIO_HOST', 'redisbox')
        monkeypatch.setenv('COSIM_MINIO_PORT', '9500')
        assert ports.minio_endpoint() == 'http://redisbox:9500'

    def test_helics_range_override(self, monkeypatch):
        monkeypatch.setenv('COSIM_HELICS_PORT_MIN', '25000')
        monkeypatch.setenv('COSIM_HELICS_PORT_MAX', '26000')
        assert ports.helics_port_range() == (25000, 26000)


class TestEnvFileLoader:

    def test_env_file_loaded_without_override(self, tmp_path, monkeypatch):
        """_load_env_file uses setdefault → an existing export wins over the file."""
        env_file = tmp_path / '.env'
        env_file.write_text('COSIM_REDIS_PORT=7000\n# comment\n\nCOSIM_MQTT_PORT="7100"\n')
        monkeypatch.setattr(ports, '_ENV_FILE', env_file)

        _clear(monkeypatch, 'COSIM_REDIS_PORT', 'COSIM_MQTT_PORT', 'REDIS_PORT', 'MQTT_PORT')
        # export wins over file:
        monkeypatch.setenv('COSIM_MQTT_PORT', '9999')

        ports._load_env_file()
        assert os.environ['COSIM_REDIS_PORT'] == '7000'   # from file (quotes stripped works too)
        assert os.environ['COSIM_MQTT_PORT'] == '9999'    # export preserved, not clobbered
        assert ports.redis_port() == 7000
