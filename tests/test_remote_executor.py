"""
test_remote_executor.py — command construction + quoting for RemoteExecutor.

No live ssh required for the command-construction tests: they call the pure
`_*_cmd`/`_build_remote_command` builders directly and assert on argv lists / strings.

An optional live integration test (ssh to 127.0.0.1) is gated behind the
COSIMGYM_TEST_SSH_LOCALHOST env var and skipped by default.

Run: pytest tests/test_remote_executor.py -v
"""

import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.config_dataclasses import MachineConfig
from core.remote_executor import RemoteExecutor


def _executor(**machine_overrides) -> RemoteExecutor:
    defaults = dict(host='10.0.0.5', workdir='/home/rando/cosimgym_rt')
    defaults.update(machine_overrides)
    machine_conf = MachineConfig.model_validate(defaults)
    return RemoteExecutor(
        machine_alias='gpu_box',
        machine_conf=machine_conf,
        manager_address='192.168.1.10',
        logger=logging.getLogger('test_remote_executor'),
        control_dir='/tmp/cosimgym-test-control',
    )


class TestTargetAndControlPath:

    def test_target_with_user(self):
        ex = _executor(user='rando')
        assert ex._target == 'rando@10.0.0.5'

    def test_target_without_user(self):
        ex = _executor()
        assert ex._target == '10.0.0.5'

    def test_control_path_scoped_to_alias(self):
        ex = _executor()
        assert ex._control_path == '/tmp/cosimgym-test-control/cm-gpu_box'


class TestPythonCmd:

    def test_conda_env_default(self):
        ex = _executor()
        assert ex._python_cmd() == ['conda', 'run', '--no-capture-output', '-n', 'cosim_gym', 'python']

    def test_conda_env_explicit(self):
        ex = _executor(conda_env='my_env')
        assert ex._python_cmd() == ['conda', 'run', '--no-capture-output', '-n', 'my_env', 'python']

    def test_explicit_python_overrides_conda(self):
        ex = _executor(python='/usr/bin/python3.12', conda_env='ignored_env')
        assert ex._python_cmd() == ['/usr/bin/python3.12']


class TestMasterCmd:

    def test_master_cmd_flags(self):
        ex = _executor(user='rando', ssh_port=2222)
        cmd = ex._master_cmd()
        assert cmd[0] == 'ssh'
        assert '-o' in cmd and 'ControlMaster=auto' in cmd
        assert f'ControlPath={ex._control_path}' in cmd
        assert 'ControlPersist=60' in cmd
        assert 'BatchMode=yes' in cmd
        assert 'ConnectTimeout=10' in cmd
        assert cmd[cmd.index('-p') + 1] == '2222'
        # Master runs a trivial `true` over the target (NOT `-nNf` background fork,
        # which deadlocks subprocess.run(capture_output=True) on the inherited pipes).
        assert cmd[-2] == 'rando@10.0.0.5'
        assert cmd[-1] == 'true'
        assert '-nNf' not in cmd


class TestBuildRemoteCommand:

    def test_basic_shape(self):
        ex = _executor()
        cmd = ex._build_remote_command(['src/core/federate_launcher.py', '--name', 'fed1'])
        assert cmd == "cd /home/rando/cosimgym_rt && conda run --no-capture-output -n cosim_gym python src/core/federate_launcher.py --name fed1"

    def test_with_remote_log_redirect(self):
        ex = _executor()
        cmd = ex._build_remote_command(['x.py'], remote_log_file='/home/rando/cosimgym_rt/logs/f.log')
        assert cmd.endswith(">> /home/rando/cosimgym_rt/logs/f.log 2>&1")

    def test_explicit_python_path_used(self):
        ex = _executor(python='/opt/conda/envs/x/bin/python')
        cmd = ex._build_remote_command(['x.py'])
        assert '/opt/conda/envs/x/bin/python' in cmd
        assert 'conda run' not in cmd

    def test_quoting_hostile_workdir(self):
        ex = _executor(workdir='/home/rando/dir with space; rm -rf /')
        cmd = ex._build_remote_command(['x.py'])
        assert "'/home/rando/dir with space; rm -rf /'" in cmd
        # the injected `;` must stay INSIDE the quoted workdir token, not become a
        # shell command separator that would run after the legitimate cd.
        assert cmd.startswith("cd '/home/rando/dir with space; rm -rf /' && ")

    def test_quoting_hostile_args(self):
        ex = _executor()
        cmd = ex._build_remote_command(['x.py', '--name', 'foo; rm -rf /', '--val', '$(whoami)'])
        assert "'foo; rm -rf /'" in cmd
        assert "'$(whoami)'" in cmd

    def test_quoting_dollar_sign(self):
        ex = _executor()
        cmd = ex._build_remote_command(['x.py', '--key', '$HOME'])
        assert "'$HOME'" in cmd


class TestRunCmd:

    def test_run_cmd_no_tty_by_default(self):
        ex = _executor(ssh_port=2222)
        cmd = ex._run_cmd('true')
        assert '-tt' not in cmd
        assert cmd[0] == 'ssh'
        assert '-S' in cmd and ex._control_path in cmd
        assert cmd[-1] == 'true'
        assert cmd[-2] == ex._target

    def test_run_cmd_with_tty(self):
        ex = _executor()
        cmd = ex._run_cmd('true', tty=True)
        assert '-tt' in cmd


class TestRsyncCmds:

    def test_deploy_cmd_excludes_and_source_dest(self):
        ex = _executor(user='rando')
        cmd = ex._rsync_deploy_cmd('/local/project/root')
        assert cmd[0] == 'rsync'
        assert '--delete' in cmd
        for pattern in ['__pycache__', '.git', 'results', 'logs', 'graphify-out', '*.pyc']:
            assert pattern in cmd
        assert cmd[-2] == '/local/project/root/src'
        assert cmd[-1] == 'rando@10.0.0.5:/home/rando/cosimgym_rt/'

    def test_collect_cmd_source_dest(self):
        ex = _executor()
        cmd = ex._rsync_collect_cmd('/remote/results/scenario/sim1', '/local/results/scenario/sim1')
        assert cmd[-2] == '10.0.0.5:/remote/results/scenario/sim1/'
        assert cmd[-1] == '/local/results/scenario/sim1/'

    def test_close_cmd(self):
        ex = _executor()
        cmd = ex._close_cmd()
        assert cmd == ['ssh', '-S', ex._control_path, '-O', 'exit', '-p', '22', '10.0.0.5']


class TestSpawnUsesPtyAndDetachedGroup:
    """spawn_many() itself calls subprocess.Popen — verify the argv it builds without launching ssh."""

    def _spawn_many(self, monkeypatch):
        captured = {}

        class _FakeProcess:
            pid = 12345

        def _fake_popen(cmd, **kwargs):
            captured['cmd'] = cmd
            captured['kwargs'] = kwargs
            return _FakeProcess()

        monkeypatch.setattr('core.remote_executor.subprocess.Popen', _fake_popen)
        ex = _executor()
        proc = ex.spawn_many(
            'cosim:spawn:sim1:m1',
            redis_url='redis://10.0.0.1:6379/0',
            remote_log_file='/remote/logs/_spawner_m1.log',
        )
        return ex, proc, captured

    def test_spawn_many_builds_tty_command(self, monkeypatch):
        ex, proc, captured = self._spawn_many(monkeypatch)
        assert proc is not None
        cmd = captured['cmd']
        assert '-tt' in cmd
        assert cmd[-2] == ex._target
        remote_cmd = cmd[-1]
        assert remote_cmd.startswith('cd /home/rando/cosimgym_rt &&')
        assert remote_cmd.endswith('>> /remote/logs/_spawner_m1.log 2>&1')
        assert captured['kwargs']['preexec_fn'] is os.setsid

    def test_spawn_many_runs_spawner_with_redis_key_not_federate_argv(self, monkeypatch):
        """The command must stay a fixed size: the federate list travels via Redis, not argv."""
        _ex, _proc, captured = self._spawn_many(monkeypatch)
        remote_cmd = captured['cmd'][-1]
        assert 'src/core/remote_spawner.py' in remote_cmd
        assert 'cosim:spawn:sim1:m1' in remote_cmd
        assert 'federate_launcher.py' not in remote_cmd


@pytest.mark.skipif(
    not os.environ.get('COSIMGYM_TEST_SSH_LOCALHOST'),
    reason="set COSIMGYM_TEST_SSH_LOCALHOST=1 to run live ssh-to-localhost integration test",
)
class TestLiveLoopback:

    def test_open_master_verify_close_against_localhost(self, tmp_path):
        machine_conf = MachineConfig.model_validate({
            'host': '127.0.0.1',
            'workdir': str(tmp_path / 'remote_workdir'),
        })
        ex = RemoteExecutor(
            machine_alias='loopback',
            machine_conf=machine_conf,
            manager_address='127.0.0.1',
            logger=logging.getLogger('test_remote_executor_live'),
            control_dir=str(tmp_path / 'control'),
        )
        try:
            ex.open_master()
            rc, out, err = ex.run(['true'])
            assert rc == 0
        finally:
            ex.close()
