"""
remote_executor.py

Wraps plain OpenSSH client (subprocess.Popen, ControlMaster multiplexing) for spawning
CosimGym federates on remote machines. One RemoteExecutor per machine alias declared
under a scenario's `deployment.machines` block.

No ScenarioManager import here on purpose — this module is standalone and testable
without a running scenario. All ssh/rsync command construction is factored into pure
`_*_cmd`/`_build_remote_command` methods (return argv lists / strings, no subprocess
calls) so command shape and quoting can be unit-tested without a live ssh connection.
"""

import logging
import os
import shlex
import subprocess
from typing import List, Optional, Tuple

from utils.config_dataclasses import MachineConfig
from utils.ports import redis_port

# rsync excludes for the code sync step: build artifacts and run-generated
# directories never need to travel to the remote machine.
_DEPLOY_EXCLUDES = ['__pycache__', '.git', 'results', 'logs', 'graphify-out', '*.pyc']


class RemoteExecutor:
    """SSH/rsync handle for one remote machine (one `deployment.machines` alias)."""

    def __init__(
        self,
        machine_alias: str,
        machine_conf: MachineConfig,
        manager_address: str,
        logger: logging.Logger,
        control_dir: str,
    ):
        self.alias = machine_alias
        self.machine_conf = machine_conf
        self.manager_address = manager_address
        self.logger = logger
        self.control_dir = control_dir
        self._master_opened = False

    # ------------------------------------------------------------------
    # Target / control-path helpers
    # ------------------------------------------------------------------

    @property
    def _target(self) -> str:
        """`user@host` (or bare `host`, letting ssh apply the local OS user)."""
        if self.machine_conf.user:
            return f'{self.machine_conf.user}@{self.machine_conf.host}'
        return self.machine_conf.host

    @property
    def _control_path(self) -> str:
        """One multiplexed control socket per alias, scoped to the caller's control_dir."""
        return os.path.join(self.control_dir, f'cm-{self.alias}')

    def _python_cmd(self) -> List[str]:
        """Interpreter invocation on the remote machine: explicit path wins, else conda run."""
        if self.machine_conf.python:
            return [self.machine_conf.python]
        return ['conda', 'run', '--no-capture-output', '-n', self.machine_conf.conda_env, 'python']

    # ------------------------------------------------------------------
    # Pure command builders (no subprocess calls — unit-testable)
    # ------------------------------------------------------------------

    def _master_cmd(self) -> List[str]:
        # Establish the ControlMaster by running a trivial `true` over it. NOT `-nNf`:
        # a backgrounded (`-f`) master keeps the inherited stdout/stderr pipes open, so
        # subprocess.run(capture_output=True) would block on those pipes until timeout
        # even after the connection succeeds. Running `true` instead lets the client exit
        # immediately (returns rc + captured stderr on auth failure) while
        # ControlPersist=60 keeps the master socket alive for the reused connections.
        return [
            'ssh',
            '-o', 'ControlMaster=auto',
            '-o', f'ControlPath={self._control_path}',
            '-o', 'ControlPersist=60',
            '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=10',
            '-p', str(self.machine_conf.ssh_port),
            self._target,
            'true',
        ]

    def _build_remote_command(self, args_list: List[str], remote_log_file: Optional[str] = None) -> str:
        """Build the single remote shell command string: cd workdir && <python invocation>, quoted.

        Every argv token is individually `shlex.quote`d before joining — no unvalidated
        YAML value (host/workdir/args) is ever interpolated into the shell string unquoted.
        """
        tail = ' '.join(shlex.quote(p) for p in (self._python_cmd() + list(args_list)))
        cmd = f'cd {shlex.quote(self.machine_conf.workdir)} && {tail}'
        if remote_log_file:
            cmd += f' >> {shlex.quote(remote_log_file)} 2>&1'
        return cmd

    def _run_cmd(self, remote_cmd: str, tty: bool = False) -> List[str]:
        cmd = ['ssh', '-S', self._control_path, '-o', 'BatchMode=yes', '-p', str(self.machine_conf.ssh_port)]
        if tty:
            cmd.append('-tt')
        cmd += [self._target, remote_cmd]
        return cmd

    def _rsync_deploy_cmd(self, project_root: str) -> List[str]:
        src_dir = os.path.join(project_root, 'src')
        ssh_transport = f'ssh -S {self._control_path} -p {self.machine_conf.ssh_port} -o BatchMode=yes'
        cmd = ['rsync', '-az', '--delete']
        for pattern in _DEPLOY_EXCLUDES:
            cmd += ['--exclude', pattern]
        cmd += ['-e', ssh_transport, src_dir, f'{self._target}:{self.machine_conf.workdir}/']
        return cmd

    def _rsync_collect_cmd(self, remote_path: str, local_path: str) -> List[str]:
        ssh_transport = f'ssh -S {self._control_path} -p {self.machine_conf.ssh_port} -o BatchMode=yes'
        return [
            'rsync', '-az',
            '-e', ssh_transport,
            f'{self._target}:{remote_path.rstrip("/")}/',
            local_path.rstrip('/') + '/',
        ]

    def _close_cmd(self) -> List[str]:
        return ['ssh', '-S', self._control_path, '-O', 'exit', '-p', str(self.machine_conf.ssh_port), self._target]

    # ------------------------------------------------------------------
    # Execution methods
    # ------------------------------------------------------------------

    def open_master(self, timeout: float = 15) -> None:
        """Open the ControlMaster connection. All later ssh/rsync calls reuse this socket."""
        os.makedirs(self.control_dir, exist_ok=True)
        cmd = self._master_cmd()
        self.logger.info(f"[{self.alias}] opening ssh control master: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"[{self.alias}] ssh control master timed out connecting to {self._target}") from e
        if result.returncode != 0:
            raise RuntimeError(
                f"[{self.alias}] failed to open ssh control master to {self._target}: {result.stderr.strip()}"
            )
        self._master_opened = True

    def run(self, cmd: List[str], timeout: Optional[float] = None) -> Tuple[int, str, str]:
        """Run a one-shot remote command (relative to no particular cwd), return (rc, stdout, stderr)."""
        remote_cmd = ' '.join(shlex.quote(p) for p in cmd)
        ssh_cmd = self._run_cmd(remote_cmd)
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr

    def verify(self) -> None:
        """Preflight checks: control connection alive, workdir writable, env sane, Redis reachable.

        Raises RuntimeError with the machine alias, failing check, and a remediation hint.
        Never falls back to local silently — the caller must abort the whole scenario.
        """
        rc, _, err = self.run(['true'], timeout=10)
        if rc != 0:
            raise RuntimeError(f"[{self.alias}] ssh control connection check failed: {err.strip()}")

        rc, _, err = self.run(['mkdir', '-p', self.machine_conf.workdir], timeout=10)
        if rc != 0:
            raise RuntimeError(f"[{self.alias}] workdir '{self.machine_conf.workdir}' not writable: {err.strip()}")

        env_check = self._python_cmd() + ['-c', 'import helics, redis, pydantic']
        rc, _, err = self.run(env_check, timeout=30)
        if rc != 0:
            raise RuntimeError(
                f"[{self.alias}] remote env sane-check failed (import helics/redis/pydantic) "
                f"in conda_env='{self.machine_conf.conda_env}': {err.strip()}. "
                "Verify the conda env exists on the remote machine and has cosim_gym deps installed."
            )

        rport = redis_port()
        redis_check_code = f"import socket; socket.create_connection(('{self.manager_address}', {rport}), 5)"
        redis_check = self._python_cmd() + ['-c', redis_check_code]
        rc, _, err = self.run(redis_check, timeout=15)
        if rc != 0:
            raise RuntimeError(
                f"[{self.alias}] cannot reach Redis at {self.manager_address}:{rport} from remote machine: "
                f"{err.strip()}. Check deployment.manager_address and firewall rules (port {rport})."
            )

    def deploy(self, project_root: str) -> None:
        """rsync `src/` (delta transfer) into `<workdir>/src`, create remote logs/ and results/ dirs."""
        cmd = self._rsync_deploy_cmd(project_root)
        self.logger.info(f"[{self.alias}] deploying code: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"[{self.alias}] rsync deploy failed: {result.stderr.strip()}")

        remote_dirs = [
            os.path.join(self.machine_conf.workdir, 'logs'),
            os.path.join(self.machine_conf.workdir, 'results'),
        ]
        rc, _, err = self.run(['mkdir', '-p'] + remote_dirs, timeout=10)
        if rc != 0:
            raise RuntimeError(f"[{self.alias}] failed to create remote logs/results dirs: {err.strip()}")

    def spawn_many(self, spawn_key: str, redis_url: str, remote_log_file: str) -> subprocess.Popen:
        """Start this machine's federate supervisor, returning the local ssh child as its handle.

        ONE ssh session per machine, not per federate. `remote_spawner.py` reads the machine's
        federate list from Redis (`spawn_key`) and supervises them, so this command line is a
        fixed size no matter how many federates the machine hosts, and the manager holds one
        ssh client per *machine*. Opening a session per federate instead made large runs fail
        nondeterministically once federates-per-machine passed sshd's `MaxSessions` — see
        `remote_spawner`'s module docstring for the full mechanism.

        `-tt` allocates a pty so a SIGHUP (manager kills this Popen / process group) propagates
        to the remote supervisor — same cleanup-for-free property as local process groups. The
        supervisor's children share its process group, so the hangup reaches the federates too.
        Supervisor stdout/stderr are appended into `remote_log_file`.

        stdin MUST NOT be inherited. `-tt` forces pty allocation, and when ssh's stdin is the
        manager's terminal that also drags the *local* terminal into raw mode — where the tty
        driver stops turning Ctrl+C into SIGINT and forwards a raw 0x03 to the remote instead.
        The manager then never sees the interrupt and the run cannot be stopped by hand.
        Pointing stdin at /dev/null makes ssh's tcgetattr fail, so it leaves the terminal alone
        while `-tt` still forces the remote pty. Federates never read stdin anyway.
        """
        args_list = [
            'src/core/remote_spawner.py',
            '--redis-url', redis_url,
            '--spawn-key', spawn_key,
            '--machine', self.alias,
        ]
        remote_cmd = self._build_remote_command(args_list, remote_log_file)
        ssh_cmd = self._run_cmd(remote_cmd, tty=True)
        self.logger.info(f"[{self.alias}] spawn_many: {remote_cmd}")
        process = subprocess.Popen(
            ssh_cmd,
            preexec_fn=os.setsid,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return process

    def collect(self, remote_path: str, local_path: str) -> None:
        """rsync a remote directory back to a local one (results/ or logs/ after run end)."""
        os.makedirs(local_path, exist_ok=True)
        cmd = self._rsync_collect_cmd(remote_path, local_path)
        self.logger.info(f"[{self.alias}] collecting: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(
                f"[{self.alias}] rsync collect failed ({remote_path} -> {local_path}): {result.stderr.strip()}"
            )

    def close(self) -> None:
        """Close the ControlMaster socket. Never raises — cleanup must not block scenario teardown."""
        if not self._master_opened:
            return
        try:
            subprocess.run(self._close_cmd(), capture_output=True, timeout=10)
        except Exception as e:
            self.logger.warning(f"[{self.alias}] failed to close ssh control master: {e}")
        self._master_opened = False
