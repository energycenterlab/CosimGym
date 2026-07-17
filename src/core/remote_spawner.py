"""
remote_spawner.py

Per-machine federate supervisor for distributed (`deployment:`) runs. ScenarioManager
starts exactly one of these per remote machine, over a single ssh session, and it
launches that machine's federates as local children and supervises them.

Why one supervisor per machine rather than one ssh session per federate:

* sshd caps concurrent sessions on a multiplexed connection (`MaxSessions`, default 10).
  Past that the ssh client silently falls back to opening a fresh TCP connection per
  federate, and sshd's `MaxStartups` (default 10:30:100) drops those *probabilistically*.
  Federates then failed to start at random, ssh exited 255, and the whole federation
  aborted. The failure rate scaled with federates-per-machine, not with anything in the
  co-simulation itself.
* Every ssh client also cost the manager a local process for the entire run, making the
  manager's footprint O(federates) instead of O(machines).

The federate list is read from Redis -- already the channel every federate uses to fetch
its own config -- rather than passed as argv, so the ssh command line stays a fixed size
no matter how many federates the machine hosts.

Process-group note: children are deliberately NOT setsid'd. They stay in this process's
group, so the SIGHUP raised when the manager's ssh child dies (the `-tt` pty hangs up)
reaches them too. That is the same "cleanup for free" property the per-federate ssh
spawn had, and it is what keeps remote federates from outliving a killed manager.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Fix imports to work when invoked as `python src/core/remote_spawner.py` from workdir
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from utils.redis_client import RedisClient
from utils.ports import redis_port as default_redis_port

# How often the supervisor polls its children for exits.
POLL_INTERVAL_S = 0.5

# How long a federate gets to honour SIGTERM before it is SIGKILLed.
TERM_GRACE_S = 5


def parse_redis_url(url):
    """`redis://host:port/db` -> (host, port, db). Mirrors federate_launcher's parsing."""
    parts = url.replace('redis://', '').split('/')
    host_port = parts[0].split(':')
    host = host_port[0]
    port = int(host_port[1]) if len(host_port) > 1 else default_redis_port()
    db = int(parts[1]) if len(parts) > 1 else 0
    return host, port, db


def spawn_federates(specs):
    """Launch every federate in `specs`, returning {name: Popen}.

    Each spec is {'name', 'args', 'stdio'}: `args` is the federate_launcher argv tail,
    `stdio` the file its stdout/stderr are appended to (mirrors the manager's local
    stdio-capture file, which catches tracebacks that bypass the federate's own logger).
    """
    running = {}
    for spec in specs:
        stdio = open(spec['stdio'], 'a')
        proc = subprocess.Popen(
            [sys.executable] + spec['args'],
            stdin=subprocess.DEVNULL,
            stdout=stdio,
            stderr=subprocess.STDOUT,
        )
        running[spec['name']] = proc
        print(f"spawned federate '{spec['name']}' pid={proc.pid}", flush=True)
    return running


def kill_all(running):
    """SIGTERM every child, then SIGKILL whatever is left after TERM_GRACE_S.

    Signals are sent to all children *before* waiting on any of them, so teardown costs
    one grace period in total rather than one per federate.
    """
    for proc in running.values():
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
    deadline = time.time() + TERM_GRACE_S
    for proc in running.values():
        try:
            proc.wait(timeout=max(0, deadline - time.time()))
        except (subprocess.TimeoutExpired, OSError):
            pass
    for proc in running.values():
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass


def supervise(running):
    """Block until every federate exits, or until one fails.

    A HELICS federation is all-or-nothing: once one federate dies the survivors block
    forever inside HELICS waiting for a peer that will never arrive. So the first
    non-zero exit tears down this machine's federates and propagates that code out
    through ssh, letting the manager fail the run fast instead of hanging.
    """
    while running:
        for name, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[name]
            if rc != 0:
                print(f"federate '{name}' exited {rc} — tearing down this machine", flush=True)
                kill_all(running)
                return rc
            print(f"federate '{name}' completed", flush=True)
        if not running:
            break
        time.sleep(POLL_INTERVAL_S)
    return 0


def main():
    parser = argparse.ArgumentParser(description='Supervise this machine share of a distributed run')
    parser.add_argument('--redis-url', required=True, help='Redis connection URL (on the manager)')
    parser.add_argument('--spawn-key', required=True, help='Redis key holding this machine federate list')
    parser.add_argument('--machine', required=True, help='machine alias, for log messages')
    args = parser.parse_args()

    host, port, db = parse_redis_url(args.redis_url)
    client = RedisClient(host=host, port=port, db=db)
    payload = client.get_json(args.spawn_key)
    if not payload or not payload.get('federates'):
        print(f"[{args.machine}] no federate list at Redis key {args.spawn_key}", flush=True)
        return 1

    specs = payload['federates']
    print(f"[{args.machine}] starting {len(specs)} federate(s)", flush=True)
    running = spawn_federates(specs)

    # Tear the children down on a manager-initiated stop. SIGHUP arrives when the ssh
    # pty hangs up (manager killed / Ctrl+C), SIGTERM when cleanup terminates the ssh
    # child's process group. Children share this process group, so they usually get the
    # signal directly too — handling it here is what makes teardown *bounded* rather
    # than relying on each federate's own handler.
    def _stop(signum, _frame):
        print(f"[{args.machine}] signal {signum} — stopping {len(running)} federate(s)", flush=True)
        kill_all(running)
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _stop)

    rc = supervise(running)
    print(f"[{args.machine}] done, exit={rc}", flush=True)
    return rc


if __name__ == '__main__':
    sys.exit(main())
