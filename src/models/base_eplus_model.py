"""
base_eplus_model.py

Generic CosimGym model that co-simulates ANY EnergyPlus IDF through the EnergyPlus Python
API (``pyenergyplus``). One generic class and one catalog entry (like ``BaseCSVReader``: the resource decides the interface).

One generic catalog entry (``eplus_model``); the building is chosen and its interface declared in the
scenario, in the model's ``user_defined`` (like ``csv_path`` for the CSV reader). The names under
``eplus_io`` are the scenario's ``subscribes`` / ``publishes`` keys; ``eplus_io`` entries that no
connection uses are dropped, a connection without an entry is an error.

  eplus:
    idf: path                 # repo-relative, absolute or ~; the file is never modified
    epw: path                 # weather file (repo-relative / absolute / ~)
    version: "24.1"           # optional: force an install (default: from the IDF, see eplus_registry)
    eplus_dir: path           # optional: force an install directory
    skip_sizing: false        # opt-in: skip the sizing runs when nothing is autosized (faster start)
    burn_in_days: 0           # after a start/reset mid-run: free-run this many whole days before it (thermal history)
    startup_timeout: 1800     # s until the first zone timestep is reached
    step_timeout: 600         # s per tick
  eplus_io:
    inputs:
      Setpoint: {actuator: {component_type: "Zone Temperature Control", control_type: "Heating Setpoint", key: "ZONE1"}, default: 20}
      Sched:    {schedule: "MY_SCHEDULE"}    # shorthand: the actuator of that IDF Schedule:Constant/Compact/Year/File
      Level:    {global: "MY_EMS_GLOBAL"}
    outputs:
      TZone:    {variable: {name: "Zone Mean Air Temperature", key: "ZONE1"}, agg: last}   # agg: last|mean|sum|min|max
      Heat:     {meter: "Heating:EnergyTransfer"}
      Sp:       {actuator: {...}}   # read an actuator back;  {global: NAME}: read an EMS global
    from_external_interface: false   # also take the I/O an FMU-export IDF declares (FMU variable names)

``key`` may hold ``*`` / ``?`` (actuators and variables): it expands at start-up to every key the IDF offers
(first use runs EnergyPlus once to list them, cached), each as ``<name>_<key>`` with non-alphanumerics -> ``_``;
the connections pick which of those are used. ``init_state`` / ``default`` is what an input holds from the first tick;
during the skipped (burn-in) zone steps of a restart only the spec ``default`` applies. ``max_sim_time`` / ``sim_start_date`` default to the IDF RunPeriod
(a scenario may give a shorter window, never a longer one).

The IDF is prepared on a copy per instance (see ``eplus_idf``): upgraded to the install's
version if older, ``ExternalInterface*`` stripped, weather run forced, run period rewritten
for the model's position. EnergyPlus cannot rewind: a reset restarts the worker.

``real_period`` must be the IDF zone timestep (3600 / Timestep) or a whole multiple of it.

Author: Pietro Rando Mazzarino
Organization: EC-Lab Politecnico di Torino
"""
import atexit
import copy
import fcntl
import functools
import json
import math
import os
import select
import signal
import socket
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from .base_model import BaseModel
from .eplus_idf import Idf, IdfError, content_hash, expand_io, has_pattern, normalize_io, weekday_after
from .eplus_registry import cache_root, resolve, upgrade_idf

_DAY_S = 86400.0
_WORKER = Path(__file__).with_name('eplus_worker.py')
_REPO = Path(__file__).resolve().parents[2]


class EnergyPlusError(RuntimeError):
    pass


class _Worker:
    """One ``eplus_worker.py`` process and the socket to it."""

    def __init__(self, cfg: dict, workdir: Path, label: str):
        self.label, self.workdir = label, workdir
        self.sock, child = socket.socketpair()
        cfg = dict(cfg, fd=child.fileno(), parent_pid=os.getpid())
        cfg_path = workdir / 'worker.json'
        cfg_path.write_text(json.dumps(cfg))
        self.log_path = workdir / 'worker.log'
        with open(self.log_path, 'wb') as log:
            self.proc = subprocess.Popen([sys.executable, '-I', str(_WORKER), str(cfg_path)],
                                         pass_fds=[child.fileno()], stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=workdir)   # own process group
        child.close()
        self._buf = b''

    def send(self, obj) -> None:
        try:
            self.sock.sendall((json.dumps(obj) + '\n').encode())
        except OSError as exc:
            raise EnergyPlusError(f'{self.label}: worker is gone ({exc}).{self._tail()}')

    def recv(self, timeout: float, what: str) -> dict:
        end = time.monotonic() + timeout
        while b'\n' not in self._buf:
            left = end - time.monotonic()
            if left <= 0:
                self.kill()
                raise EnergyPlusError(f'{self.label}: no answer from EnergyPlus within {timeout:.0f}s while {what}.{self._tail()}')
            if select.select([self.sock], [], [], min(left, 1.0))[0]:
                chunk = self.sock.recv(65536)
                if not chunk:
                    raise EnergyPlusError(f'{self.label}: EnergyPlus worker died while {what} '
                                          f'(exit code {self.proc.poll()}).{self._tail()}')
                self._buf += chunk
            elif self.proc.poll() is not None and not self._buf:
                raise EnergyPlusError(f'{self.label}: EnergyPlus worker exited (code {self.proc.returncode}) while {what}.{self._tail()}')
        line, self._buf = self._buf.split(b'\n', 1)
        msg = json.loads(line)
        if msg['t'] == 'error':
            self.kill()
            raise EnergyPlusError(f'{self.label}: {msg["msg"]}')
        if msg['t'] == 'done':
            err = '\n'.join(msg.get('err') or [])
            raise EnergyPlusError(f'{self.label}: EnergyPlus ended (rc={msg["rc"]}) while {what}'
                                  + (f':\n{err}' if err else ' (end of its RunPeriod reached)'))
        return msg

    def _tail(self) -> str:
        try:
            lines = self.log_path.read_text(errors='replace').splitlines()[-8:]
            return '\nworker.log tail:\n' + '\n'.join(lines)
        except OSError:
            return ''

    def kill(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send({'t': 'stop'})
                end = time.monotonic() + 10
                while self.proc.poll() is None and time.monotonic() < end:
                    time.sleep(0.05)
            except EnergyPlusError:
                pass
        self.kill()
        self.sock.close()


def discover_listing(idf: Idf, install, epw, workdir: Path, label: str = 'discover', timeout: float = 3600) -> str:
    """Everything the EnergyPlus API exposes for this IDF (``list_available_api_data_csv``). Runs EnergyPlus up to
    its first timestep (sizing + warmup), so the result is cached per prepared-IDF text + install version."""
    key = content_hash(idf.text(), install.label)
    cache = cache_root() / 'listings'
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f'{key}.csv'
    with open(cache / f'{key}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists():
            wd = Path(workdir) / 'discover'
            (wd / 'run').mkdir(parents=True, exist_ok=True)
            run = Idf(copy.deepcopy(idf.objects), idf.tail)
            run.force_weather_run()
            run.write(wd / 'model.idf')
            cfg = {'install': str(install.path), 'idf': str(wd / 'model.idf'), 'epw': str(epw) if epw else None,
                   'outdir': str(wd / 'run'), 'inputs': {}, 'outputs': {}, 'group': 1, 'skip_ticks': 0, 'discover': True}
            w = _Worker(cfg, wd, label)
            try:
                text = Path(w.recv(timeout, 'discovering the IDF interface')['path']).read_text()
            finally:
                w.close()
            out.write_text(text)
    return out.read_text()


def _logged(fn):
    """Errors inside EnergyPlus callbacks / the worker surface here: log them (the core may only print the traceback)."""
    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        try:
            return fn(self, *a, **kw)
        except Exception as exc:
            self.logger.error(f'{self.name}: {type(exc).__name__}: {exc}')
            raise
    return wrapper


class EnergyPlusModel(BaseModel):
    def __init__(self, name, metadata, config, logger):
        self._worker = None
        self._inputs, self._outputs = {}, {}
        super().__init__(name, metadata, config, logger)

    # ------------------------------------------------------------------
    # BaseModel interface
    # ------------------------------------------------------------------
    @_logged
    def initialize(self) -> None:
        self._load_config()
        self._prepare_idf()
        self._init_state_from_io()
        self._adopt_horizon()
        self._start()
        atexit.register(self._stop)          # core never calls finalize(): do not leak the process
        self.logger.info(f'EnergyPlus model {self.name} ready: EnergyPlus {self._install.label}, '
                         f'zone step {self._zone_step_s:.0f}s x{self._group}, '
                         f'{len(self._inputs)} inputs / {len(self._outputs)} outputs')

    def _init_state_from_io(self) -> None:
        """The catalog declares no I/O: the state holds the names of ``eplus_io`` (as the CSV reader does for its
        columns). Initial value: scenario ``init_state``, else the input's ``default``; none -> not sent until received."""
        init = self.config.init_state or {}
        for name, spec in self._inputs.items():
            v = init.get(name, spec.get('default'))
            v = None if v is None else float(v)
            self.init_state.inputs[name] = self.state.inputs[name] = v
        for name in self._outputs:
            v = float(init.get(name, 0.0))
            self.init_state.outputs[name] = self.state.outputs[name] = v

    @_logged
    def step(self) -> None:
        ins = {}
        for name, spec in self._inputs.items():
            v = self.state.inputs.get(name, spec.get('default'))
            if v is not None:
                ins[name] = v
        self._worker.send({'t': 'step', 'in': ins})
        msg = self._worker.recv(self._step_timeout, f'stepping tick {self.state.rel_ts}')
        self._check_clock(msg['clk'])
        self._ticks_done += 1
        self.state.outputs.update(msg['out'])

    @_logged
    def reset(self, mode: str = 'full') -> None:
        self.logger.debug(f'Reset {mode!r}: restarting EnergyPlus at the model position')
        self._stop()
        self._start()

    def finalize(self) -> None:
        self._stop()

    # ------------------------------------------------------------------
    # Configuration and IDF preparation (once per instance)
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        catalog = (self.metadata.user_defined or {}) if self.metadata else {}
        scenario = self.config.user_defined or {}
        cfg = {**(catalog.get('eplus') or {}), **(scenario.get('eplus') or {})}
        self._cfg = cfg
        self._io_spec = scenario.get('eplus_io', catalog.get('eplus_io')) or {}
        if not cfg.get('idf'):
            raise ValueError(f'{self.name}: user_defined.eplus.idf is required')
        self._idf_src = self._find_file(cfg['idf'], 'IDF')
        self._epw = self._find_file(cfg['epw'], 'weather file') if cfg.get('epw') else None
        self._burn_in_days = int(cfg.get('burn_in_days', 0))
        self._startup_timeout = float(cfg.get('startup_timeout', 1800))
        self._step_timeout = float(cfg.get('step_timeout', 600))
        self._workdir = self._resolve_workdir()

    def _find_file(self, p, what) -> Path:
        path = Path(str(p)).expanduser()
        for cand in ([path] if path.is_absolute() else [Path.cwd() / path, _REPO / path]):
            if cand.exists():
                return cand.resolve()
        raise FileNotFoundError(f'{self.name}: {what} {p!r} not found (tried absolute, cwd and repo root)')

    def _resolve_workdir(self) -> Path:
        """Next to the scenario logs (same rule as the FMU model); one dir per instance, reused by restarts."""
        log_file = os.environ.get('COSIM_FEDERATE_LOG_FILE')
        base = Path(log_file).resolve().parent.parent if log_file else Path('logs').resolve()
        d = base / 'eplus_output' / self.name.replace('/', '_')
        (d / 'run').mkdir(parents=True, exist_ok=True)
        return d

    def _prepare_idf(self) -> None:
        raw = Idf.parse(self._idf_src)
        self._install = resolve(raw.version(), self._cfg.get('version'), self._cfg.get('eplus_dir'))
        path, log = upgrade_idf(self._idf_src, self._install)
        for line in log:
            self.logger.info(line)
        idf = Idf.parse(path)
        ext = idf.convert_external_interface()
        spec = dict(self._io_spec)
        if has_pattern(spec):
            self.logger.info('eplus_io has key patterns: listing the IDF interface (one EnergyPlus run, cached)')
            listing = discover_listing(idf, self._install, self._epw, self._workdir, self.name)
            spec = expand_io(spec, listing, idf.zones())
        ins, outs = normalize_io(spec, ext if spec.get('from_external_interface') else None, idf)
        self._inputs, self._outputs = self._bind_io(ins, outs)
        for o in self._outputs.values():
            if o['kind'] == 'meter':
                idf.ensure_meter(o['name'])
        if idf.force_weather_run(bool(self._cfg.get('skip_sizing'))) is False and self._cfg.get('skip_sizing'):
            self.logger.warning('skip_sizing ignored: the IDF has autosized fields and needs the sizing runs')
        self._base = idf
        self._zone_step_s = idf.zone_step_s()
        ratio = self.real_period / self._zone_step_s
        if ratio < 1 or abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f'{self.name}: real_period={self.real_period}s must be the IDF zone timestep '
                             f'({self._zone_step_s:.0f}s = 3600/Timestep {idf.timestep_per_hour()}) or a whole multiple of it')
        self._group = int(round(ratio))
        self._rp = idf.run_period()
        if self._rp is None:
            raise IdfError(f'{self.name}: the IDF has no RunPeriod')

    def _bind_io(self, ins: dict, outs: dict):
        """Match ``eplus_io`` with this federate's connections: unused entries are dropped (EnergyPlus then
        resolves and reads fewer handles), a connection without an entry is an error. An input with a ``default``
        is kept even when unconnected: the default is an explicit instruction, dropping it would leave the IDF's own value."""
        want_in, want_out = list(self.config.inputs or []), list(self.config.outputs or [])
        if not (want_in or want_out):
            return ins, outs
        for what, want, have in (('subscribes', want_in, ins), ('publishes', want_out, outs)):
            miss = [n for n in want if n not in have]
            if miss:
                raise ValueError(f'{self.name}: connections.{what} {miss} are not declared in user_defined.eplus_io '
                                 f'(declared: {sorted(have)}). Inputs need an EnergyPlus actuator / schedule / EMS global '
                                 f'that exists in the IDF: run eplus_discover.py to list them.')
        return ({n: s for n, s in ins.items() if n in want_in or s.get('default') is not None},
                {n: outs[n] for n in want_out})

    def _adopt_horizon(self) -> None:
        """The model's window is its RunPeriod unless the catalog / scenario declare a (shorter) one."""
        rp = self._rp
        if not self.max_sim_time:
            self.max_sim_time = rp.days * _DAY_S
        if not self.sim_start_date:
            self.sim_start_date = rp.begin.strftime('%m-%d')
        self._origin_s = self._origin_seconds()
        self._epoch = self._horizon_epoch(self.state.rel_ts)
        m, d = (int(x) for x in str(self.sim_start_date)[-5:].split('-'))
        origin = date(rp.begin.year, m, d)
        if origin < rp.begin:
            origin = date(rp.begin.year + 1, m, d)
        self._origin_date = origin
        self._window_end = origin + timedelta(days=math.ceil(float(self.max_sim_time) / _DAY_S) - 1)
        if self._window_end > rp.end:
            raise ValueError(f'{self.name}: horizon {self.max_sim_time}s from {origin} ends {self._window_end}, '
                             f'after the IDF RunPeriod end {rp.end}')

    # ------------------------------------------------------------------
    # Worker lifecycle (one start per episode / horizon restart)
    # ------------------------------------------------------------------
    def _start(self) -> None:
        pos = self.horizon_pos_s()
        day0, off = divmod(pos, _DAY_S)
        skip_f = off / self._zone_step_s
        if abs(skip_f - round(skip_f)) > 1e-6:
            raise ValueError(f'{self.name}: model position {pos}s is not a multiple of the zone timestep {self._zone_step_s}s')
        # thermal history: start burn_in_days whole days earlier (never before the RunPeriod), inputs at defaults
        back = min(self._burn_in_days, int(day0) + (self._origin_date - self._rp.begin).days)
        skip = int(round(skip_f)) + back * int(round(_DAY_S / self._zone_step_s))
        start = self._origin_date + timedelta(days=int(day0) - back)
        if self._span_end_rel is None:        # span unknown: run to the end of the window (stopped early anyway)
            end = self._window_end
        else:
            need_s = skip * self._zone_step_s + max(1, self._span_end_rel - self.state.rel_ts) * self.real_period
            end = min(start + timedelta(days=max(1, math.ceil(need_s / _DAY_S)) - 1), self._window_end)
        if start > end:
            raise ValueError(f'{self.name}: start {start} is past the model window end {self._window_end}')
        shift = (start - self._rp.begin).days
        dow = weekday_after(self._rp.start_dow, shift)
        if shift and self._rp.start_dow is None:
            self.logger.warning('RunPeriod has no explicit start weekday: after a restart at another date the '
                                'weekday of day 1 is whatever EnergyPlus infers (weekday schedules may shift)')
        idf = Idf(copy.deepcopy(self._base.objects), self._base.tail)
        idf.set_run_period(start, end, dow)
        idf_path = self._workdir / 'model.idf'
        idf.write(idf_path)

        self._start_date, self._skip, self._ticks_done = start, skip, 0
        cfg = {'install': str(self._install.path), 'idf': str(idf_path), 'epw': str(self._epw) if self._epw else None,
               'outdir': str(self._workdir / 'run'), 'inputs': self._inputs, 'outputs': self._outputs,
               'group': self._group, 'skip_ticks': skip}
        t0 = time.time()
        self._worker = _Worker(cfg, self._workdir, self.name)
        self._worker.recv(self._startup_timeout, 'starting (sizing / warmup)')
        self.logger.info(f'EnergyPlus started at {start} (+{skip} burn-in zone steps), run to {end}, '
                         f'startup {time.time() - t0:.1f}s')

    def _stop(self) -> None:
        w, self._worker = getattr(self, '_worker', None), None
        if w is not None:
            w.close()

    def _check_clock(self, clk) -> None:
        """EnergyPlus's own (day of year, hour, zone step) must match our tick counter, every tick."""
        per_hour = round(3600.0 / self._zone_step_s)
        idx = self._skip + (self._ticks_done + 1) * self._group - 1      # zone step just completed
        day = self._start_date + timedelta(days=idx // (per_hour * 24))
        want = (day.timetuple().tm_yday, (idx // per_hour) % 24, idx % per_hour + 1)
        if tuple(clk) != want:
            raise EnergyPlusError(f'{self.name}: clock desync at tick {self._ticks_done}: EnergyPlus (day, hour, step) '
                                  f'{tuple(clk)} != expected {want}')
