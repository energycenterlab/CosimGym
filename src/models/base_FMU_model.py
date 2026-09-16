"""
base_FMU_model.py

Wrapper that integrates any FMU (FMI 2.0 and FMI 3.0 co-simulation, with partial
FMI 1.0 support) into the CosimGym model framework.

At initialization the model resolves the FMU binary from one of three sources
declared in the catalog entry's ``user_defined.fmu_source`` block:

  type: "local"  → path on the local filesystem
  type: "minio"  → download from MinIO / S3-compatible store into local cache
  type: "http"   → download via HTTP into local cache

Downloaded FMUs are cached in ``~/.cosimgym/fmu_cache/<model_name>/<version>/``
and re-used on subsequent runs (no re-download unless the file is missing).

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
"""

import contextlib
import logging
import os
import re
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from .base_model import BaseModel
from fmpy import read_model_description, extract, dump
from fmpy.fmi1 import FMU1Slave
from fmpy.fmi2 import FMU2Slave
from fmpy.fmi3 import FMU3Slave


# Map FMI variable type strings to fmpy get/set method names for dispatch
_FMI_TYPE_GETSET = {
    # FMI 1.0 / 2.0
    'Real':        ('getReal',    'setReal'),
    'Integer':     ('getInteger', 'setInteger'),
    'Boolean':     ('getBoolean', 'setBoolean'),
    'String':      ('getString',  'setString'),
    'Enumeration': ('getInteger', 'setInteger'),
    # FMI 3.0 (Boolean/String/Enumeration share the FMI 1.0/2.0 entries above)
    'Float32':     ('getFloat32', 'setFloat32'),
    'Float64':     ('getFloat64', 'setFloat64'),
    'Int8':        ('getInt8',    'setInt8'),
    'UInt8':       ('getUInt8',   'setUInt8'),
    'Int16':       ('getInt16',   'setInt16'),
    'UInt16':      ('getUInt16',  'setUInt16'),
    'Int32':       ('getInt32',   'setInt32'),
    'UInt32':      ('getUInt32',  'setUInt32'),
    'Int64':       ('getInt64',   'setInt64'),
    'UInt64':      ('getUInt64',  'setUInt64'),
    'Binary':      ('getBinary',  'setBinary'),
}

# Map FMI type strings to catalog schema type strings
_FMI_TO_CATALOG_TYPE = {
    # FMI 1.0 / 2.0
    'Real':        'float',
    'Integer':     'int',
    'Boolean':     'bool',
    'String':      'string',
    'Enumeration': 'int',
    # FMI 3.0
    'Float32':     'float',
    'Float64':     'float',
    'Int8':        'int',
    'UInt8':       'int',
    'Int16':       'int',
    'UInt16':      'int',
    'Int32':       'int',
    'UInt32':      'int',
    'Int64':       'int',
    'UInt64':      'int',
    'Binary':      'string',
}


class BaseFMUModel(BaseModel):

    def __init__(self, name, metadata, config, logger):
        self.fmu = None
        self.model_description = None
        self.unzipdir = None
        self.fmiVersion = None
        self._fmu_path = None

        # How many slaves have been instantiated so far (0 before the first
        # ``initialize``). Each restart gets its own output subdir so successive
        # instances do not overwrite each other's files.
        self._instance_count = 0

        # Input values seen at each tick of the current epoch, kept only when a
        # rolling reset can ask this model to move backwards: replaying the FMU
        # forward to an earlier start point has to feed it the values it actually
        # saw, otherwise it arrives there in a state it never really had.
        self._input_history = []
        # 'history' (default) replays the recorded values; 'hold' keeps the initial
        # inputs frozen through the replay. Set per model in the catalog entry under
        # user_defined.fmu_reset.replay_inputs. Only used on the replay path.
        self._replay_inputs = 'history'

        # Saved FMU states, keyed by the model tick they were taken at. An FMU that
        # implements fmi2GetFMUstate/fmi2SetFMUstate can be put back at any saved
        # moment instantly - no restart, no replay, and nothing to remember about
        # its inputs, because the state blob already contains everything. FMUs
        # without that capability (EnergyPlus among them) get the restart path.
        self._can_snapshot = False
        self._state_snapshots = {}
        self._snapshot_target_ts = None

        # Model tick the slave has last stepped, so a reposition can tell whether
        # it is already where it is being asked to go. A rolling reset with
        # rolling_window == episode_length asks for the tick the slave is about to
        # run anyway; restarting for that would be a physical discontinuity bought
        # for nothing.
        self._slave_tick = 0

        # value-reference maps: var_name → (vref, fmi_type_str)
        self.vars = {}
        self.in_vars = {}
        self.ou_vars = {}
        self.params_vars = {}

        # Directory the FMU runtime runs in. EnergyPlus exports drop an
        # ``Output_EPExport_<instanceName>`` folder into the process CWD; we
        # point that at the scenario log dir instead of the workspace root.
        self._fmu_workdir = None

        super().__init__(name, metadata, config, logger)

    # ------------------------------------------------------------------
    # BaseModel abstract interface
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self.logger.debug(f"Initializing FMU model {self.name}")
        self._load_fmu()
        self._resolve_replay_mode()
        self._start_instance()
        self._warn_rolling_replay_cost()
        self._warn_rolling_snapshot_gap()
        self._check_horizon_granularity()
        self.logger.info(f"FMU model {self.name} initialized (FMI {self.fmiVersion})")

    def _check_horizon_granularity(self) -> None:
        """EnergyPlus refuses a run period that is not a whole number of days.

        The horizon is handed to the FMU as its stop time, and an EnergyPlus
        export rejects initialization outright with
        'the delta between the FMU stop time and the FMU start time must be a
        multiple of 86400', so say which value is wrong before the FMU does.
        """
        if not self.max_sim_time:
            return
        if float(self.max_sim_time) % 86400 != 0:
            self.logger.warning(
                f"FMU model {self.name}: max_sim_time={self.max_sim_time}s is not a whole "
                f"number of days ({self.max_sim_time / 86400:.3f} days). EnergyPlus-exported "
                "FMUs require their run period to be a multiple of 86400 s and will fail to "
                "initialize. Round the horizon to a whole number of days."
            )

    def _warn_rolling_replay_cost(self) -> None:
        """Say up front what a rolling run will cost on this FMU.

        An FMU that cannot save and restore its state has to be restarted and
        re-simulated from the beginning of its run period to reach an earlier
        start point, and that start point slides forward every episode, so the
        total grows with the square of the episode count. The run still goes
        ahead - a long training is often worth waiting for - but the number
        should not be a surprise discovered hours in.
        """
        if self.reset_mode != 'rolling' or self._can_snapshot:
            return
        window = self.rolling_window or 0
        episodes = self.n_episodes or 0
        if not window or not episodes:
            self.logger.warning(
                f"FMU model {self.name}: rolling resets restart this FMU on every episode, "
                "because it does not support state save/restore."
            )
            return

        replay_steps = window * episodes * (episodes - 1) // 2
        self.logger.warning(
            f"FMU model {self.name}: rolling resets on an FMU without state save/restore cost "
            f"a restart plus a replay from the start of the run period on every episode. "
            f"Estimated total for {episodes} episodes with a {window}-step window: "
            f"{episodes} restarts and ~{replay_steps} replayed steps "
            f"(~{replay_steps * self.real_period / 86400:.1f} days of extra simulated time). "
            f"This grows with the square of the episode count. The run will proceed. "
            f"Setting rolling_window equal to the reset period removes the rewind entirely."
        )

    def _warn_rolling_snapshot_gap(self) -> None:
        """A rolling window wider than the episode outruns the saved start point.

        The next episode's start point is saved *in passing*, so the slave has to
        step through it during the current episode. It only does that when the
        episode is at least as long as the window; otherwise the save never
        happens and every rewind falls back to a restart with no input history to
        replay, which is far slower and physically approximate.
        """
        if self.reset_mode != 'rolling' or not self._can_snapshot:
            return
        window = self.rolling_window or 0
        episode = self.reset_period or self.episode_length or 0
        if not window or not episode or window <= episode:
            return
        self.logger.warning(
            f"FMU model {self.name}: rolling_window ({window}) is longer than the reset period "
            f"({episode}), so the slave never reaches the next episode's start point and no "
            "state can be saved in passing. Every rewind will fall back to a restart with the "
            "initial inputs held constant, which is slow and physically approximate. Set "
            "rolling_window <= the reset period."
        )

    def _fmu_reset_options(self) -> dict:
        """The model's ``fmu_reset`` block: catalog entry, scenario on top.

        The catalog describes the FMU, the scenario describes this study - the
        same building is restarted differently in a long RL training and in a
        validation run - so a scenario's ``user_defined.fmu_reset`` overrides the
        catalog key by key, exactly as it already overrides ``max_sim_time``.
        """
        catalog = (self.metadata.user_defined or {}).get('fmu_reset', {}) if self.metadata else {}
        scenario = (getattr(self.config, 'user_defined', None) or {}).get('fmu_reset', {})
        return {**(catalog or {}), **(scenario or {})}

    def _resolve_replay_mode(self) -> None:
        """Read the replay policy from the catalog entry.

        'history' replays the inputs the FMU actually saw at those ticks, so a
        rewind lands it in the state it really had. 'hold' freezes the initial
        inputs instead: no memory, but the replayed span is fiction.
        """
        fmu_reset = self._fmu_reset_options()
        mode = fmu_reset.get('replay_inputs', 'history')
        if mode not in ('history', 'hold'):
            self.logger.warning(
                f"Unknown fmu_reset.replay_inputs '{mode}'; using 'history'. "
                "Valid values: 'history', 'hold'."
            )
            mode = 'history'
        self._replay_inputs = mode

    def _load_fmu(self) -> None:
        """Resolve, read and unpack the FMU archive. Runs once per model lifetime.

        A restart re-instantiates the slave from the cached unzip directory and
        never re-downloads or re-extracts.
        """
        if self.unzipdir is not None:
            return
        self._fmu_path = self._resolve_fmu_path()
        self._unpack_fmu(self._fmu_path)
        self._resolve_snapshot_support()

    def _resolve_snapshot_support(self) -> None:
        """Can this FMU save and restore its own state?

        Taken from the FMU's modelDescription, which is authoritative; the catalog
        can only turn it off (for an FMU that advertises the capability but does
        not honour it), never on.
        """
        cosim = getattr(self.model_description, 'coSimulation', None)
        declared = bool(getattr(cosim, 'canGetAndSetFMUstate', False)) if cosim else False

        fmu_reset = self._fmu_reset_options()
        if fmu_reset.get('supports_rollback') is False and declared:
            self.logger.info(
                f"FMU model {self.name} advertises state save/restore but the catalog "
                "disables it; using restart-and-replay instead."
            )
            declared = False

        self._can_snapshot = declared
        self.logger.info(
            f"FMU model {self.name}: state save/restore "
            f"{'available - restarts are instant' if declared else 'unavailable - restarts re-instantiate the slave'}"
        )

    def _start_instance(self) -> None:
        """Instantiate a slave and drive it through initialization mode.

        Re-runnable: every call builds a fresh slave from the cached unzip dir,
        so ``reset`` restarts the binary without touching the filesystem.
        """
        # EnergyPlus FMUs create their Output_EPExport_<instance> folder in the
        # CWD active during instantiate/doStep, so run the lifecycle from the
        # log dir to keep the workspace root clean.
        self._fmu_workdir = self._resolve_fmu_workdir()
        with self._in_fmu_workdir():
            self._instantiate_fmu()
            self._setup_experiment()
            self._enter_initialization_mode()
            self._push_initial_state_to_fmu()
            self._exit_initialization_mode()
        self._instance_count += 1
        self._slave_tick = 0
        # The state at the first tick is what every full reset and every horizon
        # restart goes back to, so save it once and reuse it forever.
        self._take_snapshot(1)

    def _teardown(self) -> None:
        """Terminate and free the current slave. Idempotent, never raises."""
        if self.fmu is None:
            return
        self._free_all_snapshots()
        try:
            with self._in_fmu_workdir():
                self.fmu.terminate()
                self.fmu.freeInstance()
        except Exception as exc:
            self.logger.warning(f"FMU terminate/free raised: {exc}")
        finally:
            self.fmu = None

    def step(self) -> None:
        self.logger.debug(f"Stepping FMU model {self.name} at ts={self.state.ts}")
        # Model-local time, not federation time: the two diverge as soon as the
        # slave is restarted at its horizon or rewound by a rolling reset.
        # BaseModel._enforce_sim_horizon has already restarted the slave if this
        # step would have run past the declared max_sim_time.
        current_time = self.local_time()
        self._slave_tick = self.local_ts()
        self._maybe_snapshot_next_rolling_start()
        self._record_inputs()
        self._inputs_to_fmu()
        with self._in_fmu_workdir():
            self._do_step(current_time)
        self._outputs_from_fmu()

    def _do_step(self, current_time: float) -> None:
        """One FMU step, telling the slave whether it may discard rollback data.

        ``noSetFMUStatePriorToCurrentPoint`` is a promise to the FMU that the
        master will never put it back before this point - fmpy defaults it to
        true. That promise is false for us whenever states are being saved, and
        an FMU is entitled to free what a rewind needs once it is given.
        """
        if self.fmiVersion == '1.0':
            self.fmu.doStep(
                currentCommunicationPoint=current_time,
                communicationStepSize=self.real_period,
            )
            return
        self.fmu.doStep(
            currentCommunicationPoint=current_time,
            communicationStepSize=self.real_period,
            noSetFMUStatePriorToCurrentPoint=not self._can_snapshot,
        )

    def finalize(self) -> None:
        self.logger.info(f"Finalizing FMU model {self.name}")
        self._teardown()
        if self.unzipdir and os.path.isdir(self.unzipdir):
            try:
                shutil.rmtree(self.unzipdir)
            except PermissionError as exc:
                self.logger.error(f"Could not remove unzip dir: {exc}")

    def reset(self, mode: str = 'full', ts=None, time=None) -> None:
        """Reset interfaces and move the slave to the start point this mode asks for.

        ``full``    — restart at the FMU's own start date (model-local time 0).
        ``rolling`` — move to the episode's start point. An FMU cannot be stepped
                      backwards, so a rewind means restoring a saved state, or
                      restarting the slave and replaying to the target from the
                      beginning of its run period. A start point the slave is
                      already standing on costs nothing.
        ``none``/``soft`` — interfaces only, the slave keeps running.

        The clock bookkeeping lives in BaseModel.reset, which calls back into
        ``_reposition_backend`` below to do the FMU-specific work.
        """
        super().reset(mode=mode, ts=ts, time=time)
        if mode in ('none', 'soft'):
            self.logger.debug(
                f"Reset '{mode}' on FMU model {self.name}: interfaces only, slave untouched"
            )

    # ------------------------------------------------------------------
    # Model-local clock: repositioning the slave
    # ------------------------------------------------------------------

    def _reposition_backend(self, target_ts: int) -> None:
        """Bring the slave to its own tick *target_ts*.

        An FMU co-simulation slave has no seek and (here) no state snapshot, so
        the only way to reach any point is to restart at the beginning of its run
        period and step forward. Restarting at the first tick is therefore the
        cheap case; anything later costs a replay.
        """
        if self.fmu is None and self.unzipdir is None:
            # Called from BaseModel.__init__ before the FMU is loaded; nothing to do.
            return

        if self._slave_tick == int(target_ts) - 1:
            # Already standing exactly where it is asked to stand: a rolling reset
            # whose window is the episode length continues rather than rewinds.
            self.logger.debug(
                f"FMU model {self.name}: already positioned for tick {target_ts}; "
                "the slave keeps running"
            )
            return

        # A saved state puts the slave back instantly and carries its whole
        # internal state with it, so there is nothing to replay and nothing to
        # remember about its inputs.
        if self._can_snapshot and self._restore_snapshot(int(target_ts)):
            self._slave_tick = int(target_ts) - 1
            self._consume_rolling_snapshot(int(target_ts))
            self._input_history = []
            return

        self._teardown()
        self._start_instance()

        n_steps = max(0, int(target_ts) - 1)
        if n_steps:
            self._advance(n_steps)
        else:
            # Restarted at the beginning of the run period: everything recorded
            # after it belongs to a span the slave no longer has.
            self._input_history = []

    def _take_snapshot(self, tick: int) -> None:
        """Save the slave's complete internal state as of model tick *tick*."""
        if not self._can_snapshot or self.fmu is None:
            return
        try:
            with self._in_fmu_workdir():
                state = self.fmu.getFMUState()
        except Exception as exc:
            self.logger.warning(
                f"FMU model {self.name}: getFMUState failed ({exc}); falling back to "
                "restart-and-replay for this model."
            )
            self._can_snapshot = False
            return
        self._free_snapshot(tick)
        self._state_snapshots[tick] = state
        self.logger.debug(f"FMU model {self.name}: saved state at tick {tick}")

    def _restore_snapshot(self, tick: int) -> bool:
        """Put the slave back at the state saved for *tick*. True when it worked."""
        state = self._state_snapshots.get(tick)
        if state is None:
            return False
        try:
            with self._in_fmu_workdir():
                self.fmu.setFMUState(state)
        except Exception as exc:
            self.logger.warning(
                f"FMU model {self.name}: setFMUState failed ({exc}); restarting instead."
            )
            return False
        self.logger.debug(f"FMU model {self.name}: restored saved state at tick {tick}")
        return True

    def _free_snapshot(self, tick: int) -> None:
        state = self._state_snapshots.pop(tick, None)
        if state is None or self.fmu is None:
            return
        try:
            with self._in_fmu_workdir():
                self.fmu.freeFMUState(state)
        except Exception as exc:
            self.logger.debug(f"freeFMUState({tick}) raised: {exc}")

    def _free_all_snapshots(self) -> None:
        """Saved states belong to a live slave; freeing it invalidates all of them."""
        for tick in list(self._state_snapshots):
            self._free_snapshot(tick)
        self._state_snapshots.clear()

    def _maybe_snapshot_next_rolling_start(self) -> None:
        """Save the state at the tick the next rolling episode will start from.

        The start points are known in advance (they slide forward by
        `rolling_window` every episode) and the slave passes through the next one
        while the current episode runs, so one state saved in passing turns the
        next rewind into a restore.

        The pending target only moves on when a rewind has consumed it, so at
        most two states are ever held: the tick the slave was started at, which
        every full reset and horizon restart goes back to, and the start point of
        the next episode. The episode is usually longer than the window, so the
        slave runs well past the pending target before the reset that asks for
        it - keeping the earliest pending one is what makes that work.
        """
        if not self._can_snapshot or self.reset_mode != 'rolling':
            return
        if self._snapshot_target_ts is None:
            # Same arithmetic as BaseFederate._reset: start points are 1, 1+W, 1+2W.
            self._snapshot_target_ts = (1 + self.rolling_window) if self.rolling_window else None
            if self._snapshot_target_ts is None:
                return
        if self.local_ts() == self._snapshot_target_ts and \
                self._snapshot_target_ts not in self._state_snapshots:
            self._take_snapshot(self._snapshot_target_ts)

    def _consume_rolling_snapshot(self, tick: int) -> None:
        """A rewind has used the saved start point; keep the next one instead.

        Rolling start points slide forward and are never revisited, so the state
        just restored from is dead weight - unless it is the tick the slave was
        started at, which every full reset still needs.
        """
        if tick != 1:
            self._free_snapshot(tick)
        if self.reset_mode == 'rolling' and self.rolling_window:
            self._snapshot_target_ts = tick + self.rolling_window

    def _advance(self, n_steps: int) -> None:
        """Replay *n_steps* silently from model-local time 0.

        The steps are real FMU steps - they cost the same as simulated ones - but
        their outputs are discarded and nothing is published: no other federate is
        stepping while this runs.
        """
        history = self._input_history
        use_history = self._replay_inputs == 'history' and len(history) >= n_steps
        if not use_history:
            self.logger.warning(
                f"FMU model {self.name}: replaying {n_steps} steps with the initial "
                f"inputs held constant (recorded history covers {len(history)} steps). "
                "The replayed span is physically approximate."
            )
        self.logger.info(
            f"FMU model {self.name}: replaying {n_steps} steps "
            f"({n_steps * self.real_period} s of model-local time) to reach the requested start point"
        )
        with self._in_fmu_workdir():
            for i in range(n_steps):
                if use_history:
                    self._push_inputs(history[i])
                self._do_step(i * self.real_period)
        self._slave_tick = n_steps
        # The replayed prefix is now the history of the current epoch.
        self._input_history = history[:n_steps] if use_history else []

    def _record_inputs(self) -> None:
        """Remember this tick's inputs so a later rewind can replay them exactly.

        Only kept when the federate runs rolling resets, the one policy that can
        ask a model to move backwards. Cleared whenever the slave restarts at
        local time 0, since that discards everything after it.
        """
        if self._replay_inputs != 'history' or self.reset_mode != 'rolling':
            return
        if self._can_snapshot:
            return  # a saved state carries everything; no need to remember inputs
        idx = max(0, self.local_ts() - 1)
        if idx == len(self._input_history):
            self._input_history.append(dict(self.state.inputs))
        elif idx < len(self._input_history):
            self._input_history[idx] = dict(self.state.inputs)

    def _push_inputs(self, values: dict) -> None:
        """Write a recorded input snapshot straight into the slave."""
        for var_name, (vref, vtype) in self.in_vars.items():
            value = values.get(var_name)
            if value is not None:
                self._set_var(vref, vtype, value)

    # ------------------------------------------------------------------
    # FMU working directory (keeps Output_EPExport_* out of workspace root)
    # ------------------------------------------------------------------

    def _resolve_fmu_workdir(self) -> Path:
        """Directory the FMU runtime should execute in.

        Derived from the federate logger's file handler so the EnergyPlus
        ``Output_EPExport_<instance>`` folder lands next to the scenario logs.
        Falls back to ``./logs`` when no file handler is found.
        """
        base = None

        # Primary: federate log path exported by federate_launcher. Layout is
        # .../<scenario>/<timestamp>/federates/<name>.log -> run dir is parent.parent.
        log_file = os.environ.get('COSIM_FEDERATE_LOG_FILE')
        if log_file:
            base = Path(log_file).resolve().parent.parent

        # Secondary: walk the logger chain for a FileHandler.
        if base is None:
            lg = self.logger
            while lg is not None and base is None:
                for handler in getattr(lg, 'handlers', []):
                    if isinstance(handler, logging.FileHandler):
                        base = Path(handler.baseFilename).resolve().parent.parent
                        break
                lg = getattr(lg, 'parent', None)

        # Fallback: ./logs
        if base is None:
            base = Path('logs').resolve()

        workdir = base / 'fmu_output'
        if self._instance_count:
            # Second and later slaves write to their own subdir; the first keeps
            # the historical path so existing result layouts are unchanged.
            workdir = workdir / f"restart_{self._instance_count}"
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    @contextlib.contextmanager
    def _in_fmu_workdir(self):
        if self._fmu_workdir is None:
            self._fmu_workdir = self._resolve_fmu_workdir()
        prev_cwd = os.getcwd()
        os.chdir(self._fmu_workdir)
        try:
            yield
        finally:
            os.chdir(prev_cwd)

    # ------------------------------------------------------------------
    # FMU source resolution (local / MinIO / HTTP)
    # ------------------------------------------------------------------

    def _resolve_fmu_path(self) -> str:
        fmu_source = self.metadata.user_defined.get('fmu_source', {})
        src_type = fmu_source.get('type', 'local')

        if src_type == 'local':
            path = fmu_source.get('path', '')
            if not path:
                raise ValueError(
                    f"Model '{self.metadata.name}': fmu_source.type is 'local' "
                    "but fmu_source.path is empty. Set it in the catalog entry."
                )
            if not os.path.exists(path):
                raise FileNotFoundError(f"FMU not found at local path: {path}")
            return path

        elif src_type == 'minio':
            return self._download_from_minio(fmu_source)

        elif src_type == 'http':
            return self._download_from_http(fmu_source)

        else:
            raise ValueError(
                f"Unknown fmu_source.type '{src_type}'. "
                "Valid values: 'local', 'minio', 'http'."
            )

    def _cache_dir(self) -> Path:
        d = Path.home() / '.cosimgym' / 'fmu_cache' / self.metadata.name / self.metadata.version
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _download_from_minio(self, fmu_source: dict) -> str:
        try:
            from minio import Minio
        except ImportError:
            raise ImportError(
                "minio package not installed. Run: pip install minio>=7.0.0"
            )

        from utils.ports import minio_endpoint  # lazy: this module can load before src is on path
        raw_endpoint = fmu_source.get('endpoint', minio_endpoint())
        secure = raw_endpoint.startswith('https://')
        endpoint = raw_endpoint.replace('https://', '').replace('http://', '')
        bucket = fmu_source.get('bucket', 'fmus')
        object_key = fmu_source['object_key']
        access_key = fmu_source.get('access_key') or os.getenv('MINIO_ACCESS_KEY', 'cosimgym')
        secret_key = fmu_source.get('secret_key') or os.getenv('MINIO_SECRET_KEY', 'cosimgym123')

        local_path = self._cache_dir() / os.path.basename(object_key)
        if local_path.exists():
            self.logger.info(f"FMU cache hit: {local_path}")
            return str(local_path)

        self.logger.info(f"Downloading FMU from MinIO {endpoint}/{bucket}/{object_key}")
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        client.fget_object(bucket, object_key, str(local_path))
        self.logger.info(f"FMU downloaded to {local_path}")
        return str(local_path)

    def _download_from_http(self, fmu_source: dict) -> str:
        url = fmu_source['url']
        local_path = self._cache_dir() / url.split('/')[-1]
        if local_path.exists():
            self.logger.info(f"FMU cache hit: {local_path}")
            return str(local_path)

        self.logger.info(f"Downloading FMU from {url}")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        self.logger.info(f"FMU downloaded to {local_path}")
        return str(local_path)

    # ------------------------------------------------------------------
    # FMU lifecycle helpers
    # ------------------------------------------------------------------

    def _unpack_fmu(self, fmu_path: str) -> None:
        self.model_description = read_model_description(fmu_path, validate=True)
        self._get_vars_from_fmu()
        self.unzipdir = extract(fmu_path)
        self.fmiVersion = self.model_description.fmiVersion
        self.logger.debug(f"FMU unpacked: {dump(fmu_path)}")

    def _get_vars_from_fmu(self) -> None:
        for v in self.model_description.modelVariables:
            vtype = v.type if v.type else 'Real'
            self.vars[v.name] = (v.valueReference, vtype, v.causality, v.variability)
            if v.causality == 'parameter':
                self.params_vars[v.name] = (v.valueReference, vtype)
            elif v.causality == 'input':
                self.in_vars[v.name] = (v.valueReference, vtype)
            elif v.causality == 'output':
                self.ou_vars[v.name] = (v.valueReference, vtype)

    def _instantiate_fmu(self) -> None:
        guid = self.model_description.guid
        model_id = self.model_description.coSimulation.modelIdentifier

        if self.fmiVersion == '1.0':
            self.fmu = FMU1Slave(
                guid=guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=model_id,
                instanceName=self.name,
            )
            self.fmu.instantiate(loggingOn=False)

        elif self.fmiVersion == '2.0':
            self.fmu = FMU2Slave(
                guid=guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=model_id,
                instanceName=self.name,
            )
            self.fmu.instantiate()

        elif self.fmiVersion == '3.0':
            self.fmu = FMU3Slave(
                guid=guid,
                unzipDirectory=self.unzipdir,
                modelIdentifier=model_id,
                instanceName=self.name,
            )
            self.fmu.instantiate()

        else:
            raise RuntimeError(f"Unsupported FMI version: {self.fmiVersion}")

    def _stop_time_seconds(self):
        """Total simulation horizon in seconds.

        Some co-sim FMUs (notably EnergyPlus exports) require a *defined* stop
        time: with stopTime=None fmpy sets stopTimeDefined=False and EnergyPlus
        clamps the stop time to 0, so the second doStep fails with fmi2Error.
        Derive it from the scenario start/end, falling back to time_stop*period.

        A declared ``max_sim_time`` wins: it is the horizon of a single epoch, and
        the slave is restarted rather than stepped past it, so the scenario span is
        irrelevant to the FMU.
        """
        if self.max_sim_time:
            return float(self.max_sim_time)
        try:
            start = datetime.fromisoformat(self.config.start_time)
            end = datetime.fromisoformat(self.config.end_time)
            return (end - start).total_seconds()
        except (TypeError, ValueError):
            pass
        if self.config.time_stop is not None and self.real_period is not None:
            return float(self.config.time_stop) * float(self.real_period)
        return None

    def _setup_experiment(self) -> None:
        if self.fmiVersion == '2.0':
            self.fmu.setupExperiment(startTime=0.0, stopTime=self._stop_time_seconds())
        # FMI 1.0 has no setupExperiment; initialization happens in _exit_initialization_mode
        # FMI 3.0 folds setupExperiment into enterInitializationMode(startTime, stopTime)

    def _enter_initialization_mode(self) -> None:
        if self.fmiVersion == '2.0':
            self.fmu.enterInitializationMode()
        elif self.fmiVersion == '3.0':
            self.fmu.enterInitializationMode(startTime=0.0, stopTime=self._stop_time_seconds())

    def _push_initial_state_to_fmu(self) -> None:
        for param_name, (vref, vtype) in self.params_vars.items():
            value = self.state.parameters.get(param_name)
            if value is not None:
                self._set_var(vref, vtype, value)

        for inp_name, (vref, vtype) in self.in_vars.items():
            value = self.state.inputs.get(inp_name)
            if value is not None:
                self._set_var(vref, vtype, value)

    def _exit_initialization_mode(self) -> None:
        if self.fmiVersion == '2.0':
            status = self.fmu.exitInitializationMode()
            if status != 0:
                raise RuntimeError(f"FMU exitInitializationMode returned status {status}")
        elif self.fmiVersion == '1.0':
            self.fmu.initialize(tStart=0.0, stopTime=None)
        elif self.fmiVersion == '3.0':
            self.fmu.exitInitializationMode()  # raises FMICallException on non-OK status

    # ------------------------------------------------------------------
    # Per-step I/O transfer
    # ------------------------------------------------------------------

    def _inputs_to_fmu(self) -> None:
        for var_name, (vref, vtype) in self.in_vars.items():
            value = self.state.inputs.get(var_name)
            if value is not None:
                self._set_var(vref, vtype, value)

    def _outputs_from_fmu(self) -> None:
        for var_name, (vref, vtype) in self.ou_vars.items():
            if var_name in self.state.outputs:
                self.state.outputs[var_name] = self._get_var(vref, vtype)

    # ------------------------------------------------------------------
    # Generic get / set via FMI type dispatch
    # ------------------------------------------------------------------

    def _set_var(self, vref: int, vtype: str, value) -> None:
        getter, setter = _FMI_TYPE_GETSET.get(vtype, ('getReal', 'setReal'))
        try:
            getattr(self.fmu, setter)([vref], [value])
        except Exception as exc:
            self.logger.error(f"setVar vref={vref} type={vtype} value={value}: {exc}")

    def _get_var(self, vref: int, vtype: str):
        getter, setter = _FMI_TYPE_GETSET.get(vtype, ('getReal', 'setReal'))
        try:
            return getattr(self.fmu, getter)([vref])[0]
        except Exception as exc:
            self.logger.error(f"getVar vref={vref} type={vtype}: {exc}")
            return None
