"""
parallel_executor.py

Persistent-worker-process parallel execution of a federate's model instances.

The target physical models are pure-Python and GIL-bound, so threads give no
speedup for CPU-heavy `step()` implementations — this module fans the per-tick
`model._step()` compute out to a small pool of persistent `multiprocessing`
worker processes (spawn context), each owning a stateful shard of model
instances that persists across ticks (mirrors how the sequential loop in
BaseFederate.run() keeps each model instance alive for the whole simulation).

The main federate process is unaffected otherwise: all HELICS I/O, storage,
publishing and initialization stay in the main process. Only the heavy
per-instance `step()` compute is delegated.

Author: Pietro Rando Mazzarino
Email: pietro.randomazzarino@polito.it
Organization: EC-Lab Politecnico di Torino
created: 2026-03-17

"""
import importlib
import logging
import multiprocessing as mp
import os
import signal


def _build_models(build_spec, entity_ids, worker_logger):
    """Rebuild a shard of model instances from a picklable build spec.

    Mirrors the exact construction recipe used by
    BaseFederate._register_entities(): resolve module_path/class_name from
    catalog metadata, import the module, instantiate the model class per
    entity id. Must be called inside the worker process — model objects
    are not reliably picklable (they hold a live logger).
    """
    module_path, class_name, metadata, model_configs = build_spec
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    models = {}
    for entity_id in entity_ids:
        models[entity_id] = model_class(entity_id, metadata, model_configs, worker_logger)
    return models


def _worker_loop(conn, build_spec, entity_ids):
    """Entry point run inside each persistent worker process.

    Rebuilds its shard of models once, then loops: receive (ts, inputs) ->
    step each owned model -> send back {entity_id: outputs}. Exits cleanly
    on the sentinel (None) or if the parent process goes away (EOFError /
    BrokenPipeError) -- it must never spin forever.
    """
    # Each worker gets its own plain logger; the parent's logger (and any
    # non-picklable handlers) is never shipped across the process boundary.
    worker_logger = logging.getLogger(f"parallel_worker.{os.getpid()}")
    if not worker_logger.handlers:
        worker_logger.addHandler(logging.NullHandler())

    # Ignore SIGINT in the worker: the parent's signal handler owns shutdown
    # and will send us the sentinel / terminate us explicitly. Without this,
    # a Ctrl-C storm can hit the worker mid-send and corrupt the pipe.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        models = _build_models(build_spec, entity_ids, worker_logger)
    except Exception as exc:  # noqa: BLE001 - report back, then exit
        try:
            conn.send(('__error__', repr(exc)))
        except Exception:
            pass
        return

    while True:
        try:
            msg = conn.recv()
        except (EOFError, BrokenPipeError, OSError):
            break

        if msg is None:
            break

        ts, in_shard = msg
        try:
            out = {}
            for entity_id, model in models.items():
                out[entity_id] = model._step(ts, in_shard.get(entity_id, {}))
            conn.send(out)
        except Exception as exc:  # noqa: BLE001
            try:
                conn.send(('__error__', repr(exc)))
            except Exception:
                pass
            break


class ParallelModelExecutor:
    """Owns a pool of persistent worker processes stepping a partition of a
    federate's model instances in parallel.

    Usage:
        executor = ParallelModelExecutor(build_spec, entity_ids, max_workers, logger)
        executor.start()
        ...
        outputs = executor.step(ts, inputs)   # each tick
        ...
        executor.close()                      # idempotent, escalating shutdown
    """

    def __init__(self, build_spec, entity_ids, max_workers, logger=None):
        self.build_spec = build_spec
        self.entity_ids = list(entity_ids)
        self.logger = logger or logging.getLogger(__name__)

        resolved_max = max_workers or os.cpu_count() or 1
        self.n_workers = max(1, min(len(self.entity_ids), resolved_max))

        self._ctx = mp.get_context('spawn')
        self._workers = []       # list of Process
        self._parent_conns = []  # list of Connection (main-side)
        self._shards = []        # list of list[entity_id]
        self._closed = False
        self._started = False
        self._prev_sigint = None
        self._prev_sigterm = None

    def _partition_entities(self):
        shards = [[] for _ in range(self.n_workers)]
        for i, entity_id in enumerate(self.entity_ids):
            shards[i % self.n_workers].append(entity_id)
        return [s for s in shards if s]

    def start(self):
        if self._started:
            return
        self._shards = self._partition_entities()
        self.n_workers = len(self._shards)

        for shard in self._shards:
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            proc = self._ctx.Process(
                target=_worker_loop,
                args=(child_conn, self.build_spec, shard),
                daemon=True,  # auto-reaped if the parent dies unexpectedly
            )
            proc.start()
            # The child's end of the pipe must be closed in the parent, or
            # the parent keeps a dangling reference to a fd it never uses.
            child_conn.close()
            self._workers.append(proc)
            self._parent_conns.append(parent_conn)

        self._started = True
        self._register_cleanup_hooks()
        self.logger.info(
            f"ParallelModelExecutor started {self.n_workers} worker process(es) "
            f"for {len(self.entity_ids)} model instance(s)."
        )

    def _register_cleanup_hooks(self):
        import atexit
        atexit.register(self.close)

        def _handler(signum, frame):
            self.logger.warning(f"ParallelModelExecutor caught signal {signum}, shutting down workers.")
            self.close()
            prev = self._prev_sigint if signum == signal.SIGINT else self._prev_sigterm
            if callable(prev):
                prev(signum, frame)
            elif prev == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            else:
                raise SystemExit(128 + signum)

        try:
            self._prev_sigint = signal.getsignal(signal.SIGINT)
            self._prev_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except (ValueError, RuntimeError):
            # signal() only works in the main thread — if we're not there,
            # atexit + daemon=True is still a valid, if partial, safety net.
            self.logger.warning(
                "ParallelModelExecutor: could not install SIGINT/SIGTERM handlers "
                "(not in main thread). Relying on atexit + daemon workers."
            )

    def step(self, ts, inputs):
        """Fan out this tick's inputs to workers, gather and merge outputs."""
        if not self._started:
            raise RuntimeError("ParallelModelExecutor.step() called before start()")

        for shard, conn in zip(self._shards, self._parent_conns):
            in_shard = {eid: inputs.get(eid, {}) for eid in shard}
            try:
                conn.send((ts, in_shard))
            except (BrokenPipeError, OSError) as exc:
                self.close()
                raise RuntimeError(f"ParallelModelExecutor: worker pipe broken while sending (ts={ts}): {exc}")

        merged = {}
        for proc, conn in zip(self._workers, self._parent_conns):
            try:
                result = conn.recv()
            except (EOFError, BrokenPipeError, OSError) as exc:
                self.close()
                raise RuntimeError(
                    f"ParallelModelExecutor: worker (pid={proc.pid}) died or closed its "
                    f"pipe while stepping (ts={ts}): {exc}"
                )
            if isinstance(result, tuple) and len(result) == 2 and result[0] == '__error__':
                self.close()
                raise RuntimeError(f"ParallelModelExecutor: worker (pid={proc.pid}) raised: {result[1]}")
            merged.update(result)
        return merged

    def close(self):
        """Idempotent, escalating shutdown: sentinel -> join -> terminate -> join -> kill.
        Safe to call multiple times (atexit, signal handler, and explicit
        finalize() may all call this)."""
        if self._closed:
            return
        self._closed = True

        # 1) ask nicely
        for conn in self._parent_conns:
            try:
                conn.send(None)
            except Exception:
                pass

        for proc in self._workers:
            proc.join(timeout=5)

        # 2) escalate to terminate for anything still alive
        for proc in self._workers:
            if proc.is_alive():
                self.logger.warning(f"ParallelModelExecutor: worker pid={proc.pid} did not exit, terminating.")
                proc.terminate()

        for proc in self._workers:
            proc.join(timeout=5)

        # 3) last resort
        for proc in self._workers:
            if proc.is_alive():
                self.logger.error(f"ParallelModelExecutor: worker pid={proc.pid} still alive, killing.")
                try:
                    proc.kill()
                except Exception:
                    pass
                proc.join(timeout=5)

        for conn in self._parent_conns:
            try:
                conn.close()
            except Exception:
                pass

        # restore any signal handlers we replaced
        try:
            if self._prev_sigint is not None:
                signal.signal(signal.SIGINT, self._prev_sigint)
            if self._prev_sigterm is not None:
                signal.signal(signal.SIGTERM, self._prev_sigterm)
        except (ValueError, RuntimeError):
            pass

        self.logger.info("ParallelModelExecutor closed (all workers stopped).")
