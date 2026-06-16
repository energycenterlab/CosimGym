"""
checkpoint_manager.py — resolve checkpoint paths from the new experiment.checkpoint config.

New schema: experiment.checkpoint = { dir, best }. `CheckpointConfig.best_path` already resolves
`best` against `dir`. This thin manager wraps path resolution + dir creation so agents (and
custom agents) don't re-derive it. Save/load remain backend-specific and stay in the agent.
"""
from __future__ import annotations

import os
from typing import Optional


class CheckpointManager:
    def __init__(self, experiment_cfg, run_cfg=None, logger=None):
        # experiment_cfg: ExperimentConfig (has .checkpoint: CheckpointConfig)
        # run_cfg: RunConfig (optional) — used to honor an explicit test checkpoint override
        self.experiment_cfg = experiment_cfg
        self.run_cfg = run_cfg
        self.logger = logger

    @property
    def best_path(self) -> Optional[str]:
        """Path of the single best checkpoint produced by training (dir-resolved)."""
        ckpt = getattr(self.experiment_cfg, "checkpoint", None)
        return ckpt.best_path if ckpt is not None else None

    def ensure_dir(self, path: Optional[str] = None) -> Optional[str]:
        """Create the parent directory of `path` (default: best_path). Returns the path."""
        path = path or self.best_path
        if path:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
        return path

    def test_checkpoint(self) -> Optional[str]:
        """Checkpoint to load for testing: explicit run.test.checkpoint if set, else best_path."""
        test = getattr(self.run_cfg, "test", None) if self.run_cfg is not None else None
        if test is not None and getattr(test, "checkpoint", None):
            return test.checkpoint
        return self.best_path
