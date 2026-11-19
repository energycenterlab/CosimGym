# Possible migration of the Python environment from conda to uv

**Status:** idea, not decided. The repo works on conda (`environment.yml`, env `cosim_gym`) and
stays on conda until the user explicitly decides to migrate.

## Why consider it

| Problem today (conda) | What uv would give |
|---|---|
| `environment.yml` pins few versions (`pandas`, `scipy`, … unpinned): Mac, server and remote machines can silently diverge | `uv.lock` pins every transitive dependency with hashes: identical env everywhere |
| `environment.yml` drifted from the code (`ray`, `pandapower`, `pandapipes`, `ijson`, `pytest` were missing until 2026-11-19, now added by hand) | `uv add` writes the dependency to `pyproject.toml` and the lock in one step, so drift is harder |
| `conda activate` silently fails from scripts, and `conda run` has been killed early on long jobs (`scripts/scaling_study/RUNBOOK.md`) | `uv run python …` needs no activation and always uses the project `.venv` |
| Env creation takes minutes | Resolution and install take seconds |
| conda must be installed on every remote machine of a distributed run | only a single static `uv` binary is needed |

Every dependency in `environment.yml` is available on PyPI (HELICS included, already installed via
`pip:`), so there is no technical blocker on the conda side.

## What a migration would touch

- `pyproject.toml` + `uv.lock` (new), `environment.yml` (remove or keep as a fallback)
- `Makefile` (`setup-env`, `run`, `dashboard`, `check`, `clean-env` targets use `conda`)
- `tests/regression_suite.py` invocation and `CLAUDE.md` setup commands (`conda run -n cosim_gym`)
- Distributed deployment: the interpreter used on remote machines (check `MachineConfig` /
  `remote_executor.py` and `docs/user_guide/distributed_deployment.md`)
- `scripts/scaling_study/RUNBOOK.md`, `CONTRACT.md`
- ~16 pages under `docs/` that mention conda (`git grep -l -i conda -- docs`)

## Risks / open questions

- The remote machines in the paper case studies: does each have conda today, and who controls them?
- Binary wheels for `helics`, `fmpy` and EnergyPlus FMUs on every target platform (Mac arm64, Linux x86_64).
- Results reproducibility for papers already run on the conda env: pin the same versions in the
  first lock.

## If adopted

Write `docs/changes_reports/uv_migration_plan.md` first (plan of changes), and run the full
regression suite on both conda and uv envs before switching the docs.
