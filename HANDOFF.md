# HANDOFF

State of the last session, for the next one. Rewritten every session — do not append.
Rules that govern this file and all support documents live in `CLAUDE.md` →
**Working Rules**. Read that section first.

---

## Goal (this session)

Define the standing working agreement for how Claude sessions operate in this repo:
where support documents go, which files are reference documents, when documentation and
`CLAUDE.md` get updated, and how the handoff routine works — so future sessions stop
generating scattered `.md` files and stay consistent.

## Current Progress

- **`CLAUDE.md`**: new `## Working Rules (READ FIRST — apply to every session)` section
  inserted between *What This Project Is* and *Setup & Common Commands*. It fixes:
  - the only two reference documents: `CLAUDE.md` + `HANDOFF.md`;
  - the only three support-document types and their single homes —
    plan (`docs/changes_reports/<feature>_plan.md`),
    changes report (`docs/changes_reports/<feature>_changes.md`),
    future/TODO (`docs/future_and_TODOs/<topic>_followups.md`);
  - one document per feature per type, no `_v2`/`_final` variants, DONE-marking;
  - documentation upkeep duty (update `docs/` always on fundamentals/UX changes,
    `CLAUDE.md` only when it states something that changed, plus a
    `tests/regression_suite.py` entry per new feature);
  - the handoff routine (rewrite, not append);
  - scope discipline (no drive-by refactors, no unasked commits, ask before big runs).
- **`docs/KNOWN_ISSUES.md`** (new, third reference document): the single register of
  everything currently broken or limited. Absorbed
  `docs/future_and_TODOs/known_issues_from_regression.md` (deleted) plus an entry for the
  FMU-horizon limitations. Added to the mkdocs nav; the 5 references to the old path were
  rewritten (`CLAUDE.md`, `src/scenarios/cs_s{1,4}_*.yaml`,
  `docs/future_and_TODOs/scaling_study_plan.md`,
  `scripts/scaling_study/findings/bottlenecks.md`).
- **`HANDOFF.md`**: this file, rewritten from the stale
  `distributed-ssh-spawning-plan` handoff (that work is long merged into `main`).
- **`CLAUDE.original.md`**: deleted — a pre-compression backup of `CLAUDE.md` from commit
  `285ca01`, 130 lines vs the current 231, missing every feature added since.
  Recoverable from git history if ever needed.
- Verified the two document folders already exist and already follow the naming
  convention (`*_plan.md`, `*_changes.md`, `*_followups.md`), so no files were moved.

- **`CLAUDE.md` slimmed** 5.3k → 3.1k tokens: feature detail replaced by a *Feature map →
  documentation* table, plus a new `## Where the truth lives` section setting the authority
  order **code graph → code → `docs/`** (docs are a map, not the territory; a doc that
  disagrees with the code is a bug to fix, not a source to quote).
- **Documentation audit** — see
  `docs/changes_reports/session_rules_and_doc_audit_changes.md` for the full account. The
  substantive finds: ports/`src/.env` were undocumented outside `CLAUDE.md` (now a section
  in `Installation_Setup.md`), and the **cross-federation subscription target format was
  wrong in two doc pages** — there is no federation prefix, and the prefixed form fails
  silently. Verified against `BaseFederate._register_pubs/_register_subs` and
  `simple_test_multifederations.yaml`.

## What Worked

Auditing the docs *against the code* rather than against `CLAUDE.md` — it immediately
found that `CLAUDE.md` was the stale party twice (it claimed the dashboard cannot read
parquet; it can), and that a doc claim nobody had questioned was wrong in a way that fails
silently at runtime.

Rules go in `CLAUDE.md` rather than a separate rules file — it is the one doc loaded into
every session automatically, so a separate `CONVENTIONS.md` would itself violate the rule
it defines.

## What Didn't Work / Watch Out

Nothing failed. Note for the next session: the previous handoff had grown to 330 lines by
accumulating multiple sessions' history. That is exactly what the new rule forbids —
durable facts belong in `CLAUDE.md` or a type-2 changes report.

## In-flight, uncommitted state (NOT this session's work)

Branch `main`, working tree dirty with the **FMU horizon & reset** work from the prior
session — model-local clock, automatic restart at `max_sim_time`, rolling reset for FMUs:

- modified: `src/core/{BaseFederate,ScenarioManager}.py`,
  `src/models/{base_model,base_FMU_model,base_csv_reader}.py`,
  `src/models/model_catalog/{ModelCatalog,RedisCatalog,catalog.yaml,fmu_catalog_register.py,model_template.yaml}`,
  `src/utils/config_dataclasses.py`, `tests/regression_suite.py`,
  `docs/user_guide/fmu_models.md`, `docs/user_guide/scenario_configuration/general.md`
- new: `src/scenarios/fmu_horizon_smoketest.yaml`,
  `tests/test_{model_local_clock,fmu_state_snapshot,fmu_horizon_detection}.py`,
  `docs/changes_reports/fmu_horizon_and_reset_changes.md`,
  `docs/future_and_TODOs/fmu_horizon_and_reset_followups.md`
- deleted (cleanup toward the new rules): `EXPERIMENTS_INSTRUCTIONS.md`,
  `docs/handoffs/digitaltwin_interfaces.md`, `docs/handoffs/nonblocking_storage.md`

Nothing committed. Details of that work: `docs/changes_reports/fmu_horizon_and_reset_changes.md`.

## Next Steps

1. Decide whether to commit the FMU horizon/reset work and the doc cleanup (user's call —
   nothing is committed without an explicit ask).
2. Open question left with the user: delete the gitignored `site/` build directory (7.1 MB,
   stale since 2026-07-06, regenerable with `mkdocs build`). Not deleted without an ask.
3. `EXPERIMENTS_INSTRUCTIONS.md` — **closed, no action.** The user confirmed the old spec
   is obsolete and is writing a new experiment instruction document against the current
   case studies. All references to it were removed from the three `cs_*.yaml` comments;
   the file stays in git history at `77143ad`.
4. Run the pre-merge gate before any merge:
   `conda run -n cosim_gym python tests/regression_suite.py`.
5. Open items for the FMU work are listed in
   `docs/future_and_TODOs/fmu_horizon_and_reset_followups.md`.
