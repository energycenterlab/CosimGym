# Session working rules, known-issues register, CLAUDE.md slimming, doc audit

Change report for the 2026-09-16 session. No runtime code was modified — the changes are
to the working agreement, the documentation, and three scenario comments.

Related: `CLAUDE.md` → *Working Rules*, `docs/KNOWN_ISSUES.md`, `HANDOFF.md`.

---

## 1. Working rules in `CLAUDE.md`

**What.** New `## Working Rules (READ FIRST — apply to every session)` section fixing:
the set of reference documents; the three support-document types and their single homes
(`docs/changes_reports/` for plans and reports, `docs/future_and_TODOs/` for future work);
one document per feature per type; documentation-upkeep duty; the handoff routine; scope
discipline.

**Why.** Sessions were generating scattered `.md` files — a `docs/handoffs/` directory, an
`EXPERIMENTS_INSTRUCTIONS.md` at the repo root, a 330-line `HANDOFF.md` that had accreted
several sessions of history. Nothing said where a document belonged, so each session
invented a location.

**Alternative rejected.** A separate `CONVENTIONS.md`. It would itself be a fourth
root-level reference document — the exact thing the rules forbid — and it would not be
loaded automatically into a session the way `CLAUDE.md` is.

## 2. `docs/KNOWN_ISSUES.md` — the single defect register

**What.** One file for everything currently broken or limited, added to the mkdocs nav.
Absorbed `docs/future_and_TODOs/known_issues_from_regression.md` (deleted) and gained an
entry for the FMU-horizon limitations. The five references to the old path were rewritten.

**Why.** Known bugs were living in a `future_and_TODOs` file, which conflates "this is
broken" with "we would like to build this". They are different questions with different
lifetimes: an enhancement is *added to*, a defect is *deleted when fixed*.

**Design decision.** The file is explicitly **not a changelog**: an entry is deleted the
moment the bug is fixed, and the story of the fix lives in that feature's changes report.
Keeping fixed entries would make it grow without bound and stop answering the one question
it exists to answer — *what is broken right now?* It is kept in sync with
`tests/regression_suite.py`'s `KNOWN_FAIL` dict; an UNEXPECTED-PASS means removing both.

## 3. `CLAUDE.md` slimmed: 5.3k → 3.1k tokens

**What.** Feature-level detail removed and replaced with a *Feature map → documentation*
table. Dropped: the storage-sink internals, the multi-federation broker description, the
digital-twin mechanism description, the full scenario-YAML config reference, the RL config
schema, the ports/`.env` reference. Kept (nothing else records them): the working rules,
the execution-flow orientation, the config pipeline, the catalog, the regression-suite dev
workflow. New `## Where the truth lives` section.

**Why.** `CLAUDE.md` is loaded into every turn of every session, so every line is paid for
continuously — and a duplicated feature description is a line that goes stale silently,
because nothing tests it. §5 below is a worked example of exactly that happening.

**The rule that came out of it.** Authority order is **code graph → code → `docs/`**.
`docs/` is a map, not the territory; a doc that disagrees with the code is a bug to fix,
not a source to quote. Behavior questions are not answered from prose alone.

**Alternative rejected.** Keeping the detail and accepting the token cost. Rejected because
the cost is not really tokens — it is a second, untested copy of the truth.

## 4. Ports were undocumented — `docs/Installation_Setup.md`

**What.** New *Configuring ports* section: the `src/.env` single-source-of-truth model, the
full key table, the resolution order (explicit export > `src/.env` > built-in default), the
legacy aliases, and the two things it does not cover (container-internal ports, a
scenario's own `broker_config.port`).

**Why (a real gap).** This existed only in `CLAUDE.md`. The installation guide still
presented `6379` and `11883` as fixed, which is wrong and matters on the shared machine
this project runs on. Slimming `CLAUDE.md` would have deleted the only copy.

## 5. Cross-federation subscription targets — the docs were wrong

**What was wrong.** `federation.md` and `federate.md` both documented a cross-federation
target as `<federation_name>.<federate_name>.<instance_id>/<pub_key>`.

**What the code does.** `BaseFederate._register_pubs` registers
`register_global_publication(f"{federate_name}.{instance}/{pub_key}")` — a flat global
namespace with no federation component — and `_register_subs` passes the YAML target string
to `helicsInputAddTarget()` **unchanged**. There is no place where a federation name is
inserted on either side. `src/scenarios/simple_test_multifederations.yaml` confirms it:
`spring_federate` (federation_1) subscribes to `input_federate.0/force` (federation_2) with
a bare key, and **no scenario in the repo uses the prefixed form**.

**Why it stayed wrong.** The failure is silent. A federation-prefixed target does not
raise — it matches no published key, and the subscription sits at its default for the
entire run. Nothing fails, so nothing corrected the doc.

**Fixed.** Both pages now state the single bare format, explain why there is no prefix
(citing the two functions), and warn explicitly that the prefixed form silently receives
nothing. Also noted: the federation-qualified form *is* understood by
`ScenarioManager._resolve_target_federate_node`, but only for the `auto_offset` dependency
graph — it never reaches the HELICS bind — and the dotted
`<federation>.<federate>.<instance>.<variable>` form is the RL key namespace, unrelated. The
same wrong format was corrected in `docs/future_and_TODOs/scaling_study_plan.md`.

## 6. Other documentation fixes

- `docs/models_reference.md`: broken link `../pandapower_extension.md` →
  `future_and_TODOs/pandapower_extension.md`.
- `docs/user_guide/scenario_configuration/general.md`: added a warning that `sink: parquet`
  SIGSEGVs on some machines, pointing at `KNOWN_ISSUES.md`. The sink was documented as
  usable with no hint that it is a tracked HIGH-severity bug.
- Verified and *not* changed: the sink docs' claim that the dashboard reads parquet. It is
  correct — `dashboard_data.py` has parquet filename patterns and `_read_parquet_records`.
  **`CLAUDE.md` was the stale one**, still claiming "JSON results only"; that claim is now
  deleted rather than propagated.
- Three scenario comments (`cs_s1_models`, `cs_s4_topo`, `cs_s5_dt`) referenced the deleted
  `EXPERIMENTS_INSTRUCTIONS.md`. The user confirmed that spec is obsolete — superseded by
  the case studies now being written — so the references were removed outright rather than
  redirected. Each comment now states the fact it needed the citation for (the dotted form
  is the RL key namespace; the S5 wait-for-external-value is expected behavior), so nothing
  depends on the old document.

## Verification

- `mkdocs build` completes; the `KNOWN_ISSUES` page renders and is in the nav.
- `--strict` fails on 17 pre-existing warnings, all from `scaling_study_plan.md` linking to
  files under `scripts/` — real files, but outside mkdocs' `docs_dir`, so unresolvable.
  Untouched: out of scope, and that page is not in the nav.
- An internal-link scan across `docs/` reports zero broken relative links.
- The three edited scenario YAMLs parse (`yaml.safe_load`).
- Note: `!!!` admonition syntax does not work here — `mkdocs.yml` declares no
  `markdown_extensions`. The parquet warning uses a blockquote, matching the rest of `docs/`.

## Closed: the old experiment spec

`EXPERIMENTS_INSTRUCTIONS.md` was deleted from the repo root (correctly — it was not one of
the allowed document types). The user confirmed it is **obsolete**: it describes an earlier
round of experiments, and a new experiment instruction document is being written by hand
against the current case studies. Nothing in the repo references it any more. It remains in
git history at commit `77143ad` if an old detail is ever needed.
