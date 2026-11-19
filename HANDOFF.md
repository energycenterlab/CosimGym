# HANDOFF

State of the last session, for the next one. Rewritten every session — do not append.
Rules that govern this file and all support documents live in `CLAUDE.md` →
**Working Rules**. Read that section first.

---

## Goal (this session, 2026-11-19)

The author moved to a new machine (`eclab-cloud1`, Linux) after a hardware crash on the old one.
This session rebuilt the **Claude Code setup** (plugins and tools) and resolved, one at a time
and together with the author, the contradictions between rules and plugins. **No project
code was changed.**

## Current Progress — done, uncommitted

### Plugins and tools (machine state, outside git)
| Tool | Result |
|---|---|
| graphify | `uv tool install graphifyy` (0.9.112), skill updated. **Docs only.** |
| code-review-graph | `uv tool install code-review-graph` (2.4.0), graph built (157 files, 6.1k nodes, 27 MB). **Code only.** MCP in `.mcp.json` (local, gitignored). Git hooks pre-commit/post-checkout/post-merge/post-rewrite installed. |
| paper-search | repo cloned in `~/paper-search-mcp` + `uv sync`, arXiv search tested OK |
| pyright-lsp | `uv tool install pyright` + plugin enabled; clangd-lsp disabled |
| zotero MCP | **Mac only** (author's choice): not installed here |
| `claude-code-setup@anthropic-plugin-directory` | broken duplicate removed from `enabledPlugins` |
| claude-mem | worker OK (v16); context lowered 50→20 observations (`~/.claude-mem/settings.json`); stale `CAPTURE_BROKEN` deleted |

### Rule decisions (taken with the author)
1. **Commits**: `Bash(git commit:*)` removed from the global allow list (commits ask again). For autonomous projects/sessions: project `allow` in `.claude/settings.local.json` or `claude --allowedTools "Bash(git commit:*)"`. Written rule updated in both CLAUDE.md files. (Verified in the docs: deny→ask→allow regardless of scope, so a global `ask` could not be overridden; that is why it was not used.)
2. **Ponytail**: default `lite` (`env.PONYTAIL_DEFAULT_MODE` in the global settings) + global rule "ponytail governs diff size, not the choice of solution".
3. **Superpowers**: kept; global rule "plans go to the project's convention (`docs/changes_reports/<feature>_plan.md`), never `docs/superpowers/`; commit steps = checkpoints to propose; worktrees only on request".
4. **claude-mem**: recall only. Global rule: plans = superpowers, handoff = the author's `handoff` skill, to-dos = native tools, durable memory = native `MEMORY.md`. (`skillOverrides` does NOT work on plugin skills: verified in the docs.)
5. **Graphs**: code → code-review-graph, docs → graphify. Removed the project "MANDATORY graphify" PreToolUse hooks (`.claude/settings.json` is now empty); CRG hooks stay only in the global settings; project CLAUDE.md "Using the graphs" section rewritten; `graphify update .` removed from `model_reset_redesign_plan.md`; graphify-skill description restricted to docs.
6. **Python**: global rule rewritten: new projects → uv; repos with `environment.yml` stay on conda. Project CLAUDE.md uses `conda run -n cosim_gym` instead of `conda activate`. Possible migration recorded in `docs/future_and_TODOs/uv_migration_followups.md`.

### Files touched
- Repo (uncommitted): `CLAUDE.md`, `.claude/settings.json`, `.gitignore`, `.graphifyignore` (now excludes code and images), `.code-review-graphignore` (new: excludes scenario YAMLs), `docs/changes_reports/model_reset_redesign_plan.md`, `docs/future_and_TODOs/uv_migration_followups.md` (new), `HANDOFF.md`; `graphify-out/`: cache, dated snapshots and `.graphify_python`/`.graphify_root` removed from tracking (`git rm --cached`, files still on disk) → ~470 `D` entries in `git status`.
- Outside the repo, **synced to the Mac via `~/claude-sync` (cron every 30 min)**: `~/.claude/CLAUDE.md`, `~/.claude/settings.json`, `~/.claude/skills/{graphify,debug-issue,explore-codebase,refactor-safely,review-changes}`.

## What Worked
- `code-review-graph install --dry-run` before installing: it showed the conflicts (duplicate skills/hooks, CLAUDE.md injection) → final install with `--no-skills --no-instructions`, then project hooks removed by hand.
- `.code-review-graphignore` on `src/scenarios/*.yaml`: from 82k to 6k nodes. After the bloated build the SQLite DB was 322 MB: rebuilt from scratch (`rm -rf .code-review-graph && code-review-graph build`) → 27 MB.

## What Didn't Work / Watch Out
- `code-review-graph install` rewrites `.mcp.json` with `uvx` every time: put `command: code-review-graph` back.
- `graphify install` appends a `# graphify` section to the global CLAUDE.md: remove it after every graphify update; the skill description also gets overwritten (restrict it to docs again).
- `graphify-out/.graphify_python` was a Mac path (`/opt/anaconda3`): use `$(uv tool dir)/graphifyy/bin/python`.
- False alarm: `remote_executor` rsyncs only `src/`, so `.code-review-graph/` never travels to remote machines; no change needed.

## Next Steps
1. **Restart Claude Code**, then check `/mcp` (code-review-graph connected) and that pyright works.
2. **Build the graphify docs graph** (author: "as soon as possible"): `/graphify .` with the new `.graphifyignore` (61 `.md` files, ~100–150k tokens one-off; 0 Claude tokens if `GEMINI_API_KEY` is set). Today `graphify-out/graph.json` still holds the old **code** graph: do not use it.
3. **Rebuild the conda `cosim_gym` env** on this machine (missing; `conda` does not respond in non-interactive shells). `environment.yml` lacks `ray`, `pandapower`, `pandapipes`, `ijson`, `pytest` (inferred from imports, to verify). Then run pytest + the regression suite.
4. Commit the setup changes (only on the author's explicit ok). Remember on the Mac: `uv tool install pyright code-review-graph graphifyy` (pyright-lsp is now enabled in the synced settings).
5. Then resume the reset refactor: **F5** (parallel reset forwarding) and **F6** (closure) from `docs/changes_reports/model_reset_redesign_plan.md`. F0–F4 are committed (`225cb7f`). Open question for the author from F4: one `fmu_output/restart_<n>/` per reset → hundreds of directories; reuse one per instance?
6. 9 latent bugs found in the full code read (native memory `latent-bugs-codebase-read`) are not yet in `docs/KNOWN_ISSUES.md`: decide with the author.
