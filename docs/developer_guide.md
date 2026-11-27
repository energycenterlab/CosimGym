# Developer Command Reference

Every command you need day to day, in one place. Run all of them **from the project root**.
Each section links to the page with the full explanation.

> `conda run -n cosim_gym <cmd>` runs a command inside the env without activating it — use it in
> scripts and one-off commands. In your own interactive terminal `conda activate cosim_gym` once
> and then drop the prefix.

---

## 1. Set up a machine (once)

```bash
git clone <repo> && cd CosimGym
conda env create -f environment.yml            # create the cosim_gym env
conda env update -f environment.yml --prune    # later: sync it after environment.yml changes
docker compose version                         # must say v2.x (v1 is not supported)
```

Ports (only if a default is taken on this machine — e.g. a shared server):

```bash
cp src/.env.example src/.env      # then edit the COSIM_*_PORT values; never commit src/.env
```

| Variable | Default | What |
|---|---|---|
| `COSIM_REDIS_PORT` | 6379 | Redis |
| `COSIM_MQTT_PORT` | 11884 | Mosquitto (streaming, interface federates, Live dashboard) |
| `COSIM_MINIO_PORT` / `COSIM_MINIO_CONSOLE_PORT` | 9000 / 9101 | MinIO API / web console |
| `COSIM_HELICS_PORT_MIN` / `_MAX` | 20000 / 30000 | range HELICS broker ports are drawn from |

Docker and Python both read `src/.env`; an exported environment variable wins over the file.
→ [Installation → Configuring ports](Installation_Setup.md#configuring-ports)

Distributed runs (and the regression suite's distributed scenarios) also need passwordless SSH,
including to this machine itself → [Adding a machine (checklist)](user_guide/distributed_deployment.md#adding-a-machine-checklist).

EnergyPlus FMU scenarios (`bui0_*`, `fmu_*`, and their regression-suite entries) also need the
`energyplus` executable (23.1.0) on `PATH`, otherwise they hang until timeout →
[Installation → pre-requisites](Installation_Setup.md#pre-requisites), [Known Issues §14](KNOWN_ISSUES.md).

EnergyPlus **Python-API** models (`bui0_eplus_*`, `eplus_horizon_smoketest`, `eplus_ec_pattern_test`) need an install with `pyenergyplus`
under `~/.cosimgym/energyplus/<any>/` or listed in `COSIMGYM_ENERGYPLUS_DIRS` (`:`-separated); upgraded IDFs are cached in
`COSIMGYM_EPLUS_CACHE` (default `~/.cosimgym/eplus_cache`). `helics_broker` must be on `PATH` (`export PATH=/opt/conda/envs/cosim_gym/bin:$PATH`
when running outside `conda run`). List what an IDF exposes / preview key patterns: `python src/models/model_catalog/eplus_discover.py --idf B.idf --epw W.epw [--io io.yaml]`
→ [EnergyPlus Models](user_guide/eplus_models.md).

## 2. Services (Docker)

`src/docker-compose.yaml` starts: **Redis** (config + model catalog store — required),
**catalog-loader** (one-shot: copies `catalog.yaml` into Redis), **Mosquitto** (MQTT),
**MinIO** + **minio-init** (FMU file store, creates the `fmus` bucket).

```bash
docker compose -f src/docker-compose.yaml up -d          # start (also reloads the catalog)
docker compose -f src/docker-compose.yaml ps             # status
docker compose -f src/docker-compose.yaml logs -f redis  # follow a service log
docker compose -f src/docker-compose.yaml stop           # pause (keeps containers + data)
docker compose -f src/docker-compose.yaml down           # remove containers, keep data
docker compose -f src/docker-compose.yaml down -v        # ⚠ also deletes Redis + MinIO data
```

After editing `catalog.yaml` with the services already up:

```bash
conda run -n cosim_gym python src/models/model_catalog/catalog_loader.py
```

## 3. Run scenarios

A scenario is `src/scenarios/<name>.yaml`; run it by name (no `.yaml`):

```bash
conda run -n cosim_gym python -c "import sys; sys.path.insert(0,'src'); from core.ScenarioManager import main; main('simple_test')"
```

`src/test_script.py` / `src/test_script_rl.py` run whichever `main('...')` line is uncommented in
the file (`make run` = `test_script.py`).

Results land in `results/<scenario>/<sim_id>/`, logs in `logs/<scenario>/<timestamp>/`.
Catalogues of what each scenario tests: `src/scenarios/scenario_tests.md`,
`src/scenarios/paper_casestudies.md`. → [Running a Scenario](user_guide/running_scenarios.md)

## 4. Dashboard

```bash
./src/dashboard/run_dashboard.sh     # http://localhost:8052 — Results + Live pages
make run-dashboard                   # same app on http://localhost:8501
ssh -L 8052:localhost:8052 <you>@<server>   # view a server's dashboard from your laptop
```

→ [Dashboard & Analytics](user_guide/dashboard.md)

## 5. Tests

| What | Command | Needs |
|---|---|---|
| Unit tests (fast feedback, ~7 min) | `conda run -n cosim_gym python -m pytest tests/ -q --ignore=tests/regression_suite.py` | nothing — Redis/HELICS/MQTT are mocked |
| One test file | `conda run -n cosim_gym python -m pytest tests/test_ports.py -q` | — |
| **Regression suite** (pre-merge gate) | `conda run -n cosim_gym python tests/regression_suite.py` | services up + SSH to `127.0.0.1` |
| Suite + real remote machines | `RUN_CLOUD=1 conda run -n cosim_gym python tests/regression_suite.py` | the cloud machines reachable |

The regression suite runs pytest, then ~44 scenarios (one per feature + feature combinations),
each **shortened on a temporary copy** (1 simulated hour, 2 RL episodes, auto ports) so your
YAML files are never touched. Reading the result:

- `PASS` / `FAIL` — expected to work; any `FAIL` makes the suite exit 1.
- `xfail` — a tracked bug from `KNOWN_FAIL` (listed in [Known Issues](KNOWN_ISSUES.md)); expected.
- `UNEXPECTED-PASS` — a tracked bug is fixed: delete it from `KNOWN_FAIL` **and** from Known Issues.

Green = `N passed, 0 FAILED, 8 xfail`. Add a scenario to its `SCENARIOS`/`COMBOS` lists
whenever you add a feature.

## 6. Make shortcuts

```bash
make help          # list targets
make setup         # create env + start services + validate
make validate      # check conda, docker, env, redis
make status        # env + container status
make clean         # docker compose down (keeps data)
make teardown      # ⚠ removes the cosim_gym env AND the Redis data volume
```

## 7. Docs

```bash
conda run -n cosim_gym mkdocs serve      # preview the documentation at http://127.0.0.1:8000
```

Where things go (rules in `CLAUDE.md`): user docs in `docs/`, bugs/limitations only in
[Known Issues](KNOWN_ISSUES.md), plans and change reports in `docs/changes_reports/`, future work
in `docs/future_and_TODOs/`.

## 8. When something breaks

| Symptom | Look at |
|---|---|
| `port already in use` | leftover brokers: `pkill -f helics_broker; pkill -f federate_launcher` → [Troubleshooting](user_guide/troubleshooting.md) |
| Federate cannot reach Redis | services up? `docker compose ... ps`; port in `src/.env` |
| Distributed run aborts at preflight | the [checklist](user_guide/distributed_deployment.md#adding-a-machine-checklist) checks |
| Known crash / limitation | [Known Issues](KNOWN_ISSUES.md) |
