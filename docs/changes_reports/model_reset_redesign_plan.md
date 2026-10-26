**STATUS: IN PROGRESS** — F0, F1, F2 DONE (2026-10-26). Next: F3 / F4 / F5 (independent), then F6.

# Piano — Obiettivo 1: refactor del reset (BaseModel → FMU)

## Context

Il refactor del reset (`208f9fb`) è a metà: `BaseModel.reset` è abstract e non implementato in ~10 modelli (26/32 scenari non partono), `BaseFMUModel` usa snapshot dopo `freeInstance`, referenzia attributi rimossi, calcola il rolling start in modo errato e passa uno stopTime incoerente con startTime > 0 (KNOWN_ISSUES §5, §7). In più: reset di episodio in tick per-federate con `real_period` diversi → federate resettano a istanti simulati diversi; `state.time` sovrascritto col tempo assoluto → feeder e FMU su date diverse dopo un rolling; `parallel_execution` non inoltra mai i reset ai worker.

Obiettivo: un reset generico in `BaseModel`, specializzato in `BaseFMUModel` (caso E+ dichiarato in catalogo), con un solo meccanismo semplice ("ripartire da zero alla data X"), meno codice di oggi.

## Decisioni prese (dall'autore)

| Tema | Scelta |
|---|---|
| Stato a inizio episodio rolling | Ripartenza da `init_state` alla data X (niente snapshot, niente replay) |
| Clock | Due grandezze: assoluta (tick federazione, monotona) e relativa (tempo del mondo rappresentato, si sposta coi reset) |
| Unità reset | YAML invariato (tick agente). ScenarioManager converte in secondi via `rl_period`, verifica divisibilità per ogni federate fisico → **errore** se non multiplo |
| Orizzonte (`max_sim_time`) | Restart per singolo modello + warning se orizzonti diversi |
| Meccanismo FMU | Un solo path: teardown + nuova istanza con startTime = X. Snapshot eliminato |
| E+ riconoscimento | Chiave catalogo `user_defined.fmu_engine: energyplus` (auto-discovery → followups) |
| E+ startTime | Secondi dal 1 gen dell'anno, qualunque istante; prima partenza = `start_time` dello scenario |
| E+ stopTime | start + span necessario (fino al prossimo reset o fine scenario), arrotondato per eccesso al giorno, limitato dalla fine RunPeriod |
| E+ fine RunPeriod | Wrap all'inizio RunPeriod (`sim_start_date`) |
| FMU generica startTime | Secondi relativi (`rel_ts·period`) |
| `BaseModel.reset` | Hook concreto no-op; via i `pass` e il modo `soft` |
| parallel_execution | Implementare inoltro reset ai worker |
| CSV | Riga derivata da `rel_ts`; `WeatherCSVReader` diventa sottoclasse di `BaseCSVReader` |
| `simulation_horizon` | Rimosso (chiave, metodo, campo, attributi) |
| Test | Via test stantii, un modulo nuovo, 2 scenari regression |
| Rolling oltre orizzonte | Rolling oltre l'orizzonte: il clock relativo **non** si ripiega (`rel_ts = k·W`); il backend usa la posizione ripiegata `horizon_pos_s` (modulo `max_sim_time` dall'origine). Cambiato in F1, vedi «Chiarimenti emersi in F1» |
| `rolling_window == period` | Restart comunque (continuità = `mode: none`) |
| CSV orizzonte | `max_sim_time` derivato = righe·period (entra nel warning) |

## Design

### 1. Clock in `BaseModel` (`src/models/base_model.py`)
- `state.ts` / `state.time`: assoluti (`start_time + ts·period`), monotoni. Già esistenti.
- `state.rel_ts` / **`state.rel_time`** (nuovo campo in `State`): tempo rappresentato, `start_time + rel_ts·period`. Modelli dipendenti dalla data (`bui0_input_feeder`, CSV) usano `rel_time`.
- Origine modello: `sim_start_date` (risolta nell'anno di `start_time`) se dichiarata, altrimenti `start_time`.
- Orizzonte generico, nessun codice E+: il clock relativo prosegue (dopo dicembre viene il gennaio successivo); `horizon_pos_s()` = clock ripiegato su [origine, origine+`max_sim_time`), usato dal backend (startTime FMU, riga CSV). Quando il clock entra in una nuova finestra (`_epoch` cambia) → `_reset('horizon')` che riporta il modello a `init_state` e riavvia il backend senza toccare `rel_ts` (per E+ = inizio RunPeriod).
- Rolling target: `rel_ts = k·W_ticks`, anche oltre l'orizzonte; lo ripiega il backend con `horizon_pos_s`.
- Un solo metodo `_reset(mode, rel_ts_target, span_ticks)`: `state = deepcopy(init_state)` (tranne `none`), imposta `rel_ts`, salva `span_ticks` (tick che l'istanza dovrà coprire, `None` = fino a fine scenario), chiama hook `reset(mode)`. Elimina `local_time(mode=...)` multi-ramo, blocchi commentati, `ts_shift`, `epoch_index`, args `ts/time` inutilizzati.
- `reset()` concreto no-op. Modi: `full`, `rolling`, `horizon` (interno). `none` = nessuna chiamata.

### 2. Federate (`src/core/BaseFederate.py`, `src/core/ScenarioManager.py`, `src/core/federate_launcher.py`)
- ScenarioManager (dopo `_create_RL_federation`, ha `rl_period`): `P_s = period·rl_period`, `W_s = window·rl_period`, `E_s = episode_length·rl_period`; per ogni federate fisico verifica `% real_period == 0` (errore chiaro altrimenti) e scrive in `federate.rl_config` i valori già in tick propri. `federate_launcher` smette di appiattire se `rl_config` è già presente (oppure il flatten si sposta interamente in ScenarioManager — un solo punto).
- `BaseFederate._reset`: `none` → return silenzioso (oggi warning "Unknown"); `full` → `rel_ts=0`; `rolling` → target k·W (modulare, vedi §1); **entrambi** chiamano `_publish_init_state`. Via `new_starting_point`. `_track_episodes` usa `E_ticks` convertito.
- **Span calcolato dal federate** (sa modo ed episodi): `span_ticks = P_ticks` se dopo questo reset ci sarà un altro reset in train (`reset_count < n_resets_train`), altrimenti `time_stop − ts` (fine scenario). Passato in `_reset(...)` e, all'avvio, nel model config (`initial_span_ticks`). Risolve il passaggio train→test: l'ultimo reset dà span fino a fine scenario. Niente `reset_period` nel modello.
- Rimuovi `_apply_simulation_horizon`, `_shortest_declared_horizon`, `FedTimingConfig.max_sim_time`, `ScenarioConfig.simulation_horizon`, `BaseFederate.max_sim_time/horizon_steps`, `_reset_on_horizon` commentato. Tieni `_report_simulation_horizons` (warning).

### 3. Parallel (`src/core/parallel_executor.py`)
- Nuovo messaggio `('reset', mode, rel_ts_target, span_ticks)` sulla pipe esistente; worker chiama `model._reset(...)` su ogni modello, risponde ack. `ParallelModelExecutor.reset(...)` speculare a `step()` (stesso error handling). `BaseFederate._reset` lo chiama quando `parallel_executor` attivo, oltre alle copie main (usate per `init_state`/storage). Orizzonte già funziona nei worker (dentro `_step`).

### 4. FMU (`src/models/base_FMU_model.py`)
- `reset(mode)` per `full|rolling|horizon`: `_teardown()` + `_start_instance()`. Fine.
- `_fmi_start_time()`: generica `rel_ts·period`; E+ secondi dal 1 gen dell'anno di `start_time` fino a `origine + horizon_pos_s()` (la posizione ripiegata, non `rel_time`: il clock non si ripiega, vedi «Chiarimenti emersi in F1»).
- `_fmi_stop_time()`: `span_s = span_ticks·period` (dal federate, mai calcolato dal modello). Generica = start + span_s. E+ = start + ceil(span_s/86400)·86400, cap a fine RunPeriod (origine+`max_sim_time`); se il cap non lascia giorni interi sufficienti → durata = giorni interi disponibili (se 0 → restart immediato a origine).
- **Istanza esaurita**: `BaseFMUModel` tiene `_instance_stop_ts`; se il prossimo step supererebbe lo stop dell'istanza (coda del RunPeriod dopo il cap, o span finito) → restart: a origine se in coda al RunPeriod (wrap), altrimenti nuova istanza da `rel_ts` corrente con nuovo span. Copre il gap fra stop arrotondato per difetto e orizzonte.
- `_do_step(t)` usa `_fmi_start_time()` corrente + passi fatti nell'istanza (un contatore `_steps_in_instance`), non `rel_ts` assoluto.
- Validazione in `initialize` per E+: `real_period ≥ 60`; start dentro [origine, origine+max).
- Elimina: `snapshot`, `initial_snapshot`, `snapshot_custom`, `_take_snapshot`, `_restore_snapshot`, `_free_*`, `_maybe_snapshot_*`, `_consume_*`, `_advance`, `_record_inputs`, `_push_inputs`, `_input_history`, `_replay_inputs`, `_resolve_snapshot_support`, `_can_snapshot`, `_warn_*` commentati, ramo snapshot in `_push_initial_state_to_fmu` (→ push di `init_state` sempre). `noSetFMUStatePriorToCurrentPoint` → sempre `True` (default fmpy, nessun rollback più). Rimuovi `supports_rollback` da `fmu_catalog_register.py`.
- Catalogo: `bui0_building_fmu` + entry E+ esistenti → `user_defined.fmu_engine: energyplus`; `fmu_catalog_register.py` accetta `--engine`.

### 5. CSV (`src/models/base_csv_reader.py`, `weather_csv_reader.py`)
- `step`: `row = (starting_row + rel_ts) % len(data)`; via `_row_idx`, logica rolling in `reset`, `_reposition_backend` commentato. `reset` = no-op (riga deriva dal clock) → nessun override. In `initialize`, se nessun `max_sim_time` dichiarato: `max_sim_time = len(data)·real_period` (wrap gestito dall'orizzonte generico, entra nel warning di `_report_simulation_horizons`, che va chiamato dopo `initialize`).
- `WeatherCSVReader(BaseCSVReader)`: tiene solo ciò che differisce (colonne/parametri); verificare a implementazione cosa è specifico.

### 6. Altri modelli
- Rimuovi `reset(): pass` da `simple_pid_controller`, `simple_building`, `simple_heatpump`, `bui0_input_feeder`. `rc_building`, `battery_dest` tengono l'override (stato interno) togliendo args inutili. `bui0_input_feeder` passa a `state.rel_time`.

## Test & docs
- Elimina `tests/test_fmu_state_snapshot.py`, `tests/test_model_local_clock.py`; aggiorna `tests/test_fmu_horizon_detection.py` (no `supports_rollback`).
- Nuovo `tests/test_model_reset.py`: clock abs/rel, wrap orizzonte con `sim_start_date`, conversione+errore unità (ScenarioManager helper), stop/start E+ (dummy, senza binario), reset nei worker (executor con modello dummy).
- `tests/regression_suite.py`: scenario E+ BUI0 rolling breve (verifica `.eio` `Environment` = data attesa) + combo parallel+reset RL. Rimuovi/aggiorna voci legate a `simulation_horizon`.
- Docs: `fmu_models.md` §3–4 (meccanica definitiva, `fmu_engine`), `rl.md` reset (unità in tick agente + vincolo divisibilità; W == P = restart, continuità = `mode: none`; wrap modulare), KNOWN_ISSUES §5 via la voce "always tears down", `general.md` (via `simulation_horizon`), `federate.md`, `custom_models.md` (hook `reset` + `rel_time`), KNOWN_ISSUES §5/§7 → eliminati se risolti; followups: aggiungi auto-discovery engine FMU; changes report `docs/changes_reports/model_reset_redesign_changes.md` a fine lavoro. Plan repo: `docs/changes_reports/model_reset_redesign_plan.md` (copia di questo, primo step dell'implementazione).

## Esecuzione a fasi (ognuna una sessione, ripartibile da zero)

Regole comuni a ogni fase:
- **Inizio**: leggi `HANDOFF.md` + `docs/changes_reports/model_reset_redesign_plan.md` (sezione della fase), poi solo i file elencati per la fase. Advisor (Opus) all'inizio e prima di chiudere.
- **Fine**: repo in stato coerente (pytest della fase verde, nessun import rotto), segna `STATUS: DONE` sulla fase nel plan doc, `graphify update .`, riscrivi `HANDOFF.md` (skill handoff) con fase completata + prossima fase. Proponi commit di fase; commit solo su ok esplicito.
- Una fase non anticipa lavoro della successiva. Se una fase sfora, si chiude a metà con HANDOFF preciso (file toccati, cosa manca).

| Fase | Contenuto | File | Fatto quando |
|---|---|---|---|
| **F0** setup — **DONE** | Copia del piano in `docs/changes_reports/model_reset_redesign_plan.md` con tabella fasi + STATUS; HANDOFF punta lì | plan doc, `HANDOFF.md` | file scritti |
| **F1** clock + reset base — **DONE** | §1 (State `rel_time`, origine, orizzonte generico, `_reset(mode, rel_ts_target, span_ticks)`, hook no-op); §6 (via i `pass`, feeder su `rel_time`, rc_building/battery firme); `BaseFederate._reset` minimo: `none` silenzioso, publish anche su rolling, target k·W in tick propri (ancora senza conversione); elimina `test_model_local_clock.py` | `base_model.py`, `BaseFederate.py` (solo `_reset`), modelli §6 | pytest verde esclusi test FMU; `rc_building_test_base` + uno scenario RL non-FMU partono; KNOWN_ISSUES §7 eliminato |
| **F2** unità + span + pulizia orizzonte scenario — **DONE** | §2: conversione/validazione in ScenarioManager, `rl_config` per federate, launcher, span dal federate, `_track_episodes`, rimozione `simulation_horizon` & co. Elenco scenari rifiutati → chiedere all'utente come correggerli | `ScenarioManager.py`, `federate_launcher.py`, `BaseFederate.py`, `config_dataclasses.py` | unit test conversione/errore; scenari RL esistenti validi partono; `general.md` aggiornato |
| **F3** CSV — TODO | §5: riga da `rel_ts`, `max_sim_time` derivato, `WeatherCSVReader(BaseCSVReader)` | `base_csv_reader.py`, `weather_csv_reader.py` | unit test cursore/wrap/rolling; scenario con weather parte |
| **F4** FMU — TODO | §4 completo + catalogo `fmu_engine` + `fmu_catalog_register.py --engine`; elimina `test_fmu_state_snapshot.py`, aggiorna `test_fmu_horizon_detection.py` | `base_FMU_model.py`, `catalog.yaml`, `fmu_catalog_register.py` | `bui0_fmu_test`, `fmu_horizon_smoketest`, smoke E+ rolling breve con `.eio` alla data attesa; `fmu_models.md` aggiornato |
| **F5** parallel — TODO | §3 | `parallel_executor.py`, `BaseFederate.py` | unit test executor reset con modello dummy |
| **F6** chiusura — TODO | `tests/test_model_reset.py` consolidato, 2 scenari regression (go-ahead prima della run completa), docs restanti, KNOWN_ISSUES §5, followups (auto-discovery engine), changes report, plan → DONE | tests, docs | gate regression verde (salvo KNOWN_FAIL) |

**Chiarimenti emersi in F1** (precisano il design, non riaprono decisioni):
- Convenzione `rel_time` = quella di `state.time`: durante uno step indica la *fine* dello step, `start + (rel_ts+1)·period` (con `rel_ts` 0-based, incrementato dopo `step()`); senza reset `rel_time == time`. Subito dopo un reset a `T`: `rel_time = start + T·period`. Per F4: lo startTime FMU è `start + rel_ts·period` (inizio step), non `rel_time`.
- `_reset` non tocca mai `state.ts/time` (assoluti): li salva e li ripristina dopo il deepcopy di `init_state`.
- **Il clock relativo non si ripiega mai sull'orizzonte** (scelta dell'autore, dopo F1 iniziale): al superamento di `max_sim_time` il modello si riavvia all'origine (E+: inizio RunPeriod) ma `rel_ts`/`rel_time` proseguono nel mondo rappresentato (dicembre → gennaio successivo); il backend legge la posizione `horizon_pos_s(rel_ts) = (rel_ts·period − origine_s) mod max_sim_time`. Niente `rel_ts` negativo, niente formula di wrap sul rolling (`rel_ts = k·W` grezzo). Il rilevamento è un contatore d'epoca `floor((rel_ts·period − origine_s)/max)`, ricalcolato a ogni `_reset`. Per F3: riga CSV = `(starting_row + rel_ts) % len`; per F4: startTime FMU = `horizon_pos_s()` (E+: origine + pos, in secondi dal 1 gen), stop cap = fine finestra. Effetto collaterale accettato: `rel_time.year` avanza mentre il modello ripete i dati del suo anno, quindi il giorno della settimana può differire fra modello e feeder dopo il wrap.
- Modo interno rinominato `horizon_limit` → `horizon` (anche nei confronti di `base_csv_reader.py` e `base_FMU_model.py`, 1 riga ciascuno; il resto di quei file è F3/F4). Firme `reset(self, mode, ts=None, time=None)` di CSV/FMU invariate (compatibili con la chiamata `reset(mode)`).
- Attributi `reset_mode/rolling_window/episode_length/reset_period/n_episodes` restano su `BaseModel` fino a F2/F4 (li leggono FMU e CSV).
- Verifica F1: pytest senza nuovi fallimenti (da 25 a 11: restano i 9 di `test_fmu_state_snapshot.py`, che sparisce in F4, e i 2 di KNOWN_ISSUES §6); scenari `rc_building_test_base`, `bui_hp_DQN_rollingreset`, `simple_DQN_test`, `bui0_fmu_test`, `fmu_horizon_smoketest` partono e passano (copie accorciate della suite); con `log_level: INFO` si vedono 2 rolling reset (tick relativi 10, 20) in `bui_hp_DQN_rollingreset` e il restart d'orizzonte nell'FMU di `fmu_horizon_smoketest`. Tutti i 27 modelli del catalogo importano e non sono abstract. `base_FMU_model.py` usa già `horizon_pos_s()` al posto di `rel_ts*real_period` (4 punti: setupExperiment, enterInitializationMode, initialize, `_do_step`) perché il clock non si ripiega più.
- Ambiente: i compose images `redis:8.2.10-alpine` / `minio/minio` non erano scaricabili (pull denied) in questa sessione; per i test scenario è stato usato un `redis:latest` locale (8.10) con `catalog_loader.py`, poi rimosso.

**Fatto in F2** (precisazioni per le fasi successive):
- `ScenarioManager.reset_in_federate_ticks` (funzione di modulo, testata) + `_assign_federate_rl_configs()` (fine di `_scenario_setup_timing_vars`): scrive `federate.rl_config = {mode:'train', reset_type, n_episodes, episode_length, reset_period, rolling_window}` già nei tick del federate; solo con training online (`run.train` e `run.mode != 'offline'`); salta `rl_federation`; valida `episode_length` sempre, `reset_period` se mode ≠ none, `rolling_window` se rolling. Errore = `ValueError` loggato come `Scenario error: ...` (lo scenario non parte; `main()` non esce con codice ≠ 0, comportamento preesistente). `federate_launcher` non appiattisce più: usa `rl_config` o `{}`.
- Span: `BaseFederate._next_span()` = `reset_length` se restano reset in training (`n_resets_train = n_episodes·E_f // P_f`, l'ultimo cade al confine train/test), altrimenti `stop_time − ts` (anche con `reset: none` o senza RL: tutto lo scenario). Il valore iniziale va in `model_configs.initial_span_ticks` → `BaseModel.span_ticks`; i successivi in `_reset(..., span_ticks=...)`. Mai 0: `_reset` non scatta se `ts >= stop_time`. Per F4: `span_ticks` è in tick del modello, `None` solo se il modello è costruito fuori dal federate.
- Rimossi: `_apply_simulation_horizon`, `_shortest_declared_horizon`, `_model_catalog`, `FedTimingConfig.max_sim_time`, `ScenarioConfig.simulation_horizon`, `BaseFederate.max_sim_time/horizon_steps/horizon_count`, blocco commentato `_reset_on_horizon`. Un YAML con la chiave `simulation_horizon` è ignorato senza errore (`ScenarioConfig` ha `extra='ignore'`; nessuno scenario la usa). La suite vede il rifiuto: `run_scenario('cs_s2_reset_rolling')` → `(False, …, 'Scenario error: …')`.
- **Scenari rifiutati** (sweep su tutti gli YAML RL, versione intera): `cs_s2_reset_rolling{,_s42,_s43,_s44}`, `cs_s4a_dqn_s42_rolling`, `cs_s4b_sac_s42_rolling` — `rolling_window: 10` passi da 60 s = 600 s, non multiplo del `real_period: 900` del meteo. Gli altri `cs_*` sono validi interi ma le copie accorciate di `regression_suite._shorten` (episode_length 20 → 1200 s) sarebbero rifiutate; i `cs_*` non sono nella suite. Tutti gli altri scenari RL (bui*, pv_batt, simple_*, smoke) validi anche accorciati. YAML e `_shorten` NON modificati: decide l'autore (es. window 15 → 900 s).
- Verifica: 310 passati / 11 falliti (gli stessi di F1); scenari `rc_building_test_base`, `simple_DQN_test`, `bui0_setpoint_DQN`, `fmu_horizon_smoketest`, `bui_hp_DQN_rollingreset` passano; `cs_s2_reset_rolling` con episode 30/window 15 gira e il federate meteo (900 s) resetta al tick relativo 1, 2 mentre gli altri a 15, 30; l'originale è rifiutato con messaggio chiaro.

Dipendenze: F1 → F2 → (F3, F4, F5 indipendenti fra loro) → F6.

### Avvio di una fase in sessione nuova

Prompt: *"Leggi HANDOFF.md e docs/changes_reports/model_reset_redesign_plan.md ed esegui la fase F<n>."*
Prerequisiti: `docker compose -f src/docker-compose.yaml up -d`, env `cosim_gym`. Un singolo
scenario si prova con `conda run -n cosim_gym python -c "import sys; sys.path.insert(0,'tests'); import regression_suite as r; print(r.run_scenario('<nome>'))"` (copia accorciata, originale intatto).
Dopo modifiche a `catalog.yaml`: `python src/models/model_catalog/catalog_loader.py` (ricarica in Redis).

### Mappa del codice (verificata 2026-10-26, controllare che le righe non siano cambiate)

| Cosa | Dove |
|---|---|
| Clock attuale, `local_time(mode=…)`, `_reset`, `reset` abstract | `src/models/base_model.py` ~326–461 (`State` in cima al file) |
| Config modello iniettata dal federate (`reset_mode`, `rolling_window`, `episode_length`, `reset_period`, `n_episodes`) | `BaseFederate.py` ~350–365; letti in `base_model.py` ~128–143 |
| Reset di episodio del federate, `new_starting_point`, `_track_episodes` | `BaseFederate._reset` ~1085, chiamato nel loop `run` ~584; `_track_episodes` ~1143; init attributi RL ~133–150 |
| Publish stato iniziale | `BaseFederate._publish_init_state` ~973 |
| Warning orizzonti diversi | `BaseFederate._report_simulation_horizons` ~398 (chiamato ~395) |
| Flatten `rl_config` per federate non-RL | `src/core/federate_launcher.py` ~136–150 |
| `rl_period` = min periodo dei modelli controllati | `ScenarioManager._create_RL_federation` ~727, `_get_rl_period` |
| Timing federate, `time_stop`, `min_real_period` | `ScenarioManager` ~1440–1466 |
| Da rimuovere: `_apply_simulation_horizon`, `_shortest_declared_horizon` | `ScenarioManager` ~1466–1530; `FedTimingConfig.max_sim_time` `config_dataclasses.py` ~72; `ScenarioConfig.simulation_horizon` ~773; `BaseFederate.max_sim_time/horizon_steps` ~120–127 |
| Schema reset YAML (`mode` full/rolling/none, `period`, `rolling_window`, `force_defaults`) | `config_dataclasses.ResetConfig` ~331 |
| Reset lato agente (osservazioni, default) | `RL_Federate.reset_episode` ~699 — non va toccato |
| Worker parallel: loop messaggi `(ts, in_shard)` / `None` sentinel | `src/core/parallel_executor.py` `_worker_loop` ~48, `ParallelModelExecutor.step` ~191; rifiuto combinazioni `BaseFederate._setup_parallel_executor` ~174 |
| FMU: start/stop/do_step, workdir `restart_<n>` | `src/models/base_FMU_model.py` (`_setup_experiment`, `_enter_initialization_mode`, `_stop_time_seconds`, `_do_step`, `_resolve_fmu_workdir`) |
| E+ BUI0: `generationTool="idf-to-fmu-export-prep-linux"`, `canGetAndSetFMUstate=false`, step 600 s, RunPeriod 1/1–31/12 | `physical_models/resources/BUI0.fmu`, `BUI0.idf`; entry catalogo `bui0_building_fmu` (`max_sim_time: 31536000`, `sim_start_date: '01-01'`) |
| Register script (`supports_rollback`, rilevamento orizzonte) | `src/models/model_catalog/fmu_catalog_register.py` ~93–265, ~450–510 |

Modelli e reset oggi: override reale in `rc_building` (`t_m_prev`), `battery_dest` (`_model.setSOC`), `base_csv_reader`, `weather_csv_reader`; `reset(): pass` in `simple_pid_controller`, `simple_building`, `simple_heatpump`, `bui0_input_feeder`; **nessun** `reset` (quindi non istanziabili) in `exchange_dummy`, `heavy_compute_dummy`, `inputs4spring`, `light_compute_dummy`, `pandapipes_grid`, `pandapower_grid`, `pv_dest`, `rb_bems`, `spring_mass_damper`, `test_input_model`. (L'elenco in KNOWN_ISSUES §7 include per errore i tre `simple_*`.) `bui0_input_feeder` usa `state.time` per orari/stagioni → passa a `rel_time`.

Scenari utili per fase: F1 `rc_building_test_base` (base), `bui_hp_DQN_rollingreset` (RL rolling, CSV + simple_building/heatpump, no FMU), `simple_DQN_test` (RL, spring_mass_damper). F2: `cs_s2_reset_rolling` (federate 900 s + 60 s, window 10 → atteso **rifiuto**: usare come test dell'errore). F3: `bui_hp_DQN_rollingreset`, `cs_s2_*` dopo correzione. F4: `bui0_fmu_test`, `fmu_horizon_smoketest`, `bui0_setpoint_DQN`.

## Rischi / da verificare in implementazione
- Scenari esistenti (`cs_s2_*`, meteo 900 s, window 10 a 60 s) verranno rifiutati dal controllo divisibilità → vanno corretti (elenco da produrre).
- Ogni restart FMU crea `fmu_output/restart_<n>/`: con centinaia di episodi molte cartelle. Proposta: riusare una cartella per istanza modello (decidere in implementazione, chiedere se costoso).
- Discontinuità al restart non segnalata all'agente (followups §2) — fuori perimetro.

## Verifica
- `conda run -n cosim_gym pytest tests/` verde (salvo noti §6 KNOWN_ISSUES).
- `conda run -n cosim_gym python tests/regression_suite.py` (richiede docker + ssh localhost; chiedere ok prima, run lunga).
- Scenario E+ rolling: controllare `.eio` `Environment` line e `TBuilding` coerente con la data attesa a ogni episodio.
