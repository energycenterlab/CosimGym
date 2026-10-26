# Scenari da rivedere (rolling window)

Questi 6 scenari non partono più: `rolling_window: 10` passi agente × 60 s = 600 s,
ma il federate meteo ha `real_period: 900`. Dal refactor del reset (F2) `ScenarioManager`
rifiuta le lunghezze di reset che non sono multiple intere del `real_period` di ogni federate.

Fix possibile: `rolling_window` (e `period`/`episode_length`) tali che la durata sia multipla di 900 s
(es. 15 o 30 passi da 60 s). Non portare il meteo a 60 s.

Per rilanciarli, rimetterli in `src/scenarios/` dopo la correzione.
