# Эксперимент 20 — выезд на двухпутный (knowledge.md §36)

| файл | что |
|---|---|
| `split.json` | разбиение: отложенные и разработка (заморожено коммитом 4f97e9c до первой правки) |
| `checksums_last_synth.sha256` | контрольные суммы записей Last_synth_data и `roundT_doubleT` (исходной и зеркальной) |
| `scenes.json` | пикеты сцен из README выгрузки: портал, «нос», слияние, рабочий |
| `dev_last_synth.json` / `.txt` | разработка: база против кандидата на 5 синтетических записях |
| `dev_regress_v2.json` | разработка: регрессия мерилом §34 (New_synth, старая синтетика, реальные) |
| `base_cache_eval.json` | база на кэше эксперимента 18 (прежний Δs) |
| `holdout_roundT_doubleT.json` | отложенный замер: кадры подтверждённых и сырых находок по вариантам |
| `holdout_last_synth.json`, `holdout_last_synth_mirror.json` | отложенный замер на синтетике и её зеркалах |

GIF (не в git): `output/Opus 5.5/double_track/` — `base` (база), `ds_moving`,
`ds_mov_med`, `v1` (новый Δs, пересечение), `v2` (кандидат, вся разработка),
`v1_mirror` / `v2_mirror` (зеркала), `holdout/{base,cand}` (отложенный замер).
Код отложенного замера — тег `double-track-v1`.
