# research/ — эксперименты

Скрипты, которыми выбирался и проверялся алгоритм: замеры, мерила, эталоны
«ответ тот же», GIF для отсмотра. Итоги каждого — в
[../docs/knowledge.md](../docs/knowledge.md) (§N), планы — в
[../docs/experiments/](../docs/experiments/README.md). Само решение от этой
папки не зависит.

## Как запускать

Все скрипты — из этой папки, с корнем репозитория в `PYTHONPATH`: они пишут и
читают относительно текущей папки (`results/`, `output/`, `*.json`) и
импортируют друг друга как соседей.

```bash
cd research
export PYTHONPATH=..
```

Нужен диск с записями `/Volumes/T7` (см. [../AGENTS.md](../AGENTS.md)); без него
работают только то, что читает кэш и эталоны в `output/`.

| что | где |
|---|---|
| `results/` | замеры экспериментов, разметка, разбиения, контрольные суммы замороженных эталонов (в git) |
| `output/` | кэши (`exp18_cache`, `exp21_cache`), эталоны сверки (`exp19_golden`, `exp22_golden`), обёртки записей (`new_synth`, `last_synth`), GIF (`Opus 5.5/`) — не в git, пересоздаются скриптами |
| `*.json` в этой папке | замороженные выборки ранних экспериментов (§15–§31) |

## Главное

```bash
# «ответ тот же» после любой правки конвейера — эталон текущего кода (§38–§39)
python exp_speed.py check --all --golden output/exp22_golden --variants low_rest_b0
python exp_speed.py check --all --golden output/exp22_golden --variants low_rest_b0 --workers 4 --process
#   между платформами (Mac ↔ Linux) — с --ignore alt_x (§39)
python exp_speed.py golden ...                 # новый эталон (до первой правки, заморозить суммами)
python exp_speed.py bench --label my --workers 4 --process    # время кадра по стадиям

# мерило обнаружения по кэшу: дальность, непрерывность, ложные (§34, §37)
python exp_far_eval.py --summary --variants low_rest_b0
python exp21_eval.py --variants low_rest_b0   # низкие предметы, все 32 записи

# GIF по записи — отсматривать самому до вывода о результате
RAIL_WORKERS=4 RAIL_PROCESS=1 python make_far_gifs.py --dataset /Volumes/T7/Dataset \
    --bags doubleT_obstacle --variant low_rest_b0 --out "output/Opus 5.5/check"

# зеркальная проверка (§32)
python exp_far_mirror.py --variant low_rest_b0

# numba против numpy бит в бит (§39)
python exp23_jit_check.py
```

Остальные скрипты и что они делают — [../AgentReadme.md](../AgentReadme.md),
раздел «Структура репозитория».
