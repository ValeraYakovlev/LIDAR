"""Эксперимент 20: разбиение данных и защита отложенных записей.

Разбиение — `results/exp20/split.json`, зафиксировано коммитом до первой правки.
Отложенные записи (и их зеркальные копии — имя то же) скрипты берут только с
флагом --holdout, на финальном замере.
"""

import json
from pathlib import Path

SPLIT = json.load(open(Path(__file__).resolve().parent / "results/exp20/split.json"))
HOLDOUT = sorted({b for bags in SPLIT["holdout"].values() for b in bags})
DEV_LAST_SYNTH = SPLIT["dev"]["last_synth"]
HOLDOUT_LAST_SYNTH = SPLIT["holdout"]["last_synth"]


def guard(bags, allow=False):
    """Останавливает скрипт, если среди записей есть отложенная, а --holdout не дан."""
    bad = [b for b in bags if b in HOLDOUT]
    if bad and not allow:
        raise SystemExit(f"{', '.join(bad)}: отложено (эксперимент 20, results/exp20/split.json) — "
                         "только на финальном замере, с --holdout")
