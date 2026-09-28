#!/usr/bin/env python3
"""Эксперимент 20: обёртка записей Last_synth_data для загрузчика.

В выгрузке `.db3` и `metadata.yaml` лежат в разных подпапках (`db3/`, `yaml/`),
а rosbags ждёт их в одной, и на exFAT диска T7 ссылок не сделать. Обёртка —
`output/last_synth/<conv_rR_aA>/`: ссылка на `.db3`, копии `metadata.yaml` и
`labels.json`. Дальше везде `--dataset output/last_synth`.

    python exp20_wrap.py
"""

import argparse
import os
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/Volumes/T7/Last_synth_data")
    p.add_argument("--out", default="output/last_synth")
    a = p.parse_args()
    for d in sorted(Path(a.src).glob("conv_*_obst_bags")):
        w = Path(a.out) / d.name.replace("_obst_bags", "")
        w.mkdir(parents=True, exist_ok=True)
        db = next((d / "db3").glob("*.db3"))
        link = w / db.name
        if not link.exists():
            os.symlink(db, link)
        shutil.copy2(d / "yaml" / "metadata.yaml", w / "metadata.yaml")
        shutil.copy2(d / "yaml" / "labels.json", w / "labels.json")
        print(f"  {w} -> {db}")


if __name__ == "__main__":
    main()
