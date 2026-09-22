#!/usr/bin/env python3
"""Эксперимент 17: кэш покадровых измерений для быстрой отладки подгонки.

Один проход по записи ПОДРЯД (шаг 1, как на поезде) текущим методом §28 и
сохранение того, из чего строится геометрия, — чтобы подгонку можно было гонять
десятки раз за секунды, а не за минуты чтения bag:

  img    — контраст вида сверху (views.contrast_image), uint8 0..255
  rails  — центры колеи по срезам (tunnel_frame.rail_samples) и колея
  shift  — Δs, измеренный базовым методом (продольное смещение к поперечной
           геометрии нечувствительно, §28)
  base   — выход базового метода: форма контраста и путь — для сравнения

Отложенный прогон сюда не попадает никогда.

    python exp_parallel_cache.py
    python exp_parallel_cache.py --bags roundT_doubleT
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from rail_detection import bag_path, iter_frames
from rail_detection.contrast_gauge import ContrastGauge
from rail_detection.tunnel_frame import rail_samples

DEV_RUNS = ["doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
            "squareT_platform_squareT_switch", "doubleT_obstacle"]
VALIDATION_RUN = "roundT_squareT_pressureGate_squareT"


def cache_run(dataset, bag, out_dir):
    cg = ContrastGauge(smooth=True)
    frames = []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=1):
        res = cg.update(points, steps=1)
        rd, rx, _, gauge = rail_samples(points)
        rec = {"idx": idx, "rail_d": rd.astype(np.float32), "rail_x": rx.astype(np.float32),
               "gauge": gauge}
        if res is not None:
            fit = res["fit"]
            rec.update({
                "img": np.round(res["silhouette"]["image"] * 255).astype(np.uint8),
                "shift": res["shift"],
                "floor": np.asarray(res["floor"], float),
                "base": {"shape": np.asarray(fit["shape"]), "offsets": list(fit["offsets"]),
                         "deg": fit["deg"], "reach": fit["reach"],
                         "reach_side": dict(fit["reach_side"]),
                         "path_x": res["path"]["x"].astype(np.float32),
                         "mode": res["path"]["mode"],
                         "n_clusters": len(res["clusters"]),
                         "confirmed": res["confirmed"]},
            })
        frames.append(rec)
        print(f"\r  {bag}: кадр {idx}/{n_total}", end="", flush=True)
    print()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{bag}.pkl", "wb") as f:
        pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=None)
    p.add_argument("--out", default="output/exp17_cache")
    a = p.parse_args()
    bags = a.bags or DEV_RUNS
    if VALIDATION_RUN in bags:
        raise SystemExit(f"{VALIDATION_RUN} заморожен (validation_run_v6.json)")
    for b in bags:
        cache_run(a.dataset, b, Path(a.out))


if __name__ == "__main__":
    main()
