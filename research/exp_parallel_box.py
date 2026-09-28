#!/usr/bin/env python3
"""Эксперимент 17: сколько точек попадает в коробку габарита — база §28 против новой.

Самое прямое мерило «путь смотрит в стену»: на чистом прогоне в коробке
2.2 × 3.3 м вдоль пути точек быть не должно, и каждая точка там — это стена,
опора или оборудование, в которое коридор въехал. Кластеров из них может и не
собраться (находок 0), а коридор всё равно уже в стене.

Оба метода идут по одним и тем же кадрам записи подряд (шаг 1), каждый со своим
состоянием. Заодно — ошибка пути по рельсам (4–25 м и 27–40 м, дальние — только
где срезы согласны между собой) и число скачков стены у поезда больше 0.5 м
между соседними кадрами (у базы стена — форма контраста на 3 м, у новой —
отступ; сюда входят и законные смены сечения). Считается число точек в коробке
по дальности и доля кадров, где их больше MIN_POINTS (несколько точек — шум
двойного эха, §29).

    python exp_parallel_box.py --bags roundT_doubleT
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from rail_detection import bag_path, iter_frames
from rail_detection.contrast_gauge import ContrastGauge, path_at
from rail_detection.parallel_path import ParallelGauge
from rail_detection.tunnel_frame import rail_samples
from rail_detection.views import shape_x

warnings.filterwarnings("ignore", message=".*encountered in matmul")

DEV_RUNS = ["doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
            "squareT_platform_squareT_switch", "doubleT_obstacle"]
VALIDATION_RUN = "roundT_squareT_pressureGate_squareT"
BINS = [2.0, 40.0, 80.0, 120.0, 150.0]
MIN_POINTS = 20


def run(dataset, bag, max_frames=None):
    methods = {"base": ContrastGauge(smooth=True), "new": ParallelGauge()}
    rows = []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=1,
                                           max_frames=max_frames):
        row = {"idx": idx}
        rd, rx, _, _ = rail_samples(points)
        for name, g in methods.items():
            res = g.update(points, steps=1)
            if res is None:
                row[name] = None
                continue
            s = res["s"][res["inside"]]
            # стены у поезда (3 м): у базы — форма контраста, у новой — отступы
            if name == "base":
                wall = [float(shape_x(res["fit"], np.array([3.0]), side)[0])
                        for side in ("left", "right")]
            else:
                wall = list(res["track"]["curve"]["w0"])
            rail = {}
            for lo, hi, key in ((4, 25, "near"), (27, 40, "far")):
                m = (rd >= lo) & (rd <= hi)
                if m.sum() >= 2 and (hi < 30 or np.ptp(rx[m]) <= 0.3):
                    rail[key] = float(np.median(np.abs(rx[m] - path_at(res["path"], rd[m])[0])))
            row[name] = {"box": np.histogram(s, bins=BINS)[0].tolist(),
                         "reach": res["reach"], "raw": bool(res["clusters"]),
                         "conf": bool(res["confirmed"]), "wall": wall, "rail": rail}
        rows.append(row)
        print(f"\r  {bag}: кадр {idx}/{n_total}", end="", flush=True)
    print()
    return rows


def summarize(bag, rows):
    out = {"bag": bag, "frames": len(rows)}
    for name in ("base", "new"):
        rr = [r[name] for r in rows if r.get(name)]
        box = np.array([r["box"] for r in rr])
        out[name] = {
            "ok": len(rr),
            "frames_with_box": [int((box[:, k] >= MIN_POINTS).sum()) for k in range(box.shape[1])],
            "points_mean": [float(box[:, k].mean()) for k in range(box.shape[1])],
            "raw": int(sum(r["raw"] for r in rr)), "conf": int(sum(r["conf"] for r in rr)),
            "reach_med": float(np.median([r["reach"] for r in rr])),
        }
        if rr and "wall" in rr[0]:
            jumps = 0
            for a, b in zip(rows[:-1], rows[1:]):
                pa, pb = a.get(name), b.get(name)
                if pa and pb:
                    jumps += sum(1 for x, y in zip(pa["wall"], pb["wall"])
                                 if np.isfinite(x) and np.isfinite(y) and abs(x - y) > 0.5)
            out[name]["wall_jumps"] = int(jumps)
            for key in ("near", "far"):
                v = [r["rail"][key] for r in rr if key in r["rail"]]
                out[name][f"rail_{key}"] = float(np.median(v)) if v else float("nan")
                out[name][f"rail_{key}_p90"] = (float(np.percentile(v, 90)) if v
                                                else float("nan"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=DEV_RUNS)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--out", default="output/exp17_eval")
    a = p.parse_args()
    bags = [VALIDATION_RUN] if a.validate else a.bags
    if not a.validate and VALIDATION_RUN in bags:
        raise SystemExit(f"{VALIDATION_RUN} заморожен: только через --validate")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for bag in bags:
        rows = run(a.dataset, bag)
        s = summarize(bag, rows)
        with open(Path(a.out) / f"box_{bag}.json", "w") as f:
            json.dump({"summary": s, "rows": rows}, f)
        for name in ("base", "new"):
            q = s[name]
            print(f"{bag:36s} {name:4s} кадров со стеной в коробке по дальности "
                  f"{'/'.join(f'{int(a)}-{int(b)}' for a, b in zip(BINS[:-1], BINS[1:]))} м: "
                  f"{q['frames_with_box']}  сырых {q['raw']} подтв {q['conf']}  "
                  f"дальность {q['reach_med']:.0f} м  скачков стен {q.get('wall_jumps')}  "
                  f"рельсы {q.get('rail_near', float('nan')):.3f} / "
                  f"{q.get('rail_far', float('nan')):.3f} (p90 {q.get('rail_far_p90', float('nan')):.3f})")


if __name__ == "__main__":
    main()
