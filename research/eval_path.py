#!/usr/bin/env python3
"""Эксперимент 16: насколько путь дёргается между кадрами и чего это стоит.

Путь строится в каждом кадре заново, и это видно на GIF: центр коридора
подрагивает там, где физически ничего не меняется — поезд за кадр проезжает
1.5 м, и путь под ним тот же самый. Здесь это измеряется тремя числами, и любое
изменение алгоритма обязано улучшить первое, не испортив второе и третье.

  дрожание  — главный замер. Путь ПРОШЛОГО кадра переносится в текущий кадр
              (поворот и сдвиг на измеренное Δs, `advance_path`) и сравнивается
              с путём текущего кадра на 5, 20, 40 и 80 м. Для идеально
              устойчивого метода это ноль; всё остальное — дрожание.
  рельсы    — цена в точности: медиана |путь − центр колеи| на рельсовых срезах
              4-25 м. Гладкий, но уехавший от рельсов путь не годится.
  находки   — цена в детекторе: сколько кадров дали кластер в габарите (сырых и
              подтверждённых). На чистых прогонах любая находка ложная, на
              doubleT_obstacle находки в кадрах 4-75 — это человек на путях.

Запуск:
    python eval_path.py                       # 4 чистых прогона + doubleT_obstacle
    python eval_path.py --validate            # отложенный прогон, один раз
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rail_detection import bag_path, iter_frames
from rail_detection.contrast_gauge import ContrastGauge, advance_path, path_at
from rail_detection.tunnel_frame import rail_samples

DEV_RUNS = ["doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
            "squareT_platform_squareT_switch", "doubleT_obstacle"]
VALIDATION_RUN = "roundT_squareT_pressureGate_squareT"
PROBE = (5.0, 20.0, 40.0, 80.0)
OBSTACLE_FRAMES = (4, 75)   # кадры doubleT_obstacle, где человек на путях (§25)


def run(dataset, bag, stride, smooth=False, rail_refine=False, history=5):
    cg = ContrastGauge(smooth=smooth, rail_refine=rail_refine, history=history)
    prev = None
    rows = []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=stride):
        res = cg.update(points, steps=stride)
        rec = {"idx": idx, "ok": res is not None}
        if res is not None:
            rd, rx, _, _ = rail_samples(points)
            m = (rd >= 4) & (rd <= 25)
            rec["rail_err"] = (float(np.median(np.abs(rx[m] - path_at(res["path"], rd[m])[0])))
                               if m.sum() >= 3 else None)
            rec["n_clusters"] = len(res["clusters"])
            rec["dist"] = res["clusters"][0]["dist"] if res["clusters"] else None
            rec["confirmed"] = res["confirmed"]
            rec["reach"] = res["reach"]
            rec["mode"] = res["path"]["mode"]
            shift = res["shift"]
            if prev is not None and shift is not None and np.isfinite(shift):
                moved = advance_path(prev, shift)
                a = path_at(moved, PROBE)[0]
                b = path_at(res["path"], PROBE)[0]
                rec["jump"] = np.abs(a - b).tolist()
            prev = res["path"]
        else:
            prev = None
        rows.append(rec)
        print(f"\r  {bag}: кадр {idx}/{n_total}", end="", flush=True)
    print()
    return rows


def summarize(bag, rows):
    jumps = np.array([r["jump"] for r in rows if r.get("jump") is not None], float)
    rail = np.array([r["rail_err"] for r in rows if r.get("rail_err") is not None], float)
    n = len(rows)
    raw = [r for r in rows if r.get("n_clusters")]
    conf = [r for r in rows if r.get("confirmed")]
    out = {"bag": bag, "frames": n, "ok": float(np.mean([r["ok"] for r in rows])),
           "raw": len(raw), "confirmed": len(conf),
           "rail_med": float(np.median(rail)) if len(rail) else float("nan"),
           "rail_p90": float(np.percentile(rail, 90)) if len(rail) else float("nan"),
           "pairs": len(jumps)}
    for k, D in enumerate(PROBE):
        out[f"jump{int(D)}_med"] = float(np.median(jumps[:, k])) if len(jumps) else float("nan")
        out[f"jump{int(D)}_p90"] = (float(np.percentile(jumps[:, k], 90)) if len(jumps)
                                    else float("nan"))
    if bag == "doubleT_obstacle":
        lo, hi = OBSTACLE_FRAMES
        inside = [r for r in rows if lo <= r["idx"] <= hi]
        out["person_frames"] = len(inside)
        out["person_found"] = sum(1 for r in inside if r.get("n_clusters"))
        out["person_confirmed"] = sum(1 for r in inside if r.get("confirmed"))
        out["outside_found"] = len(raw) - out["person_found"]
    return out


def print_table(rows):
    print(f"\n{'прогон':38s} {'кадров':>6s} {'есть':>5s} {'рельсы':>7s} "
          + " ".join(f"{'скач' + str(int(D)):>6s}" for D in PROBE)
          + f" {'p90@20':>7s} {'сырых':>6s} {'подтв':>6s}")
    for s in rows:
        print(f"{s['bag']:38s} {s['frames']:6d} {s['ok']:5.2f} {s['rail_med']:7.3f} "
              + " ".join(f"{s[f'jump{int(D)}_med']:6.3f}" for D in PROBE)
              + f" {s['jump20_p90']:7.3f} {s['raw']:6d} {s['confirmed']:6d}")
        if "person_found" in s:
            print(f"{'':38s} человек: найден в {s['person_found']}/{s['person_frames']} "
                  f"кадрах, подтверждён в {s['person_confirmed']}, "
                  f"вне отрезка с человеком {s['outside_found']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--tag", default="dev")
    p.add_argument("--smooth", nargs="?", const=True, default=False,
                   choices=[True, "bend"],
                   help="путь как состояние: без значения — медиана по всему пути, "
                        "bend — медиана только для изгиба за точкой привязки к рельсам")
    p.add_argument("--rail-refine", action="store_true",
                   help="пересобрать путь по головкам рельсов до 34 м")
    p.add_argument("--history", type=int, default=5, help="кадров в окне медианы пути")
    p.add_argument("--out", default="output/path_eval")
    a = p.parse_args()
    if a.validate:
        bags = [VALIDATION_RUN]
    else:
        bags = a.bags or DEV_RUNS
        if VALIDATION_RUN in bags:
            raise SystemExit(f"{VALIDATION_RUN} отложен: только через --validate")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = [summarize(b, run(a.dataset, b, a.stride, a.smooth, a.rail_refine, a.history))
             for b in bags]
    print_table(table)
    with open(out / f"summary_{a.tag}.json", "w") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)
    print(f"\nзаписано: {out}/summary_{a.tag}.json")


if __name__ == "__main__":
    main()
