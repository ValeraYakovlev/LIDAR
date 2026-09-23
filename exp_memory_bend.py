#!/usr/bin/env python3
"""Память держит изгиб на 70–80 м: замер того, что заметили глазами на GIF (§32).

На каждом кадре WallParallelTracker строит две гипотезы — «память» (решение
прошлого кадра, перенесённое на Δs, с приором) и «заново» (путь этого кадра без
памяти) — и берёт меньшую цену с гистерезисом PRIOR_CAP. Здесь обе гипотезы
перехватываются в момент выбора (обёртка над `parallel_path.total_cost`,
алгоритм не меняется) и сравниваются на 75 м:

  расхождение — поперечная разность путей гипотез на 75 м по длине пути; со
                знаком поворота (+ = память загнута сильнее в сторону поворота);
  кто лучше   — средняя |невязка| кромок стен на 60–90 м (обрезанная на 1 м)
                у каждой гипотезы; меньше — путь лучше ложится на стены.

Систематическая ошибка видна не в среднем (медиана расхождения — сантиметры), а
там, где гипотезы расходятся больше чем на 0.3 м: у wall-parallel-v1 память там
выбирается почти всегда, хотя в трёх записях из четырёх её путь ложится на стены
60–90 м хуже.

Работает на кэше (`exp_parallel_cache.py`); отложенный прогон туда не попадает.

    python exp_memory_bend.py
    python exp_memory_bend.py --bags roundT_doubleT
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

import rail_detection.parallel_path as pp
from exp_parallel_dev import DEV_RUNS, run_method

warnings.filterwarnings("ignore", message=".*encountered in matmul")

PROBE = 75.0             # м по длине пути
BAND = (60.0, 90.0)      # м: где сравнивается согласие с кромками
DIVERGE = 0.3            # м: гипотезы разошлись
STRAIGHT_R = 3000.0      # м: радиус, выше которого участок считается прямым


def capture():
    """Обёртка над total_cost: запоминает обе гипотезы кадра в порядке оценки
    (сначала «память», если она есть, затем «заново»)."""
    seen = []
    real = pp.total_cost

    def spy(st, m, edges, rails, prior_m, prior_sig):
        c = pp.track_curve(st, m)
        ed, ex, es, _ = edges
        s, u = pp.project(c, ex, ed)
        hid = pp.hidden_edges(c, ed, es)
        r = u - pp._wall_at(st, m, s, es)
        band = (~hid) & (s > BAND[0]) & (s < BAND[1])
        mid = (c["s"] > 40) & (c["s"] < 120)
        seen.append({"x": float(np.interp(PROBE, c["s"], c["x"])),
                     "psi": float(np.interp(PROBE, c["s"], c["psi"])),
                     "kappa": float(np.mean(c["kappa"][mid])),
                     "fit": float(np.mean(np.minimum(np.abs(r[band]), 1.0))) if band.any()
                     else float("nan")})
        return real(st, m, edges, rails, prior_m, prior_sig)

    pp.total_cost = spy
    return seen, real


def measure(frames):
    seen, real = capture()
    try:
        outs, _ = run_method(frames, "new")
    finally:
        pp.total_cost = real
    rows, i = [], 0
    for k, o in enumerate(outs):
        if o is None or "res" not in o:
            continue
        # на первом кадре и после сброса памяти нет — одна гипотеза
        n = 2 if k > 0 and outs[k - 1] is not None else 1
        cand, i = seen[i:i + n], i + n
        if n < 2:
            continue
        mem, fresh = cand
        turn = np.sign(fresh["kappa"]) if abs(fresh["kappa"]) > 1 / STRAIGHT_R else 0.0
        dx = (mem["x"] - fresh["x"]) * np.cos(fresh["psi"]) * (turn if turn else 1.0)
        rows.append({"idx": k, "diff": float(dx), "curve": bool(turn),
                     "fit_mem": mem["fit"], "fit_fresh": fresh["fit"],
                     "chosen": o["res"]["origin"]})
    return rows


def summarize(bag, rows):
    d = np.array([r["diff"] for r in rows])
    curve = np.array([r["curve"] for r in rows])
    big = np.abs(d) > DIVERGE
    mem_big = [r for r, b in zip(rows, big) if b and r["chosen"] == "память"]
    out = {"bag": bag, "pairs": len(rows), "curve_pairs": int(curve.sum()),
           "divergent": int(big.sum()), "divergent_memory_chosen": len(mem_big),
           "fit_memory": float(np.nanmean([r["fit_mem"] for r, b in zip(rows, big) if b]))
           if big.any() else None,
           "fit_fresh": float(np.nanmean([r["fit_fresh"] for r, b in zip(rows, big) if b]))
           if big.any() else None}
    if curve.any():
        out["curve_diff"] = {q: float(np.percentile(d[curve], p))
                             for q, p in (("p10", 10), ("median", 50), ("p90", 90))}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bags", nargs="*", default=DEV_RUNS)
    p.add_argument("--cache", default="output/exp17_cache")
    p.add_argument("--out", default="output/exp17_eval/memory_bend.json")
    a = p.parse_args()
    table = []
    for bag in a.bags:
        frames = pickle.load(open(Path(a.cache) / f"{bag}.pkl", "rb"))
        s = summarize(bag, measure(frames))
        table.append(s)
        line = f"{bag}: пар {s['pairs']}, на кривых {s['curve_pairs']}"
        if "curve_diff" in s:
            q = s["curve_diff"]
            line += (f"; на кривых память минус заново на {PROBE:.0f} м: медиана "
                     f"{q['median']:+.2f}, p10 {q['p10']:+.2f}, p90 {q['p90']:+.2f} м")
        print(line)
        if s["divergent"]:
            print(f"   расходятся > {DIVERGE} м: {s['divergent']} кадров, память выбрана в "
                  f"{s['divergent_memory_chosen']}; |невязка| кромок на {BAND[0]:.0f}–"
                  f"{BAND[1]:.0f} м: память {s['fit_memory']:.3f}, заново {s['fit_fresh']:.3f} м",
                  flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(table, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
