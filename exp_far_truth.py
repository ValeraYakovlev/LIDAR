#!/usr/bin/env python3
"""Эксперимент 18: где стоят предметы 1–5 на New_synth_data и граница S_B.

Разметки в записи нет, есть порядок предметов от разработчиков (план 18).
Положение предмета — по длине пути: S(k) + дальность, где S(k) — сумма Δs
трекера эталона. Предметы размечены ПО ПОРЯДКУ, окно за окном (`--list`):
предмет k+1 — только в окне за предметом k; дальше пятого не смотрел.
S_B = s_5 + 50 м, всё, что дальше, — отложенная часть. Как найден каждый
предмет — поле `how` в OBJECTS.

Скопления собираются вблизи (4–40 м), где точек на предмете много и путь
точен, в коробке шире габарита (|u| <= 2.0 м, 0.08–3.8 м над головками — выше
головок рельсов, ниже свода); компактные — не длиннее 2.5 м вдоль пути.

    python exp_far_truth.py --list 0 170 1.3   # скопления в окне длины пути, |u| центра <= 1.3
    python exp_far_truth.py                    # -> results/exp18/new_synth_objects.json
"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

from exp_far_cache import load

CACHE = "output/exp18_cache/new_synth"
BAG = "cloud_with_fake_obj"
OUT = Path("results/exp18/new_synth_objects.json")
MARGIN = 50.0

TYPES = ["2 × 2 м посередине габарита", "0.3 × 0.3 м посередине габарита",
         "0.3 × 0.3 м на рельсах", "0.3 × 0.3 м с краю габарита",
         "0.3 × 0.3 м за пределами габарита, близко"]
POSITIVE = [True, True, True, True, False]


def track_length(cols):
    """S(k): длина пути, пройденная к кадру k, — сумма Δs трекера. Кадр без
    геометрии берёт последнее известное Δs (поезд не останавливается мгновенно)."""
    ds = cols["ds"].astype(float).copy()
    last = 0.0
    for k in range(len(ds)):
        if np.isfinite(ds[k]):
            last = ds[k]
        else:
            ds[k] = last
    ds[0] = 0.0
    return np.cumsum(ds)


def candidates(cols, pts, S):
    rows = []
    for k in range(len(S)):
        if not cols["ok"][k]:
            continue
        s, u, v = pts(k)
        m = (np.abs(u) <= 2.0) & (v >= 0.08) & (v <= 3.8) & (s >= 4.0) & (s <= 40.0)
        if m.sum() < 5:
            continue
        P = np.column_stack([s[m], u[m], v[m]])
        lab = DBSCAN(eps=0.3, min_samples=3).fit_predict(P)
        for L in np.unique(lab[lab >= 0]):
            c = P[lab == L]
            # предмет компактен; стены, контактный рельс и кабели тянутся вдоль
            # пути на десятки метров (первая попытка поиска нашла именно их)
            if len(c) < 5 or np.ptp(c[:, 0]) > 2.5:
                continue
            rows.append({"k": k, "s_glob": float(S[k] + c[:, 0].min()), "dist": float(c[:, 0].min()),
                         "n": len(c), "u": (float(c[:, 1].min()), float(c[:, 1].max())),
                         "v": (float(c[:, 2].min()), float(c[:, 2].max()))})
    return rows


def groups(rows, lo, hi, umax):
    """Скопления в окне [lo, hi] длины пути, собранные по кадрам: одно место
    тоннеля (±3 м вдоль, ±0.6 м поперёк, ±0.8 м по высоте) — одна строка."""
    sel = sorted((r for r in rows if lo <= r["s_glob"] <= hi), key=lambda r: r["s_glob"])
    out = []
    for r in sel:
        uc, vc = sum(r["u"]) / 2, sum(r["v"]) / 2
        for g in out:
            if abs(g["s"] - r["s_glob"]) < 3 and abs(g["uc"] - uc) < 0.6 and abs(g["vc"] - vc) < 0.8:
                g["rows"].append(r)
                break
        else:
            out.append({"s": r["s_glob"], "uc": uc, "vc": vc, "rows": [r]})
    res = []
    for g in out:
        R = g["rows"]
        ks = sorted({r["k"] for r in R})
        if len(ks) < 3 or abs(g["uc"]) > umax:
            continue
        res.append({"s": float(np.median([r["s_glob"] for r in R])), "frames": [ks[0], ks[-1]],
                    "n_frames": len(ks),
                    "u": [float(np.median([r["u"][0] for r in R])), float(np.median([r["u"][1] for r in R]))],
                    "v": [float(np.median([r["v"][0] for r in R])), float(np.median([r["v"][1] for r in R]))],
                    "n": int(np.median([r["n"] for r in R]))})
    return res


# Разметка по спискам `--list` (2026-09-24), окно за окном; дальше предмета 5
# не смотрел. Поиск предметов 1–4 — у пути (|u| центра <= 1.3 м): при |u| >= 1.35
# тоннель полон повторяющихся элементов (секции контактного рельса через ~3.5 м
# на u +1.36…+1.6, кронштейны, свод), первая автоматическая попытка нашла
# именно их. Окно [s_k + 50, s_k + 150] пришлось расширить до +250 для
# предмета 2: между 1 и 2 — 204 м, а не «примерно 100». Предмет 5 — единственное
# неповторяющееся скопление во всю ширину коробки в окне [557, 657].
OBJECTS = [
    dict(n=1, s_obj=103.0, frames=[185, 223], u=[-0.85, 1.12], v=[0.10, 1.76],
         how="--list 0 170 1.3: единственное скопление у пути, ~975 вокселей"),
    dict(n=2, s_obj=307.2, frames=[347, 367], u=[0.24, 0.53], v=[1.07, 1.33],
         how="--list 153 353 1.3 (окно 153–253 пусто)"),
    dict(n=3, s_obj=408.0, frames=[407, 424], u=[0.73, 1.01], v=[0.18, 0.35],
         how="--list 357 557 1.3: на правом рельсе (u = +0.76)"),
    dict(n=4, s_obj=507.1, frames=[460, 478], u=[1.02, 1.30], v=[1.05, 1.31],
         how="--list 357 557 1.3 (виден в окне предмета 3), окно 458–658 — он же"),
    dict(n=5, s_obj=607.1, frames=[513, 532], u=[-1.35, -1.03], v=[1.06, 1.31],
         how="--list 557 657 2.0: остальное в окне — контактный рельс и стена"),
]


def main():
    import sys
    if len(sys.argv) == 5 and sys.argv[1] == "--list":
        cols, pts = load(CACHE, BAG)
        rows = candidates(cols, pts, track_length(cols))
        for g in groups(rows, float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])):
            print(f"s={g['s']:6.1f}  кадров {g['n_frames']:3d} ({g['frames'][0]}–{g['frames'][1]})  "
                  f"u {g['u'][0]:+.2f}…{g['u'][1]:+.2f}  v {g['v'][0]:.2f}…{g['v'][1]:.2f}  n~{g['n']}")
        return
    objs = [{**o, "type": TYPES[o["n"] - 1], "positive": POSITIVE[o["n"] - 1]} for o in OBJECTS]
    S_B = objs[-1]["s_obj"] + MARGIN
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"bag": BAG, "S_B": S_B, "objects": objs,
                   "note": "S(k) — сумма Δs трекера эталона wall-parallel-v1; "
                           "всё с длиной пути >= S_B — отложенная часть"},
                  f, ensure_ascii=False, indent=1)
    for o in objs:
        print(f"предмет {o['n']} ({o['type']}): s = {o['s_obj']:.1f} м, u {o['u'][0]:+.2f}…{o['u'][1]:+.2f}, "
              f"v {o['v'][0]:.2f}…{o['v'][1]:.2f}")
    print(f"S_B = {S_B:.1f} м")


if __name__ == "__main__":
    main()
