#!/usr/bin/env python3
"""Эксперимент 17: замер на кэше (exp_parallel_cache.py) — база §28 против новой модели.

Одно и то же мерило для обоих методов:

  дрож. стен — стены ПРОШЛОГО кадра переносятся в текущий на измеренное Δs
               (`contrast_gauge.advance_path`, тот же перенос, что в §28) и
               сравниваются с текущими на 20, 40, 60, 80, 100 м. Меряется только
               там, где стена наблюдалась в обоих кадрах.
  дрож. пути — то же для пути (§28).
  стена в коридоре — сколько кромок контраста, извлечённых ОДНИМ И ТЕМ ЖЕ
               способом для обоих методов (как в §26, без знания о пути),
               оказались ближе 1.1 м к пути по нормали, то есть внутри
               габарита: «путь смотрит в стену». На чистом прогоне таких быть
               не должно. Кромка считается, только если рядом (±5 м вдоль) есть
               ещё хотя бы две такие же — одиночный выброс не в счёт. Кромки на
               линии взгляда (внутренняя стена поворота за точкой касания)
               отбрасываются по углу, без модели (sight_lines): путь за 80-110 м
               на кривой R = 400-600 м законно проходит за этой линией.
  рельсы     — медиана |путь − центр колеи| на 4–25 м (срезы, по которым
               подгоняется новая модель — это проверка, что она их не теряет) и
               на 27–40 м (НЕ участвуют в подгонке ни у одного метода) — только
               на кадрах, где дальние срезы согласны между собой в пределах
               0.3 м: на двухпутном участке детектор там хватает то свою, то
               чужую пару, и такой «эталон» портит оба метода одинаково.

Отложенный прогон в кэш не попадает и здесь не мерится.

    python exp_parallel_dev.py
    python exp_parallel_dev.py --bags roundT_doubleT --frames 0 120
"""

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np

from rail_detection.contrast_gauge import PATH_GRID, advance_path, build_path, path_at
from rail_detection.parallel_path import (WallParallelTracker, edges_from_bands,
                                          offset_curve, to_path_dict)
from rail_detection.views import (CELL_D, CELL_X, D_MAX, D_MIN, X_HALF, band_edges,
                                  shape_x)

warnings.filterwarnings("ignore", message=".*encountered in matmul")

DEV_RUNS = ["doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
            "squareT_platform_squareT_switch", "doubleT_obstacle"]
PROBE = (20.0, 40.0, 60.0, 80.0, 100.0)
CORRIDOR = 1.10


def make_grid():
    nx = int(round(2 * X_HALF / CELL_X))
    nd = int(round((D_MAX - D_MIN) / CELL_D))
    return {"xc": -X_HALF + (np.arange(nx) + 0.5) * CELL_X,
            "dc": D_MIN + (np.arange(nd) + 0.5) * CELL_D,
            "cell_x": CELL_X, "cell_d": CELL_D}


def base_fit(rec):
    b = rec["base"]
    return {"shape": b["shape"], "offsets": b["offsets"], "deg": b["deg"],
            "reach": b["reach"], "reach_side": b["reach_side"]}


def sight_lines(d, x, side, tol=0.002, lag=5.0):
    """Кромки, лежащие на линии взгляда, а не на стене — без всякой модели.

    Луч, скользящий по внутренней стене поворота, за точкой касания уходит в
    противоположную стену, и наблюдаемая кромка дальше — прямая через сенсор:
    её угол равен уже достигнутому ближе максимуму (для левой стороны; для
    правой — минимуму). Настоящая стена, пока видна, даёт с глубиной НОВЫЙ
    максимум угла. tol — 0.002 рад, это 0.2 м на 100 м.
    """
    ang = np.arctan2(x, d)
    order = np.argsort(d)
    out = np.zeros(len(d), bool)
    dd, aa = d[order], ang[order]
    for k in range(len(dd)):
        near = dd < dd[k] - lag
        if not near.any():
            continue
        if side == "left":
            out[order[k]] = aa[near].max() >= aa[k] - tol
        else:
            out[order[k]] = aa[near].min() <= aa[k] + tol
    return out


def ref_edges(grid, img, fit):
    """Кромки для мерила: как в §26, с ведением оси от ближних полос к дальним
    и прошлой формой в качестве подсказки — одинаково для обоих методов."""
    d, l, r, _ = band_edges(img >= 0.25, grid, "silhouette", weight=img, mode="mass",
                            prior=fit)
    return d, l, r


def wall_path_dict(xw, dw):
    """Кривая стены как «путь» на сетке глубин — чтобы перенести её тем же
    advance_path, что и путь."""
    o = np.argsort(dw)
    x = np.interp(PATH_GRID, dw[o], xw[o])
    psi = np.arctan(np.gradient(x, PATH_GRID))
    return {"d": PATH_GRID, "x": x, "psi": psi, "arc": PATH_GRID.copy(), "mode": "", "d0": None,
            "offset": 0.0}


def run_method(frames, method):
    """Прогон метода по кэшу. Возвращает список кадров с путём, стенами, дальностью."""
    grid = make_grid()
    tr = WallParallelTracker()
    out = []
    prev_fit = None
    offset = 0.0
    t0 = time.time()
    for k, rec in enumerate(frames):
        if "img" not in rec:
            out.append(None)
            tr.reset()
            continue
        img = rec["img"].astype(float) / 255.0
        fit = base_fit(rec)
        if method == "base":
            path = {"d": PATH_GRID, "x": rec["base"]["path_x"].astype(float)}
            walls = {}
            for side in ("left", "right"):
                walls[side] = (shape_x(fit, PATH_GRID, side), PATH_GRID.copy(),
                               fit["reach_side"].get(side))
            out.append({"path_x": path["x"], "walls": walls, "reach": fit["reach"],
                        "shift": rec["shift"], "fit": fit})
            continue
        rd, rx = rec["rail_d"].astype(float), rec["rail_x"].astype(float)
        d, l, r, L = band_edges(img >= 0.25, grid, "silhouette", weight=img, mode="mass",
                                prior=prev_fit)
        edges = edges_from_bands(d, l, r, L)
        fresh = build_path(fit, rd, rx, offset)
        if fresh["mode"] == "rails+contrast":
            offset = fresh["offset"]
        res = tr.step(edges, rd, rx, rec["shift"], fresh_path=(PATH_GRID, fresh["x"]),
                      fresh_walls=lambda dd, side, f=fit: shape_x(f, dd, side))
        prev_fit = fit
        if res is None:
            out.append(None)
            continue
        c = res["curve"]
        pd = to_path_dict(c, PATH_GRID, "")
        walls = {}
        for side, w in (("left", c["wl"]), ("right", c["wr"])):
            xw, dw = offset_curve(c, w)
            walls[side] = (xw, dw, res["reach_side"].get(side))
        out.append({"path_x": pd["x"], "walls": walls, "reach": res["reach"],
                    "shift": rec["shift"], "fit": fit, "res": res})
    return out, time.time() - t0


def wall_on_grid(w):
    xw, dw, _ = w
    o = np.argsort(dw)
    return np.interp(PATH_GRID, dw[o], xw[o])


def measure(frames, outs):
    grid = make_grid()
    jw, jp = [], []
    intr, intr_frames, n_frames = [], 0, 0
    rail_near, rail_far = [], []
    for k, (rec, o) in enumerate(zip(frames, outs)):
        if o is None:
            continue
        n_frames += 1
        px = o["path_x"]
        # рельсы
        rd, rx = rec["rail_d"].astype(float), rec["rail_x"].astype(float)
        for lo, hi, acc in ((4, 25, rail_near), (27, 40, rail_far)):
            m = (rd >= lo) & (rd <= hi)
            # дальние срезы — только если согласны между собой: на двухпутном
            # участке детектор там скачет между своей и чужой парой (±1.5 м)
            if m.sum() >= 2 and (hi < 30 or np.ptp(rx[m]) <= 0.3):
                acc.append(float(np.median(np.abs(rx[m] - np.interp(rd[m], PATH_GRID, px)))))
        # стена в коридоре
        img = rec["img"].astype(float) / 255.0
        d, l, r = ref_edges(grid, img, o["fit"])
        path = {"d": PATH_GRID, "x": px, "psi": np.arctan(np.gradient(px, PATH_GRID)),
                "arc": np.r_[0, np.cumsum(np.hypot(np.diff(px), np.diff(PATH_GRID)))]}
        reach = min(o["reach"] or 0.0, 120.0)
        cnt = 0
        for xs, side in ((l, "left"), (r, "right")):
            m = (d <= reach) & ~sight_lines(d, xs, side)
            if not m.any():
                continue
            x_p, psi, _ = path_at(path, d[m])
            u = (xs[m] - x_p) * np.cos(psi)
            bad = np.abs(u) < CORRIDOR
            dd = d[m][bad]
            for q in dd:
                if np.sum(np.abs(dd - q) <= 5.0) >= 3:
                    cnt += 1
        intr.append(cnt)
        intr_frames += cnt > 0
        # дрожание
        if k > 0 and outs[k - 1] is not None and o["shift"] is not None \
                and np.isfinite(o["shift"]):
            p = outs[k - 1]
            ds = float(o["shift"])
            prev_path = {"d": PATH_GRID, "x": p["path_x"],
                         "psi": np.arctan(np.gradient(p["path_x"], PATH_GRID)), "mode": ""}
            mv = advance_path(prev_path, ds)
            reach_pp = min((p["reach"] or 0) - ds, o["reach"] or 0)
            jp.append([abs(np.interp(D, PATH_GRID, mv["x"]) - np.interp(D, PATH_GRID, px))
                       if D <= reach_pp else np.nan for D in PROBE])
            # стены переносятся с поворотом пути (движение поезда задаёт путь)
            for side in ("left", "right"):
                wa, wb = p["walls"][side], o["walls"][side]
                if wa[2] is None or wb[2] is None:
                    continue
                xa, xb = wall_on_grid(wa), wall_on_grid(wb)
                mvw = _advance_with(prev_path, xa, ds)
                lim = min(wa[2] - ds, wb[2])
                jw.append([abs(np.interp(D, PATH_GRID, mvw) - np.interp(D, PATH_GRID, xb))
                           if D <= lim else np.nan for D in PROBE])
    jw, jp = np.array(jw, float), np.array(jp, float)
    q = lambda a, k, f: float(f(a[np.isfinite(a[:, k]), k])) if len(a) and \
        np.isfinite(a[:, k]).any() else float("nan")
    return {
        "frames": n_frames,
        "wall_med": [q(jw, k, np.median) for k in range(len(PROBE))],
        "wall_p90": [q(jw, k, lambda v: np.percentile(v, 90)) for k in range(len(PROBE))],
        "path_med": [q(jp, k, np.median) for k in range(len(PROBE))],
        "path_p90": [q(jp, k, lambda v: np.percentile(v, 90)) for k in range(len(PROBE))],
        "intr_frames": int(intr_frames), "intr_mean": float(np.mean(intr)) if intr else 0.0,
        "rail_near": float(np.median(rail_near)) if rail_near else float("nan"),
        "rail_far": float(np.median(rail_far)) if rail_far else float("nan"),
        "rail_far_p90": float(np.percentile(rail_far, 90)) if rail_far else float("nan"),
    }


def _advance_with(path, x_curve, ds):
    """Перенос произвольной кривой x(d) движением поезда вдоль path на ds —
    ровно то преобразование кадра, которое делает advance_path."""
    psi0 = float(path["psi"][0])
    ox, od = ds * np.sin(psi0), ds * np.cos(psi0)
    dpsi = float(np.interp(od, path["d"], path["psi"])) - psi0
    c, sn = np.cos(dpsi), np.sin(dpsi)
    dx, dd = x_curve - ox, PATH_GRID - od
    xn = dx * c - dd * sn
    dn = dx * sn + dd * c
    o = np.argsort(dn)
    return np.interp(PATH_GRID, dn[o], xn[o])


def fmt(v):
    return " ".join(f"{x:5.2f}" if np.isfinite(x) else "  -  " for x in v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bags", nargs="*", default=DEV_RUNS)
    p.add_argument("--cache", default="output/exp17_cache")
    p.add_argument("--frames", nargs=2, type=int, default=None)
    p.add_argument("--methods", nargs="*", default=["base", "new"])
    p.add_argument("--tag", default="dev")
    a = p.parse_args()
    rows = []
    for bag in a.bags:
        frames = pickle.load(open(Path(a.cache) / f"{bag}.pkl", "rb"))
        if a.frames:
            frames = frames[a.frames[0]:a.frames[1]]
        for meth in a.methods:
            outs, dt = run_method(frames, meth)
            s = measure(frames, outs)
            s.update({"bag": bag, "method": meth, "sec_per_frame": dt / max(len(frames), 1)})
            rows.append(s)
            print(f"{bag[:30]:30s} {meth:5s} кадров {s['frames']:4d} "
                  f"стены {fmt(s['wall_med'])} | p90 {fmt(s['wall_p90'])} | "
                  f"путь {fmt(s['path_med'])} | стена в коридоре: кадров {s['intr_frames']:3d} "
                  f"({s['intr_mean']:.1f}) | рельсы {s['rail_near']:.3f} / {s['rail_far']:.3f} "
                  f"(p90 {s['rail_far_p90']:.3f}) | {s['sec_per_frame']*1000:.0f} мс", flush=True)
    out = Path("output/exp17_eval")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"summary_{a.tag}.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
