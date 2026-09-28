#!/usr/bin/env python3
"""Эксперимент 2, шаги 0-1: чем СТЕНА отличается от «начинки» перед ней.

Это замер, а не реализация. Вопрос один: несут ли неиспользованные каналы
(`intensity`, `ring`) что-нибудь, чего нет в геометрических масках. Если
распределения признака у стены и у начинки совпадают — признак пуст, и дальше
по нему идти незачем; проверить это дешевле, чем встраивать.

Как делится выборка. На кадре строится текущая геометрия, точки переводятся в
координаты пути, берётся клиренс-полоса. На каждой стороне:

  стена   — точки в ±0.25 м от НАЙДЕННОЙ границы o(d);
  начинка — точки между путём и стеной (от U_MIN до o(d) − 0.5 м): контактный
            рельс, кабельные лотки, кронштейны, торец настила.

Деление опирается на уже найденную стену и потому честно ровно настолько,
насколько верна сама стена: кадры со стороной хуже coverage 0.9 в выборку не
берутся. Вопрос сейчас не «где стена», а «отличается ли она от того, что перед
ней», и для него такого деления достаточно.

Мера разделимости — AUC: вероятность, что у случайной точки стены признак
больше, чем у случайной точки начинки. 0.5 — признак пуст. Симметричная мера,
не требующая выбирать порог, и не боящаяся того, что точек стены в разы больше.

Запуск:
    python exp_clutter_probe.py --stride 8
"""

import argparse
from pathlib import Path

import numpy as np

from rail_detection import DEFAULT_BAGS, bag_path, iter_frames, to_track_coords
from rail_detection.rangeimage import ring_elevation, surface_step, unfold
from rail_detection.tracker import TunnelTracker
from rail_detection.tunnel_frame import U_MIN, V_HI, V_LO, side_offset

WALL_TOL = 0.25      # м: полоса вокруг найденной границы, которая считается стеной
CLUTTER_GAP = 0.5    # м: настолько ближе стены должна быть точка, чтобы считаться начинкой
MIN_COVERAGE = 0.90  # кадры с худшей стеной в выборку не идут
RANGE_EDGES = np.array([3, 6, 9, 12, 16, 20, 25, 30, 36, 45])

FEATURES = ("intensity", "az_step", "ring_step")


def frame_features(points):
    """Признаки на точку: интенсивность и деприцированные скачки дальности по
    азимуту и по кольцам. Скачок = inf там, где соседа нет."""
    n = len(points)
    out = {"intensity": points['intensity'].astype(float),
           "az_step": np.full(n, np.inf),
           "ring_step": np.full(n, np.inf)}
    grid = unfold(points)
    if grid is not None:
        flat = grid["index"].ravel()
        out["az_step"][flat] = surface_step(grid, axis=0).ravel()
        out["ring_step"][flat] = surface_step(grid, axis=1,
                                              coords=ring_elevation(grid)).ravel()
    return out


def collect(dataset, bags, stride, max_frames=None):
    """Собирает (дальность, признаки) отдельно для стены и для начинки."""
    acc = {g: {"range": [], **{f: [] for f in FEATURES}} for g in ("wall", "clutter")}
    n_frames, n_used = 0, 0
    for bag in bags:
        tracker = TunnelTracker()
        for idx, points, _ in iter_frames(bag_path(dataset, bag), stride=stride,
                                          max_frames=max_frames):
            n_frames += 1
            res = tracker.update(points, steps=stride)
            if res is None:
                continue
            x = points['x'].astype(float)
            y = points['y'].astype(float)
            z = points['z'].astype(float)
            feats = frame_features(points)
            d, u, v = to_track_coords(x, y, z, res["frame"])
            rng = np.sqrt(x * x + y * y + z * z)
            band = (v >= V_LO) & (v <= V_HI) & (d > 3) & (d < 45)
            taken = False
            for side, sign in (("left", -1.0), ("right", +1.0)):
                s = res[side]
                if s is None or s["coverage"] < MIN_COVERAGE:
                    continue
                su = sign * u
                w = side_offset(res["shape"], side, d)
                sel = band & (su >= U_MIN) & np.isfinite(w)
                groups = {"wall": sel & (np.abs(su - w) < WALL_TOL),
                          "clutter": sel & (su < w - CLUTTER_GAP)}
                for g, m in groups.items():
                    acc[g]["range"].append(rng[m])
                    for f in FEATURES:
                        acc[g][f].append(feats[f][m])
                taken = True
            n_used += int(taken)
    data = {g: {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in d.items()}
            for g, d in acc.items()}
    return data, n_frames, n_used


def normalize_by_range(data, key):
    """Общая для обеих групп нормировка признака на дальность.

    Нужна для интенсивности: у Hesai она не калибрована по дальности, а начинка
    по построению ближе стены — без нормировки разница «стена vs начинка»
    смешалась бы с разницей в дальности. Нормировка считается по обеим группам
    сразу: посчитанная по каждой отдельно, она стёрла бы ровно то, что меряется.
    """
    allr = np.concatenate([data[g]["range"] for g in ("wall", "clutter")])
    allv = np.concatenate([data[g][key] for g in ("wall", "clutter")])
    ok = np.isfinite(allv)
    centers, med = [], []
    for lo, hi in zip(RANGE_EDGES[:-1], RANGE_EDGES[1:]):
        sl = ok & (allr >= lo) & (allr < hi)
        if sl.sum() > 50:
            centers.append((lo + hi) / 2)
            med.append(float(np.median(allv[sl])))
    if len(centers) < 2:
        return None
    centers, med = np.array(centers), np.array(med)
    return {g: data[g][key] / np.maximum(np.interp(data[g]["range"], centers, med), 1e-6)
            for g in ("wall", "clutter")}, centers, med


def auc(a, b, n=400000, seed=0):
    """Вероятность, что у случайной точки стены признак больше, чем у точки
    начинки. Считается по случайным парам, а не полным перебором: пар здесь
    миллиарды, а погрешность Монте-Карло при n=4·10⁵ — порядка 0.001, много
    меньше разницы, ради которой замер делается. inf-значения (нет соседа)
    участвуют как самые большие — это часть признака, а не пропуск."""
    a, b = a[np.isfinite(a) | np.isposinf(a)], b[np.isfinite(b) | np.isposinf(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    return float(np.mean(rng.choice(a, n, replace=True) > rng.choice(b, n, replace=True)))


def report(name, vals, data, note=""):
    w, c = vals["wall"], vals["clutter"]
    a = auc(w, c)
    verdict = "ПУСТ" if 0.42 <= a <= 0.58 else ("стена БОЛЬШЕ" if a > 0.5 else "стена МЕНЬШЕ")
    print(f"\n{name}  {note}")
    print(f"  медиана: стена {np.median(w[np.isfinite(w)]):.3f}   "
          f"начинка {np.median(c[np.isfinite(c)]):.3f}"
          + (f"   (нет соседа: стена {np.mean(~np.isfinite(w)):.1%}, "
             f"начинка {np.mean(~np.isfinite(c)):.1%})" if not np.all(np.isfinite(w)) else ""))
    print(f"  AUC {a:.3f}  -> {verdict}")
    print("  по дальности: ", end="")
    for lo, hi in zip(RANGE_EDGES[:-1], RANGE_EDGES[1:]):
        mw = (data["wall"]["range"] >= lo) & (data["wall"]["range"] < hi)
        mc = (data["clutter"]["range"] >= lo) & (data["clutter"]["range"] < hi)
        if mw.sum() < 200 or mc.sum() < 200:
            continue
        print(f"{lo:.0f}-{hi:.0f}м {auc(w[mw], c[mc]):.2f}  ", end="")
    print()
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=list(DEFAULT_BAGS))
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--plot", default="output/clutter_features.png")
    a = p.parse_args()

    data, n_frames, n_used = collect(a.dataset, a.bags, a.stride, a.max_frames)
    print(f"кадров просмотрено {n_frames}, в выборку вошло {n_used} "
          f"(порог coverage >= {MIN_COVERAGE})")
    print(f"точек: стена {len(data['wall']['range'])}, начинка {len(data['clutter']['range'])}")

    got = normalize_by_range(data, "intensity")
    if got is not None:
        vals, centers, med = got
        print("\nмедиана интенсивности по дальности (общая нормировка): "
              + " ".join(f"{c:.0f}м:{m:.0f}" for c, m in zip(centers, med)))
        report("ИНТЕНСИВНОСТЬ, нормированная на дальность", vals, data)

    for key, name in (("az_step", "СКАЧОК ПО АЗИМУТУ |Δ²(1/r)|·r², м"),
                      ("ring_step", "СКАЧОК ПО КОЛЬЦАМ |Δ²(1/r)|·r², м")):
        vals = {g: data[g][key] for g in ("wall", "clutter")}
        report(name, vals, data, note="(меньше = глаже)")

    _plot(a.plot, data)


def _plot(path, data):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(11, 3.4), dpi=130)
    FigureCanvasAgg(fig)
    specs = [("intensity", "интенсивность (сырая)", np.linspace(0, 40, 60), False),
             ("az_step", "скачок по азимуту, м", np.linspace(0, 0.6, 60), True),
             ("ring_step", "скачок по кольцам, м", np.linspace(0, 0.6, 60), True)]
    for i, (key, title, bins, logy) in enumerate(specs, 1):
        ax = fig.add_subplot(1, 3, i)
        for g, c, lab in (("wall", "tab:green", "стена"), ("clutter", "tab:red", "начинка")):
            v = data[g][key]
            ax.hist(v[np.isfinite(v)], bins=bins, density=True, histtype="step",
                    color=c, label=lab, lw=1.4)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        if logy:
            ax.set_yscale("log")
        if i == 1:
            ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"\nкартинка: {path}")


if __name__ == "__main__":
    main()
