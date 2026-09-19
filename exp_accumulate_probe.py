#!/usr/bin/env python3
"""Эксперимент 3, шаг 2: что накопление даёт по ПЛОТНОСТИ — до правок алгоритма.

Отсечка эксперимента. Дальность наблюдения упирается в ~45 м не потому, что
модель плоха, а потому, что глубже точек в клиренс-полосе не хватает на
локализацию границы (§17). Значит первым делом надо просто посчитать, сколько их
становится при накоплении N кадров, и на каких глубинах. Если на 50–80 м
плотность не выросла кратно — накопление не работает (скорее всего из-за ошибки
в Δs), и дальше идти незачем.

Замер намеренно не трогает подгонку: считаются точки, а не метрики. Плюс
отдельно — РАЗМАЗЫВАНИЕ: разброс накопленных точек поперёк пути в тех срезах,
где стена видна и в одиночном кадре. Если ошибка Δs велика, плотность вырастет,
но стена расплывётся, и первое без второго ничего не значит.

Запуск:
    python exp_accumulate_probe.py --bags roundT_doubleT --n-frames 1 3 5 10
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from rail_detection import DEFAULT_BAGS, bag_path, fit_tunnel_geometry, iter_frames
from rail_detection.accumulate import FrameAccumulator, track_band
from rail_detection.tunnel_frame import U_MIN, side_offset

DEPTH_SLICES = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 100)]
SPREAD_TOL = 0.5     # м: окрестность найденной границы, в которой меряется размазывание


def run(dataset, bag, n_frames, stride, max_frames):
    """Один проход: для каждого N — точки в срезах и разброс у границы."""
    counts = {n: defaultdict(list) for n in n_frames}
    spread = {n: defaultdict(list) for n in n_frames}
    accs = {n: FrameAccumulator(n_frames=n) for n in n_frames}
    n_seen = 0
    for idx, points, _ in iter_frames(bag_path(dataset, bag), stride=stride,
                                      max_frames=max_frames):
        solo = fit_tunnel_geometry(points)
        if solo is None:
            continue
        n_seen += 1
        for n in n_frames:
            merged = accs[n].push(points, solo["frame"], steps=stride)
            band = track_band(merged, solo["frame"])
            for lo, hi in DEPTH_SLICES:
                sl = (band["d"] >= lo) & (band["d"] < hi)
                counts[n][(lo, hi)].append(int(sl.sum()))
                for side, sign in (("left", -1.0), ("right", +1.0)):
                    s = solo[side]
                    if s is None or not s.get("coverage"):
                        continue
                    su = sign * band["u"][sl]
                    w = side_offset(solo["shape"], side, band["d"][sl])
                    near = (su >= U_MIN) & (np.abs(su - w) < SPREAD_TOL)
                    if near.sum() >= 30:
                        spread[n][(lo, hi)].append(float(np.std(su[near] - w[near])))
    return counts, spread, n_seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=list(DEFAULT_BAGS))
    p.add_argument("--n-frames", nargs="*", type=int, default=[1, 3, 5, 10])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=120)
    p.add_argument("--plot", default="output/accumulation_density.png")
    a = p.parse_args()

    total_c = {n: defaultdict(list) for n in a.n_frames}
    total_s = {n: defaultdict(list) for n in a.n_frames}
    for bag in a.bags:
        counts, spread, n_seen = run(a.dataset, bag, a.n_frames, a.stride, a.max_frames)
        print(f"\n=== {bag} ({n_seen} кадров, шаг {a.stride}) ===")
        print(f"{'срез, м':>10s} " + " ".join(f"{'N='+str(n):>12s}" for n in a.n_frames))
        for key in DEPTH_SLICES:
            base = np.median(counts[a.n_frames[0]][key]) if counts[a.n_frames[0]][key] else 0
            cells = []
            for n in a.n_frames:
                v = np.median(counts[n][key]) if counts[n][key] else 0
                gain = f"×{v / base:.1f}" if base > 0 else "—"
                cells.append(f"{v:7.0f} {gain:>4s}")
                total_c[n][key] += counts[n][key]
                total_s[n][key] += spread[n][key]
            print(f"{key[0]:4d}-{key[1]:<4d} " + " ".join(f"{c:>12s}" for c in cells))

    print("\n=== Все прогоны вместе: точек в клиренс-полосе (медиана по кадрам) ===")
    # Выигрыш считается ПОКАДРОВО и потом усредняется, а не как отношение медиан
    # по объединённой выборке: у прогонов разная абсолютная плотность, и медиана
    # смеси ничьему выигрышу не равна.
    print(f"{'срез, м':>10s} " + " ".join(f"{'N='+str(n):>12s}" for n in a.n_frames))
    base_n = a.n_frames[0]
    for key in DEPTH_SLICES:
        base = np.asarray(total_c[base_n][key], dtype=float)
        cells = []
        for n in a.n_frames:
            v = np.asarray(total_c[n][key], dtype=float)
            med = np.median(v) if len(v) else 0
            ok = (len(v) == len(base)) and len(base) and (base > 0).any()
            gain = f"×{np.median(v[base > 0] / base[base > 0]):.1f}" if ok else "—"
            cells.append(f"{med:7.0f} {gain:>5s}")
        print(f"{key[0]:4d}-{key[1]:<4d} " + " ".join(f"{c:>12s}" for c in cells))

    print("\n=== Размазывание: разброс точек поперёк пути у границы, м ===")
    print("(растёт — значит ошибка Δs съедает выигрыш от плотности)")
    print(f"{'срез, м':>10s} " + " ".join(f"{'N='+str(n):>8s}" for n in a.n_frames))
    for key in DEPTH_SLICES:
        cells = []
        for n in a.n_frames:
            v = total_s[n][key]
            cells.append(f"{np.median(v):8.3f}" if v else f"{'—':>8s}")
        print(f"{key[0]:4d}-{key[1]:<4d} " + " ".join(cells))

    _plot(a.plot, total_c, total_s, a.n_frames)


def _plot(path, counts, spread, n_frames):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 3.4), dpi=130)
    FigureCanvasAgg(fig)
    centers = [(lo + hi) / 2 for lo, hi in DEPTH_SLICES]
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    for n in n_frames:
        c = [np.median(counts[n][k]) if counts[n][k] else np.nan for k in DEPTH_SLICES]
        s = [np.median(spread[n][k]) if spread[n][k] else np.nan for k in DEPTH_SLICES]
        ax1.plot(centers, c, "o-", lw=1.3, label=f"N={n}")
        ax2.plot(centers, s, "o-", lw=1.3, label=f"N={n}")
    ax1.set_yscale("log")
    ax1.set_title("точек в клиренс-полосе", fontsize=9)
    ax2.set_title("разброс у границы, м (размазывание)", fontsize=9)
    for ax in (ax1, ax2):
        ax.set_xlabel("глубина, м", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"\nкартинка: {path}")


if __name__ == "__main__":
    main()
