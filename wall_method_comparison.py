#!/usr/bin/env python3
"""Сравнение 3 подходов к поиску стены тоннеля + референс (текущий метод по
плотности), визуально, по 5 бэгам. Красным — точки в полосе "высота стены",
которые метод ИСКЛЮЧИЛ (не признал стеной); остальные точки полосы — синим
(признаны стеной); серым — контекст (весь кадр целиком, вне полосы).

Подходы:
  0 (референс)  — текущий метод: ближайший к центру значимый пик гистограммы X
                  по каждому срезу глубины (см. rail_detection/walls.py).
  1 (envelope)  — внешняя граница: идём от края (макс |x|) внутрь, берём первую
                  точку, у которой есть "поддержка" (соседи рядом) — не пик
                  плотности, а именно самая дальняя ПОДТВЕРЖДЁННАЯ точка.
  2 (polar)     — полярное представление: точки переводятся в (угол, дальность)
                  относительно сенсора; для каждого углового луча берётся
                  САМЫЙ ДАЛЬНИЙ валидный возврат (луч лидара останавливается на
                  стене — дальше физически ничего быть не должно, если это
                  сплошная стена).
  1+2 (combo)   — как 2, но дальний возврат по лучу принимается только если
                  соседние угловые лучи дают близкую дальность (согласованная
                  протяжённая поверхность, а не одиночный дальний луч-фантом).

Для каждого бэга — 1 png с 4 методами (столбцы) x 2 проекции (строки):
вид сверху (X-глубина) и вид в глубину тоннеля (X-Z, все точки кадра сразу).

Запуск:
    python wall_method_comparison.py
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rail_detection import DEFAULT_BAGS, bag_path, load_frame

HEIGHT_LO, HEIGHT_HI = 0.3, 2.5   # полоса "стена" над полом
DEPTH_MAX = 40.0
TOL = 0.15                        # допуск "точка принадлежит найденной стене"


def estimate_floor_z(x, y, z):
    depth = -y
    mask = (depth > 3) & (depth < 15) & (np.abs(x) < 3)
    if mask.sum() < 100:
        mask = (depth > 0) & (depth < 40)
    return float(np.percentile(z[mask], 5))


def height_band_mask(x, y, z, floor_z):
    depth = -y
    return (depth > 0) & (depth < DEPTH_MAX) & (z > floor_z + HEIGHT_LO) & (z < floor_z + HEIGHT_HI)


# ---------- метод 0: референс (текущий, по плотности, из walls.py) ----------

def wall_x_density(depth_vals, x_side, xbin=0.1, min_points=20, min_peak_count=15):
    from scipy.signal import find_peaks
    if len(x_side) < min_points:
        return None
    bins = np.arange(x_side.min(), x_side.max() + xbin, xbin)
    if len(bins) < 3:
        return float(np.median(x_side)) if len(x_side) >= min_peak_count else None
    counts, edges = np.histogram(x_side, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    peaks_idx, _ = find_peaks(counts)
    candidates = list(peaks_idx)
    if len(counts) > 1:
        if counts[0] >= counts[1]:
            candidates.append(0)
        if counts[-1] >= counts[-2]:
            candidates.append(len(counts) - 1)
    candidates = [i for i in set(candidates) if counts[i] >= min_peak_count]
    if not candidates:
        return None
    i_best = min(candidates, key=lambda i: abs(centers[i]))
    return float(centers[i_best])


def classify_reference(x, y, z, band, depth_bins):
    depth = -y
    kept = np.zeros(band.sum(), dtype=bool)
    xb, db = x[band], depth[band]
    idx_all = np.arange(len(xb))
    for lo, hi in depth_bins:
        sl = (db >= lo) & (db < hi)
        if sl.sum() < 20:
            continue
        xs = xb[sl]
        for side_mask, side_sign in [(xs > 0, 1), (xs < 0, -1)]:
            side_vals = xs[side_mask]
            wx = wall_x_density(db[sl], side_vals)
            if wx is None:
                continue
            local_idx = idx_all[sl][side_mask]
            kept_local = np.abs(side_vals - wx) < TOL
            kept[local_idx[kept_local]] = True
    return kept


# ---------- метод 1: envelope (внешняя граница с поддержкой) ----------

def classify_envelope(x, y, z, band, depth_bins, support_radius=0.3, min_support=6):
    depth = -y
    kept = np.zeros(band.sum(), dtype=bool)
    xb, db = x[band], depth[band]
    idx_all = np.arange(len(xb))
    for lo, hi in depth_bins:
        sl = (db >= lo) & (db < hi)
        if sl.sum() < 20:
            continue
        xs = xb[sl]
        for side_mask in [(xs > 0), (xs < 0)]:
            side_vals = xs[side_mask]
            if len(side_vals) < min_support:
                continue
            order = np.argsort(-np.abs(side_vals))  # от края внутрь
            sorted_vals = side_vals[order]
            found = None
            for v in sorted_vals:
                support = np.sum(np.abs(side_vals - v) < support_radius)
                if support >= min_support:
                    found = v
                    break
            if found is None:
                continue
            local_idx = idx_all[sl][side_mask]
            kept_local = np.abs(side_vals - found) < TOL
            kept[local_idx[kept_local]] = True
    return kept


# ---------- метод 2: polar (самый дальний возврат по лучу) ----------

def to_polar(x, y):
    angle = np.degrees(np.arctan2(x, -y))  # 0 = прямо вперёд
    rng = np.sqrt(x ** 2 + y ** 2)
    return angle, rng


def classify_polar(x, y, z, band, angle_bin_deg=0.5, max_range=15.0):
    xb, yb = x[band], y[band]
    angle, rng = to_polar(xb, yb)
    kept = np.zeros(band.sum(), dtype=bool)
    valid_range = rng < max_range
    bins = np.arange(angle.min(), angle.max() + angle_bin_deg, angle_bin_deg)
    idx_bin = np.digitize(angle, bins)
    for i in range(1, len(bins)):
        sl = (idx_bin == i) & valid_range
        if sl.sum() == 0:
            continue
        i_max = np.where(sl)[0][np.argmax(rng[sl])]
        far_range = rng[i_max]
        local = np.where(idx_bin == i)[0]
        kept[local[np.abs(rng[local] - far_range) < TOL]] = True
    return kept


# ---------- метод 1+2: polar + поддержка соседних лучей ----------

def classify_polar_supported(x, y, z, band, angle_bin_deg=0.5, max_range=15.0,
                              support_bins=3, support_tol=0.5):
    xb, yb = x[band], y[band]
    angle, rng = to_polar(xb, yb)
    kept = np.zeros(band.sum(), dtype=bool)
    valid_range = rng < max_range
    bins = np.arange(angle.min(), angle.max() + angle_bin_deg, angle_bin_deg)
    idx_bin = np.digitize(angle, bins)
    n_bins = len(bins)

    far_per_bin = np.full(n_bins + 1, np.nan)
    for i in range(1, n_bins):
        sl = (idx_bin == i) & valid_range
        if sl.sum() > 0:
            far_per_bin[i] = rng[sl].max()

    for i in range(1, n_bins):
        if np.isnan(far_per_bin[i]):
            continue
        neighbors = far_per_bin[max(1, i - support_bins):min(n_bins, i + support_bins + 1)]
        neighbors = neighbors[~np.isnan(neighbors)]
        close_neighbors = np.sum(np.abs(neighbors - far_per_bin[i]) < support_tol)
        if close_neighbors < 3:  # включая себя
            continue
        sl = (idx_bin == i) & valid_range
        local = np.where(sl)[0]
        kept[local[np.abs(rng[local] - far_per_bin[i]) < TOL]] = True
    return kept


METHODS = [
    ("0: референс (плотность)", classify_reference, True),
    ("1: envelope (граница+поддержка)", classify_envelope, True),
    ("2: polar (дальний луч)", classify_polar, False),
    ("1+2: polar+поддержка", classify_polar_supported, False),
]
DEPTH_BINS = [(3, 5), (5, 7), (7, 9), (9, 11), (11, 13), (13, 15), (15, 18),
              (18, 21), (21, 24), (24, 27), (27, 30), (30, 35), (35, 40)]


def make_comparison(bag_name, points, out_path):
    x, y, z = points['x'], points['y'], points['z']
    floor_z = estimate_floor_z(x, y, z)
    band = height_band_mask(x, y, z, floor_z)
    xb, yb, zb, db = x[band], y[band], z[band], -y[band]

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    depth_all = -y
    ctx_mask = (depth_all > 0) & (depth_all < DEPTH_MAX)

    results = {}
    for name, fn, needs_bins in METHODS:
        kept = fn(x, y, z, band, DEPTH_BINS) if needs_bins else fn(x, y, z, band)
        results[name] = kept
        n_removed = int((~kept).sum())
        print(f"  [{bag_name}] {name}: исключено {n_removed}/{band.sum()} "
              f"({100*n_removed/max(1,band.sum()):.1f}%)")

    for col, (name, _, _) in enumerate(METHODS):
        kept = results[name]

        ax_top = axes[0, col]
        ax_top.scatter(x[ctx_mask], depth_all[ctx_mask], s=0.2, c='lightgray', alpha=0.3)
        ax_top.scatter(xb[~kept], db[~kept], s=2, c='red', alpha=0.6, label='исключено')
        ax_top.scatter(xb[kept], db[kept], s=2, c='steelblue', alpha=0.6, label='стена')
        ax_top.set_xlim(-6, 6); ax_top.set_ylim(0, DEPTH_MAX)
        ax_top.set_title(f"{name}\nвид сверху", fontsize=9)
        if col == 0:
            ax_top.legend(fontsize=6, loc='upper right')

        ax_cs = axes[1, col]
        ax_cs.scatter(x[ctx_mask], z[ctx_mask], s=0.2, c='lightgray', alpha=0.3)
        ax_cs.scatter(xb[~kept], zb[~kept], s=2, c='red', alpha=0.6)
        ax_cs.scatter(xb[kept], zb[kept], s=2, c='steelblue', alpha=0.6)
        ax_cs.set_xlim(-6, 6)
        ax_cs.set_title("вид вглубь тоннеля (X-Z)", fontsize=9)

    fig.suptitle(f"{bag_name} — сравнение методов поиска стены (красное = исключено из стены)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=115)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/Volumes/T7/Dataset")
    parser.add_argument("--out", default="output")
    parser.add_argument("--bags", nargs="*", default=DEFAULT_BAGS)
    parser.add_argument("--frame", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.bags:
        print(f"[{name}] загрузка кадра...")
        points, n = load_frame(bag_path(args.dataset, name), args.frame)
        out_path = out_dir / f"wall_methods_{name}.png"
        make_comparison(name, points, out_path)
        print(f"  сохранено: {out_path}")


if __name__ == "__main__":
    main()
