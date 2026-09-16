#!/usr/bin/env python3
"""Визуализация combined_wall_fit: кандидаты от 3 методов (union) + итоговая
гладкая подгонка (прямая/дуга/переход). Берёт кадры из указанного набора
(test_set.json ИЛИ hidden_test_set.json) и рисует сетку примеров.

Запуск:
    python plot_combined_walls.py --test-set hidden_test_set.json --n-sample 40
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rail_detection import DEFAULT_DEPTH_BINS, analyze_frame, bag_path, load_frame, combined_wall_fit
from eval_combined_walls import wall_is_clean, estimate_floor_z


def label_for(fit):
    if fit is None or fit["kind"] is None:
        return "нет"
    if fit["kind"] == "straight":
        return "прямая"
    if fit["kind"] == "arc":
        return f"дуга R={fit['radius']:.0f}м"
    if fit["kind"] == "transition":
        return f"пр->дуга({fit['split_depth']:.0f}м)"
    return str(fit["kind"])


def fit_curve_xy(fit, depths):
    depths = np.asarray(depths)
    if fit is None or fit["kind"] is None:
        return depths, np.full_like(depths, np.nan)
    if fit["kind"] in ("straight", "arc", "unclear"):
        return depths, np.polyval(fit["coeffs"], depths)
    if fit["kind"] == "transition":
        cc1, cc2 = fit["coeffs"]
        split = fit["split_depth"]
        d1 = depths[depths <= split]
        d2 = depths[depths > split]
        return np.concatenate([d1, d2]), np.concatenate([np.polyval(cc1, d1), np.polyval(cc2, d2)])
    return depths, np.full_like(depths, np.nan)


def plot_one(ax, bag_name, frame_idx, points, res, depth_max=45, x_lim=(-6, 6)):
    x, y = points['x'], points['y']
    depth = -y
    mask = (depth > 0) & (depth < depth_max) & (np.abs(x) < x_lim[1] + 1)
    ax.scatter(x[mask], depth[mask], s=0.25, c='lightgray', alpha=0.3)

    for side, color in [("left", "darkblue"), ("right", "purple")]:
        sd, sx = res[f"{side}_candidates"]
        ax.scatter(sx, sd, s=6, c=color, alpha=0.4)
        fit = res[f"{side}_fit"]
        style = '-' if wall_is_clean(fit) else ':'
        fd, fx = fit_curve_xy(fit, np.sort(sd) if len(sd) else np.array([]))
        ax.plot(fx, fd, c=('deepskyblue' if side == 'left' else 'magenta'), lw=2.2, ls=style)

    ax.set_xlim(*x_lim); ax.set_ylim(0, depth_max)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{bag_name} #{frame_idx}\nЛ:{label_for(res['left_fit'])} П:{label_for(res['right_fit'])}", fontsize=6.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/Volumes/T7/Dataset")
    parser.add_argument("--out", default="output")
    parser.add_argument("--test-set", default="test_set.json")
    parser.add_argument("--n-sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-name", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    with open(args.test_set) as f:
        test_set = json.load(f)["test_set"]

    items = []
    for item in test_set:
        points, n = load_frame(bag_path(args.dataset, item["bag"]), item["frame"])
        rail_records, _ = analyze_frame(points, DEFAULT_DEPTH_BINS)
        floor_z = estimate_floor_z(rail_records)
        if floor_z is None:
            continue
        res = combined_wall_fit(points, floor_z, DEFAULT_DEPTH_BINS)
        if res is None:
            continue
        items.append((item["bag"], item["frame"], points, res))

    n_sample = min(args.n_sample, len(items))
    sample = random.sample(items, n_sample)

    ncols = 8
    nrows = -(-n_sample // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, (bag_name, frame_idx, points, res) in zip(axes, sample):
        plot_one(ax, bag_name, frame_idx, points, res)
    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle(f"combined_wall_fit: {n_sample} примеров из {args.test_set}", fontsize=11)
    fig.tight_layout()
    out_name = args.out_name or f"combined_walls_{Path(args.test_set).stem}.png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=130)
    print(f"График сохранён: {out_path}")


if __name__ == "__main__":
    main()
