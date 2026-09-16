#!/usr/bin/env python3
"""Вид сверху на тоннель с найденной rail-guided геометрией стен.

Рисуются: сырое облако (серое), точки клиренс-полосы, по которым и работает
метод (песочные), ось пути по рельсам (пунктир) и итоговые стены (зелёные).
Отдельным режимом --compare рисуется то же самое вместе со старым детектором
стен, чтобы разница была видна на одном и том же кадре, а не по памяти.

Запуск:
    python plot_tunnel_geometry.py --test-set test_set.json --n-sample 40
    python plot_tunnel_geometry.py --test-set test_set.json --n-sample 16 --compare
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rail_detection import (
    DEFAULT_DEPTH_BINS,
    WALL_DEPTH_BINS,
    analyze_walls,
    bag_path,
    combined_wall_fit,
    eval_fit,
    fit_straight_or_arc,
    fit_tunnel_geometry,
    iter_selected_frames,
    to_track_coords,
    wall_x,
)
from rail_detection.tunnel_frame import V_HI, V_LO, axis_formula

DEPTH_MAX = 45.0
X_LIM = (-6.0, 6.0)


def _old_fits(points, res):
    """Стены по старым методам — для наглядного сравнения на том же кадре."""
    rail_records = res["frame"]["rail_records"]
    floor_z = float(np.mean([r["shoulder_z"] for r in rail_records]))
    out = {}
    c = combined_wall_fit(points, floor_z, DEFAULT_DEPTH_BINS)
    out["combined"] = (c["left_fit"], c["right_fit"]) if c else (None, None)
    by_depth = {(r["depth_lo"], r["depth_hi"]): r for r in rail_records}
    wr = analyze_walls(points, by_depth, DEFAULT_DEPTH_BINS)
    if len(wr) >= 4:
        d = [(w["depth_lo"] + w["depth_hi"]) / 2 for w in wr]
        out["density"] = (fit_straight_or_arc(d, [w["left_wall_x"] for w in wr]),
                          fit_straight_or_arc(d, [w["right_wall_x"] for w in wr]))
    else:
        out["density"] = (None, None)
    return out


def draw_frame(ax, bag, idx, points, res, show_old=False):
    x = points['x'].astype(float)
    y = points['y'].astype(float)
    z = points['z'].astype(float)
    depth = -y

    vis = (depth > 0) & (depth < DEPTH_MAX) & (np.abs(x) < X_LIM[1] + 1)
    ax.scatter(x[vis], depth[vis], s=0.25, c="lightgray", alpha=0.35, zorder=1)

    d, u, v = to_track_coords(x, y, z, res["frame"])
    band = vis & (v >= V_LO) & (v <= V_HI)
    ax.scatter(x[band], depth[band], s=0.5, c="#c8a45a", alpha=0.55, zorder=2)

    dd = np.linspace(3, DEPTH_MAX, 120)
    ax.plot(eval_fit(res["frame"]["axis_fit"], dd), dd, c="dimgray", lw=0.9, ls="--", zorder=3)

    if show_old:
        for key, color in (("density", "magenta"), ("combined", "deepskyblue")):
            lf, rf = _old_fits(points, res)[key]
            for fit in (lf, rf):
                if fit is None or fit.get("kind") is None:
                    continue
                ax.plot(eval_fit(fit, dd), dd, c=color, lw=1.3, ls="-", alpha=0.85, zorder=4)

    for side in ("left", "right"):
        if res[side] is None:
            continue
        ax.plot(wall_x(res, side, dd), dd, c="limegreen", lw=2.4, zorder=5)

    ax.set_xlim(*X_LIM)
    ax.set_ylim(0, DEPTH_MAX)
    ax.set_xticks([])
    ax.set_yticks([])

    def lab(side):
        s = res[side]
        if s is None:
            return "нет"
        return f"{s['offset']:.2f}м cov{s['coverage']:.2f} leak{s['leak']:.2f}"

    sh = res["shape"]
    shape_txt = f"дуга R={sh['radius']:.0f}м" if sh["kind"] == "arc" else "прямая"
    ax.set_title(f"{bag} #{idx}\n{shape_txt}\nЛ {lab('left')}\nП {lab('right')}", fontsize=5.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--out", default="output")
    p.add_argument("--test-set", default="test_set.json")
    p.add_argument("--n-sample", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare", action="store_true",
                   help="наложить старые детекторы стен (розовый — плотность, голубой — combined)")
    p.add_argument("--out-name", default=None)
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)

    with open(a.test_set) as f:
        test_set = json.load(f)["test_set"]
    chosen = random.sample(test_set, min(a.n_sample, len(test_set)))

    by_bag = defaultdict(list)
    for item in chosen:
        by_bag[item["bag"]].append(item["frame"])

    panels = []
    for bag, frames in by_bag.items():
        for idx, points in iter_selected_frames(bag_path(a.dataset, bag), frames):
            res = fit_tunnel_geometry(points, WALL_DEPTH_BINS)
            if res is None:
                continue
            panels.append((bag, idx, np.array(points), res))

    if not panels:
        print("Нечего рисовать.")
        return

    ncols = 8 if len(panels) > 16 else 4
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, (bag, idx, points, res) in zip(axes, panels):
        draw_frame(ax, bag, idx, points, res, show_old=a.compare)
    for ax in axes[len(panels):]:
        ax.axis("off")

    legend = ("зелёный — стена (rail-guided); пунктир — ось пути по рельсам\n"
              f"песочные — точки клиренс-полосы v={V_LO}..{V_HI} м над головкой рельса, по ним и работает метод")
    if a.compare:
        legend += "\nрозовый — старый детектор по плотности, голубой — старый combined_wall_fit"
    fig.suptitle(f"Геометрия тоннеля в координатах пути — {len(panels)} кадров из {a.test_set}\n{legend}",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    name = a.out_name or (f"tunnel_geometry_{'vs_old_' if a.compare else ''}{Path(a.test_set).stem}.png")
    fig.savefig(out_dir / name, dpi=130)
    print(f"График сохранён: {out_dir / name}")

    n_arc = sum(1 for *_, res in panels if res["shape"]["kind"] == "arc")
    print(f"\nФорма трассы: дуга в {n_arc} кадрах из {len(panels)}, прямая в {len(panels) - n_arc}")
    for bag, idx, _, res in panels[:10]:
        print(f"  {bag} #{idx}: {axis_formula(res)}")


if __name__ == "__main__":
    main()
