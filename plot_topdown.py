#!/usr/bin/env python3
"""Вид тоннеля сверху (план): ось глубины идёт вверх по графику, высота (Z) не
показывается. Серым — сырое облако точек, красным — centerline по рельсам,
синим пунктиром — прямая линия для сравнения (виден ли поворот пути).

Пример запуска:
    python plot_topdown.py
    python plot_topdown.py --dataset /Volumes/T7/Dataset --out output
"""

import argparse
from pathlib import Path

from rail_detection import (
    DEFAULT_BAGS,
    analyze_frame,
    bag_path,
    load_frame,
    plot_topdown_grid,
    plot_centerline_before_after,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="/Volumes/T7/Dataset", help="папка с bag-файлами")
    parser.add_argument("--out", default="output", help="куда сохранять график")
    parser.add_argument("--bags", nargs="*", default=DEFAULT_BAGS, help="какие бэги обрабатывать")
    parser.add_argument("--frame", type=int, default=None, help="индекс кадра (по умолчанию — средний)")
    parser.add_argument("--depth-max", type=float, default=60, help="до какой глубины показывать сырые точки")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    points_by_bag = {}
    results_by_bag = {}
    for name in args.bags:
        path = bag_path(args.dataset, name)
        print(f"[{name}] загрузка кадра...")
        points, n_frames = load_frame(path, args.frame)
        records, n_skip = analyze_frame(points)
        print(f"[{name}] centerline: {len(records)} точек, {n_skip} срезов пропущено")
        points_by_bag[name] = points
        results_by_bag[name] = (records, n_skip)

    out_path = out_dir / "topdown.png"
    plot_topdown_grid(points_by_bag, results_by_bag, out_path, depth_max=args.depth_max)
    print(f"График сохранён: {out_path}")

    ba_path = out_dir / "centerline_before_after.png"
    plot_centerline_before_after(results_by_bag, ba_path)
    print(f"График сохранён: {ba_path}")


if __name__ == "__main__":
    main()
