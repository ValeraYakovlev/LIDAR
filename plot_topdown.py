#!/usr/bin/env python3
"""Вид тоннеля сверху (план): ось глубины идёт вверх по графику, высота (Z) не
показывается. Серым — сырое облако точек, красным — centerline по рельсам,
синим/фиолетовым — левая/правая стена, синим пунктиром — прямая линия для
сравнения (виден ли поворот пути). Направление поворота (налево/направо)
определяется ПРЕЖДЕ ВСЕГО по кривизне стен тоннеля (независимо от рельс) —
это и выводится в заголовок каждого графика.

Пример запуска:
    python plot_topdown.py
    python plot_topdown.py --dataset /Volumes/T7/Dataset --out output
"""

import argparse
from pathlib import Path

from rail_detection import (
    DEFAULT_BAGS,
    DEFAULT_DEPTH_BINS,
    analyze_frame,
    bag_path,
    load_frame,
    plot_topdown_grid,
    plot_centerline_before_after,
    analyze_walls,
    fit_wall,
    describe_path,
    local_heading_deg,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="/Volumes/T7/Dataset", help="папка с bag-файлами")
    parser.add_argument("--out", default="output", help="куда сохранять график")
    parser.add_argument("--bags", nargs="*", default=DEFAULT_BAGS, help="какие бэги обрабатывать")
    parser.add_argument("--frame", type=int, default=None, help="индекс кадра (по умолчанию — средний)")
    parser.add_argument("--depth-max", type=float, default=60, help="до какой глубины показывать сырые точки")
    parser.add_argument("--thresh-deg", type=float, default=1.5, help="порог |курса| straight/поворот, град.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    points_by_bag = {}
    results_by_bag = {}
    wall_records_by_bag = {}
    wall_turn_desc_by_bag = {}

    for name in args.bags:
        path = bag_path(args.dataset, name)
        print(f"[{name}] загрузка кадра...")
        points, n_frames = load_frame(path, args.frame)
        records, n_skip = analyze_frame(points, DEFAULT_DEPTH_BINS)
        print(f"[{name}] centerline: {len(records)} точек, {n_skip} срезов пропущено")
        points_by_bag[name] = points
        results_by_bag[name] = (records, n_skip)

        records_by_depth = {(r["depth_lo"], r["depth_hi"]): r for r in records}
        wall_records = analyze_walls(points, records_by_depth, DEFAULT_DEPTH_BINS)
        wall_records_by_bag[name] = wall_records

        left_coeffs = fit_wall(wall_records, "left")
        right_coeffs = fit_wall(wall_records, "right")
        depths = [(w["depth_lo"] + w["depth_hi"]) / 2 for w in wall_records]

        desc_parts = []
        for side, coeffs in [("лев.", left_coeffs), ("прав.", right_coeffs)]:
            if coeffs is not None:
                desc_parts.append(f"{side} стена: {describe_path(coeffs, depths, args.thresh_deg)}")
        wall_turn_desc_by_bag[name] = "; ".join(desc_parts) if desc_parts else "стены не определены"
        print(f"[{name}] стены: {len(wall_records)} срезов, {wall_turn_desc_by_bag[name]}")

    out_path = out_dir / "topdown.png"
    plot_topdown_grid(
        points_by_bag, results_by_bag, out_path, depth_max=args.depth_max,
        wall_records_by_bag=wall_records_by_bag, wall_turn_desc_by_bag=wall_turn_desc_by_bag,
    )
    print(f"\nГрафик сохранён: {out_path}")

    ba_path = out_dir / "centerline_before_after.png"
    plot_centerline_before_after(results_by_bag, ba_path)
    print(f"График сохранён: {ba_path}")


if __name__ == "__main__":
    main()
