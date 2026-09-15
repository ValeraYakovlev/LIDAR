#!/usr/bin/env python3
"""Находит рельсы и центральный желоб на срезах лидарных кадров и строит графики
выравнивания. Методика описана в knowledge.md (§13-14).

Пример запуска:
    python run_rail_detection.py
    python run_rail_detection.py --dataset /Volumes/T7/Dataset --out output
    python run_rail_detection.py --bags roundT_doubleT doubleT_platform
"""

import argparse
import json
from pathlib import Path

from rail_detection import (
    DEFAULT_BAGS,
    analyze_frame,
    bag_path,
    load_frame,
    plot_all_bags_grid,
    plot_centerline_grid,
    plot_combined_overlay,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="/Volumes/T7/Dataset", help="папка с bag-файлами")
    parser.add_argument("--out", default="output", help="куда сохранять графики и summary.json")
    parser.add_argument("--bags", nargs="*", default=DEFAULT_BAGS, help="какие бэги обрабатывать")
    parser.add_argument("--frame", type=int, default=None, help="индекс кадра (по умолчанию — средний)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_by_bag = {}
    for name in args.bags:
        path = bag_path(args.dataset, name)
        print(f"[{name}] загрузка кадра...")
        points, n_frames = load_frame(path, args.frame)
        frame_used = n_frames // 2 if args.frame is None else args.frame
        print(f"[{name}] кадр {frame_used}/{n_frames}, анализ срезов...")
        records, n_skip = analyze_frame(points)
        print(f"[{name}] выровнено {len(records)}, пропущено {n_skip}")
        results_by_bag[name] = (records, n_skip)

    print("построение графиков...")
    summary = plot_all_bags_grid(results_by_bag, out_dir / "rail_aligned_overlay.png")
    plot_combined_overlay(results_by_bag, out_dir / "rail_aligned_allbags.png")
    plot_centerline_grid(results_by_bag, out_dir / "rail_centerline.png")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Сводка ===")
    for bag, s in summary.items():
        print(
            f"{bag}: {s['n_aligned']} выровнено, {s['n_skipped']} пропущено, "
            f"рельс-рельс {s['rail_gauge_mean']:.2f}±{s['rail_gauge_std']:.2f}м"
        )
    print(f"\nГрафики сохранены в {out_dir}/")


if __name__ == "__main__":
    main()
