#!/usr/bin/env python3
"""Срез перпендикулярно РЕАЛЬНОМУ изгибу пути (а не оси Y сенсора), и
раздельное выравнивание прямых/поворотных участков.

Метод:
  1. Первый проход (как раньше) — слепая детекция по каждому бэгу, грубая
     centerline.
  2. Подгонка гладкой кривой пути по этой centerline (robust_centerline).
  3. Второй проход — повторная детекция желоба/рельс, но точки среза заранее
     скорректированы в систему координат вдоль локального курса кривой
     (find_groove_and_rails(path_coeffs=...)) — теперь срез перпендикулярен
     реальному пути, а не фиксированной оси Y.
  4. Каждый срез второго прохода помечается 'straight' или 'curve' по
     локальному углу курса на своей глубине.
  5. Срезы объединяются по всем бэгам в 4 группы (прямые/поворотные,
     до/после коррекции) — сравниваем расстояние рельс-рельс в каждой,
     чтобы понять, было ли завышение на поворотах проекционной ошибкой.

Запуск:
    python curved_slicing.py
"""

import argparse
from pathlib import Path

import numpy as np

from rail_detection import (
    DEFAULT_BAGS,
    DEFAULT_DEPTH_BINS,
    analyze_frame,
    bag_path,
    load_frame,
    plot_bag_overlay,
    fit_path,
    classify_section,
)
from rail_detection.detector import find_groove_and_rails

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def gauge_stats(records):
    if not records:
        return float("nan"), float("nan"), 0
    g = [r["rail_gauge"] for r in records]
    return float(np.mean(g)), float(np.std(g)), len(g)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="/Volumes/T7/Dataset")
    parser.add_argument("--out", default="output")
    parser.add_argument("--bags", nargs="*", default=DEFAULT_BAGS)
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--thresh-deg", type=float, default=1.5, help="порог |курса| straight/curve, град.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = {
        ("raw", "straight"): [], ("raw", "curve"): [],
        ("corrected", "straight"): [], ("corrected", "curve"): [],
    }

    for name in args.bags:
        points, n_frames = load_frame(bag_path(args.dataset, name), args.frame)

        pass1_records, _ = analyze_frame(points, DEFAULT_DEPTH_BINS)
        coeffs = fit_path(pass1_records)
        heading_note = "coeffs=None (мало точек, коррекция = identity)" if coeffs is None else "OK"

        pass2_records = []
        for lo, hi in DEFAULT_DEPTH_BINS:
            r = find_groove_and_rails(points, lo, hi, path_coeffs=coeffs)
            if r is not None:
                pass2_records.append(r)

        for r in pass1_records:
            mid = (r["depth_lo"] + r["depth_hi"]) / 2
            label = classify_section(coeffs, mid, args.thresh_deg)
            groups[("raw", label)].append(r)
        for r in pass2_records:
            mid = (r["depth_lo"] + r["depth_hi"]) / 2
            label = classify_section(coeffs, mid, args.thresh_deg)
            groups[("corrected", label)].append(r)

        n_curve = sum(1 for r in pass2_records if classify_section(coeffs, (r["depth_lo"] + r["depth_hi"]) / 2, args.thresh_deg) == "curve")
        print(f"[{name}] pass1={len(pass1_records)} pass2={len(pass2_records)} "
              f"из них поворотных={n_curve} ({heading_note})")

    print("\n=== Расстояние рельс-рельс по группам ===")
    for (stage, label), records in groups.items():
        mean, std, n = gauge_stats(records)
        print(f"{stage:10s} {label:9s}: n={n:3d}  gauge={mean:.3f}±{std:.3f}м")

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    titles = {
        ("raw", "straight"): "ДО коррекции — прямые участки",
        ("raw", "curve"): "ДО коррекции — поворотные участки",
        ("corrected", "straight"): "ПОСЛЕ коррекции — прямые участки",
        ("corrected", "curve"): "ПОСЛЕ коррекции — поворотные участки",
    }
    positions = {
        ("raw", "straight"): (0, 0), ("raw", "curve"): (0, 1),
        ("corrected", "straight"): (1, 0), ("corrected", "curve"): (1, 1),
    }
    for key, records in groups.items():
        r, c = positions[key]
        plot_bag_overlay(axes[r, c], titles[key], records, n_skipped=0)

    fig.suptitle(
        "Срез перпендикулярно оси Y (до) vs перпендикулярно реальному курсу пути (после)\n"
        "поворотные участки — где |локальный курс| ≥ %.1f°" % args.thresh_deg,
        fontsize=13,
    )
    fig.tight_layout()
    out_path = out_dir / "curved_slicing_comparison.png"
    fig.savefig(out_path, dpi=120)
    print(f"\nГрафик сохранён: {out_path}")


if __name__ == "__main__":
    main()
