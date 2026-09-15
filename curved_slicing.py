#!/usr/bin/env python3
"""Срез перпендикулярно РЕАЛЬНОМУ изгибу пути (а не оси Y сенсора).

Метод:
  1. Первый проход (как раньше) — слепая детекция по каждому бэгу, грубая
     centerline.
  2. Подгонка гладкой кривой пути по этой centerline (robust_centerline).
  3. Второй проход — повторная детекция желоба/рельс, но точки среза заранее
     скорректированы в систему координат вдоль локального курса кривой
     (find_groove_and_rails(path_coeffs=...)) — теперь срез перпендикулярен
     реальному пути, а не фиксированной оси Y.

Результат отчитывается ПО КАЖДОМУ БЭГУ ОТДЕЛЬНО (не усреднено по всем сразу) —
таблица ДО/ПОСЛЕ + направление поворота (лево/право/прямо). Дополнительно
срезы объединяются в пары групп (прямые/поворотные, до/после) для наглядного
сравнения формы — это уже вторичный, сводный график.

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
    describe_path,
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

    per_bag = {}
    pooled = {
        ("raw", "straight"): [], ("raw", "curve"): [],
        ("corrected", "straight"): [], ("corrected", "curve"): [],
    }

    for name in args.bags:
        points, n_frames = load_frame(bag_path(args.dataset, name), args.frame)

        pass1_records, _ = analyze_frame(points, DEFAULT_DEPTH_BINS)
        coeffs = fit_path(pass1_records)

        pass2_records = []
        for lo, hi in DEFAULT_DEPTH_BINS:
            r = find_groove_and_rails(points, lo, hi, path_coeffs=coeffs)
            if r is not None:
                pass2_records.append(r)

        for r in pass1_records:
            mid = (r["depth_lo"] + r["depth_hi"]) / 2
            pooled[("raw", classify_section(coeffs, mid, args.thresh_deg))].append(r)
        for r in pass2_records:
            mid = (r["depth_lo"] + r["depth_hi"]) / 2
            pooled[("corrected", classify_section(coeffs, mid, args.thresh_deg))].append(r)

        depths_all = [(r["depth_lo"] + r["depth_hi"]) / 2 for r in pass2_records]
        turn_desc = describe_path(coeffs, depths_all, args.thresh_deg) if coeffs is not None else "н/д"

        per_bag[name] = {
            "raw": gauge_stats(pass1_records),
            "corrected": gauge_stats(pass2_records),
            "turn": turn_desc,
        }

    # --- отчёт по каждому бэгу отдельно ---
    print(f"\n{'бэг':38s} {'ДО (n)':22s} {'ПОСЛЕ (n)':22s}  направление")
    print("-" * 110)
    for name, s in per_bag.items():
        m1, s1, n1 = s["raw"]
        m2, s2, n2 = s["corrected"]
        print(f"{name:38s} {m1:.3f}±{s1:.3f} (n={n1:2d})   {m2:.3f}±{s2:.3f} (n={n2:2d})   {s['turn']}")

    fig1, ax1 = plt.subplots(figsize=(11, 5))
    names = list(per_bag.keys())
    x = np.arange(len(names))
    raw_means = [per_bag[n]["raw"][0] for n in names]
    raw_stds = [per_bag[n]["raw"][1] for n in names]
    cor_means = [per_bag[n]["corrected"][0] for n in names]
    cor_stds = [per_bag[n]["corrected"][1] for n in names]
    w = 0.35
    ax1.bar(x - w / 2, raw_means, w, yerr=raw_stds, label="до коррекции", color="crimson", alpha=0.8, capsize=3)
    ax1.bar(x + w / 2, cor_means, w, yerr=cor_stds, label="после коррекции", color="darkgreen", alpha=0.8, capsize=3)
    ax1.axhline(1.52, color="black", ls="--", lw=1, label="номинал 1520мм")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("расстояние рельс-рельс, м")
    ax1.set_title("Расстояние рельс-рельс ПО КАЖДОМУ БЭГУ отдельно: до/после коррекции на курс пути")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")
    fig1.tight_layout()
    per_bag_path = out_dir / "curved_slicing_per_bag.png"
    fig1.savefig(per_bag_path, dpi=120)
    print(f"\nГрафик сохранён: {per_bag_path}")

    # --- вторичный сводный график: прямые/поворотные (объединено по бэгам) ---
    fig2, axes = plt.subplots(2, 2, figsize=(13, 12))
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
    for key, records in pooled.items():
        r, c = positions[key]
        plot_bag_overlay(axes[r, c], titles[key], records, n_skipped=0)
    fig2.suptitle(
        "(сводно, для формы) Срез по оси Y vs по реальному курсу пути\n"
        "поворотные участки — где |локальный курс| ≥ %.1f°" % args.thresh_deg,
        fontsize=13,
    )
    fig2.tight_layout()
    pooled_path = out_dir / "curved_slicing_pooled.png"
    fig2.savefig(pooled_path, dpi=120)
    print(f"Графике сохранён: {pooled_path}")


if __name__ == "__main__":
    main()
