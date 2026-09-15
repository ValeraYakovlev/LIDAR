#!/usr/bin/env python3
"""Оценка combined_wall_fit (density+envelope+polar кандидаты -> одна гладкая
RANSAC-подгонка прямая/дуга/переход). Использует test_set.json (dev, уже
исследовался в прошлых итерациях) для разработки/итераций — НЕ hidden_test_set.json
(тот — для одной финальной, "слепой" проверки в конце).

Запуск:
    python eval_combined_walls.py --test-set test_set.json
"""

import argparse
import json

import numpy as np

from rail_detection import (
    DEFAULT_DEPTH_BINS,
    analyze_frame,
    bag_path,
    load_frame,
    combined_wall_fit,
)


def wall_is_clean(fit, resid_thresh=0.08):
    if fit is None:
        return False
    return (
        fit["kind"] in ("straight", "arc", "transition")
        and fit["residual"] is not None and fit["residual"] < resid_thresh * 2.0
        and (fit.get("inlier_frac") is None or fit["inlier_frac"] >= 0.6)
    )


def estimate_floor_z(rail_records):
    if not rail_records:
        return None
    return float(np.mean([r["shoulder_z"] for r in rail_records]))


def evaluate(dataset, test_set_path, resid_thresh=0.08, verbose=True):
    with open(test_set_path) as f:
        test_set = json.load(f)["test_set"]

    n_ok = 0
    fails = []
    for item in test_set:
        points, n = load_frame(bag_path(dataset, item["bag"]), item["frame"])
        rail_records, _ = analyze_frame(points, DEFAULT_DEPTH_BINS)
        floor_z = estimate_floor_z(rail_records)
        if floor_z is None:
            fails.append((item["bag"], item["frame"], "нет floor_z (рельсы не найдены)"))
            continue
        res = combined_wall_fit(points, floor_z, DEFAULT_DEPTH_BINS, straight_resid_thresh=resid_thresh)
        if res is None:
            fails.append((item["bag"], item["frame"], "мало точек в полосе стены"))
            continue
        left_clean = wall_is_clean(res["left_fit"], resid_thresh)
        right_clean = wall_is_clean(res["right_fit"], resid_thresh)
        ok = left_clean or right_clean
        if ok:
            n_ok += 1
        else:
            lf, rf = res["left_fit"], res["right_fit"]
            fails.append((item["bag"], item["frame"],
                          f"L={lf['kind']}({lf['residual']}) R={rf['kind']}({rf['residual']})"))

    pass_rate = n_ok / len(test_set)
    if verbose:
        print(f"{test_set_path}: {n_ok}/{len(test_set)} = {100*pass_rate:.1f}%")
        for b, fr, r in fails:
            print(f"  {b} #{fr}: {r[:90]}")
    return pass_rate, fails


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/Volumes/T7/Dataset")
    parser.add_argument("--test-set", default="test_set.json")
    args = parser.parse_args()
    evaluate(args.dataset, args.test_set)
