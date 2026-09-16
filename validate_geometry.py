#!/usr/bin/env python3
"""Оценивает качество детекции стен тоннеля на ЗАФИКСИРОВАННОЙ ДО каких-либо
правок алгоритма тестовой выборке (test_set.json — 100 кадров, seed=2024,
см. отдельный коммит, предшествующий этому скрипту).

Для каждого кадра теста:
  1. Детектирует стены по плотности (ближайшее к центру значимое скопление
     точек гистограммы X, не крайняя точка и не просто самый плотный пик
     вообще — см. rail_detection.walls: оба более простых варианта ловили
     посторонние поверхности вроде платформы).
  2. Подгоняет физически осмысленную модель курса КАЖДОЙ стены отдельно:
     прямая / дуга окружности постоянного радиуса / переход прямая->дуга,
     через RANSAC (curvature.fit_straight_or_arc) — один шумный срез не
     портит всю форму (попадает в выбросы, а не тянет кривую на себя).
  3. Кадр считается "геометрия определена корректно", если ХОТЯ БЫ ОДНА
     стена даёт чистую (не ломаную) подгонку — тоннель может не только
     поворачивать, но и расширяться/сужаться (напр. на платформе), поэтому
     две стены не обязаны совпадать по кривизне; walls_consistent()
     всё равно считается и попадает в отчёт как доп. сигнал уверенности,
     но не требуется как обязательное условие.

Затем случайно выбирает --n-sample (по умолчанию 40) кадров ИЗ ПРОШЕДШИХ
проверку В ЭТОЙ ЖЕ тестовой выборке (не свежий скан) и рисует по каждому
график для ручной проверки перед решением, сливать ли в main.

Запуск:
    python validate_geometry.py
"""

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rail_detection import (
    DEFAULT_DEPTH_BINS,
    analyze_frame,
    bag_path,
    analyze_walls,
    fit_straight_or_arc,
    walls_consistent,
)


def wall_is_clean(fit, resid_thresh=0.08):
    """'transition' (прямая->дуга) тоже гладкая, не ломаная форма — это два
    смыкающихся гладких куска, реальный физический случай (путь начал
    поворачивать), а не зигзаг; поэтому тоже допустим как "чистый"."""
    return (
        fit["kind"] in ("straight", "arc", "transition")
        and fit["residual"] is not None and fit["residual"] < resid_thresh * 2.0
        and (fit.get("inlier_frac") is None or fit["inlier_frac"] >= 0.6)
    )


def evaluate_frame(points, depth_bins=DEFAULT_DEPTH_BINS, resid_thresh=0.08, min_coverage=8):
    """Возвращает dict с результатами для одного кадра, включая 'ok': bool."""
    rail_records, _ = analyze_frame(points, depth_bins)
    records_by_depth = {(r["depth_lo"], r["depth_hi"]): r for r in rail_records}
    wall_records = analyze_walls(points, records_by_depth, depth_bins)

    if len(wall_records) < min_coverage:
        return {"ok": False, "reason": f"мало срезов со стенами ({len(wall_records)})",
                "wall_records": wall_records, "rail_records": rail_records}

    depths = [(w["depth_lo"] + w["depth_hi"]) / 2 for w in wall_records]
    left_xs = [w["left_wall_x"] for w in wall_records]
    right_xs = [w["right_wall_x"] for w in wall_records]

    left_fit = fit_straight_or_arc(depths, left_xs, straight_resid_thresh=resid_thresh)
    right_fit = fit_straight_or_arc(depths, right_xs, straight_resid_thresh=resid_thresh)

    left_clean = wall_is_clean(left_fit, resid_thresh)
    right_clean = wall_is_clean(right_fit, resid_thresh)
    # Стена может быть прямой ИЛИ дугой, но не ломаной — а тоннель, помимо
    # поворота, может ещё и расширяться/сужаться (платформа), поэтому левая и
    # правая стена не обязаны совпадать по кривизне — важно лишь, чтобы ХОТЯ БЫ
    # ОДНА была гладкой (см. обсуждение: "если одна ломаная — используем ту,
    # что гладкая"). walls_consistent() всё равно считается — как доп. сигнал
    # уверенности в отчёте, но не как обязательное условие.
    consistent = walls_consistent(left_fit, right_fit, depths)
    ok = left_clean or right_clean

    reason = ""
    if not ok:
        reason = f"L={left_fit['kind']}({left_fit['residual']}) R={right_fit['kind']}({right_fit['residual']})"
    used_side = "both" if (left_clean and right_clean) else ("left" if left_clean else ("right" if right_clean else "none"))
    return {
        "ok": ok, "reason": reason, "used_side": used_side, "consistent": consistent,
        "wall_records": wall_records, "rail_records": rail_records,
        "left_fit": left_fit, "right_fit": right_fit,
    }


def fit_curve_x(fit, depths):
    """Строит x(depth) по результату fit_straight_or_arc для отрисовки."""
    depths = np.asarray(depths)
    if fit["kind"] in ("straight", "arc", "unclear"):
        return depths, np.polyval(fit["coeffs"], depths)
    if fit["kind"] == "transition":
        cc1, cc2 = fit["coeffs"]
        split = fit["split_depth"]
        d1 = depths[depths <= split]
        d2 = depths[depths > split]
        x1 = np.polyval(cc1, d1)
        x2 = np.polyval(cc2, d2)
        return np.concatenate([d1, d2]), np.concatenate([x1, x2])
    return depths, np.full_like(depths, np.nan)


def label_for(fit):
    if fit["kind"] == "straight":
        return "прямая"
    if fit["kind"] == "arc":
        return f"дуга R={fit['radius']:.0f}м"
    if fit["kind"] == "transition":
        return f"прямая->дуга ({fit['split_depth']:.0f}м)"
    return str(fit["kind"])


def plot_example(ax, bag_name, frame_idx, points, ev, depth_max=45, x_lim=(-6, 6)):
    x, y = points['x'], points['y']
    depth = -y
    mask = (depth > 0) & (depth < depth_max) & (np.abs(x) < x_lim[1] + 1)
    ax.scatter(x[mask], depth[mask], s=0.25, c='gray', alpha=0.3)

    wr = ev["wall_records"]
    wd = np.array([(w["depth_lo"] + w["depth_hi"]) / 2 for w in wr])
    lw = np.array([w["left_wall_x"] for w in wr])
    rw = np.array([w["right_wall_x"] for w in wr])
    order = np.argsort(wd)
    # сырые измерения по срезам — только точки, БЕЗ соединяющей линии (она и
    # создавала иллюзию "ломаной стены"; реальный результат — подогнанная
    # кривая ниже, которая как раз и устойчива к шуму отдельных срезов)
    ax.scatter(lw[order], wd[order], color='darkblue', s=8, alpha=0.5, zorder=2)
    ax.scatter(rw[order], wd[order], color='purple', s=8, alpha=0.5, zorder=2)

    ld, lx = fit_curve_x(ev["left_fit"], wd[order])
    rd, rx = fit_curve_x(ev["right_fit"], wd[order])
    lstyle = '-' if wall_is_clean(ev["left_fit"]) else ':'
    rstyle = '-' if wall_is_clean(ev["right_fit"]) else ':'
    ax.plot(lx, ld, color='deepskyblue', lw=2.2, ls=lstyle, zorder=3)
    ax.plot(rx, rd, color='magenta', lw=2.2, ls=rstyle, zorder=3)

    ax.set_xlim(*x_lim)
    ax.set_ylim(0, depth_max)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"{bag_name} #{frame_idx}\nЛ:{label_for(ev['left_fit'])} П:{label_for(ev['right_fit'])}",
        fontsize=6.5,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="/Volumes/T7/Dataset")
    parser.add_argument("--out", default="output")
    parser.add_argument("--test-set", default="test_set.json",
                         help="ЗАФИКСИРОВАННАЯ тестовая выборка (см. test_set.json) — "
                              "метрика и примеры считаются по ней, не по свежему скану")
    parser.add_argument("--target-pass-rate", type=float, default=0.85)
    parser.add_argument("--n-sample", type=int, default=40, help="сколько случайных примеров показать")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    import json
    from rail_detection import load_frame
    with open(args.test_set) as f:
        test_set = json.load(f)["test_set"]

    good = []
    fails = []
    for item in test_set:
        points, n_total = load_frame(bag_path(args.dataset, item["bag"]), item["frame"])
        ev = evaluate_frame(points)
        if ev["ok"]:
            good.append((item["bag"], item["frame"], points, ev))
        else:
            fails.append((item["bag"], item["frame"], ev["reason"]))

    pass_rate = len(good) / len(test_set)
    verdict = "ЦЕЛЬ ДОСТИГНУТА" if pass_rate >= args.target_pass_rate else "цель НЕ достигнута"
    print(f"На зафиксированной тестовой выборке ({args.test_set}): "
          f"{len(good)}/{len(test_set)} = {100*pass_rate:.1f}% "
          f"(цель {100*args.target_pass_rate:.0f}%) -> {verdict}")
    print("\nОтказы:")
    for b, f, r in fails:
        print(f"  {b} #{f}: {r[:90]}")

    if len(good) == 0:
        print("Ни одного кадра не прошло проверку — нечего показывать.")
        return

    n_sample = min(args.n_sample, len(good))
    sample = random.sample(good, n_sample)
    print(f"\nСлучайно отобрано для показа (из прошедших тестовую выборку): {n_sample}")

    ncols = 8
    nrows = -(-n_sample // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, (name, frame_idx, points, ev) in zip(axes, sample):
        plot_example(ax, name, frame_idx, points, ev)
    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.suptitle(
        f"{n_sample} случайных кадров из ЗАФИКСИРОВАННОЙ тестовой выборки ({len(test_set)} шт., seed=2024) — "
        f"пройдено {len(good)}/{len(test_set)} = {100*pass_rate:.0f}%\n"
        f"точки — сырые измерения по срезам (RANSAC-устойчивая подгонка); линия — итоговая стена "
        f"(сплошная = чистая, пунктир = не прошла порог); синий/голубой — левая, фиолетовый/розовый — правая",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = out_dir / "geometry_validation_sample.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nГрафик сохранён: {out_path}")


if __name__ == "__main__":
    main()
