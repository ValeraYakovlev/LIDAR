#!/usr/bin/env python3
"""Эксперимент 3, шаг 1: продольное смещение Δs, и чем его проверить.

Накопление кадров держится целиком на том, знаем ли мы, на сколько поезд проехал
между кадрами. Продольное смещение в тоннеле — классический вырожденный случай
лидарной одометрии: тоннель через два метра выглядит так же, и полный ICP здесь
уходит в дрейф. Поэтому оценивается не шесть степеней свободы, а ОДНА, и сразу
несколькими независимыми способами — чтобы было чем проверить.

  (а) По таймстампам и скорости. Δt между кадрами известен точно из самих данных
      (пер-точечный timestamp): ровно 0.1000 с на всех шести записях. Скорость
      неизвестна, поэтому сам по себе Δs этот способ не даёт — он даёт ПЕРЕСЧЁТ
      Δs в скорость, и это первая проверка на осмысленность: метро не разгоняется
      на 5 м/с² и не едет 200 км/ч.

  (б) Корреляция профиля плотности по глубине — ОСНОВНОЙ способ.
  (в) Перекрытие занятости трёхмерной решётки — ПРОВЕРОЧНЫЙ.

Устройство обоих и почему они расставлены именно так (у (в) нашёлся системный
дефект — самоподобие на нулевом сдвиге) — в `rail_detection/shift.py`.

Что печатается и зачем: расхождение (б) и (в) — прямая мера надёжности; путь,
проинтегрированный по записи, сверяется с «скорость × длительность»; а прогон
`doubleT_obstacle`, где поезд заведомо стоит (knowledge.md §6), обязан дать ноль.

Запуск:
    python exp_shift_probe.py --bags doubleT_platform --stride 1
"""

import argparse
from pathlib import Path

import numpy as np

from rail_detection import DEFAULT_BAGS, bag_path, iter_frames, to_track_coords
from rail_detection.shift import MAX_SHIFT, estimate_shift, frame_time
from rail_detection.tracker import TunnelTracker
from rail_detection.tunnel_frame import V_HI, V_LO


def run_bag(dataset, bag, stride, max_frames=None):
    tracker = TunnelTracker()
    prev = None
    rows = []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=stride,
                                            max_frames=max_frames):
        res = tracker.update(points, steps=stride)
        if res is None:
            prev = None
            continue
        x = points['x'].astype(float)
        y = points['y'].astype(float)
        z = points['z'].astype(float)
        d, u, v = to_track_coords(x, y, z, res["frame"])
        band = (v >= V_LO) & (v <= V_HI) & (d > 3) & (d < 60)
        cur = {"d": d[band], "u": u[band], "v": v[band],
               "t": frame_time(points), "idx": idx}
        if prev is not None:
            est = estimate_shift(prev, cur, max_shift=MAX_SHIFT * stride,
                                 cross_check=True)
            rows.append({"idx": idx, "dt": cur["t"] - prev["t"],
                         "den": est["shift"], "den_q": est["contrast"],
                         "ok": est["ok"], "occ": est["shift_occupancy"],
                         "occ_q": est["occupancy_contrast"]})
        prev = cur
    return rows, n_total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=list(DEFAULT_BAGS) + ["doubleT_obstacle"])
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--plot", default="output/shift_estimates.png")
    a = p.parse_args()

    all_rows = {}
    print(f"{'прогон':38s} {'пар':>5s} {'Δt, с':>7s} {'Δs, м':>7s} {'контр.':>7s} "
          f"{'не изм.':>8s} {'расх. с занят.':>15s} {'>10см':>7s} {'путь, м':>8s} "
          f"{'v, км/ч':>8s}")
    for bag in a.bags:
        rows, n_total = run_bag(a.dataset, bag, a.stride, a.max_frames)
        if not rows:
            print(f"{bag:38s} нет пар")
            continue
        all_rows[bag] = rows
        occ = np.array([r["occ"] for r in rows])
        den = np.array([r["den"] for r in rows])
        dt = np.array([r["dt"] for r in rows])
        ok = np.array([r["ok"] for r in rows])
        both = np.isfinite(occ) & np.isfinite(den)
        diff = np.abs(occ - den)[both]
        path = float(np.nansum(np.where(ok, den, 0.0)))
        speed = float(np.nanmedian(den[ok] / dt[ok])) * 3.6 if ok.any() else float("nan")
        print(f"{bag:38s} {len(rows):5d} {np.median(dt):7.4f} {np.nanmedian(den):7.3f} "
              f"{np.nanmedian([r['den_q'] for r in rows]):7.1f} {1 - ok.mean():8.1%} "
              f"{np.median(diff):15.3f} {np.mean(diff > 0.10):7.1%} {path:8.1f} {speed:8.1f}")

    _plot(a.plot, all_rows, a.stride)


def _plot(path, all_rows, stride):
    if not all_rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n = len(all_rows)
    fig = Figure(figsize=(7.5, 1.9 * n + 0.6), dpi=130)
    FigureCanvasAgg(fig)
    for i, (bag, rows) in enumerate(all_rows.items(), 1):
        ax = fig.add_subplot(n, 1, i)
        xs = [r["idx"] for r in rows]
        ax.plot(xs, [r["occ"] / r["dt"] for r in rows], c="tab:blue", lw=0.9,
                alpha=0.6, label="по занятости (сверка)")
        ax.plot(xs, [r["den"] / r["dt"] for r in rows], c="tab:orange", lw=1.3,
                label="по плотности (основной)")
        bad = [r["idx"] for r in rows if not r["ok"]]
        if bad:
            ax.scatter(bad, np.zeros(len(bad)) - 0.5, s=9, c="crimson", marker="|",
                       label="не измерено")
        ax.axhline(0, c="gray", lw=0.6)
        ax.set_ylabel("скорость, м/с", fontsize=7)
        ax.set_title(bag, fontsize=8)
        ax.tick_params(labelsize=6.5)
        ax.set_ylim(-1, MAX_SHIFT / (0.1 * stride))
        if i == 1:
            ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"\nкартинка: {path}")


if __name__ == "__main__":
    main()
