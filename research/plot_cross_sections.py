#!/usr/bin/env python3
"""Срезы тоннеля ПЕРПЕНДИКУЛЯРНО оси Y — профиль (X, Z) на разных глубинах.

Это диагностика для глаз: вид сверху показывает, где стены, но не показывает,
почему детектор желоба и рельсов на кадре отказал. Здесь видно сам профиль пола
и то, на каком именно условии детектор останавливается.

Каждая панель — один срез по глубине: серые точки полосы пола (то, по чему
детектор и работает), поверх — медианный профиль Z(X), и подпись с причиной
отказа либо с найденной геометрией.

Запуск:
    python plot_cross_sections.py --bag roundT_squareT_pressureGate_squareT --frames 160 164 168 172
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from rail_detection import bag_path, iter_selected_frames

DEPTHS = [(3, 5), (5, 7), (9, 11), (13, 15), (18, 21), (24, 27), (30, 35), (35, 40)]


def groove_diagnosis(points, depth_lo, depth_hi, x_range=4.0, xbin=0.05):
    """Повторяет шаги detector.find_groove_and_rails и сообщает, на каком именно
    из них всё останавливается. Сам детектор возвращает просто None, и по нему
    нельзя понять, данных не хватило или геометрия не та."""
    x, z = points['x'].astype(float), points['z'].astype(float)
    depth = -points['y'].astype(float)
    mask = (depth >= depth_lo) & (depth < depth_hi) & (np.abs(x) < x_range)
    out = {"prof": None, "reason": None, "rec": None}
    if mask.sum() < 200:
        out["reason"] = f"точек в срезе {mask.sum()} < 200"
        return out
    xr, zr = x[mask], z[mask]
    zlo, zhi = np.percentile(zr, [1, 45])
    band = (zr >= zlo) & (zr <= zhi)
    out["raw"] = (xr[band], zr[band])
    if band.sum() < 100:
        out["reason"] = f"точек в полосе пола {band.sum()} < 100"
        return out
    xb, zb = xr[band], zr[band]

    bins = np.arange(-x_range, x_range + xbin, xbin)
    idx = np.digitize(xb, bins)
    px, pz = [], []
    for i in range(1, len(bins)):
        m = idx == i
        if m.sum() >= 3:
            px.append((bins[i - 1] + bins[i]) / 2)
            pz.append(np.median(zb[m]))
    if len(px) < 15:
        out["reason"] = f"профиль из {len(px)} точек < 15"
        return out
    px, pz = np.array(px), np.array(pz)
    out["prof"] = (px, pz)

    search = np.abs(px) < 2.5
    if search.sum() < 8:
        out["reason"] = "зона поиска |x|<2.5 почти пуста"
        return out
    i_min = np.argmin(pz[search])
    x_min, z_min = px[search][i_min], pz[search][i_min]
    out["x_min"] = x_min

    lf = (px > x_min - 0.7) & (px < x_min - 0.2)
    rf = (px > x_min + 0.2) & (px < x_min + 0.7)
    if lf.sum() < 3 or rf.sum() < 3:
        out["reason"] = f"нет плеч по бокам минимума (x={x_min:+.2f})"
        return out
    shoulder_z = (np.median(pz[lf]) + np.median(pz[rf])) / 2
    notch = shoulder_z - z_min
    if notch < 0.06:
        out["reason"] = f"желоб {notch*100:.0f} см < 6 см (x={x_min:+.2f})"
        return out

    half = z_min + 0.5 * notch
    li = np.where((px < x_min) & (pz >= half))[0]
    ri = np.where((px > x_min) & (pz >= half))[0]
    if len(li) == 0 or len(ri) == 0:
        out["reason"] = "желоб не закрывается на полувысоте"
        return out
    xl, xr_ = px[li[-1]], px[ri[0]]
    gw = xr_ - xl
    if not (0.15 < gw < 2.0):
        out["reason"] = f"ширина желоба {gw:.2f} м вне 0.15..2.0"
        return out

    peaks, _ = find_peaks(pz, prominence=0.02)
    if len(peaks) == 0:
        out["reason"] = "у профиля нет пиков"
        return out
    peak_x = px[peaks]
    rc = peak_x[(peak_x > xr_) & (peak_x < xr_ + 1.0)]
    lc = peak_x[(peak_x < xl) & (peak_x > xl - 1.0)]
    if len(rc) == 0 or len(lc) == 0:
        out["reason"] = f"нет пиков рельсов у краёв желоба ({xl:+.2f}..{xr_:+.2f})"
        return out
    gauge = rc.min() - lc.max()
    if not (0.3 < gauge < 2.2):
        out["reason"] = f"колея {gauge:.2f} м вне 0.3..2.2"
        return out
    out["rec"] = {"center": (lc.max() + rc.min()) / 2, "gauge": gauge,
                  "notch": notch, "rails": (lc.max(), rc.min())}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bag", default="roundT_squareT_pressureGate_squareT")
    p.add_argument("--frames", nargs="+", type=int, default=[160, 164, 168, 172])
    p.add_argument("--out", default="output")
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for idx, points in iter_selected_frames(bag_path(a.dataset, a.bag), a.frames):
        frames[idx] = np.array(points)

    ncols, nrows = len(a.frames), len(DEPTHS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.1 * nrows), squeeze=False)
    n_ok = 0
    for ci, fr in enumerate(a.frames):
        pts = frames.get(fr)
        for ri, (lo, hi) in enumerate(DEPTHS):
            ax = axes[ri][ci]
            if pts is None:
                ax.axis("off")
                continue
            d = groove_diagnosis(pts, lo, hi)
            if "raw" in d and d["raw"] is not None:
                ax.scatter(d["raw"][0], d["raw"][1], s=0.6, c="lightgray", alpha=0.5)
            if d["prof"] is not None:
                ax.plot(d["prof"][0], d["prof"][1], c="steelblue", lw=1.2)
            if d["rec"] is not None:
                n_ok += 1
                rl, rr = d["rec"]["rails"]
                for rx in (rl, rr):
                    ax.axvline(rx, c="limegreen", lw=1.6)
                ax.axvline(d["rec"]["center"], c="crimson", lw=1.0, ls="--")
                title = (f"#{fr}  {lo}-{hi}м\nНАЙДЕНО: центр {d['rec']['center']:+.2f}, "
                         f"колея {d['rec']['gauge']:.2f}, желоб {d['rec']['notch']*100:.0f}см")
                color = "darkgreen"
            else:
                if d.get("x_min") is not None:
                    ax.axvline(d["x_min"], c="orange", lw=1.0, ls=":")
                title = f"#{fr}  {lo}-{hi}м\nотказ: {d['reason']}"
                color = "firebrick"
            ax.set_title(title, fontsize=6.5, color=color)
            ax.set_xlim(-4, 4)
            ax.tick_params(labelsize=6)
            if ri == nrows - 1:
                ax.set_xlabel("X (вбок), м", fontsize=7)
            if ci == 0:
                ax.set_ylabel("Z (высота), м", fontsize=7)

    fig.suptitle(
        f"{a.bag}: срезы перпендикулярно оси Y (профиль пола)\n"
        "серое — точки нижней полосы среза, синее — медианный профиль Z(X), "
        "зелёное — найденные рельсы, оранжевый пунктир — минимум профиля\n"
        f"срезов с найденными рельсами: {n_ok} из {ncols * nrows}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = out_dir / f"cross_sections_{a.bag}.png"
    fig.savefig(path, dpi=125)
    print(f"Сохранено: {path}  (найдено рельсов: {n_ok}/{ncols * nrows})")


if __name__ == "__main__":
    main()
