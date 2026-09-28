#!/usr/bin/env python3
"""GIF по каждому из трёх подходов эксперимента 14 — как он работает, покадрово.

На каждый прогон и каждый подход — свой GIF в своей папке:

  output/approach1_topdown_contrast/  — «смотреть с большой высоты»:
        сырой вид сверху -> контраст, нормированный по глубине -> граница
        контраста и геометрия по ней;
  output/approach2_floor/             — пол: нижняя точка каждой клетки ->
        маска пола и её кромки -> вид сбоку с уровнем пола вдоль глубины;
  output/approach3_ceiling/           — свод: верхняя точка каждой клетки ->
        маска свода и её кромки -> вид сбоку с уровнем свода и обрезом поля
        зрения лидара.

Под картинками — две ленты по всему прогону с бегунком текущего кадра:
кривизна оси на 40 м и дальность, до которой граница реально наблюдается.

Кромки раскрашены по тому, что с ними сделала подгонка: синие — легли на
форму, серые крестики — скрыты поворотом (лидар видит там линию взгляда, а не
стену, и форма это предсказывает), красные — отброшены как выбросы.

Запуск:
    python make_view_gifs.py                                  # 4 разработочных прогона
    python make_view_gifs.py --bags doubleT_obstacle
    python make_view_gifs.py --validate                       # отложенный прогон
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from rail_detection import bag_path, frame_count, iter_frames
from rail_detection.views import (FOV_UP_DEG, METHODS, curvature, fit_views, half_width,
                                  shape_x)

DEV_RUNS = ["doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
            "squareT_platform_squareT_switch"]
VALIDATION_RUN = "roundT_squareT_pressureGate_squareT"

FOLDERS = {"silhouette": "approach1_topdown_contrast",
           "floor": "approach2_floor",
           "ceiling": "approach3_ceiling"}
TITLES = {"silhouette": "Подход 1 — вид с большой высоты, граница контраста",
          "floor": "Подход 2 — геометрия по полу",
          "ceiling": "Подход 3 — геометрия по своду"}
X_LIM = (-12.0, 12.0)
D_SHOW = 150.0
N_SIDE = 7000        # точек на вид сбоку
EXTRA = 1.15         # на сколько дальше предела наблюдения рисовать продолжение


def _crop(grid, img):
    """Вырезка по X_LIM и прореживание МАКСИМУМОМ 2x2 (1 м x 0.2 м): дальние
    дуги колец занимают одну строку из десяти, и при обычном уменьшении под
    размер панели они просто исчезли бы — а показать надо именно их."""
    cols = (grid["xc"] >= X_LIM[0]) & (grid["xc"] <= X_LIM[1])
    a = np.asarray(img)[:, cols]
    a = a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2]
    if a.dtype == bool:
        return a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).any(axis=(1, 3))
    b = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # клетки 2x2 целиком пустые
        return np.nanmax(b, axis=(1, 3))


def _u8(a, lo, hi):
    """В uint8 для хранения: NaN -> 0 (рисуется белым)."""
    v = np.clip((np.nan_to_num(a, nan=lo - 1) - lo) / (hi - lo), 0, 1)
    out = np.round(v * 254).astype(np.uint8) + 1
    out[~np.isfinite(a)] = 0
    return out


def collect(dataset, bag, stride, max_frames=None):
    rng = np.random.default_rng(0)
    priors = {}
    recs = []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=stride,
                                           max_frames=max_frames):
        grid, res = fit_views(points, priors)
        x = points["x"].astype(float)
        d = -points["y"].astype(float)
        z = points["z"].astype(float)
        ok = (np.abs(x) + np.abs(d) + np.abs(z) > 0.1) & (d > 0) & (d < D_SHOW)
        sel = np.where(ok)[0]
        if len(sel) > N_SIDE:
            # вид сбоку: половина точек равномерно по глубине, а не по счёту —
            # иначе ближние 10 м забирают всё и дальний пол/свод не виден
            dq = d[sel]
            wts = np.clip(dq, 3, None) ** 2
            sel = rng.choice(sel, N_SIDE, replace=False, p=wts / wts.sum())
        rec = {"idx": idx, "n_total": n_total,
               "side": np.column_stack([d[sel], z[sel]]).astype(np.float32)}
        count = grid["count"].astype(float)
        for m in METHODS:
            r = res[m]
            fit = r["fit"]
            priors[m] = fit
            item = {"fit": None}
            if m == "silhouette":
                raw = np.log1p(count) / max(np.log1p(count.max()), 1e-9)
                item["imgA"] = _u8(_crop(grid, raw), 0, 1)
                item["imgB"] = _u8(_crop(grid, r["image"]), 0, 1)
            elif m == "floor":
                if r.get("level") is not None:
                    rel = grid["zmin"] - np.polyval(r["level"], grid["dc"])[:, None]
                    item["level"] = r["level"]
                else:
                    rel = grid["zmin"] * np.nan
                    item["level"] = None
                item["imgA"] = _u8(_crop(grid, rel), -0.6, 2.4)
                item["level_samples"] = r["level_samples"]
            else:
                ld, lz, seen = r["level_samples"]
                if len(ld) and seen.any():
                    lvl = np.interp(grid["dc"], ld[seen], lz[seen])
                    rel = grid["zmax"] - lvl[:, None]
                else:
                    rel = grid["zmax"] * np.nan
                item["imgA"] = _u8(_crop(grid, rel), -2.4, 0.6)
                item["level_samples"] = r["level_samples"]
            if r.get("mask") is not None:
                item["mask"] = _crop(grid, r["mask"])
            if fit is not None:
                sm = fit["samples"]
                item["fit"] = {
                    "shape": fit["shape"], "offsets": fit["offsets"], "deg": fit["deg"],
                    "reach": fit["reach"], "reach_side": fit["reach_side"],
                    "hw": half_width(fit), "k40": curvature(fit, 40.0),
                    "rms": fit["rms"],
                    "samples": {k: np.asarray(v) for k, v in sm.items()},
                }
            rec[m] = item
        recs.append(rec)
        print(f"\r  {bag}: кадр {idx}/{n_total}", end="", flush=True)
    print()
    return recs


SHAPE_NAME = {1: "прямая", 2: "дуга", 3: "кубика (переходная/S-кривая)"}


def _draw_fit(ax, f, dd):
    if f is None:
        return
    sm = f["samples"]
    vis = sm["inlier"] & ~sm["hidden"]
    out = ~sm["inlier"] & ~sm["hidden"]
    ax.scatter(sm["x"][vis], sm["d"][vis], s=5, c="#2a6fdb", zorder=4, linewidths=0)
    ax.scatter(sm["x"][sm["hidden"]], sm["d"][sm["hidden"]], s=9, c="#888888", marker="x",
               zorder=4, linewidths=0.7)
    ax.scatter(sm["x"][out], sm["d"][out], s=7, c="#e03131", zorder=4, linewidths=0)
    for side in ("left", "right"):
        cut = f["reach_side"].get(side)
        if cut is None:
            continue
        xx = shape_x(f, dd, side)
        seen = dd <= cut
        ahead = (~seen) & (dd <= (f["reach"] or cut) * EXTRA)
        ax.plot(xx[seen], dd[seen], c="#2fbf4a", lw=2.2, zorder=5)
        ax.plot(xx[ahead], dd[ahead], c="#2fbf4a", lw=1.2, ls=":", zorder=5)
    xc = shape_x(f, dd)
    lim = dd <= (f["reach"] or 40) * EXTRA
    ax.plot(xc[lim], dd[lim], c="#d6336c", lw=1.2, ls="--", zorder=5)
    if f["reach"]:
        ax.axhline(f["reach"], c="crimson", lw=0.8, ls="--", alpha=0.6)


def _img(ax, a, cmap, title):
    ax.imshow(np.ma.masked_equal(a, 0), origin="lower", aspect="auto", cmap=cmap,
              extent=[X_LIM[0], X_LIM[1], 2.0, 150.0], vmin=1, vmax=255,
              interpolation="nearest")
    ax.set_title(title, fontsize=7.5)


def _frame_axes(ax, ylabel=True):
    ax.set_xlim(*X_LIM)
    ax.set_ylim(0, D_SHOW)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.tick_params(labelsize=6)
    if ylabel:
        ax.set_ylabel("глубина, м", fontsize=7)
    ax.set_xlabel("вбок, м", fontsize=6.5)


def render(recs, bag, method, dpi=80):
    ks = np.array([1000 * r[method]["fit"]["k40"] if r[method]["fit"] else np.nan for r in recs])
    reach = np.array([r[method]["fit"]["reach"] if r[method]["fit"] and r[method]["fit"]["reach"]
                      else np.nan for r in recs])
    xs = np.array([r["idx"] for r in recs], float)
    klim = max(float(np.nanpercentile(np.abs(ks), 98)) * 1.2 if np.any(np.isfinite(ks)) else 1, 1.0)
    rmed = float(np.nanmedian(reach)) if np.any(np.isfinite(reach)) else 0.0
    fail = ~np.isfinite(reach)

    fig = Figure(figsize=(9.2, 9.6), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 3, height_ratios=[5.2, 0.8, 0.8], hspace=0.42, wspace=0.28,
                          left=0.07, right=0.98, top=0.83, bottom=0.05)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    axK = fig.add_subplot(gs[1, :])
    axR = fig.add_subplot(gs[2, :])
    dd = np.linspace(2, D_SHOW, 300)
    fig.text(0.5, 0.878, "зелёная — стены (пунктир — продолжение за предел наблюдения), "
             "малиновая — ось;\nсиние точки — кромки, легшие на форму, серые × — скрыты поворотом, "
             "красные — выбросы", fontsize=6.3, ha="center")
    images = []
    for i, r in enumerate(recs):
        for a in (axA, axB, axC, axK, axR):
            a.clear()
        it = r[method]
        f = it["fit"]
        if method == "silhouette":
            _img(axA, it["imgA"], "gray_r", "1. вид сверху как есть:\nплотность падает как d^-2.5")
            _img(axB, it["imgB"], "gray_r", "2. контраст по глубине:\nдальний тоннель так же ярок")
            if "mask" in it:
                axC.imshow(np.ma.masked_equal(it["mask"].astype(np.uint8), 0), origin="lower",
                           aspect="auto", cmap="Greys", vmin=0, vmax=1.6,
                           extent=[X_LIM[0], X_LIM[1], 2.0, 150.0], interpolation="nearest")
            _draw_fit(axC, f, dd)
            axC.set_title("3. граница контраста -> кромки\n-> одна форма на обе стены", fontsize=7.5)
            for a, yl in ((axA, True), (axB, False), (axC, False)):
                _frame_axes(a, yl)
        else:
            if method == "floor":
                _img(axA, it["imgA"], "viridis",
                     "1. нижняя точка клетки\nотносительно пола (-0.6…+2.4 м)")
                maskt = "2. маска пола (−0.35…+0.25 м)\n-> кромки -> форма"
            else:
                _img(axA, it["imgA"], "magma",
                     "1. верхняя точка клетки\nотносительно свода (−2.4…+0.6 м)")
                maskt = "2. маска свода (не ниже 0.5 м)\n-> кромки -> форма"
            if "mask" in it:
                axB.imshow(np.ma.masked_equal(it["mask"].astype(np.uint8), 0), origin="lower",
                           aspect="auto", cmap="Greys", vmin=0, vmax=1.6,
                           extent=[X_LIM[0], X_LIM[1], 2.0, 150.0], interpolation="nearest")
            _draw_fit(axB, f, dd)
            axB.set_title(maskt, fontsize=7.5)
            _frame_axes(axA, True)
            _frame_axes(axB, False)
            # вид сбоку: как меряется уровень поверхности вдоль глубины
            sd = r["side"]
            axC.scatter(sd[:, 1], sd[:, 0], s=0.4, c="#aaaaaa", alpha=0.5, linewidths=0)
            if method == "floor":
                ld, lz = it["level_samples"]
                axC.scatter(lz, ld, s=8, c="#1c7ed6", zorder=4)
                if it.get("level") is not None:
                    zf = np.polyval(it["level"], dd)
                    axC.plot(zf, dd, c="#1c7ed6", lw=1.4)
                    axC.fill_betweenx(dd, zf - 0.35, zf + 0.25, color="#1c7ed6", alpha=0.18)
                axC.set_title("3. вид сбоку: уровень пола\nвдоль глубины (тангаж + уклон)",
                              fontsize=7.5)
            else:
                ld, lz, seen = it["level_samples"]
                axC.scatter(lz[seen], ld[seen], s=8, c="#e8590c", zorder=4)
                axC.scatter(lz[~seen], ld[~seen], s=12, c="#868e96", marker="x", zorder=4)
                axC.plot(dd * np.tan(np.radians(FOV_UP_DEG)), dd, c="k", lw=0.9, ls=":")
                if seen.any():
                    axC.fill_betweenx(ld[seen], lz[seen] - 0.5, lz[seen], color="#e8590c",
                                      alpha=0.2)
                axC.set_title("3. вид сбоку: уровень свода; пунктир —\nверхний луч лидара +14.4°",
                              fontsize=7.5)
            axC.set_xlim(-3, 7)
            axC.set_ylim(0, D_SHOW)
            axC.set_xlabel("высота z, м", fontsize=6.5)
            axC.tick_params(labelsize=6)

        if f is None:
            head = "геометрия не построена"
        else:
            shape = SHAPE_NAME.get(f["deg"], "?")
            if f["deg"] >= 2 and abs(f["k40"]) > 1e-5:
                shape += f", R₄₀={1 / abs(f['k40']):.0f} м {'направо' if f['k40'] > 0 else 'налево'}"
            head = (f"{shape};  полуширина {f['hw']:.2f} м;  наблюдается до "
                    f"{f['reach'] or 0:.0f} м;  невязка {100 * f['rms']:.0f} см")
        fig.suptitle(f"{TITLES[method]}\n{bag}  кадр {r['idx']}/{r['n_total']}\n{head}",
                     fontsize=9)

        axK.axhline(0, c="gray", lw=0.7)
        axK.plot(xs, ks, c="#495057", lw=1.0)
        if fail.any():
            axK.scatter(xs[fail], np.full(fail.sum(), -klim * 0.8), s=14, c="crimson", marker="|")
        axK.axvline(r["idx"], c="crimson", lw=1.3)
        axK.set_xlim(xs.min(), max(xs.max(), xs.min() + 1))
        axK.set_ylim(-klim, klim)
        axK.tick_params(labelsize=6)
        axK.set_title("кривизна оси на 40 м, 1/км (плюс — направо; красные штрихи — нет геометрии)",
                      fontsize=7)
        axR.plot(xs, reach, c="crimson", lw=1.0)
        axR.axhline(rmed, c="gray", lw=0.7, ls="--")
        axR.axvline(r["idx"], c="crimson", lw=1.3)
        axR.set_xlim(xs.min(), max(xs.max(), xs.min() + 1))
        axR.set_ylim(0, D_SHOW)
        axR.tick_params(labelsize=6)
        axR.set_xlabel("кадр записи", fontsize=6.5)
        axR.set_title(f"докуда граница наблюдается, м (медиана {rmed:.0f})", fontsize=7)

        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=128))
        print(f"\r  {method}: отрисовано {i + 1}/{len(recs)}", end="", flush=True)
    print()
    return images, rmed


def build(dataset, bag, out_root, stride, target, fps, max_frames, methods):
    if stride is None:
        stride = max(1, round(frame_count(bag_path(dataset, bag)) / target))
    print(f"\n=== {bag} (шаг {stride}) ===")
    recs = collect(dataset, bag, stride, max_frames)
    rows = []
    for m in methods:
        images, rmed = render(recs, bag, m)
        folder = out_root / FOLDERS[m]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{bag}.gif"
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
        ok = sum(1 for r in recs if r[m]["fit"] is not None)
        rows.append((m, len(recs), ok, rmed, path.stat().st_size / 1e6))
        print(f"  {path} ({rows[-1][-1]:.1f} МБ)")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=None)
    p.add_argument("--validate", action="store_true",
                   help=f"GIF по отложенному прогону {VALIDATION_RUN}")
    p.add_argument("--methods", nargs="*", default=list(METHODS))
    p.add_argument("--out", default="output")
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--target-frames", type=int, default=130)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=8.0)
    a = p.parse_args()
    if a.validate:
        bags = [VALIDATION_RUN]
    else:
        bags = a.bags or DEV_RUNS
        if VALIDATION_RUN in bags:
            raise SystemExit(f"{VALIDATION_RUN} отложен для валидации: только через --validate")
    out_root = Path(a.out)
    summary = []
    for bag in bags:
        for row in build(a.dataset, bag, out_root, a.stride, a.target_frames, a.fps,
                         a.max_frames, a.methods):
            summary.append((bag,) + row)
    print("\n=== Итог ===")
    for bag, m, n, ok, rmed, mb in summary:
        print(f"{bag:38s} {m:10s} кадров {n:4d}, с геометрией {ok:4d}, "
              f"видно до {rmed:4.0f} м, {mb:5.1f} МБ")


if __name__ == "__main__":
    main()
