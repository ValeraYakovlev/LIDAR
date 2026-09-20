#!/usr/bin/env python3
"""GIF: путь по контрасту вида сверху + габарит, перпендикулярный этому пути.

Три панели на кадр:

  слева   — вид сверху, контраст нормирован по глубине. Зелёным — граница
            тоннеля, малиновым — найденный путь, красным — края габаритного
            коридора (отступ 1.1 м ПО НОРМАЛИ к пути, поэтому в проекции на
            повороте коридор шире самого габарита). Синяя черта — след той
            плоскости, которая показана на средней панели: она перпендикулярна
            пути, а не оси сенсора.
  справа  — сама эта плоскость: сечение на выбранной дальности ВДОЛЬ пути.
  сверху    Прямоугольник 2.2 × 3.3 м наклонён по измеренному крену пути
            (серый пунктир — тот же габарит без наклона, для сравнения).
  справа  — лента по всему прогону: дальность ближайшей находки в габарите на
  снизу     каждом кадре. Оранжевое — сырая находка, красное — подтверждённая
            приближением на измеренное Δs.

Срез выбирается сам: подтверждённая находка, иначе ближайшая сырая, иначе
дальность по умолчанию.

Запуск:
    python make_gauge_gifs.py                    # 5 прогонов разработки
    python make_gauge_gifs.py --validate         # отложенный прогон
    python make_gauge_gifs.py --bags doubleT_obstacle
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from PIL import Image

from rail_detection import bag_path, frame_count, iter_frames
from rail_detection.contrast_gauge import (ContrastGauge, corridor_lines, gauge_corners,
                                           path_at, pose_at, slice_points)
from rail_detection.views import shape_x

DEV_RUNS = ["doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
            "squareT_platform_squareT_switch", "doubleT_obstacle"]
VALIDATION_RUN = "roundT_squareT_pressureGate_squareT"
FOLDER = "approach4_contrast_gauge"

X_LIM = (-12.0, 12.0)
D_SHOW = 150.0
N_SLICE = 6000
DEFAULT_SLICE = 60.0


def _crop(grid, img):
    """Вырезка по X_LIM и прореживание МАКСИМУМОМ 2x2 (1 м x 0.2 м): дальняя
    стена зондируется раз в 1-3 м, и при обычном уменьшении под размер панели
    она бы пропала."""
    cols = (grid["xc"] >= X_LIM[0]) & (grid["xc"] <= X_LIM[1])
    a = np.asarray(img)[:, cols]
    a = a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2]
    b = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(b, axis=(1, 3))


def _u8(a, lo=0.0, hi=1.0):
    v = np.clip((np.nan_to_num(a, nan=lo - 1) - lo) / (hi - lo), 0, 1)
    out = np.round(v * 254).astype(np.uint8) + 1
    out[~np.isfinite(a)] = 0
    return out


def collect(dataset, bag, stride, max_frames=None):
    cg = ContrastGauge()
    rng = np.random.default_rng(0)
    recs = []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=stride,
                                           max_frames=max_frames):
        res = cg.update(points, steps=stride)
        rec = {"idx": idx, "n_total": n_total, "ok": res is not None}
        if res is None:
            recs.append(rec)
            print(f"\r  {bag}: кадр {idx}/{n_total}", end="", flush=True)
            continue
        fit = res["fit"]
        dd = np.linspace(2, D_SHOW, 300)
        left, path, right = corridor_lines(res, dd)
        x, y, _ = res["xyz"]
        inside = res["inside"]

        cl = res["clusters"]
        dist = (res["confirmed"] or (cl[0]["dist"] if cl else None)
                or min(DEFAULT_SLICE, 0.8 * res["reach"]))
        m, ht = slice_points(res, dist)
        sel = np.where(m)[0]
        if len(sel) > N_SLICE:
            sel = rng.choice(sel, N_SLICE, replace=False)
        hit = np.where(inside)[0]
        if len(hit) > 4000:
            hit = rng.choice(hit, 4000, replace=False)

        # След плоскости среза на виде сверху: путь в этой точке плюс нормаль
        pd = res["path"]["d"]
        x_p, psi, arc = path_at(res["path"], pd)
        k = int(np.argmin(np.abs(arc - dist)))
        d_c, x_c, psi_c = pd[k], x_p[k], psi[k]
        cut = np.array([[x_c - 3.0 * np.cos(psi_c), d_c + 3.0 * np.sin(psi_c)],
                        [x_c + 3.0 * np.cos(psi_c), d_c - 3.0 * np.sin(psi_c)]])

        # Положение кластеров на виде сверху: дальность вдоль пути -> глубина,
        # поперечное смещение -> x (делённое на cos ψ, потому что u — по нормали)
        marks = []
        for q in cl[:3]:
            j = int(np.argmin(np.abs(arc - q["dist"])))
            marks.append((x_p[j] + q["u"] / np.cos(psi[j]), pd[j]))

        rec.update({
            "marks": np.array(marks, dtype=np.float32).reshape(-1, 2),
            "img": _u8(_crop(res["grid"], res["silhouette"]["image"])),
            "walls": np.column_stack([shape_x(fit, dd, "left"),
                                      shape_x(fit, dd, "right"), dd]).astype(np.float32),
            "reach_side": fit["reach_side"], "reach": res["reach"],
            "corridor": np.column_stack([left, path, right, dd]).astype(np.float32),
            "hits": np.column_stack([x[hit], -y[hit]]).astype(np.float32),
            # Срез рисуется в координатах ПУТИ (u, v), а не в системе вагона:
            # иначе наклон габарита на картинке пропадёт — он ровно на него и
            # повёрнут, и прямоугольник всегда выглядел бы прямым.
            "slice": np.column_stack([res["u"][sel], res["v"][sel]]).astype(np.float32),
            "slice_in": inside[sel],
            "slice_dist": float(dist), "slice_ht": float(ht),
            "pose": pose_at(res, dist), "cut": cut.astype(np.float32),
            "clusters": cl, "confirmed": res["confirmed"], "shift": res["shift"],
            "has_pose": res["has_pose"], "offset": res["offset"],
            "mode": res["path"]["mode"],
            "deg": fit["deg"],
        })
        recs.append(rec)
        print(f"\r  {bag}: кадр {idx}/{n_total}", end="", flush=True)
    print()
    return recs


def render(recs, bag, dpi=85):
    xs = np.array([r["idx"] for r in recs], float)
    raw = np.array([(r["clusters"][0]["dist"] if r.get("clusters") else np.nan)
                    for r in recs], float)
    conf = np.array([(r["confirmed"] if r.get("confirmed") else np.nan) for r in recs], float)
    reach = np.array([r.get("reach", np.nan) for r in recs], float)

    fig = Figure(figsize=(11.2, 7.6), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.6, 1.0],
                          left=0.06, right=0.98, top=0.86, bottom=0.07,
                          wspace=0.22, hspace=0.32)
    axT = fig.add_subplot(gs[:, 0])
    axS = fig.add_subplot(gs[0, 1])
    axR = fig.add_subplot(gs[1, 1])
    images = []
    for i, r in enumerate(recs):
        for a in (axT, axS, axR):
            a.clear()
        if not r["ok"]:
            head = f"{bag}  кадр {r['idx']}/{r['n_total']}\nгеометрия не построена"
        else:
            axT.imshow(np.ma.masked_equal(r["img"], 0), origin="lower", aspect="auto",
                       cmap="gray_r", vmin=1, vmax=255, interpolation="nearest",
                       extent=[X_LIM[0], X_LIM[1], 2.0, 150.0])
            w = r["walls"]
            for j, side in ((0, "left"), (1, "right")):
                cut = r["reach_side"].get(side) or 0.0
                seen = w[:, 2] <= cut
                axT.plot(w[seen, j], w[seen, 2], c="#2fbf4a", lw=2.0, zorder=4)
            c = r["corridor"]
            vis = c[:, 3] <= r["reach"]
            axT.plot(c[vis, 1], c[vis, 3], c="#d6336c", lw=1.3, ls="--", zorder=5)
            axT.plot(c[vis, 0], c[vis, 3], c="#d1495b", lw=1.4, zorder=5)
            axT.plot(c[vis, 2], c[vis, 3], c="#d1495b", lw=1.4, zorder=5)
            if len(r["hits"]):
                axT.scatter(r["hits"][:, 0], r["hits"][:, 1], s=6, c="#ff8c1a",
                            marker="x", linewidths=0.7, zorder=6)
            if len(r["marks"]):
                axT.scatter(r["marks"][:, 0], r["marks"][:, 1], s=90, facecolors="none",
                            edgecolors="#d1495b", linewidths=1.6, zorder=9)
            axT.plot(r["cut"][:, 0], r["cut"][:, 1], c="#1c7ed6", lw=2.4, zorder=8)
            axT.axhline(r["reach"], c="crimson", lw=0.8, ls=":", alpha=0.7)
            axT.set_xlim(*X_LIM)
            axT.set_ylim(0, D_SHOW)
            axT.set_ylabel("глубина, м", fontsize=8)
            axT.set_xlabel("вбок, м", fontsize=8)
            axT.tick_params(labelsize=7)
            axT.set_title("вид сверху: контраст; зелёная — граница тоннеля, малиновая — путь,\n"
                          "красные — края коридора габарита, синяя черта — плоскость среза;\n"
                          "оранжевые × — точки в коробке, кружок — кластер (находка)",
                          fontsize=7.5)

            sl, si = r["slice"], r["slice_in"]
            axS.scatter(sl[~si, 0], sl[~si, 1], s=2.0, c="#9aa5b1", alpha=0.6, linewidths=0)
            if si.any():
                axS.scatter(sl[si, 0], sl[si, 1], s=6.0, c="#d1495b", alpha=0.95,
                            linewidths=0)
            theta, uc, vc = r["pose"]
            # Серый пунктир — тот же габарит с тем же центром, но без наклона:
            # отличаться от красного он обязан только креном (§25).
            axS.add_patch(Polygon(gauge_corners(0.0, uc, vc), closed=True, fill=False,
                                  edgecolor="#6b7280", lw=0.9, ls="--"))
            axS.add_patch(Polygon(gauge_corners(theta, uc, vc), closed=True, fill=False,
                                  edgecolor="#d1495b", lw=1.8))
            axS.axhline(0, c="#adb5bd", lw=0.6)
            axS.axvline(0, c="#adb5bd", lw=0.6)
            axS.set_xlim(-5, 5)
            axS.set_ylim(-1.2, 5.5)
            axS.set_aspect("equal", adjustable="box")
            axS.tick_params(labelsize=7)
            axS.set_xlabel("вбок от оси пути, м", fontsize=7.5)
            axS.set_ylabel("над уровнем пола, м", fontsize=7.5)
            axS.set_title(f"плоскость ⊥ пути на {r['slice_dist']:.0f} м "
                          f"(срез ±{r['slice_ht']:.1f} м)\nкрен пути "
                          f"{np.degrees(theta):+.1f}°"
                          + ("" if r["has_pose"] else ", рельсов нет — крен 0"),
                          fontsize=8)

            head = f"{bag}  кадр {r['idx']}/{r['n_total']}"
            if r["confirmed"]:
                head += f"\nПРЕПЯТСТВИЕ в габарите на {r['confirmed']:.0f} м (подтверждено)"
            elif r["clusters"]:
                q = r["clusters"][0]
                head += (f"\nкластер в габарите на {q['dist']:.0f} м "
                         f"({q['n']} вокселей, {q['size'][1]:.1f}×{q['size'][2]:.1f} м) "
                         f"— не подтверждён")
            else:
                head += "\nгабарит чист"
            head += (f";  путь наблюдается до {r['reach']:.0f} м;  "
                     + ("вблизи по рельсам, дальше по контрасту"
                        if r["mode"] == "rails+contrast" else
                        f"рельсов нет: ось тоннеля со сдвигом {r['offset']:+.2f} м"))

        axR.plot(xs, reach, c="#adb5bd", lw=0.9)
        axR.scatter(xs, raw, s=7, c="#ff8c1a", zorder=3)
        axR.scatter(xs, conf, s=12, c="#d1495b", zorder=4)
        axR.axvline(r["idx"], c="crimson", lw=1.2)
        axR.set_xlim(xs.min(), max(xs.max(), xs.min() + 1))
        axR.set_ylim(0, D_SHOW)
        axR.tick_params(labelsize=7)
        axR.set_xlabel("кадр записи", fontsize=7.5)
        axR.set_title("дальность находки в габарите по прогону, м\n"
                      "оранжевое — сырая, красное — подтверждённая; серое — предел пути",
                      fontsize=8)
        fig.suptitle(head, fontsize=9.5)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=128))
        print(f"\r  отрисовано {i + 1}/{len(recs)}", end="", flush=True)
    print()
    return images


def build(dataset, bag, out_dir, stride, target, fps, max_frames):
    if stride is None:
        stride = max(1, round(frame_count(bag_path(dataset, bag)) / target))
    print(f"\n=== {bag} (шаг {stride}) ===")
    recs = collect(dataset, bag, stride, max_frames)
    images = render(recs, bag)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{bag}.gif"
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    n_raw = sum(1 for r in recs if r.get("clusters"))
    n_conf = sum(1 for r in recs if r.get("confirmed"))
    print(f"  {path} ({path.stat().st_size / 1e6:.1f} МБ); кадров {len(recs)}, "
          f"сырых находок {n_raw}, подтверждённых {n_conf}")
    return {"bag": bag, "frames": len(recs), "raw": n_raw, "conf": n_conf,
            "mb": path.stat().st_size / 1e6}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=None)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--out", default=f"output/{FOLDER}")
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
            raise SystemExit(f"{VALIDATION_RUN} отложен: только через --validate")
    out_dir = Path(a.out)
    rows = [build(a.dataset, b, out_dir, a.stride, a.target_frames, a.fps, a.max_frames)
            for b in bags]
    print("\n=== Итог ===")
    for s in rows:
        print(f"{s['bag']:38s} кадров {s['frames']:4d}  сырых {s['raw']:4d}  "
              f"подтверждённых {s['conf']:4d}  {s['mb']:5.1f} МБ")


if __name__ == "__main__":
    main()
