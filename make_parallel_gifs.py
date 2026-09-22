#!/usr/bin/env python3
"""GIF эксперимента 17: путь и стены одной кривой, габарит вдоль этого пути.

Трекер обрабатывает КАЖДЫЙ кадр записи — так, как это было бы на поезде, — а в
GIF попадает каждый k-й (k = кадров записи / --target-frames). Поэтому связь
кадров в GIF та же, что и в работе, а не растянутая на шаг прореживания.

Панели:

  слева        — вид сверху, контраст нормирован по глубине. Зелёным — стены:
                 путь, отступленный по нормали на постоянное расстояние (оно в
                 заголовке); малиновым — путь; красным — края коридора габарита
                 (1.1 м по нормали); синяя черта — плоскость среза справа.
  справа вверху — срез перпендикулярно пути (как в §27).
  справа в центре — дальность находки в габарите по всему прогону.
  справа внизу — отступы стен от пути по всему прогону: если стены «прыгают»,
                 это видно здесь сразу.

Запуск:
    python make_parallel_gifs.py                     # 5 прогонов разработки
    python make_parallel_gifs.py --validate          # замороженный прогон, ОДИН раз
    python make_parallel_gifs.py --bags roundT_doubleT
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from PIL import Image

from make_gauge_gifs import X_LIM, _crop, _u8
from rail_detection import bag_path, frame_count, iter_frames
from rail_detection.contrast_gauge import corridor_lines, gauge_corners, path_at, pose_at, \
    slice_points
from rail_detection.parallel_path import ParallelGauge, offset_curve

warnings.filterwarnings("ignore", message=".*encountered in matmul")

DEV_RUNS = ["doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
            "squareT_platform_squareT_switch", "doubleT_obstacle"]
VALIDATION_RUN = "roundT_squareT_pressureGate_squareT"
FOLDER = "Opus 5.5"
D_SHOW = 150.0
N_SLICE = 6000
DEFAULT_SLICE = 60.0


def collect(dataset, bag, every, max_frames=None):
    """Все кадры через трекер; тяжёлые данные для рисования — только у каждого every-го."""
    pg = ParallelGauge()
    rng = np.random.default_rng(0)
    recs, stats = [], []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=1,
                                           max_frames=max_frames):
        res = pg.update(points, steps=1)
        st = {"idx": idx, "ok": res is not None}
        if res is not None:
            tr = res["track"]
            cl = res["clusters"]
            st.update({"raw": cl[0]["dist"] if cl else None, "conf": res["confirmed"],
                       "reach": res["reach"], "wl": tr["curve"]["w0"][0],
                       "wr": tr["curve"]["w0"][1], "steps": tr["curve"]["steps"],
                       "ds_measured": tr["ds_measured"], "origin": tr["origin"],
                       "mode": res["path"]["mode"], "shift": res["shift"]})
        stats.append(st)
        if idx % every == 0:
            recs.append(_render_data(idx, n_total, res, rng))
        print(f"\r  {bag}: кадр {idx}/{n_total}", end="", flush=True)
    print()
    return recs, stats


def _render_data(idx, n_total, res, rng):
    rec = {"idx": idx, "n_total": n_total, "ok": res is not None}
    if res is None:
        return rec
    tr = res["track"]
    c = tr["curve"]
    dd = np.linspace(2, D_SHOW, 300)
    left, mid, right = corridor_lines(res, dd)
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
    pd = res["path"]["d"]
    x_p, psi, arc = path_at(res["path"], pd)
    k = int(np.argmin(np.abs(arc - dist)))
    d_c, x_c, psi_c = pd[k], x_p[k], psi[k]
    cut = np.array([[x_c - 3.0 * np.cos(psi_c), d_c + 3.0 * np.sin(psi_c)],
                    [x_c + 3.0 * np.cos(psi_c), d_c - 3.0 * np.sin(psi_c)]])
    marks = []
    for q in cl[:3]:
        j = int(np.argmin(np.abs(arc - q["dist"])))
        marks.append((x_p[j] + q["u"] / np.cos(psi[j]), pd[j]))
    walls = {}
    for side, w in (("left", c["wl"]), ("right", c["wr"])):
        xw, dw = offset_curve(c, w)
        lim = tr["reach_side"].get(side) or 0.0
        start = tr["wall_from"].get(side)
        start = lim if start is None else max(2.0, start)
        keep = (dw >= start) & (dw <= min(lim, D_SHOW))
        walls[side] = np.column_stack([xw[keep], dw[keep]]).astype(np.float32)
    rec.update({
        "marks": np.array(marks, dtype=np.float32).reshape(-1, 2),
        "img": _u8(_crop(res["grid"], res["silhouette"]["image"])),
        "walls": walls, "w0": c["w0"], "steps": c["steps"],
        "wall_at_slice": (float(np.interp(dist, c["s"], c["wl"])),
                          float(np.interp(dist, c["s"], c["wr"]))),
        "reach": res["reach"],
        "corridor": np.column_stack([left, mid, right, dd]).astype(np.float32),
        "hits": np.column_stack([x[hit], -y[hit]]).astype(np.float32),
        "slice": np.column_stack([res["u"][sel], res["v"][sel]]).astype(np.float32),
        "slice_in": inside[sel], "slice_dist": float(dist), "slice_ht": float(ht),
        "pose": pose_at(res, dist), "cut": cut.astype(np.float32),
        "clusters": cl, "confirmed": res["confirmed"], "has_pose": res["has_pose"],
        "mode": res["path"]["mode"], "origin": tr["origin"],
    })
    return rec


def render(recs, stats, bag, dpi=85):
    xs = np.array([s["idx"] for s in stats], float)
    raw = np.array([s.get("raw") if s.get("raw") is not None else np.nan for s in stats], float)
    conf = np.array([s.get("conf") if s.get("conf") else np.nan for s in stats], float)
    reach = np.array([s.get("reach", np.nan) or np.nan for s in stats], float)
    wl = np.array([s.get("wl", np.nan) for s in stats], float)
    wr = np.array([s.get("wr", np.nan) for s in stats], float)
    # отступ за ПЕРВОЙ ступенькой впереди (пунктир): видно, что смена сечения
    # известна заранее и подъезжает, а не возникает скачком
    first = lambda st, j: (st["steps"][j][0][1] if st.get("steps") and st["steps"][j]
                           else np.nan)
    wl1 = np.array([first(s, 0) for s in stats], float)
    wr1 = np.array([first(s, 1) for s in stats], float)

    fig = Figure(figsize=(11.2, 8.2), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.25], height_ratios=[1.7, 0.8, 0.8],
                          left=0.06, right=0.98, top=0.87, bottom=0.06,
                          wspace=0.22, hspace=0.55)
    axT = fig.add_subplot(gs[:, 0])
    axS = fig.add_subplot(gs[0, 1])
    axR = fig.add_subplot(gs[1, 1])
    axW = fig.add_subplot(gs[2, 1])
    images = []
    for i, r in enumerate(recs):
        for a in (axT, axS, axR, axW):
            a.clear()
        if not r["ok"]:
            head = f"{bag}  кадр {r['idx']}/{r['n_total']}\nгеометрия не построена"
        else:
            axT.imshow(np.ma.masked_equal(r["img"], 0), origin="lower", aspect="auto",
                       cmap="gray_r", vmin=1, vmax=255, interpolation="nearest",
                       extent=[X_LIM[0], X_LIM[1], 2.0, 150.0])
            for side in ("left", "right"):
                w = r["walls"][side]
                if len(w):
                    axT.plot(w[:, 0], w[:, 1], c="#2fbf4a", lw=2.0, zorder=4)
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
            axT.set_title("вид сверху: контраст; зелёные — стены (путь, отступленный\n"
                          "по нормали), малиновая — путь, красные — края коридора;\n"
                          "синяя черта — плоскость среза, оранжевые × — точки в коробке",
                          fontsize=7.5)

            sl, si = r["slice"], r["slice_in"]
            axS.scatter(sl[~si, 0], sl[~si, 1], s=2.0, c="#9aa5b1", alpha=0.6, linewidths=0)
            if si.any():
                axS.scatter(sl[si, 0], sl[si, 1], s=6.0, c="#d1495b", alpha=0.95,
                            linewidths=0)
            theta, uc, vc = r["pose"]
            axS.add_patch(Polygon(gauge_corners(0.0, uc, vc), closed=True, fill=False,
                                  edgecolor="#6b7280", lw=0.9, ls="--"))
            axS.add_patch(Polygon(gauge_corners(theta, uc, vc), closed=True, fill=False,
                                  edgecolor="#d1495b", lw=1.8))
            for w in r["wall_at_slice"]:
                axS.axvline(w, c="#2fbf4a", lw=1.2, ls="--")
            axS.axhline(0, c="#adb5bd", lw=0.6)
            axS.axvline(0, c="#adb5bd", lw=0.6)
            axS.set_xlim(-5, 5)
            axS.set_ylim(-1.2, 5.5)
            axS.set_aspect("equal", adjustable="box")
            axS.tick_params(labelsize=7)
            axS.set_xlabel("вбок от оси пути, м (зелёный пунктир — стены)", fontsize=7.5)
            axS.set_ylabel("над уровнем пола, м", fontsize=7.5)
            axS.set_title(f"плоскость ⊥ пути на {r['slice_dist']:.0f} м "
                          f"(срез ±{r['slice_ht']:.1f} м), крен пути "
                          f"{np.degrees(theta):+.1f}°"
                          + ("" if r["has_pose"] else " (рельсов нет — 0)"), fontsize=8)

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
            walls_txt = []
            for j, name in ((0, "слева"), (1, "справа")):
                t = f"{name} {abs(r['w0'][j]):.2f} м"
                for sb, w in r["steps"][j]:
                    t += f", с {sb:.0f} м — {abs(w):.2f}"
                walls_txt.append(t)
            head += (f";  путь наблюдается до {r['reach']:.0f} м\n"
                     f"стены по нормали от пути: {', '.join(walls_txt)};  "
                     + ("вблизи держат рельсы" if r["mode"] == "rails+walls"
                        else "рельсов нет — путь по стенам на прежнем отступе"))

        axR.plot(xs, reach, c="#adb5bd", lw=0.9)
        axR.scatter(xs, raw, s=5, c="#ff8c1a", zorder=3)
        axR.scatter(xs, conf, s=9, c="#d1495b", zorder=4)
        axR.axvline(r["idx"], c="crimson", lw=1.2)
        axR.set_xlim(xs.min(), max(xs.max(), xs.min() + 1))
        axR.set_ylim(0, D_SHOW)
        axR.tick_params(labelsize=7)
        axR.set_title("дальность находки в габарите, м: оранжевое — сырая, "
                      "красное — подтверждённая; серое — предел пути", fontsize=8)

        axW.plot(xs, -wl, c="#2f9e44", lw=1.0, label="до левой стены")
        axW.plot(xs, wr, c="#1c7ed6", lw=1.0, label="до правой стены")
        axW.plot(xs, -wl1, c="#2f9e44", lw=0.8, ls=":", label="за ступенькой")
        axW.plot(xs, wr1, c="#1c7ed6", lw=0.8, ls=":")
        axW.axvline(r["idx"], c="crimson", lw=1.2)
        axW.set_xlim(xs.min(), max(xs.max(), xs.min() + 1))
        top = np.nanmax(np.r_[-wl, wr, -wl1, wr1, 3.0])
        axW.set_ylim(0, min(top + 0.5, 12))
        axW.tick_params(labelsize=7)
        axW.set_xlabel("кадр записи", fontsize=7.5)
        axW.set_title("расстояние от пути до стен по нормали, м (сплошные — у поезда,"
                      " пунктир — за ступенькой впереди)", fontsize=8)
        axW.legend(fontsize=6.5, loc="upper right", ncol=2)
        fig.suptitle(head, fontsize=9.2)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=128))
        print(f"\r  отрисовано {i + 1}/{len(recs)}", end="", flush=True)
    print()
    return images


def build(dataset, bag, out_dir, target, fps, max_frames):
    n = frame_count(bag_path(dataset, bag))
    every = max(1, round(n / target))
    print(f"\n=== {bag}: все {n} кадров через трекер, в GIF каждый {every}-й ===")
    recs, stats = collect(dataset, bag, every, max_frames)
    images = render(recs, stats, bag)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{bag}.gif"
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    summary = {
        "bag": bag, "frames": len(stats), "gif_frames": len(recs), "every": every,
        "ok": int(sum(s["ok"] for s in stats)),
        "raw_all": int(sum(1 for s in stats if s.get("raw") is not None)),
        "conf_all": int(sum(1 for s in stats if s.get("conf"))),
        "raw_gif": int(sum(1 for r in recs if r.get("clusters"))),
        "conf_gif": int(sum(1 for r in recs if r.get("confirmed"))),
        "fresh_frac": float(np.mean([s.get("origin") == "заново" for s in stats if s["ok"]])),
        "mb": path.stat().st_size / 1e6,
        "stats": stats,
    }
    with open(out_dir / f"{bag}.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=0, default=float)
    print(f"  {path} ({summary['mb']:.1f} МБ); кадров {len(stats)} (в GIF {len(recs)}), "
          f"сырых находок {summary['raw_all']}, подтверждённых {summary['conf_all']}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=None)
    p.add_argument("--validate", action="store_true",
                   help=f"замороженный прогон {VALIDATION_RUN} (validation_run_v6.json)")
    p.add_argument("--out", default=f"output/{FOLDER}")
    p.add_argument("--target-frames", type=int, default=130)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=8.0)
    a = p.parse_args()
    if a.validate:
        bags = [VALIDATION_RUN]
    else:
        bags = a.bags or DEV_RUNS
        if VALIDATION_RUN in bags:
            raise SystemExit(f"{VALIDATION_RUN} заморожен: только через --validate")
    rows = [build(a.dataset, b, Path(a.out), a.target_frames, a.fps, a.max_frames)
            for b in bags]
    print("\n=== Итог ===")
    for s in rows:
        print(f"{s['bag']:38s} кадров {s['frames']:4d}  сырых {s['raw_all']:4d}  "
              f"подтверждённых {s['conf_all']:4d}  заново {s['fresh_frac']:.2f}  "
              f"{s['mb']:5.1f} МБ")


if __name__ == "__main__":
    main()
