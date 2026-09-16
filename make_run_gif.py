#!/usr/bin/env python3
"""GIF одного прогона: вид сверху покадрово, с найденной геометрией тоннеля.

Снизу — лента кривизны по всему прогону с бегунком текущего кадра: по одному
кадру не видно, поворачивает путь или это шум подгонки, а на ленте поворот
виден как сплошной участок одного знака.

Алгоритм не меняется — используется тот же rail_detection.tunnel_frame.

Запуск:
    python make_run_gif.py                       # все прогоны
    python make_run_gif.py --bags roundT_doubleT --stride 2
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from rail_detection import (DEFAULT_BAGS, bag_path, fit_tunnel_geometry, frame_count,
                            iter_frames, to_track_coords, wall_x)
from rail_detection.curvature import eval_fit
from rail_detection.tunnel_frame import V_HI, V_LO

DEPTH_MAX = 45.0
X_LIM = (-6.0, 6.0)
MAX_GRAY = 14000  # точек на кадр в GIF: больше глазом не различить, а вес растёт

# doubleT_obstacle исключён из DEFAULT_BAGS как нетиповая сцена (стоящий поезд
# на пути), но для покадрового просмотра он как раз самый интересный — видно,
# как геометрия ведёт себя при реальном препятствии.
ALL_BAGS = list(DEFAULT_BAGS) + ["doubleT_obstacle"]


def collect(dataset, bag, stride, max_frames=None):
    """Один проход по bag: геометрия кадра + прореженные точки для отрисовки.

    Точки сохраняются сразу, чтобы не читать многогигабайтный bag второй раз:
    лента кривизны нужна целиком до того, как рисуется первый кадр.
    """
    rng = np.random.default_rng(0)
    out = []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=stride,
                                           max_frames=max_frames):
        x = points['x'].astype(float)
        y = points['y'].astype(float)
        z = points['z'].astype(float)
        depth = -y
        vis = (depth > 0) & (depth < DEPTH_MAX) & (np.abs(x) < X_LIM[1] + 1)

        res = fit_tunnel_geometry(points)
        band = np.zeros(len(x), dtype=bool)
        if res is not None:
            _, _, v = to_track_coords(x, y, z, res["frame"])
            band = vis & (v >= V_LO) & (v <= V_HI)

        gray_idx = np.where(vis & ~band)[0]
        if len(gray_idx) > MAX_GRAY:
            gray_idx = rng.choice(gray_idx, MAX_GRAY, replace=False)
        band_idx = np.where(band)[0]
        if len(band_idx) > MAX_GRAY // 2:
            band_idx = rng.choice(band_idx, MAX_GRAY // 2, replace=False)

        rec = {
            "idx": idx, "n_total": n_total,
            "gray": np.column_stack([x[gray_idx], depth[gray_idx]]).astype(np.float32),
            "band": np.column_stack([x[band_idx], depth[band_idx]]).astype(np.float32),
        }
        if res is None:
            rec.update(alpha=np.nan, kind=None)
        else:
            dd = np.linspace(3, DEPTH_MAX, 80)
            rec.update(
                alpha=res["shape"]["alpha"],
                kind=res["shape"]["kind"],
                radius=res["shape"]["radius"],
                axis=np.column_stack([eval_fit(res["frame"]["axis_fit"], dd), dd]).astype(np.float32),
                walls={s: np.column_stack([wall_x(res, s, dd), dd]).astype(np.float32)
                       for s in ("left", "right") if res[s] is not None},
                labels={s: (res[s]["offset"], res[s]["coverage"], res[s]["leak"])
                        for s in ("left", "right") if res[s] is not None},
            )
        out.append(rec)
        print(f"\r  кадр {idx}/{n_total}", end="", flush=True)
    print()
    return out


def render(records, bag, dpi=120):
    """Рисует кадры GIF. Фигура создаётся один раз и переиспользуется."""
    alphas = np.array([r["alpha"] for r in records], dtype=float)
    xs = np.array([r["idx"] for r in records], dtype=float)
    arc = np.array([r["kind"] == "arc" for r in records])
    lim = float(np.nanmax(np.abs(alphas))) if np.any(np.isfinite(alphas)) else 1.0
    lim = max(lim * 1.2, 0.5)

    fig = Figure(figsize=(4.6, 7.4), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(2, 1, height_ratios=[4.2, 1.0], hspace=0.30,
                          left=0.15, right=0.97, top=0.93, bottom=0.09)
    ax = fig.add_subplot(gs[0])
    ax_k = fig.add_subplot(gs[1])

    images = []
    for i, r in enumerate(records):
        ax.clear()
        ax_k.clear()

        ax.scatter(r["gray"][:, 0], r["gray"][:, 1], s=0.7, c="lightgray", alpha=0.5)
        if len(r["band"]):
            ax.scatter(r["band"][:, 0], r["band"][:, 1], s=1.2, c="#c8a45a", alpha=0.7)

        if r["kind"] is None:
            head = f"{bag}  кадр {r['idx']}/{r['n_total']}\nнет опоры по рельсам — геометрия не строится"
        else:
            ax.plot(r["axis"][:, 0], r["axis"][:, 1], c="dimgray", lw=1.0, ls="--")
            for w in r["walls"].values():
                ax.plot(w[:, 0], w[:, 1], c="limegreen", lw=2.8)
            if r["kind"] == "arc":
                shape = f"дуга R={r['radius']:.0f} м, {'направо' if r['alpha'] > 0 else 'налево'}"
            else:
                shape = "прямая"
            parts = []
            for side, name in (("left", "Л"), ("right", "П")):
                if side in r["labels"]:
                    o, cov, leak = r["labels"][side]
                    parts.append(f"{name} {o:.2f}м cov{cov:.2f} leak{leak:.2f}")
            head = f"{bag}  кадр {r['idx']}/{r['n_total']}\n{shape}\n" + "   ".join(parts)

        ax.set_xlim(*X_LIM)
        ax.set_ylim(0, DEPTH_MAX)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(head, fontsize=8.5)

        ax_k.axhline(0, c="gray", lw=0.8)
        ax_k.plot(xs, alphas, c="silver", lw=1.0)
        ax_k.scatter(xs[arc], alphas[arc], s=5, c="limegreen", zorder=3)
        ax_k.axvline(r["idx"], c="crimson", lw=1.4)
        ax_k.set_xlim(xs.min(), xs.max())
        ax_k.set_ylim(-lim, lim)
        ax_k.set_yticks([-lim, 0, lim])
        ax_k.set_yticklabels([f"{lim:.1f}\nналево", "0", f"{lim:.1f}\nнаправо"], fontsize=6.5)
        ax_k.tick_params(axis="x", labelsize=7)
        ax_k.set_xlabel("кадр прогона", fontsize=7.5)
        ax_k.set_title("увод пути вбок на 40 м вперёд, м (зелёное — распознано как дуга)",
                       fontsize=7.5)

        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=96))
        print(f"\r  отрисовано {i + 1}/{len(records)}", end="", flush=True)
    print()
    return images


def build_one(dataset, bag, out_dir, stride, target_frames, fps, max_frames):
    if stride is None:
        # Записи различаются по длине втрое (201 против 877 кадров), поэтому
        # шаг подбирается под целевую длину GIF: иначе один прогон вышел бы
        # втрое длиннее и тяжелее остальных при той же сути.
        n = frame_count(bag_path(dataset, bag))
        stride = max(1, round(n / target_frames))
    print(f"\n=== {bag} (шаг {stride}) ===")
    records = collect(dataset, bag, stride, max_frames)
    if not records:
        print("  кадры не прочитались")
        return None

    n_arc = sum(1 for r in records if r["kind"] == "arc")
    n_none = sum(1 for r in records if r["kind"] is None)
    print(f"  кадров: {len(records)}, дуга в {n_arc}, без опоры {n_none}")

    images = render(records, bag)
    path = out_dir / f"run_{bag}.gif"
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    size = path.stat().st_size / 1e6
    print(f"  GIF: {path} ({size:.1f} МБ)")
    return {"bag": bag, "frames": len(records), "arc": n_arc, "no_anchor": n_none,
            "stride": stride, "size_mb": size}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=ALL_BAGS, help="какие прогоны обрабатывать")
    p.add_argument("--out", default="output")
    p.add_argument("--stride", type=int, default=None,
                   help="брать каждый N-й кадр (по умолчанию подбирается под --target-frames)")
    p.add_argument("--target-frames", type=int, default=150, help="желаемая длина GIF в кадрах")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=10.0)
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for bag in a.bags:
        got = build_one(a.dataset, bag, out_dir, a.stride, a.target_frames, a.fps, a.max_frames)
        if got:
            summary.append(got)

    print("\n=== Итог ===")
    print(f"{'прогон':38s} {'шаг':>4s} {'кадров':>7s} {'дуга':>6s} {'без опоры':>10s} {'МБ':>6s}")
    for s in summary:
        print(f"{s['bag']:38s} {s['stride']:4d} {s['frames']:7d} "
              f"{s['arc']:6d} {s['no_anchor']:10d} {s['size_mb']:6.1f}")


if __name__ == "__main__":
    main()
