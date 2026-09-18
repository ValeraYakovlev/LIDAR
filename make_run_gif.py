#!/usr/bin/env python3
"""GIF одного прогона: вид сверху на ВЕСЬ кадр, с найденной геометрией тоннеля.

Три панели:
  сверху  — вид сверху на всю глубину кадра. Граница нарисована сплошной там,
            где она реально наблюдается, и пунктиром дальше — так сразу видно,
            докуда геометрию можно отслеживать, а где это уже продолжение
            модели в пустоту.
  средняя — лента кривизны по всему прогону с бегунком текущего кадра: по
            одному кадру не видно, поворачивает путь или это шум подгонки, а на
            ленте поворот виден как сплошной участок одного знака.
  нижняя  — лента дальности наблюдения: до какой глубины держится граница.

Кадры обрабатываются ПОДРЯД, с переносом геометрии между ними
(rail_detection.tracker) — отключается флагом --no-track.

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
                            iter_frames, to_track_coords, tracked_depth, wall_x)
from rail_detection.tunnel_frame import tunnel_center_coeffs
from rail_detection.tracker import TunnelTracker
from rail_detection.tunnel_frame import V_HI, V_LO

DEPTH_MAX = 120.0
X_LIM = (-10.0, 10.0)
MAX_GRAY = 16000  # точек на кадр в GIF: больше глазом не различить, а вес растёт
EXTRAPOLATION_SHOW = 1.5  # во сколько раз за предел наблюдения показывать продолжение

# doubleT_obstacle исключён из DEFAULT_BAGS как нетиповая сцена (стоящий поезд
# на пути), но для покадрового просмотра он как раз самый интересный — видно,
# как геометрия ведёт себя при реальном препятствии.
ALL_BAGS = list(DEFAULT_BAGS) + ["doubleT_obstacle"]


def collect(dataset, bag, stride, max_frames=None, track=True, depth_max=DEPTH_MAX):
    """Один проход по bag: геометрия кадра + прореженные точки для отрисовки.

    Точки сохраняются сразу, чтобы не читать многогигабайтный bag второй раз:
    ленты нужны целиком до того, как рисуется первый кадр.
    """
    rng = np.random.default_rng(0)
    tracker = TunnelTracker() if track else None
    out = []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=stride,
                                           max_frames=max_frames):
        x = points['x'].astype(float)
        y = points['y'].astype(float)
        z = points['z'].astype(float)
        depth = -y
        vis = (depth > 0) & (depth < depth_max) & (np.abs(x) < X_LIM[1] + 2)

        res = tracker.update(points, steps=stride) if tracker else fit_tunnel_geometry(points)
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
            rec.update(alpha=np.nan, kind=None, reach=np.nan)
        else:
            dd = np.linspace(3, depth_max, 160)
            reach = [tracked_depth(res, s) for s in ("left", "right")]
            reach = [r for r in reach if r is not None]
            rec.update(
                alpha=res["shape"]["alpha"],
                kind=res["shape"]["kind"],
                radius=res["shape"]["radius"],
                limited=bool(res["shape"].get("smoothed")),
                reach=float(np.mean(reach)) if reach else np.nan,
                # Ось ИТОГОВАЯ (опора плюс найденная форма), а не опорная: опора —
                # это лишь прямая по рельсам, и когда рельсы находятся плохо, она
                # улетает вбок, а модель компенсирует это полуширинами. Рисовать
                # надо то, что метод утверждает про путь, а не промежуточную величину.
                axis=np.column_stack([np.polyval(tunnel_center_coeffs(res), dd), dd]).astype(np.float32),
                walls={s: np.column_stack([wall_x(res, s, dd), dd]).astype(np.float32)
                       for s in ("left", "right") if res[s] is not None},
                reach_side={s: tracked_depth(res, s) for s in ("left", "right")},
                labels={s: (res[s]["offset"], res[s]["coverage"], res[s]["leak"])
                        for s in ("left", "right") if res[s] is not None},
            )
        out.append(rec)
        print(f"\r  кадр {idx}/{n_total}", end="", flush=True)
    print()
    return out


def render(records, bag, depth_max=DEPTH_MAX, dpi=120):
    """Рисует кадры GIF. Фигура создаётся один раз и переиспользуется."""
    alphas = np.array([r["alpha"] for r in records], dtype=float)
    reach = np.array([r["reach"] for r in records], dtype=float)
    xs = np.array([r["idx"] for r in records], dtype=float)
    arc = np.array([r["kind"] == "arc" for r in records])
    no_anchor = np.array([r["kind"] is None for r in records])
    lim = float(np.nanmax(np.abs(alphas))) if np.any(np.isfinite(alphas)) else 1.0
    lim = max(lim * 1.2, 0.5)
    reach_med = float(np.nanmedian(reach)) if np.any(np.isfinite(reach)) else 0.0

    fig = Figure(figsize=(4.8, 8.6), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 1, height_ratios=[4.6, 1.0, 1.0], hspace=0.42,
                          left=0.15, right=0.97, top=0.93, bottom=0.06)
    ax = fig.add_subplot(gs[0])
    ax_k = fig.add_subplot(gs[1])
    ax_r = fig.add_subplot(gs[2])

    images = []
    for i, r in enumerate(records):
        for a in (ax, ax_k, ax_r):
            a.clear()

        ax.scatter(r["gray"][:, 0], r["gray"][:, 1], s=0.6, c="lightgray", alpha=0.5)
        if len(r["band"]):
            ax.scatter(r["band"][:, 0], r["band"][:, 1], s=1.1, c="#c8a45a", alpha=0.7)

        if r["kind"] is None:
            head = f"{bag}  кадр {r['idx']}/{r['n_total']}\nнет опоры — геометрия не строится"
        else:
            ax.plot(r["axis"][:, 0], r["axis"][:, 1], c="dimgray", lw=0.9, ls="--")
            for side, w in r["walls"].items():
                # Сплошная — пока граница наблюдается, пунктир — дальше уже
                # продолжение модели, а не измерение. Продолжение рисуется лишь
                # немного за предел наблюдения: парабола, продлённая втрое
                # дальше своих данных, разлетается на десятки метров и забивает
                # картинку, ничего при этом не сообщая.
                cut = r["reach_side"].get(side) or 0.0
                seen = w[:, 1] <= cut
                ahead = (~seen) & (w[:, 1] <= cut * EXTRAPOLATION_SHOW)
                ax.plot(w[seen, 0], w[seen, 1], c="limegreen", lw=2.6)
                ax.plot(w[ahead, 0], w[ahead, 1], c="limegreen", lw=1.4, ls=":", alpha=0.75)
            if np.isfinite(r["reach"]):
                ax.axhline(r["reach"], c="crimson", lw=0.9, ls="--", alpha=0.7)
                ax.text(X_LIM[0] + 0.3, r["reach"] + 1.5, f"наблюдается до {r['reach']:.0f} м",
                        fontsize=6.5, c="crimson")
            shape = (f"дуга R={r['radius']:.0f} м, {'направо' if r['alpha'] > 0 else 'налево'}"
                     if r["kind"] == "arc" else "прямая")
            if r.get("limited"):
                shape += " (выброс сглажен)"
            parts = []
            for side, name in (("left", "Л"), ("right", "П")):
                if side in r["labels"]:
                    o, cov, leak = r["labels"][side]
                    parts.append(f"{name} {o:.2f}м cov{cov:.2f} leak{leak:.2f}")
            head = f"{bag}  кадр {r['idx']}/{r['n_total']}\n{shape}\n" + "   ".join(parts)

        ax.set_xlim(*X_LIM)
        ax.set_ylim(0, depth_max)
        ax.set_xticks([])
        ax.set_yticks(np.arange(0, depth_max + 1, 20))
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_ylabel("глубина, м", fontsize=7)
        ax.set_title(head, fontsize=8.5)

        ax_k.axhline(0, c="gray", lw=0.8)
        ax_k.plot(xs, alphas, c="silver", lw=1.0)
        ax_k.scatter(xs[arc], alphas[arc], s=5, c="limegreen", zorder=3)
        if no_anchor.any():
            ax_k.scatter(xs[no_anchor], np.full(no_anchor.sum(), -lim * 0.82),
                         s=18, c="crimson", marker="|", zorder=3)
        ax_k.axvline(r["idx"], c="crimson", lw=1.4)
        ax_k.set_xlim(xs.min(), xs.max())
        ax_k.set_ylim(-lim, lim)
        ax_k.set_yticks([-lim, 0, lim])
        ax_k.set_yticklabels([f"{lim:.1f}\nналево", "0", f"{lim:.1f}\nнаправо"], fontsize=6.5)
        ax_k.tick_params(axis="x", labelsize=7)
        ax_k.set_title("увод пути вбок на 40 м вперёд, м\n"
                       "зелёное — дуга, красные штрихи — нет опоры", fontsize=7.0)

        ax_r.plot(xs, reach, c="crimson", lw=1.1)
        ax_r.axhline(reach_med, c="gray", lw=0.8, ls="--")
        ax_r.axvline(r["idx"], c="crimson", lw=1.4)
        ax_r.set_xlim(xs.min(), xs.max())
        ax_r.set_ylim(0, depth_max)
        ax_r.set_yticks([0, depth_max / 2, depth_max])
        ax_r.tick_params(labelsize=6.5)
        ax_r.set_xlabel("кадр прогона", fontsize=7.5)
        ax_r.set_title(f"докуда наблюдается граница, м (медиана {reach_med:.0f})", fontsize=7.0)

        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=96))
        print(f"\r  отрисовано {i + 1}/{len(records)}", end="", flush=True)
    print()
    return images, reach_med


def build_one(dataset, bag, out_dir, stride, target_frames, fps, max_frames, track, depth_max):
    if stride is None:
        # Записи различаются по длине втрое (201 против 877 кадров), поэтому
        # шаг подбирается под целевую длину GIF: иначе один прогон вышел бы
        # втрое длиннее и тяжелее остальных при той же сути.
        n = frame_count(bag_path(dataset, bag))
        stride = max(1, round(n / target_frames))
    print(f"\n=== {bag} (шаг {stride}) ===")
    records = collect(dataset, bag, stride, max_frames, track=track, depth_max=depth_max)
    if not records:
        print("  кадры не прочитались")
        return None

    n_arc = sum(1 for r in records if r["kind"] == "arc")
    n_none = sum(1 for r in records if r["kind"] is None)
    n_lim = sum(1 for r in records if r.get("limited"))
    print(f"  кадров: {len(records)}, дуга в {n_arc}, без опоры {n_none}, скачок срезан в {n_lim}")

    images, reach_med = render(records, bag, depth_max=depth_max)
    path = out_dir / f"run_{bag}.gif"
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    size = path.stat().st_size / 1e6
    print(f"  GIF: {path} ({size:.1f} МБ), граница наблюдается до {reach_med:.0f} м (медиана)")
    return {"bag": bag, "frames": len(records), "arc": n_arc, "no_anchor": n_none,
            "limited": n_lim, "stride": stride, "size_mb": size, "reach": reach_med}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=ALL_BAGS, help="какие прогоны обрабатывать")
    p.add_argument("--out", default="output")
    p.add_argument("--stride", type=int, default=None,
                   help="брать каждый N-й кадр (по умолчанию подбирается под --target-frames)")
    p.add_argument("--target-frames", type=int, default=140, help="желаемая длина GIF в кадрах")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--depth-max", type=float, default=DEPTH_MAX, help="глубина показа, м")
    p.add_argument("--no-track", action="store_true", help="обрабатывать кадры независимо")
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for bag in a.bags:
        got = build_one(a.dataset, bag, out_dir, a.stride, a.target_frames, a.fps,
                        a.max_frames, not a.no_track, a.depth_max)
        if got:
            summary.append(got)

    print("\n=== Итог ===")
    print(f"{'прогон':38s} {'шаг':>4s} {'кадров':>7s} {'дуга':>6s} {'без опоры':>10s} "
          f"{'срезано':>8s} {'видно, м':>9s} {'МБ':>6s}")
    for s in summary:
        print(f"{s['bag']:38s} {s['stride']:4d} {s['frames']:7d} {s['arc']:6d} "
              f"{s['no_anchor']:10d} {s['limited']:8d} {s['reach']:9.0f} {s['size_mb']:6.1f}")


if __name__ == "__main__":
    main()
