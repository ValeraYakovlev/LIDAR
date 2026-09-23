#!/usr/bin/env python3
"""GIF одного прогона: вид сверху на ВЕСЬ кадр, с найденной геометрией тоннеля.

Три панели:
  сверху  — вид сверху на всю глубину кадра, точки раскрашены по высоте над
            головкой рельса (см. HEIGHT_BANDS). Граница нарисована сплошной там,
            где она реально наблюдается, и пунктиром дальше — так сразу видно,
            докуда геометрию можно отслеживать, а где это уже продолжение
            модели в пустоту. Точки клиренс-полосы жёлтые, если они из самого
            кадра, и синие, если подклеены из прошлых накоплением: видно, где
            метод смотрит сам, а где опирается на накопленное.
  средняя — лента кривизны по всему прогону с бегунком текущего кадра: по
            одному кадру не видно, поворачивает путь или это шум подгонки, а на
            ленте поворот виден как сплошной участок одного знака.
  нижняя  — лента дальности наблюдения: до какой глубины держится граница.

Кадры обрабатываются ПОДРЯД, со связыванием геометрии и накоплением облаков
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
from matplotlib.lines import Line2D
from PIL import Image

from rail_detection import (DEFAULT_BAGS, bag_path, fit_tunnel_geometry, frame_count,
                            iter_frames, to_track_coords, tracked_depth, wall_x)
from rail_detection.tunnel_frame import tunnel_center_coeffs
from rail_detection.tracker import TunnelTracker
from rail_detection.tunnel_frame import V_HI, V_LO
from rail_detection.gauge import (TRUST_DEPTH, ObstacleWatch, cluster_obstacles,
                                  corridor_halfwidth)

DEPTH_MAX = 180.0  # тоннель просматривается дальше 120 м, и это видно на картинке
X_LIM = (-10.0, 10.0)
MAX_GRAY = 16000  # точек на кадр в GIF: больше глазом не различить, а вес растёт
EXTRAPOLATION_SHOW = 1.5  # во сколько раз за предел наблюдения показывать продолжение

# Полосы высот над головкой рельса, каждая своим цветом.
#
# Зачем. Зелёная линия — граница тоннеля В КЛИРЕНС-ПОЛОСЕ (v = 0.2…1.1 м), а
# рисуется она поверх вида СВЕРХУ, где высота не видна вовсе. Поэтому настил
# платформы и свод, которые законно шире полосы, выглядели точками «за стеной», и
# картинка сообщала об ошибке там, где её нет: замер на кадре 48 doubleT_platform
# дал 25 937 точек снаружи левой границы и НОЛЬ из них в клиренс-полосе — 48%
# настил и лотки, 52% свод. Раскраска по высоте убирает это недоразумение,
# ничего не меняя в самом методе.
HEIGHT_BANDS = [
    (-9.0, V_LO, "#ddd7cc", "пол, желоб"),
    (V_HI, 2.0, "#d97b7b", "настил, лотки"),
    (2.0, 9.0, "#7f9ec4", "свод"),
]

# doubleT_obstacle исключён из DEFAULT_BAGS как нетиповая сцена (стоящий поезд
# на пути), но для покадрового просмотра он как раз самый интересный — видно,
# как геометрия ведёт себя при реальном препятствии.
ALL_BAGS = list(DEFAULT_BAGS) + ["doubleT_obstacle"]


def collect(dataset, bag, stride, max_frames=None, track=True, depth_max=DEPTH_MAX,
            accumulate=None):
    """Один проход по bag: геометрия кадра + прореженные точки для отрисовки.

    Точки сохраняются сразу, чтобы не читать многогигабайтный bag второй раз:
    ленты нужны целиком до того, как рисуется первый кадр.

    accumulate: сколько кадров складывать со сдвигом на пройденный путь;
        None — как настроен tracker по умолчанию (rail_detection.accumulate).
    """
    rng = np.random.default_rng(0)
    tracker = (TunnelTracker(**({} if accumulate is None else {"accumulate": accumulate}))
               if track else None)
    watch = ObstacleWatch()
    out = []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=stride,
                                           max_frames=max_frames):
        res = tracker.update(points, steps=stride) if tracker else fit_tunnel_geometry(points)
        # Рисуется то облако, по которому шла подгонка: при накоплении это
        # несколько кадров, сложенных со сдвигом на пройденный путь.
        if tracker is not None:
            points, n_native = tracker.merged, tracker.n_native
        else:
            n_native = len(points)

        x = points['x'].astype(float)
        y = points['y'].astype(float)
        z = points['z'].astype(float)
        depth = -y
        vis = (depth > 0) & (depth < depth_max) & (np.abs(x) < X_LIM[1] + 2)
        band = np.zeros(len(x), dtype=bool)
        v = None
        if res is not None:
            _, _, v = to_track_coords(x, y, z, res["frame"])
            band = vis & (v >= V_LO) & (v <= V_HI)

        def pick(mask, cap):
            i = np.where(mask)[0]
            return rng.choice(i, cap, replace=False) if len(i) > cap else i

        # Точки, подклеенные из прошлых кадров, рисуются отдельным цветом: иначе
        # по картинке не отличить, где метод видит сам, а где опирается на
        # накопленное, а это ровно то, что проверяется.
        cap = MAX_GRAY // 2
        own = pick(band & (np.arange(len(x)) < n_native), cap)
        past = pick(band & (np.arange(len(x)) >= n_native), cap)

        # Остальное разбирается по полосам высот, а не валится в один серый ком
        # (см. HEIGHT_BANDS). Если геометрии нет, высоту отсчитывать не от чего —
        # тогда всё идёт в первую полосу как было.
        share = MAX_GRAY // len(HEIGHT_BANDS)
        if v is None:
            layers = [pick(vis, MAX_GRAY)] + [np.zeros(0, dtype=int)] * (len(HEIGHT_BANDS) - 1)
        else:
            layers = [pick(vis & ~band & (v >= lo) & (v < hi), share)
                      for lo, hi, _, _ in HEIGHT_BANDS]

        # Габаритный коридор и вторжения в него (rail_detection.gauge): то, ради
        # чего вся геометрия и строилась — препятствие определяется как то, что
        # мешает проехать, а не как то, что выглядит необычно.
        gauge = None
        if res is not None:
            clusters, inside = cluster_obstacles(points, res)
            sh = (tracker.accumulator.last.get("shift")
                  if tracker is not None and tracker.accumulator is not None
                  and tracker.accumulator.last.get("ok") else None)
            confirmed = watch.update(clusters[0]["depth"] if clusters else None, sh)
            hit = np.where(inside)[0]
            if len(hit) > MAX_GRAY // 4:
                hit = rng.choice(hit, MAX_GRAY // 4, replace=False)
            reach = [tracked_depth(res, sd) for sd in ("left", "right")]
            reach = [q for q in reach if q]
            gauge = {"pts": np.column_stack([x[hit], depth[hit]]).astype(np.float32),
                     "n_clusters": len(clusters), "confirmed": confirmed,
                     "reach": float(np.mean(reach)) if reach else None}

        rec = {
            "idx": idx, "n_total": n_total, "gauge": gauge,
            "layers": [np.column_stack([x[i], depth[i]]).astype(np.float32) for i in layers],
            "band": np.column_stack([x[own], depth[own]]).astype(np.float32),
            "past": np.column_stack([x[past], depth[past]]).astype(np.float32),
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


def _verdict(g):
    """Что метод утверждает про габарит. Различаются три состояния, и это не
    придирка: «кластер есть, но не подтверждён» и «подтверждён» — разные вещи,
    и смешивать их значило бы выдавать шум за находку."""
    if g is None:
        return "\nгабарит не построен"
    if g.get("confirmed"):
        return f"\nПРЕПЯТСТВИЕ в габарите на {g['confirmed']:.0f} м"
    if g.get("n_clusters"):
        return f"\nкластер в габарите ({g['n_clusters']}), не подтверждён по движению"
    return "\nгабарит чист"


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

        for pts, (_, _, colour, _) in zip(r["layers"], HEIGHT_BANDS):
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=0.7, c=colour, alpha=0.7)
        if len(r.get("past", ())):
            ax.scatter(r["past"][:, 0], r["past"][:, 1], s=1.1, c="#5a8fc8", alpha=0.55)
        if len(r["band"]):
            ax.scatter(r["band"][:, 0], r["band"][:, 1], s=1.1, c="#c8a45a", alpha=0.7)

        if r["kind"] is None:
            head = f"{bag}  кадр {r['idx']}/{r['n_total']}\nнет опоры — геометрия не строится"
        else:
            # Ось рисуется не на всю глубину показа: парабола, продлённая
            # втрое дальше своих данных, уезжает за край кадра и ничего не
            # сообщает. То же правило, что и для стен.
            a_cut = (r["reach"] if np.isfinite(r["reach"]) else 40.0) * EXTRAPOLATION_SHOW
            a_sel = r["axis"][:, 1] <= a_cut
            ax.plot(r["axis"][a_sel, 0], r["axis"][a_sel, 1], c="dimgray", lw=0.9, ls="--")
            # Габаритный коридор: сплошной там, где путь измерен, и расходящийся
            # конусом дальше — за пределом наблюдения поезд может уйти вбок не
            # больше, чем позволяет минимальный радиус, и это честнее одной
            # продлённой кривой.
            g = r.get("gauge")
            if g is not None:
                dd = r["axis"][:, 1]
                cw = corridor_halfwidth(dd, g.get("reach"))
                # Конус рисуется, пока он уже тоннеля: шире 3 м он перестаёт
                # что-либо утверждать — там уже вся ширина тоннеля, и «возможное
                # положение поезда» совпадает с «где угодно».
                show = cw < 3.0
                for sgn in (-1.0, +1.0):
                    line = r["axis"][:, 0] + sgn * cw
                    trust = show & (dd <= TRUST_DEPTH)
                    cone = show & (dd > TRUST_DEPTH)
                    ax.plot(line[trust], dd[trust], c="#d1495b", lw=1.4, alpha=0.9)
                    ax.plot(line[cone], dd[cone], c="#d1495b", lw=0.8, ls="--", alpha=0.45)
                if len(g["pts"]):
                    ax.scatter(g["pts"][:, 0], g["pts"][:, 1], s=7, c="#d1495b",
                               marker="x", linewidths=0.8, zorder=5)
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
            head = (f"{bag}  кадр {r['idx']}/{r['n_total']}\n{shape}\n"
                    + "   ".join(parts)
                    + f"\nзелёная — граница тоннеля, красная — габарит поезда"
                    + _verdict(r.get("gauge")))

        # Подпись полос: без неё раскраска ничего не объясняет, а именно
        # объяснение здесь и есть цель.
        handles = [Line2D([], [], marker="o", ls="", ms=3, c=c, label=n)
                   for _, _, c, n in HEIGHT_BANDS]
        handles += [Line2D([], [], marker="o", ls="", ms=3, c="#c8a45a",
                           label=f"клиренс {V_LO}-{V_HI} м"),
                    Line2D([], [], marker="o", ls="", ms=3, c="#5a8fc8",
                           label="то же, из прошлых кадров")]
        ax.legend(handles=handles, fontsize=5.4, loc="upper left", framealpha=0.85,
                  handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
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


def build_one(dataset, bag, out_dir, stride, target_frames, fps, max_frames, track,
              depth_max, accumulate=None):
    if stride is None:
        # Записи различаются по длине втрое (201 против 877 кадров), поэтому
        # шаг подбирается под целевую длину GIF: иначе один прогон вышел бы
        # втрое длиннее и тяжелее остальных при той же сути.
        n = frame_count(bag_path(dataset, bag))
        stride = max(1, round(n / target_frames))
    print(f"\n=== {bag} (шаг {stride}) ===")
    records = collect(dataset, bag, stride, max_frames, track=track, depth_max=depth_max,
                      accumulate=accumulate)
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
    p.add_argument("--accumulate", type=int, default=None,
                   help="сколько кадров складывать со сдвигом на Δs "
                        "(по умолчанию как в rail_detection.tracker; 1 — без накопления)")
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for bag in a.bags:
        got = build_one(a.dataset, bag, out_dir, a.stride, a.target_frames, a.fps,
                        a.max_frames, not a.no_track, a.depth_max, a.accumulate)
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
