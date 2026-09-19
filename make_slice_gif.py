#!/usr/bin/env python3
"""GIF прогона в плоскости СРЕЗА, а не сверху.

Вид сверху не отвечает на вопрос «есть ли предмет в габарите»: он проецирует
высоту, и настил платформы со сводом ложатся на ту же картинку, что и путь.
Здесь тоннель режется ПЕРПЕНДИКУЛЯРНО траектории, и на каждом срезе рисуется
прямоугольник габарита вагона. Предмет в габарите на таком срезе видно глазом,
без всякого детектора.

Как строится срез. Кадры уже сложены со сдвигом на пройденный путь
(`rail_detection.accumulate`), то есть выровнены друг относительно друга вдоль
оси движения. Дальше облако переводится в координаты пути `(d, u, v)`: `u` —
смещение по НОРМАЛИ к оси пути, `v` — высота над плоскостью головок рельсов.
Срез постоянной глубины `d` в этих координатах и есть сечение, перпендикулярное
траектории: на повороте оно режет тоннель не наискось, а поперёк, потому что
поправка на курс уже внесена.

Габарит — прямоугольник 2.2 × 3.3 м, центр — между головками рельсов, низ —
на плоскости головок, и он НАКЛОНЁН по реальному крену пути: на кривой наружный
рельс поднят, вагон стоит на рельсах, значит и его сечение повёрнуто так же
(`rail_detection.roll`). Для сравнения тем же размером рисуется и ненаклонённый
габарит — тонким пунктиром.

Запуск:
    python make_slice_gif.py --bag doubleT_obstacle
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

from rail_detection import bag_path, iter_frames, side_offset, to_track_coords
from rail_detection.roll import frame_roll
from rail_detection.tracker import TunnelTracker

# Габарит: 2.2 м в ширину, 3.3 м в высоту, центр между рельсами.
HALF_WIDTH = 1.10
GAUGE_H = 3.30
POSE_WINDOW = 10.0   # м: крен на срезе — медиана по рельсовым срезам в этом окне

DEPTHS = [5, 10, 20, 30, 40, 50, 55, 60]   # на каких глубинах резать, м

def half_thick(D):
    """Полутолщина среза растёт с глубиной: на 55 м плотность в сотни раз
    меньше, чем на 5 м, и тонкий срез там оказывается пустым независимо от
    того, есть в нём что-нибудь или нет."""
    return 0.8 if D <= 20 else (1.5 if D <= 40 else 2.5)

# Низ ЗОНЫ РАЗБОРА. Паспортный габарит начинается почти от головки рельса, но
# сами головки, накладки и скрепления лежат на v = 0 ± 0.15 м, и при низе 0.10 м
# подсвечивается путь: замер дал скопление «в габарите» в 101 кадре из 101, все
# на v ≈ 0. Порог 0.25 м проходит над путём и оставляет видимым всё, что выше
# четверти метра. Ниже — слепая зона, и это записано честно.
GAUGE_BOTTOM = 0.25   # м над плоскостью головок, в НАКЛОНЁННОЙ системе
# Зона прохода — не константа, а то, что ОСТАЛОСЬ между габаритом и найденной
# стеной. Фиксированную ширину задать нельзя: в этом тоннеле стена стоит на
# 2.37 м, и зона в 2.6 м подсвечивала бы саму стену на каждом срезе. Граница
# берётся от геометрии кадра, а полоса WALL_MARGIN у стены исключается — это
# обделка, лотки и кронштейны, а не предмет.
#
# Человек рядом с путём опасен, даже когда он формально вне габарита: замер на
# этой записи показал людей на u = +1.95…+2.25 м, то есть в 0.6–0.9 м от края
# кузова, в служебном проходе.
WALL_MARGIN = 0.45
U_LIM, V_LIM = 5.0, 5.0
MIN_CLUSTER = 8         # вокселей 10 см в срезе, чтобы назвать это скоплением


def gauge_pose(points, res):
    """Крен и центр габарита по рельсам этого кадра: список (d, крен, u_c, v_c).

    Центр — середина между головками рельсов, переведённая в координаты пути;
    крен — по разнице высот двух головок (два пика профиля пола). Меряется по
    СЫРОМУ кадру, а не по слитому: накопленные точки прошлых кадров лежат только
    в клиренс-полосе, рельсов в них нет.
    """
    fr = frame_roll(points, res["frame"]["rail_records"])
    out = []
    for (d, roll), (_, xc, zh) in zip(fr["rail"], fr["centers"]):
        _, uc, vc = to_track_coords(np.array([xc]), np.array([-d]), np.array([zh]),
                                    res["frame"])
        out.append((d, roll, float(uc[0]), float(vc[0])))
    return out


def pose_at(pose, D):
    """Крен и центр на глубине D: медиана по рельсовым срезам в окне ±POSE_WINDOW.

    Медиана, а не значение ближайшего среза: по одному срезу крен меряется с
    шумом (см. exp_roll_correlation.py), а возвышение рельса меняется плавно —
    на переходной кривой за десятки метров. Там, где рельсов не видно (дальше
    40 м), берётся медиана по всему кадру: это последнее, что про путь известно.
    """
    if not pose:
        return 0.0, 0.0, 0.0
    P = np.array(pose)
    near = np.abs(P[:, 0] - D) <= POSE_WINDOW
    use = P[near] if near.sum() >= 3 else P
    return float(np.median(use[:, 1])), float(np.median(use[:, 2])), float(np.median(use[:, 3]))


def to_gauge_frame(u, v, theta, uc, vc):
    """Координаты в системе вагона: u' — вдоль плоскости головок рельсов,
    v' — по нормали к ней. Поворот на крен вокруг середины между рельсами."""
    du, dv = u - uc, v - vc
    c, s = np.cos(theta), np.sin(theta)
    return du * c + dv * s, -du * s + dv * c


def gauge_corners(theta, uc, vc, bottom=0.0):
    """Углы наклонённого прямоугольника габарита в координатах пути."""
    c, s = np.cos(theta), np.sin(theta)
    pts = [(-HALF_WIDTH, bottom), (HALF_WIDTH, bottom), (HALF_WIDTH, GAUGE_H),
           (-HALF_WIDTH, GAUGE_H)]
    return np.array([(uc + a * c - b * s, vc + a * s + b * c) for a, b in pts])


def slice_clusters(u, v, half_width=HALF_WIDTH):
    """Скопления в заданной зоне на одном срезе. DBSCAN по (u, v): на срезе
    предмет — это связное пятно, а не набор одиночных отражений."""
    from sklearn.cluster import DBSCAN

    # u, v здесь уже в системе вагона (to_gauge_frame)
    inside = (np.abs(u) < half_width) & (v > GAUGE_BOTTOM) & (v < GAUGE_H)
    if inside.sum() < MIN_CLUSTER:
        return inside, []
    pts = np.column_stack([u[inside], v[inside]])
    keys = np.floor(pts / 0.10).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    small = pts[keep]
    if len(small) < MIN_CLUSTER:
        return inside, []
    lab = DBSCAN(eps=0.30, min_samples=4).fit_predict(small)
    out = []
    for L in np.unique(lab[lab >= 0]):
        m = lab == L
        if m.sum() < MIN_CLUSTER:
            continue
        c = small[m]
        out.append({"n": int(m.sum()), "u": float(np.median(c[:, 0])),
                    "v0": float(c[:, 1].min()), "v1": float(c[:, 1].max()),
                    "w": float(np.ptp(c[:, 0]))})
    return inside, out


def slice_walkway(u, v, walls, ug, vg):
    """Скопления в проходе: между габаритом и стеной, но не у самой стены.

    Граница берётся от найденной геометрии, а не задаётся числом: ширина прохода
    в тоннеле не постоянна, и на двухпутном участке её вообще нет с одной
    стороны, а с другой она шесть метров.
    """
    from sklearn.cluster import DBSCAN

    band = np.zeros(len(u), dtype=bool)
    for side, sgn in (("left", -1.0), ("right", +1.0)):
        w = walls.get(side)
        if w is None:
            continue
        band |= (sgn * ug > HALF_WIDTH) & (sgn * u < w - WALL_MARGIN)
    inside = band & (vg > GAUGE_BOTTOM) & (vg < GAUGE_H)
    if inside.sum() < MIN_CLUSTER:
        return inside, []
    pts = np.column_stack([u[inside], v[inside]])
    keys = np.floor(pts / 0.10).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    small = pts[keep]
    if len(small) < MIN_CLUSTER:
        return inside, []
    lab = DBSCAN(eps=0.30, min_samples=4).fit_predict(small)
    out = []
    for L in np.unique(lab[lab >= 0]):
        m = lab == L
        if m.sum() < MIN_CLUSTER:
            continue
        c = small[m]
        h = float(np.ptp(c[:, 1]))
        if h < 0.5:                      # плоское — это настил или лоток
            continue
        out.append({"n": int(m.sum()), "u": float(np.median(c[:, 0])),
                    "v0": float(c[:, 1].min()), "v1": float(c[:, 1].max()),
                    "w": float(np.ptp(c[:, 0]))})
    return inside, out


def collect(dataset, bag, stride, max_frames):
    tracker = TunnelTracker()
    rng = np.random.default_rng(0)
    out = []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=stride,
                                            max_frames=max_frames):
        res = tracker.update(points, steps=stride)
        if res is None:
            continue
        pose = gauge_pose(points, res)
        pts = tracker.merged
        x = pts['x'].astype(float)
        y = pts['y'].astype(float)
        z = pts['z'].astype(float)
        d, u, v = to_track_coords(x, y, z, res["frame"])

        panels = []
        for D in DEPTHS:
            ht = half_thick(D)
            sel = (d > D - ht) & (d < D + ht) & (np.abs(u) < U_LIM) \
                & (v > -1.0) & (v < V_LIM)
            uu, vv = u[sel], v[sel]
            if len(uu) > 6000:
                k = rng.choice(len(uu), 6000, replace=False)
                uu, vv = uu[k], vv[k]
            theta, uc, vc = pose_at(pose, D)
            ug, vg = to_gauge_frame(uu, vv, theta, uc, vc)
            in_g, cl_g = slice_clusters(ug, vg, HALF_WIDTH)
            # Где стоят стены на этой глубине — по геометрии кадра
            walls = {}
            for side in ("left", "right"):
                if res[side] is None:
                    continue
                w = side_offset(res["shape"], side, np.array([float(D)]))
                if w is not None and np.isfinite(w[0]):
                    walls[side] = float(w[0])
            in_s, cl_s = slice_walkway(uu, vv, walls, ug, vg)
            panels.append({"u": uu.astype(np.float32), "v": vv.astype(np.float32),
                           "in_g": in_g, "in_s": in_s & ~in_g,
                           "cl_g": cl_g, "cl_s": cl_s, "depth": D,
                           "pose": (theta, uc, vc)})
        rolls = [r for _, r, _, _ in pose]
        out.append({"idx": idx, "n_total": n_total, "panels": panels,
                    "roll": float(np.median(rolls)) if rolls else None})
        print(f"\r  кадр {idx}/{n_total}", end="", flush=True)
    print()
    return out


def render(records, bag, dpi=110):
    cols = 4
    rows = (len(DEPTHS) + cols - 1) // cols
    fig = Figure(figsize=(3.1 * cols, 3.0 * rows + 0.6), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    axes = [fig.add_subplot(rows, cols, i + 1) for i in range(len(DEPTHS))]
    images = []
    for n, r in enumerate(records):
        found = []
        for ax, p in zip(axes, r["panels"]):
            ax.clear()
            rest = ~(p["in_g"] | p["in_s"])
            ax.scatter(p["u"][rest], p["v"][rest], s=1.2, c="#9aa5b1", alpha=0.55)
            if p["in_s"].any():
                ax.scatter(p["u"][p["in_s"]], p["v"][p["in_s"]], s=2.6,
                           c="#e58f2a", alpha=0.85)
            if p["in_g"].any():
                ax.scatter(p["u"][p["in_g"]], p["v"][p["in_g"]], s=3.2,
                           c="#d1495b", alpha=0.95)
            theta, uc, vc = p["pose"]
            # Тот же габарит без наклона и С ТЕМ ЖЕ центром: отличаться от
            # красного он обязан только креном, иначе глаз увидит две разницы.
            ax.add_patch(Polygon(gauge_corners(0.0, uc, vc), closed=True, fill=False,
                                 edgecolor="#6b7280", lw=0.8, ls="--"))
            ax.add_patch(Polygon(gauge_corners(theta, uc, vc), closed=True, fill=False,
                                 edgecolor="#d1495b", lw=1.7))
            # плоскость головок рельсов — наклонённая, по ней и стоит габарит
            c_, s_ = np.cos(theta), np.sin(theta)
            ax.plot([uc - 0.8 * c_, uc + 0.8 * c_], [vc - 0.8 * s_, vc + 0.8 * s_],
                    c="#7a5c3a", lw=2.0)
            ax.set_xlim(-U_LIM, U_LIM)
            ax.set_ylim(-1.0, V_LIM)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            title = (f"{p['depth']} м  (±{half_thick(p['depth']):.1f}), "
                     f"крен {np.degrees(p['pose'][0]):+.1f}°")
            colour = "black"
            if p["cl_g"]:
                big = max(p["cl_g"], key=lambda c: c["n"])
                title += f"\nВ ГАБАРИТЕ: u={big['u']:+.2f}, {big['n']} яч."
                colour = "#d1495b"
                found.append(("габарит", p["depth"]))
            elif p["cl_s"]:
                big = max(p["cl_s"], key=lambda c: c["n"])
                title += (f"\nрядом с путём: u={big['u']:+.2f}, "
                          f"h={big['v1'] - big['v0']:.1f} м")
                colour = "#e58f2a"
                found.append(("рядом", p["depth"]))
            ax.set_title(title, fontsize=7.5, color=colour)
        in_gauge = [str(dep) for kind, dep in found if kind == "габарит"]
        near_track = [str(dep) for kind, dep in found if kind == "рядом"]
        parts = []
        if in_gauge:
            parts.append("В ГАБАРИТЕ на " + ", ".join(in_gauge) + " м")
        if near_track:
            parts.append("рядом с путём на " + ", ".join(near_track) + " м")
        verdict = "   |   ".join(parts) if parts else "чисто на всех срезах"
        roll = f"{np.degrees(r['roll']):+.2f}°" if r.get("roll") is not None else "—"
        fig.suptitle(f"{bag}  кадр {r['idx']}/{r['n_total']}   —   срезы ПЕРПЕНДИКУЛЯРНО "
                     f"траектории. Красный — габарит {2 * HALF_WIDTH:.1f}×{GAUGE_H:.1f} м, "
                     f"наклонён по крену рельсов (кадр {roll}); пунктир — тот же без "
                     f"наклона\n{verdict}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=96))
        print(f"\r  отрисовано {n + 1}/{len(records)}", end="", flush=True)
    print()
    return images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bag", default="doubleT_obstacle")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out", default="output")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--tag", default="_AFTER", help="суффикс имени GIF")
    a = p.parse_args()

    print(f"=== {a.bag} (шаг {a.stride}) ===")
    recs = collect(a.dataset, a.bag, a.stride, a.max_frames)
    if not recs:
        print("кадры не обработались")
        return
    imgs = render(recs, a.bag)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"slices_{a.bag}{a.tag}.gif"
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / a.fps), loop=0, optimize=True)
    n_hit = sum(1 for r in recs if any(p["cl_g"] for p in r["panels"]))
    n_near = sum(1 for r in recs if any(p["cl_s"] for p in r["panels"]))
    rolls = [np.degrees(r["roll"]) for r in recs if r.get("roll") is not None]
    if rolls:
        print(f"крен рельсов по кадрам: медиана {np.median(rolls):+.2f}°, "
              f"5-95% {np.percentile(rolls, 5):+.2f}…{np.percentile(rolls, 95):+.2f}°")
    print(f"GIF: {path} ({path.stat().st_size / 1e6:.1f} МБ)")
    print(f"кадров со скоплением В ГАБАРИТЕ: {n_hit}/{len(recs)}")
    print(f"кадров с объектом РЯДОМ С ПУТЁМ: {n_near}/{len(recs)}")


if __name__ == "__main__":
    main()
