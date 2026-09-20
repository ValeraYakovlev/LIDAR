"""Габарит вдоль пути, найденного по контрасту вида сверху — на всю дальность.

Эксперимент 15 (ветка `feature/contrast-gauge`). Собирает вместе две вещи,
которые до сих пор жили порознь:

* **путь берётся у контраста** (`views.run_silhouette`, эксперимент 14): вид
  сверху нормируется по глубине, и граница тоннеля — граница контраста. Она
  наблюдается до 120–150 м, тогда как клиренс-полоса §15–§21 кончается на 78 м.
* **поперечное положение и крен берутся у рельсов** — там, где они видны
  (5–34 м). Контраст даёт ось ТОННЕЛЯ, а она на двухпутном участке и у платформы
  смещена от оси ПУТИ на 1–2 м, и габарит, поставленный по центру тоннеля, стоял
  бы не над путём. Сдвиг между осью тоннеля и осью пути меряется по ближним
  рельсам и переносится на всю глубину: дальше тоннель и путь идут параллельно.

## Плоскость габарита перпендикулярна пути

Точка кадра переводится в координаты пути `(s, u, v)`, где `s` — длина ВДОЛЬ
пути, а не глубина `d`: на повороте с курсом 10° плоскость постоянной глубины
режет путь наискось, и габарит в ней оказывается шире настоящего на 1.5%, а
предмет — смещённым по дальности. Проекция делается на касательную:

    s = S(d) + (x − x_path(d))·sin ψ(d),    u = (x − x_path(d))·cos ψ(d)

где `S(d)` — длина дуги пути до глубины `d`, `ψ(d)` — курс пути. Срез
постоянного `s` и есть сечение, перпендикулярное аппроксимированному пути.

Сам габарит — жёсткий прямоугольник 2.2 × 3.3 м (§25), центр между головками
рельсов, наклонён по измеренному крену пути. Он не подстраивается под тоннель:
если бы подстраивался, узкое место перестало бы быть препятствием.

## Честная цена

Ниже 0.25 м над головками рельсов разбора нет: там сам путь (головки, накладки,
скрепления) занят на 100% кадров. Лежащий предмет ниже четверти метра этот
детектор не увидит — ровно как в §25.

Дальше предела наблюдения контраста путь не измерен, и находок там не ищем.
Внутри предела находки делятся на сырые (что попало в коробку на этом кадре) и
подтверждённые: неподвижное препятствие приближается ровно на измеренное Δs за
кадр (`gauge.ObstacleWatch`, §24), а дрожание оси этому правилу не подчиняется.
"""

import numpy as np

from .gauge import ObstacleWatch
from .roll import rail_pose_track
from .shift import estimate_shift
from .tunnel_frame import rail_samples
from .curvature import ransac_poly_fit
from .views import D_MAX, floor_level, rasterize, run_silhouette, shape_x

# Габарит: 2.2 м в ширину, 3.3 м в высоту от плоскости головок (§25).
HALF_WIDTH = 1.10
GAUGE_H = 3.30
GAUGE_BOTTOM = 0.25    # м над головками: ниже лидар видит сам путь (§25)

NEAR_LIMIT = 2.0       # ближе — собственная конструкция поезда (§12)
RAIL_NEAR = 25.0       # м: докуда рельсам верим при замере сдвига оси
POSE_MAX_DEPTH = 34.0  # м: предел замера крена (§25)
POSE_DEPTHS = np.arange(4.0, POSE_MAX_DEPTH + 0.1, 2.0)
POSE_WINDOW = 10.0     # м: окно медианы крена
FALLBACK_HEAD = 0.15   # м: головки над плато пола, когда рельсов не нашлось

# Полоса высот для замера Δs — та же клиренс-полоса, что и в §20.
BAND_LO, BAND_HI = 0.2, 1.1

VOXEL = 0.05
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 12
MIN_CLUSTER_POINTS = 25      # вокселей
MIN_CLUSTER_EXTENT = 0.25    # м


def half_thick(s):
    """Полутолщина среза растёт с дальностью: на 100 м плотность в сотни раз
    меньше, чем на 5 м, и тонкий срез там пуст независимо от содержимого."""
    return 0.8 if s <= 20 else (1.5 if s <= 40 else (2.5 if s <= 70 else 4.0))


PATH_GRID = np.arange(0.0, D_MAX + 0.25, 0.25)
RAIL_TOL = 0.08        # м: допуск RANSAC при подгонке прямой по рельсам


def build_path(fit, rail_d, rail_x, prev_offset=0.0, near=RAIL_NEAR):
    """Путь кадра: рельсы вблизи, кривизна контраста дальше.

    Постоянного сдвига между осью тоннеля и осью пути НЕ ХВАТАЕТ, и это замер, а
    не предположение. На станции `doubleT_platform` рельсы стоят на x = -0.05 от
    4 до 20 м (прямая), а ось контраста на том же участке идёт наискось: путь,
    выровненный по ней одним сдвигом, пересекал рельсы и уезжал на ±0.5 м. В
    габарит из-за этого попадал контактный рельс — 42 ложных находки из 115
    кадров, все на 2 м, все одной формы 4 × 0.2 × 0.2 м.

    Поэтому рельсы задают у пути и ПОЛОЖЕНИЕ, и НАПРАВЛЕНИЕ до глубины d0, а
    контраст добавляет только то, чего у рельсов нет, — искривление дальше d0:

        x(d) = c0 + c1*d                                          d <= d0
        x(d) = c0 + c1*d + [x_c(d) - x_c(d0) - x_c'(d0)*(d-d0)]   d > d0

    Скобка — это в точности «форма контраста за вычетом её касательной в d0»,
    то есть чистая кривизна. Склейка гладкая: в d0 совпадают и значение, и наклон.

    Без рельсов (а таких кадров сотни подряд на гермозатворе) путь — ось
    контраста плюс последний известный сдвиг; это честно записывается в mode.
    """
    x_c = shape_x(fit, PATH_GRID)
    line = None
    if len(rail_d):
        m = rail_d <= near
        if m.sum() >= 3:
            line = ransac_poly_fit(rail_d[m], rail_x[m], 1, RAIL_TOL)
            if line is None:
                line = {"coeffs": np.polyfit(rail_d[m], rail_x[m], 1)}
    if line is None:
        x = x_c + prev_offset
        mode, d0 = "contrast+offset", None
    else:
        c1, c0 = float(line["coeffs"][0]), float(line["coeffs"][1])
        d0 = float(min(near, rail_d[rail_d <= near].max()))
        k = int(np.searchsorted(PATH_GRID, d0))
        sc = np.gradient(x_c, PATH_GRID)
        bend = x_c - x_c[k] - sc[k] * (PATH_GRID - PATH_GRID[k])
        x = c0 + c1 * PATH_GRID + np.where(PATH_GRID > d0, bend, 0.0)
        mode = "rails+contrast"
    psi = np.arctan(np.gradient(x, PATH_GRID))
    arc = np.concatenate([[0.0], np.cumsum(np.sqrt(1 + np.tan(psi[1:]) ** 2)
                                           * np.diff(PATH_GRID))])
    return {"d": PATH_GRID, "x": x, "psi": psi, "arc": arc, "mode": mode, "d0": d0,
            "offset": float(x[0] - x_c[0])}


def path_at(path, depths):
    """(x_path, курс ψ, длина дуги S) на заданных глубинах."""
    depths = np.asarray(depths, float)
    return (np.interp(depths, path["d"], path["x"]),
            np.interp(depths, path["d"], path["psi"]),
            np.interp(depths, path["d"], path["arc"]))


def to_path_coords(x, y, z, path, floor_coeffs):
    """(s, u, v): вдоль пути, по нормали к нему, над уровнем пола.

    Ровно та проекция, которая делает плоскость постоянного s перпендикулярной
    пути. Высота отсчитывается от профиля пола кадра; плоскость головок рельсов
    добавляется позже, вместе с креном (rail_pose_track).
    """
    d = -y
    x_p, psi, arc = path_at(path, d)
    dx = x - x_p
    return arc + dx * np.sin(psi), dx * np.cos(psi), z - np.polyval(floor_coeffs, d)


def _pose_curves(pose, grid):
    """Крен, центр и уровень головок вдоль пути — медианой по окну ±POSE_WINDOW.

    Дальше предела замера (POSE_MAX_DEPTH) значения не экстраполируются, а
    держатся такими, какими были на самом дальнем надёжном участке: это
    последнее, что про путь известно (§25).
    """
    if not pose:
        return (np.zeros_like(grid), np.zeros_like(grid),
                np.full_like(grid, FALLBACK_HEAD), False)
    P = np.array(pose)
    th, uc, vc = [], [], []
    for s in grid:
        near = np.abs(P[:, 0] - s) <= POSE_WINDOW
        use = P[near] if near.sum() >= 3 else P[np.argsort(np.abs(P[:, 0] - s))[:3]]
        th.append(np.median(use[:, 1]))
        uc.append(np.median(use[:, 2]))
        vc.append(np.median(use[:, 3]))
    return np.array(th), np.array(uc), np.array(vc), True


def _clean_pose(pose):
    """Срез, где одна головка не нашлась, даёт крен −9…−10° при соседях −3…−5°.
    Возвышение рельса так не скачет — такие срезы отбрасываются до медианы."""
    if len(pose) < 4:
        return pose
    r = np.array([q[1] for q in pose])
    med = np.median(r)
    mad = 1.4826 * np.median(np.abs(r - med))
    tol = max(3.0 * mad, np.radians(1.5))
    return [q for q in pose if abs(q[1] - med) <= tol]


def gauge_corners(theta, uc, vc, bottom=GAUGE_BOTTOM, half=HALF_WIDTH, top=GAUGE_H):
    """Углы наклонённого прямоугольника габарита в координатах (u, v)."""
    c, s = np.cos(theta), np.sin(theta)
    pts = [(-half, bottom), (half, bottom), (half, top), (-half, top)]
    return np.array([(uc + a * c - b * s, vc + a * s + b * c) for a, b in pts])


def _voxel(pts, size=VOXEL):
    keys = np.floor(pts / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def cluster(su, uv, vv, min_points=MIN_CLUSTER_POINTS):
    """DBSCAN по точкам внутри габарита, в координатах (s, u', v')."""
    from sklearn.cluster import DBSCAN

    pts = np.column_stack([su, uv, vv])
    if len(pts) < DBSCAN_MIN_SAMPLES:
        return []
    small = _voxel(pts)
    if len(small) < DBSCAN_MIN_SAMPLES:
        return []
    lab = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(small)
    out = []
    for L in np.unique(lab[lab >= 0]):
        m = lab == L
        if m.sum() < min_points:
            continue
        c = small[m]
        size = c.max(axis=0) - c.min(axis=0)
        # Скользящий вдоль стены шум даёт много точек, но он плоский; предмет на
        # путях имеет габарит во всех трёх измерениях (§24).
        if size.max() < MIN_CLUSTER_EXTENT:
            continue
        out.append({"dist": float(c[:, 0].min()), "n": int(m.sum()),
                    "size": tuple(float(t) for t in size),
                    "u": float(np.median(c[:, 1])), "v": float(np.median(c[:, 2]))})
    out.sort(key=lambda q: q["dist"])
    return out


class ContrastGauge:
    """Последовательный проход по записи: путь по контрасту, габарит по рельсам.

    Состояние между кадрами: подсказка оси для контраста, последний известный
    сдвиг оси пути (на случай кадров без рельсов), профиль плотности для Δs и
    подтверждение находки по приближению.
    """

    def __init__(self, confirm=3):
        self.prior = None
        self.offset = 0.0
        self.prev_band = None
        self.watch = ObstacleWatch(confirm=confirm)

    def update(self, points, steps=1):
        x = points['x'].astype(float)
        y = points['y'].astype(float)
        z = points['z'].astype(float)
        keep = (np.abs(x) + np.abs(y) + np.abs(z)) > 0.1
        x, y, z = x[keep], y[keep], z[keep]

        grid = rasterize(points)
        sil = run_silhouette(grid, self.prior)
        fit = sil["fit"]
        self.prior = fit
        if fit is None or not fit.get("reach"):
            self.prev_band = None
            self.watch.reset()
            return None

        floor, _, _ = floor_level(grid, lambda d: float(shape_x(fit, d)))
        if floor is None:
            floor = np.array([0.0, float(np.percentile(z, 2))])

        rail_d, rail_x, _, gauge = rail_samples(points)
        path = build_path(fit, rail_d, rail_x, self.offset)
        if path["mode"] == "rails+contrast":
            self.offset = path["offset"]
        s, u, v = to_path_coords(x, y, z, path, floor)

        pose = _clean_pose(rail_pose_track(s, u, v, gauge or 1.60, POSE_DEPTHS,
                                           lambda D: max(1.0, half_thick(D))))
        pgrid = np.arange(0.0, D_MAX + 1.0, 1.0)
        theta, uc, vc, has_pose = _pose_curves(pose, pgrid)

        # Каждая точка поворачивается на крен СВОЕГО сечения: коробка едет вдоль
        # пути, а не стоит одна на весь кадр.
        th_p = np.interp(s, pgrid, theta)
        uc_p = np.interp(s, pgrid, uc)
        vc_p = np.interp(s, pgrid, vc)
        du, dv = u - uc_p, v - vc_p
        ug = du * np.cos(th_p) + dv * np.sin(th_p)
        vg = -du * np.sin(th_p) + dv * np.cos(th_p)

        limit = min(float(fit["reach"]), D_MAX)
        inside = ((np.abs(ug) <= HALF_WIDTH) & (vg >= GAUGE_BOTTOM) & (vg <= GAUGE_H)
                  & (s > NEAR_LIMIT) & (s <= limit))
        clusters = cluster(s[inside], ug[inside], vg[inside])

        band = {"d": s[(v >= BAND_LO) & (v <= BAND_HI) & (s > 2) & (s < 60)],
                "u": u[(v >= BAND_LO) & (v <= BAND_HI) & (s > 2) & (s < 60)],
                "v": v[(v >= BAND_LO) & (v <= BAND_HI) & (s > 2) & (s < 60)]}
        shift = None
        if self.prev_band is not None and len(band["d"]) > 500:
            est = estimate_shift(self.prev_band, band, max_shift=2.2 * max(steps, 1))
            if est.get("ok"):
                shift = float(est["shift"])
        self.prev_band = band if len(band["d"]) > 500 else None
        confirmed = self.watch.update(clusters[0]["dist"] if clusters else None, shift)

        return {
            "fit": fit, "grid": grid, "silhouette": sil, "offset": self.offset,
            "path": path,
            "floor": floor, "pose": pose, "has_pose": has_pose,
            "pose_curves": (pgrid, theta, uc, vc), "gauge_value": gauge,
            "s": s, "u": u, "v": v, "ug": ug, "vg": vg, "xyz": (x, y, z),
            "inside": inside, "clusters": clusters, "confirmed": confirmed,
            "shift": shift, "limit": limit, "reach": float(fit["reach"]),
        }


def corridor_lines(res, depths, half=HALF_WIDTH):
    """Края габаритного коридора на виде сверху — ровно то, что проверяется.

    Коридор отступает от пути на `half` ПО НОРМАЛИ, поэтому в координатах x
    отступ равен half/cos ψ: на повороте коридор в проекции шире самого габарита,
    и рисовать его постоянной шириной значило бы показывать не то, что проверяется.

    Центр берётся с той же поправкой u_c, с какой идёт и детекция: середина
    между головками рельсов может не совпасть с осью пути на пару сантиметров,
    и коробка стоит по головкам, а не по оси.
    """
    x_p, psi, arc = path_at(res["path"], depths)
    pgrid, _, uc, _ = res["pose_curves"]
    centre = x_p + np.interp(arc, pgrid, uc) / np.cos(psi)
    return centre - half / np.cos(psi), centre, centre + half / np.cos(psi)


def slice_points(res, dist, u_lim=5.0, v_lim=5.5):
    """Точки в плоскости, ПЕРПЕНДИКУЛЯРНОЙ пути, на дальности dist вдоль пути."""
    s, ug, vg = res["s"], res["ug"], res["vg"]
    ht = half_thick(dist)
    m = (np.abs(s - dist) < ht) & (np.abs(ug) < u_lim) & (vg > -1.2) & (vg < v_lim)
    return m, ht


def pose_at(res, dist):
    """Крен, центр колеи и уровень головок на заданной дальности вдоль пути."""
    pgrid, theta, uc, vc = res["pose_curves"]
    return (float(np.interp(dist, pgrid, theta)), float(np.interp(dist, pgrid, uc)),
            float(np.interp(dist, pgrid, vc)))
