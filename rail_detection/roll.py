"""Крен пути и крен свода — поперечный наклон в плоскости среза.

На кривой наружный рельс поднят (возвышение наружного рельса), и плоскость
головок рельсов наклонена поперёк пути. Вагон стоит на рельсах, значит и его
габарит наклонён так же: прямоугольник, поставленный «ровно», на кривой режет
лишний угол у одной стены и не досматривает у другой.

Координаты пути (`to_track_coords`) этот наклон НЕ снимают: высота `v`
отсчитывается от профиля пола `z_f(d)`, который зависит только от глубины. Крен
приходится мерить отдельно, и здесь он меряется двумя независимыми способами.

  по рельсам — две головки рельса дают два пика в профиле пола Z(X); разница их
      высот на расстоянии колеи и есть крен. Детектор рельсов эти пики уже
      находит (`find_rails`), здесь по ним только уточняется высота.

  по своду — нижняя поверхность свода над путём, прямая по ней. У квадратного
      тоннеля это прямо крен потолка. У круглого — ослабленный: поворот свода
      смещает вершину дуги вбок, а наклон прямой по центральной части дуги
      меньше самого угла примерно во столько раз, во сколько центр дуги ниже её
      радиуса. Поэтому сравнивать надо корреляцию, а не сами значения.
"""

import numpy as np

HEAD_WINDOW = 0.05     # м: окрестность рельса по x, в которой ищется головка
HEAD_PCT = 90.0        # перцентиль высоты — верх головки, а не подошва
CEIL_V_MIN = 3.0       # м над головкой рельса: ниже свода не бывает
CEIL_V_MAX = 6.0
CEIL_HALF_SPAN = 1.2   # м вбок от центра колеи: центральная часть свода
CEIL_BIN = 0.10        # м: шаг профиля свода по x
MIN_BINS = 8


def rail_roll(points, rec):
    """Крен по двум головкам рельса в одном срезе, радианы. None — не измерено.

    Возвращает (крен, x центра колеи, z головок в центре) — всё в координатах
    сенсора: срез постоянной глубины, так что пара рельсов в нём на одной глубине.
    """
    x = points['x']
    z = points['z']
    depth = -points['y']
    sl = (depth >= rec["depth_lo"]) & (depth < rec["depth_hi"])
    lo, hi = rec["shoulder_z"] - 0.25, rec["shoulder_z"] + 0.35
    heads = []
    for xr in (rec["x_rail_left"], rec["x_rail_right"]):
        m = sl & (np.abs(x - xr) < HEAD_WINDOW) & (z > lo) & (z < hi)
        if m.sum() < 8:
            return None
        heads.append(float(np.percentile(z[m], HEAD_PCT)))
    dx = rec["x_rail_right"] - rec["x_rail_left"]
    if dx < 1.2:
        return None
    return (float(np.arctan2(heads[1] - heads[0], dx)),
            float((rec["x_rail_left"] + rec["x_rail_right"]) / 2),
            float((heads[0] + heads[1]) / 2))


def ceiling_tilt(points, rec, x_center, z_heads):
    """Наклон нижней поверхности свода над путём в том же срезе, радианы.

    Профиль свода — медиана высоты по бинам x (медиана, а не нижний перцентиль:
    ниже свода висят светильники и кабели, и нижний край ловил бы их). Прямая
    подгоняется дважды: второй раз без бинов, отстоящих больше чем на 10 см, —
    иначе одиночный кронштейн перекашивает наклон.
    """
    x = points['x']
    z = points['z']
    depth = -points['y']
    m = ((depth >= rec["depth_lo"]) & (depth < rec["depth_hi"])
         & (np.abs(x - x_center) < CEIL_HALF_SPAN)
         & (z > z_heads + CEIL_V_MIN) & (z < z_heads + CEIL_V_MAX))
    if m.sum() < 30:
        return None
    xs, zs = x[m], z[m]
    edges = np.arange(x_center - CEIL_HALF_SPAN, x_center + CEIL_HALF_SPAN + CEIL_BIN, CEIL_BIN)
    idx = np.digitize(xs, edges)
    bx, bz = [], []
    for i in range(1, len(edges)):
        k = idx == i
        if k.sum() >= 3:
            bx.append((edges[i - 1] + edges[i]) / 2)
            bz.append(float(np.median(zs[k])))
    bx, bz = np.array(bx), np.array(bz)
    if len(bx) < MIN_BINS or bx.min() > x_center - 0.6 or bx.max() < x_center + 0.6:
        return None
    a, b = np.polyfit(bx, bz, 1)
    keep = np.abs(bz - (a * bx + b)) < 0.10
    if keep.sum() < MIN_BINS:
        return None
    a, b = np.polyfit(bx[keep], bz[keep], 1)
    return float(np.arctan(a))


def frame_roll(points, rail_records):
    """Крен кадра по рельсам и по своду: списки по срезам и медиана.

    Медиана по срезам, а не каждое значение по отдельности: на колее 1.6 м
    ошибка высоты головки в 2 см даёт 0.7° крена, а срезов в кадре десяток.
    """
    rails, ceils, pairs, centers = [], [], [], []
    for rec in rail_records:
        r = rail_roll(points, rec)
        if r is None:
            continue
        roll, xc, zh = r
        d = (rec["depth_lo"] + rec["depth_hi"]) / 2
        rails.append((d, roll))
        centers.append((d, xc, zh))
        c = ceiling_tilt(points, rec, xc, zh)
        if c is not None:
            ceils.append((d, c))
            pairs.append((d, roll, c))
    med = lambda v: float(np.median([t[1] for t in v])) if v else None
    return {"rail": rails, "ceil": ceils, "pairs": pairs, "centers": centers,
            "rail_med": med(rails), "ceil_med": med(ceils)}


# --- Крен по рельсам В ИЗВЕСТНОМ МЕСТЕ -----------------------------------------
#
# Детектор рельсов (`find_rails`) ищет пару пиков заново в каждом срезе, и дальше
# 20–25 м на двухпутном участке регулярно хватает ЧУЖУЮ пару: на doubleT_obstacle
# в кадре 8 срезы 32.5 и 37.5 м дали центр колеи на u = +1.52 и +2.35 м при оси на
# нуле. Геометрия такие срезы отбрасывает (RAIL_WINDOW = 0.30 м от оси), а поза
# габарита их брала — и на 30 м медиана из трёх срезов, два из которых чужие,
# уводила габарит на полтора метра вбок.
#
# Но искать пару заново и не нужно: ось пути и колея уже известны. Поэтому
# головки меряются там, где им положено быть, — у u = ±колея/2 от оси, — и это
# работает и там, где детектор промахнулся.

HEAD_SEARCH = 0.25     # м: окно поиска головки вокруг ожидаемого положения
HEAD_BIN = 0.02        # м: шаг профиля по u внутри окна
GAUGE_TOL = 0.15       # м: найденная колея не может отличаться от известной больше


def _head(u, v, u_expected):
    """Головка рельса в окне вокруг ожидаемого положения: самый высокий бин
    профиля. В узком окне у пола выше головки рельса ничего нет — скрепления и
    шпала ниже, контактный рельс снаружи окна."""
    m = np.abs(u - u_expected) < HEAD_SEARCH
    if m.sum() < 10:
        return None
    uu, vv = u[m], v[m]
    edges = np.arange(u_expected - HEAD_SEARCH, u_expected + HEAD_SEARCH + HEAD_BIN, HEAD_BIN)
    idx = np.digitize(uu, edges)
    best = None
    for i in range(1, len(edges)):
        k = idx == i
        if k.sum() < 3:
            continue
        h = float(np.percentile(vv[k], HEAD_PCT))
        if best is None or h > best[1]:
            best = ((edges[i - 1] + edges[i]) / 2, h)
    return best


def rail_pose_track(d, u, v, gauge, depths, half_thick):
    """Крен и середина между головками по срезам, в КООРДИНАТАХ ПУТИ.

    d, u, v — точки кадра в координатах пути; gauge — колея кадра.
    Для каждой глубины: головки ищутся у u = ±gauge/2, пара принимается, только
    если найденная колея не дальше GAUGE_TOL от известной. Возвращает список
    (d, крен, u_c, v_c).
    """
    sel = (v > -0.5) & (v < 0.8) & (np.abs(u) < gauge / 2 + HEAD_SEARCH + 0.1)
    d, u, v = d[sel], u[sel], v[sel]
    out = []
    for D in depths:
        ht = half_thick(D)
        m = (d > D - ht) & (d < D + ht)
        if m.sum() < 30:
            continue
        L = _head(u[m], v[m], -gauge / 2)
        R = _head(u[m], v[m], +gauge / 2)
        if L is None or R is None:
            continue
        g = R[0] - L[0]
        if abs(g - gauge) > GAUGE_TOL:
            continue
        out.append((float(D), float(np.arctan2(R[1] - L[1], g)),
                    float((L[0] + R[0]) / 2), float((L[1] + R[1]) / 2)))
    return out
