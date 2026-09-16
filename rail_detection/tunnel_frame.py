"""Геометрия тоннеля в системе координат, связанной С ПУТЁМ (rail-guided).

Зачем отдельный подход. Прежние детекторы стен искали стену независимо в каждом
срезе прямо в координатах сенсора (X вбок, глубина по -Y) и брали то ближайшее
к центру значимое скопление точек, то самое дальнее. И то и другое неустойчиво:
в тоннеле метро между путём и стеной есть своя "начинка" (контактный рельс,
кабельные лотки, настил служебного прохода), а сама стена местами уходит в нишу
или раскрывается в соседний путь. Никакого признака "это именно стена" у
одиночного среза нет, поэтому ошибка среза ничем не ограничена.

Здесь вместо этого используются два физических ограничения, которых у прежних
методов не было.

1. Стена идёт ПАРАЛЛЕЛЬНО пути. В криволинейных координатах пути (u — смещение
   вбок от оси, v — высота над головкой рельса) стена перестаёт быть кривой и
   становится линией почти постоянного смещения. Задача "восстановить кривую по
   шумным срезам" превращается в "найти одно число, согласованное по всему
   кадру", а это решается голосованием по всем срезам сразу.

2. Обе стены и путь между ними — ОДНА И ТА ЖЕ кривая, сдвинутая вбок. Трасса
   железнодорожного пути состоит из прямых, круговых кривых постоянного радиуса
   и переходных кривых между ними; в окне 40 м круговая кривая с точностью до
   малых углов описывается параболой x = d²/(2R). Поэтому форма (прямая или
   дуга и её радиус) подгоняется ОДНА на обе стены сразу, а стороны отличаются
   только постоянной полушириной. Это убирает главный артефакт независимой
   подгонки — расхождение стен "домиком", когда левая и правая уезжают в разные
   стороны, чего в тоннеле быть не может.

Кривизна берётся именно со стен, а не с рельсов. Рельсы надёжно детектируются
лишь вблизи (до ~20 м), а на такой базе поворот неотличим от шума: при радиусе
300 м стрелка прогиба на 17 м составляет 12 см — меньше разброса самого
детектора. Стены же видны на всю глубину кадра и с обеих сторон, и на базе 40 м
тот же поворот даёт уже 0.67 м. Поэтому ось пути по рельсам используется только
как ПРЯМАЯ опора (положение и курс вблизи), а всю кривизну определяют стены.

Полоса поиска по высоте выбрана по данным, а не наугад: настил платформы лежит
на v ≈ +1.2 м над головкой рельса и вбок уходит на 4-5 м, кабельные лотки и
своды — ещё выше, поэтому широкая полоса (прежние floor+0.3..+2.5) неизбежно
захватывает их вместо стены. В полосе v ≈ 0.2..1.1 м остаётся ровно то, что и
образует боковую границу: вертикальная грань стены тоннеля либо торец
платформы. Это же и есть полоса, важная для габарита: препятствие на пути
находится именно здесь.
"""

import numpy as np

from .curvature import eval_fit, ransac_poly_fit, slope_from_fit
from .detector import DEFAULT_DEPTH_BINS, analyze_frame

# Более частые срезы, чем DEFAULT_DEPTH_BINS: стена — сплошная поверхность, по
# ней точек хватает и на узких срезах, а частая сетка даёт гладкую кривую и
# больше голосов для согласования.
WALL_DEPTH_BINS = (
    [(d, d + 2) for d in range(3, 21, 2)]
    + [(d, d + 3) for d in range(21, 33, 3)]
    + [(33, 37), (37, 42)]
)

V_LO, V_HI = 0.2, 1.1      # полоса высот над головкой рельса (см. docstring)
U_MIN, U_MAX = 1.0, 6.5    # коридор поиска стены вбок от оси пути, м
NEAR_DEPTH = 20.0          # "ближняя зона": там рельсы видно надёжно
DEPTH_SCALE = 40.0         # нормировка глубины в модели формы (обусловленность)
MIN_RADIUS = 120.0         # м: круче метро не поворачивает
CURVE_GAIN = 0.85          # дуга обязана сбить невязку хотя бы во столько раз
CURVE_SNR = 4.0            # и увести стену вбок хотя бы во столько раз сильнее шума


def fit_track_frame(points, depth_bins=DEFAULT_DEPTH_BINS, near_depth=NEAR_DEPTH):
    """Шаг 1: прямая опора по рельсам + профиль уровня пола по глубине.

    Ось подгоняется ПРЯМОЙ и только ПО БЛИЖНЕЙ ЗОНЕ (до near_depth): вблизи
    точек на порядок больше и желоб виден надёжно, а на 30-40 м детектор рельс
    уже регулярно промахивается. Кривизну здесь не ищем намеренно — на базе
    ближней зоны она неотличима от шума детектора (см. docstring модуля), её
    определят стены.

    Возвращает dict {axis_fit, floor_coeffs, gauge, rail_records} или None,
    если рельсы не нашлись даже вблизи.
    """
    rail_records, _ = analyze_frame(points, depth_bins)
    if len(rail_records) < 3:
        return None

    d = np.array([(r["depth_lo"] + r["depth_hi"]) / 2 for r in rail_records])
    xc = np.array([r["rail_center"] for r in rail_records])
    zf = np.array([r["shoulder_z"] for r in rail_records])
    gauge = np.array([r["rail_gauge"] for r in rail_records])

    # Колея физически постоянна: срез, где она заметно отличается от медианы —
    # это промах детектора (зацепил не тот пик), а не реальная геометрия.
    g_med = float(np.median(gauge))
    keep = np.abs(gauge - g_med) < 0.25
    if keep.sum() >= 3:
        d, xc, zf = d[keep], xc[keep], zf[keep]

    near = d <= near_depth
    if near.sum() >= 3:
        d, xc, zf = d[near], xc[near], zf[near]
    if len(d) < 3:
        return None

    axis = ransac_poly_fit(d, xc, 1, 0.15)
    coeffs = axis["coeffs"] if axis is not None else np.array([0.0, float(np.median(xc))])
    floor = ransac_poly_fit(d, zf, 1, 0.06)
    floor_coeffs = floor["coeffs"] if floor is not None else np.array([0.0, float(np.median(zf))])

    return {
        "axis_fit": {"kind": "straight", "coeffs": coeffs, "radius": None,
                     "residual": axis["residual"] if axis is not None else None,
                     "inlier_frac": axis["inlier_frac"] if axis is not None else None,
                     "split_depth": None},
        "floor_coeffs": floor_coeffs,
        "gauge": g_med,
        "rail_records": rail_records,
    }


def to_track_coords(x, y, z, frame):
    """Переход в координаты пути: (d — глубина, u — вбок от оси пути, v — высота
    над головкой рельса). Множитель cos(курс) делает u расстоянием ПО НОРМАЛИ к
    пути: на повороте срез постоянной глубины режет тоннель наискось, и без этой
    поправки ширина тоннеля кажется больше, чем она есть."""
    d = -y
    xc = eval_fit(frame["axis_fit"], d)
    heading = np.arctan(slope_from_fit(frame["axis_fit"], d))
    u = (x - xc) * np.cos(heading)
    v = z - np.polyval(frame["floor_coeffs"], d)
    return d, u, v


def _side_envelope(d, u, side_sign, depth_bins, u_min=U_MIN, u_max=U_MAX,
                   min_points=12, pct=97.0):
    """Шаг 2: по каждому срезу — боковая граница тоннеля на этой стороне.

    Оценка — ВЫСОКИЙ ПЕРЦЕНТИЛЬ |u| (не максимум и не пик плотности). Максимум
    ловит единичный выброс или залёт луча в нишу; пик плотности ловит самую
    "отражающую" поверхность, которая часто оказывается не стеной, а лотком или
    торцом настила перед ней. Перцентиль же отвечает на нужный вопрос — "докуда
    доходит основная масса точек" — и по построению не боится нескольких
    промахов.

    Возвращает (depths, offsets) — по одному значению на срез.
    """
    su = side_sign * u
    band = (su >= u_min) & (su <= u_max)
    depths, offsets = [], []
    for lo, hi in depth_bins:
        sl = band & (d >= lo) & (d < hi)
        if sl.sum() < min_points:
            continue
        depths.append((lo + hi) / 2)
        offsets.append(float(np.percentile(su[sl], pct)))
    return np.array(depths), np.array(offsets)


def _solve_shape(data, curvature):
    """МНК модели параллельных стен по срезам ОБЕИХ сторон сразу.

    Модель: смещение оси тоннеля от прямой опоры s(t) = α·t² + β·t, где
    t = d/DEPTH_SCALE, а боковые границы отстоят от неё на постоянные полуширины:
        слева   o_L(d) = W_L − s(d)
        справа  o_R(d) = W_R + s(d)
    (o — положительное расстояние вбок, поэтому знак s у сторон разный.)

    Все четыре параметра входят линейно, поэтому это обычный МНК, а не перебор.
    Существенно, что α и β ОБЩИЕ для сторон: именно это запрещает стенам
    разъезжаться "домиком" и позволяет набрать кривизну по точкам с обеих
    сторон, а не по одной.

    data: {"left": (depths, offsets, mask), "right": (...)} — сторона без
        достаточного числа срезов просто не даёт строк, и её полуширина из
        модели исключается.
    Возвращает (params, rms) либо None.
    """
    present = [name for name in ("left", "right") if data[name][2].sum() >= 3]
    if not present:
        return None
    n_shape = 2 if curvature else 1          # [α, β] либо только [β]
    n_par = n_shape + len(present)

    rows, rhs = [], []
    for name, sgn in (("left", -1.0), ("right", +1.0)):
        if name not in present:
            continue
        d, o, m = data[name]
        for di, oi in zip(d[m], o[m]):
            t = di / DEPTH_SCALE
            row = np.zeros(n_par)
            row[:n_shape] = ([sgn * t * t, sgn * t] if curvature else [sgn * t])
            row[n_shape + present.index(name)] = 1.0
            rows.append(row)
            rhs.append(oi)
    if len(rows) < n_par + 1:
        return None
    A, b = np.array(rows), np.array(rhs)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    rms = float(np.sqrt(np.mean((A @ sol - b) ** 2)))

    alpha = float(sol[0]) if curvature else 0.0
    beta = float(sol[1]) if curvature else float(sol[0])
    widths = {name: float(sol[n_shape + i]) for i, name in enumerate(present)}
    return {"alpha": alpha, "beta": beta, "widths": widths, "rms": rms}


def _offset_coeffs(shape, side):
    """Полином положительного смещения стены o(d) в СЫРОЙ глубине (м), чтобы
    дальше везде работал обычный np.polyval."""
    sgn = -1.0 if side == "left" else +1.0
    w = shape["widths"].get(side)
    if w is None:
        return None
    return np.array([
        sgn * shape["alpha"] / DEPTH_SCALE ** 2,
        sgn * shape["beta"] / DEPTH_SCALE,
        w,
    ])


def _fit_parallel_walls(data, window=0.55):
    """Шаг 3: согласованная по всему кадру форма тоннеля.

    Опора — ближняя зона: там точек на порядок больше и оценка среза устойчива,
    поэтому медиана ближних срезов задаёт стартовые полуширины. Дальше идёт
    попеременное уточнение "модель -> какие срезы ей соответствуют -> модель":
    к текущим линиям притягиваются срезы в окне ±window, по ним подгоняется новая
    форма, и так несколько раз. Срез, улетевший в нишу или в открытый соседний
    путь, отстоит от модели скачком и в окно не попадает.

    Прямая или дуга — решается по данным, а не назначается заранее. Лишний
    свободный параметр всегда хоть немного уменьшает невязку, поэтому само по
    себе "с дугой лучше" ничего не значит — на замусоренном кадре модель так
    находит поворот там, где его нет. Дуга принимается при трёх условиях сразу:
    невязка падает заметно (CURVE_GAIN), изгиб превышает уровень шума подгонки
    в CURVE_SNR раз (α — это и есть увод стены вбок в метрах на глубине
    DEPTH_SCALE, его прямо и сравниваем с невязкой), и радиус физически возможен
    (MIN_RADIUS).
    """
    seeded = {}
    for name in ("left", "right"):
        d, o = data[name]
        if len(d) < 4:
            seeded[name] = (d, o, np.zeros(len(d), dtype=bool))
            continue
        near = d <= NEAR_DEPTH
        seed = float(np.median(o[near])) if near.sum() >= 2 else float(np.median(o))
        seeded[name] = (d, o, np.abs(o - seed) < window)

    shape = None
    for _ in range(4):
        straight = _solve_shape(seeded, curvature=False)
        if straight is None:
            return None, seeded
        arc = _solve_shape(seeded, curvature=True)
        pick, kind = straight, "straight"
        if arc is not None and arc["rms"] < CURVE_GAIN * straight["rms"] \
                and abs(arc["alpha"]) > CURVE_SNR * arc["rms"]:
            radius = DEPTH_SCALE ** 2 / (2 * abs(arc["alpha"])) if abs(arc["alpha"]) > 1e-9 else np.inf
            if radius >= MIN_RADIUS:
                pick, kind = arc, "arc"
        pick["kind"] = kind
        pick["radius"] = (DEPTH_SCALE ** 2 / (2 * abs(pick["alpha"]))
                          if kind == "arc" else None)

        changed = False
        for name in ("left", "right"):
            d, o, m = seeded[name]
            c = _offset_coeffs(pick, name)
            if c is None:
                continue
            new_m = np.abs(o - np.polyval(c, d)) < window
            if not np.array_equal(new_m, m):
                changed = True
            seeded[name] = (d, o, new_m)
        shape = pick
        if not changed:
            break
    return shape, seeded


def wall_metrics(d, u, side_sign, predict, depth_bins, margin=0.3, min_points=12):
    """Честные метрики качества стены — не "гладкая ли подогнанная кривая"
    (гладкой она будет и по неверным точкам), а насколько она согласована с
    самими точками:

    coverage — доля срезов, где стена действительно НАБЛЮДАЕТСЯ: рядом с
        предсказанным положением (±0.2 м) реально есть точки. Ловит случай,
        когда кривая проведена через пустоту.

    leak — насколько стена "протекает": доля точек ЗА ней дальше чем на margin,
        посчитанная ВНУТРИ каждого среза и усреднённая по срезам. Настоящая
        стена непрозрачна, за ней точек почти нет, поэтому заметная утечка
        означает, что настоящая граница дальше найденной — ровно та ошибка, что
        видна на старых картинках, где линия проведена в метре от центра.

        Усреднение именно по срезам, а не по всем точкам сразу: плотность точек
        падает как 1/r², и ближние 5 метров дают их на порядок-два больше, чем
        вся остальная глубина. Общая доля по точкам — это метрика ближнего
        плана, и одна локальная ниша у самого поезда утаскивает её к 0.6 при
        совершенно правильной стене на всей остальной глубине.

    predict: функция глубины -> ожидаемое боковое смещение стены (положительное).
        Через неё той же меркой меряется любой метод, а не только этот.
    """
    su = side_sign * u
    n_seen, n_total = 0, 0
    leaks = []
    for lo, hi in depth_bins:
        sl = (d >= lo) & (d < hi) & (su >= U_MIN) & (su <= U_MAX)
        if sl.sum() < min_points:
            continue
        w = float(np.asarray(predict(np.array([(lo + hi) / 2]))).ravel()[0])
        if not np.isfinite(w):
            continue
        n_total += 1
        vals = su[sl]
        if np.sum(np.abs(vals - w) < 0.2) >= 5:
            n_seen += 1
        leaks.append(float(np.mean(vals > w + margin)))
    return {
        "coverage": (n_seen / n_total) if n_total else 0.0,
        "leak": float(np.mean(leaks)) if leaks else 1.0,
        "n_slices": n_total,
    }


def fit_tunnel_geometry(points, depth_bins=WALL_DEPTH_BINS, v_lo=V_LO, v_hi=V_HI):
    """Полный проход: рельсы -> координаты пути -> общая форма тоннеля.

    Возвращает dict:
        frame  — прямая опора (ось по рельсам, профиль пола, колея)
        shape  — общая для обеих стен форма: kind ('straight'/'arc'), radius, rms
        left/right — по стороне: offset_coeffs (полином смещения o(d)), offset,
                     метрики coverage/leak, сырые оценки по срезам
    Положение стены в координатах сенсора считает wall_x().
    """
    frame = fit_track_frame(points)
    if frame is None:
        return None

    x = points['x'].astype(float)
    y = points['y'].astype(float)
    z = points['z'].astype(float)

    d, u, v = to_track_coords(x, y, z, frame)
    band = (v >= v_lo) & (v <= v_hi) & (d > 2) & (d < 45)
    db, ub = d[band], u[band]

    data = {name: _side_envelope(db, ub, sign, depth_bins)
            for name, sign in (("left", -1.0), ("right", +1.0))}
    shape, seeded = _fit_parallel_walls(data)
    if shape is None:
        return None

    sides = {}
    for name, sign in (("left", -1.0), ("right", +1.0)):
        coeffs = _offset_coeffs(shape, name)
        if coeffs is None:
            sides[name] = None
            continue
        m = wall_metrics(db, ub, sign, lambda dd, _c=coeffs: np.polyval(_c, dd), depth_bins)
        dd, oo, inl = seeded[name]
        sides[name] = {
            "offset_coeffs": coeffs, "sign": sign,
            "offset": float(np.median(np.polyval(coeffs, np.linspace(4, 40, 20)))),
            "depths": dd, "offsets": oo, "inliers": inl,
            **m,
        }
    return {"frame": frame, "shape": shape, "left": sides["left"], "right": sides["right"]}


def wall_x(result, side, depths):
    """Положение стены в координатах СЕНСОРА (то, что рисуется на виде сверху):
    прямая опора по рельсам плюс боковое смещение, пересчитанное через курс."""
    s = result[side]
    depths = np.asarray(depths, dtype=float)
    if s is None:
        return np.full(depths.shape, np.nan)
    frame = result["frame"]
    xc = eval_fit(frame["axis_fit"], depths)
    heading = np.arctan(slope_from_fit(frame["axis_fit"], depths))
    return xc + s["sign"] * np.polyval(s["offset_coeffs"], depths) / np.cos(heading)


def tunnel_center_coeffs(result, depths=None):
    """Полином оси ТОННЕЛЯ в координатах сенсора: прямая опора по рельсам плюс
    общая для стен форма. Это и есть итоговая формула трассы в кадре."""
    if depths is None:
        depths = np.linspace(4, 42, 40)
    frame = result["frame"]
    shape = result["shape"]
    xc = eval_fit(frame["axis_fit"], depths)
    heading = np.arctan(slope_from_fit(frame["axis_fit"], depths))
    t = depths / DEPTH_SCALE
    s = shape["alpha"] * t ** 2 + shape["beta"] * t
    return np.polyfit(depths, xc + s / np.cos(heading), 2)


def geometry_formula(result, side):
    """Формула боковой границы: смещение от прямой опоры по рельсам, метры."""
    s = result[side]
    if s is None:
        return "нет данных"
    a, b, w = s["offset_coeffs"]
    if abs(a) < 1e-7 and abs(b) < 1e-4:
        return f"o = {w:.2f} м (постоянная полуширина)"
    return f"o = {w:.2f} {b:+.4f}·d {a:+.6f}·d²"


def axis_formula(result):
    """Формула трассы в кадре: прямая или дуга с радиусом."""
    shape = result["shape"]
    c = tunnel_center_coeffs(result)
    if shape["kind"] == "arc":
        side = "направо" if shape["alpha"] > 0 else "налево"
        return (f"x = {c[2]:.2f} {c[1]:+.3f}·d {c[0]:+.5f}·d² — "
                f"дуга R={shape['radius']:.0f} м ({side})")
    return f"x = {c[2]:.2f} {c[1]:+.3f}·d — прямая"
