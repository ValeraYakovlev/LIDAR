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

3. Рельсы — это тоже наблюдение той же самой кривой, только самой оси, а не её
   сдвига вбок. Поэтому рельсы и стены решаются ОДНОЙ системой МНК, а не по
   очереди. Они дополняют друг друга: рельсы точно держат ось вблизи (там их
   видно надёжно), стены дают длинную базу для кривизны. При радиусе 300 м
   стрелка прогиба на 17 м рельсов — 12 см, меньше разброса детектора, а на 40 м
   по стенам — уже 0.67 м; зато стены сами по себе ничего не говорят о том, где
   именно проходит путь между ними.

   Раньше рельсы работали отдельным предварительным шагом: по ним подгонялась
   прямая, и на повороте её наклон оказывался наклоном ХОРДЫ, а не касательной,
   а эта ошибка потом целиком уходила в форму стен. В общей системе такого
   перекоса не возникает: оба свидетельства объясняются одной кривой.

Опора при этом не обязана быть рельсовой. Уровень пола, по которому отбирается
полоса высот, восстанавливается и из самого облака (см. floor_profile), а грубый
курс — по стенам (_bootstrap_heading). Это важно не теоретически: в записях
встречаются участки в сотни кадров подряд, где рельсы не детектируются вовсе,
хотя стены видны прекрасно, и раньше такие кадры не обрабатывались вообще.

Полоса поиска по высоте выбрана по данным, а не наугад: настил платформы лежит
на v ≈ +1.2 м над головкой рельса и вбок уходит на 4-5 м, кабельные лотки и
своды — ещё выше, поэтому широкая полоса (прежние floor+0.3..+2.5) неизбежно
захватывает их вместо стены. В полосе v ≈ 0.2..1.1 м остаётся ровно то, что и
образует боковую границу: вертикальная грань стены тоннеля либо торец
платформы. Это же и есть полоса, важная для габарита: препятствие на пути
находится именно здесь.
"""

from itertools import combinations

import numpy as np

from .curvature import eval_fit, ransac_poly_fit, slope_from_fit
from .detector import DEFAULT_DEPTH_BINS, analyze_frame
from .rangeimage import surface_mask

# Более частые срезы, чем DEFAULT_DEPTH_BINS: стена — сплошная поверхность, по
# ней точек хватает и на узких срезах, а частая сетка даёт гладкую кривую и
# больше голосов для согласования.
WALL_DEPTH_BINS = (
    [(d, d + 2) for d in range(3, 21, 2)]
    + [(d, d + 3) for d in range(21, 33, 3)]
    + [(33, 37), (37, 42)]
)

# Срезы ТОЛЬКО для замера дальности наблюдения. В подгонку они не идут
# намеренно: замер показал, что дальние разрежённые срезы в подгонке ничего не
# добавляют в среднем и заметно вредят на станциях, где за 40 м начинается
# открытое пространство. Но ответить "докуда геометрию вообще видно" без них
# нельзя, поэтому они считаются отдельно и только как диагностика.
REACH_DEPTH_BINS = [(42, 48), (48, 55), (55, 63), (63, 72), (72, 85), (85, 100)]

# Те же срезы, продлённые до 62 м. §17 такое продление отверг: дальние срезы были
# слишком разрежены и на станциях описывали уже не стену, а открытое пространство.
# С накоплением кадров (§20) причина отпала — плотность на 50–60 м выросла вчетверо,
# — поэтому вариант заведён заново и меряется отдельно.
WALL_DEPTH_BINS_DEEP = WALL_DEPTH_BINS + [(42, 48), (48, 55), (55, 62)]

V_LO, V_HI = 0.2, 1.1      # полоса высот над головкой рельса (см. docstring)
U_MIN, U_MAX = 1.0, 6.5    # коридор поиска стены вбок от оси пути, м
NEAR_DEPTH = 20.0          # "ближняя зона": там рельсы видно надёжно
DEPTH_SCALE = 40.0         # нормировка глубины в модели формы (обусловленность)
MIN_RADIUS = 120.0         # м: круче метро не поворачивает
CURVE_GAIN = 0.85          # дуга обязана сбить невязку хотя бы во столько раз
CURVE_SNR = 4.0            # и увести стену вбок хотя бы во столько раз сильнее шума
RAIL_WEIGHT = 1.0          # вес рельсового наблюдения относительно стенного
RAIL_WINDOW = 0.30         # м: дальше этого от оси рельсовый срез — промах детектора
RAIL_FIT_DEPTH = 15.0      # м: глубже рельсы в подгонку не берутся (см. fit_tunnel_geometry)
MAX_WIDTH_BREAKS = 2       # сколько разрывов ширины разрешено на сторону
MIN_WIDTH_STEP = 0.25      # м: меньший скачок ширины не считается разрывом
WIDTH_GAIN = 0.80          # разрыв обязан сбить среднюю |невязку| во столько раз
NEAR_GATE_U = 3.0          # м: ширина ближних ворот, если они включены (по умолчанию выключены)
NEAR_GATE_MAX_DEPTH = 25.0 # м: глубже ворота не ставятся, там ось уже не так точна
PREV_WALL_MARGIN = 0.8     # м: насколько наружу от стен прошлого кадра ещё смотрим


def rail_samples(points, depth_bins=DEFAULT_DEPTH_BINS):
    """Наблюдения оси пути по рельсам: (глубины, положения центра колеи, записи).

    Срезы с колеёй, отличающейся от медианы больше чем на 25 см, отбрасываются:
    колея физически постоянна, так что это промах детектора (зацепил не тот
    пик), а не реальная геометрия. Возвращает пустые массивы, если рельсов нет —
    это не ошибка, дальше геометрия строится и без них.
    """
    rail_records, _ = analyze_frame(points, depth_bins)
    return rail_axis_samples(rail_records)


def rail_axis_samples(rail_records):
    """Та же выжимка, но из УЖЕ найденных срезов — без повторного прогона
    детектора. Нужна, когда опора кадра посчитана раньше и переиспользуется:
    детектор рельсов стоит 35 мс из 61 мс всей подгонки, и второй раз его
    гонять незачем."""
    empty = (np.zeros(0), np.zeros(0), rail_records, None)
    if not rail_records:
        return empty
    d = np.array([(r["depth_lo"] + r["depth_hi"]) / 2 for r in rail_records])
    xc = np.array([r["rail_center"] for r in rail_records])
    gauge = np.array([r["rail_gauge"] for r in rail_records])
    g_med = float(np.median(gauge))
    keep = np.abs(gauge - g_med) < 0.25
    if keep.sum() >= 3:
        d, xc = d[keep], xc[keep]
    elif len(d) < 3:
        return empty
    return d, xc, rail_records, g_med


def floor_profile(points, rail_records, depth_bins=WALL_DEPTH_BINS, x_near=2.0,
                  min_points=60, groove_lo=0.10, groove_hi=0.40):
    """Уровень пола вдоль глубины — опора для полосы высот.

    Когда рельсы найдены, берётся их shoulder_z: это прямое измерение пола
    рядом с желобом. Когда не найдены — а это происходит целыми кусками записи
    (см. knowledge.md §15), и раньше кадр из-за этого не обрабатывался вовсе —
    уровень восстанавливается из самого облака.

    Форму профиля даёт НИЗКИЙ ПЕРЦЕНТИЛЬ z у оси: по замерам он повторяет
    shoulder_z со стандартным отклонением 2-8 см, устойчивее любой другой
    простой оценки. Но смещён вниз ровно на глубину желоба, а она разная в
    разных тоннелях (18-32 см), поэтому постоянной поправкой не обойтись.
    Глубина желоба калибруется в самом кадре: пол — это широкое плато, желоб
    узкий, так что расстояние от перцентиля до самого населённого уровня среди
    нижних точек и есть искомая поправка. Медиана по срезам гасит её шум, а
    ограничение [groove_lo, groove_hi] не даёт поправке уйти в бессмыслицу на
    срезе, где плато нашлось не то.
    """
    if rail_records:
        d = np.array([(r["depth_lo"] + r["depth_hi"]) / 2 for r in rail_records])
        zf = np.array([r["shoulder_z"] for r in rail_records])
        if len(d) >= 3:
            fit = ransac_poly_fit(d, zf, 1, 0.06)
            if fit is not None:
                return fit["coeffs"]
        return np.array([0.0, float(np.median(zf))])

    x = points['x'].astype(float)
    z = points['z'].astype(float)
    depth = -points['y'].astype(float)
    near = np.abs(x) < x_near
    depths, lows, grooves = [], [], []
    for lo, hi in depth_bins:
        sl = near & (depth >= lo) & (depth < hi)
        if sl.sum() < min_points:
            continue
        zs = z[sl]
        p_low = float(np.percentile(zs, 2))
        depths.append((lo + hi) / 2)
        lows.append(p_low)
        plateau = _plateau_level(zs)
        if plateau is not None:
            grooves.append(plateau - p_low)
    if len(depths) < 3:
        return None
    depths, lows = np.array(depths), np.array(lows)
    groove = float(np.clip(np.median(grooves), groove_lo, groove_hi)) if grooves else 0.25
    fit = ransac_poly_fit(depths, lows + groove, 1, 0.08)
    if fit is not None:
        return fit["coeffs"]
    return np.array([0.0, float(np.median(lows + groove))])


def _plateau_level(zs, zbin=0.03):
    """Самый населённый уровень среди нижних точек среза — это пол: он занимает
    широкую полосу, тогда как желоб узкий и точек в нём мало."""
    lo, hi = np.percentile(zs, [1, 50])
    if hi - lo < zbin:
        return None
    counts, edges = np.histogram(zs, bins=np.arange(lo, hi + zbin, zbin))
    if counts.sum() == 0:
        return None
    centers = (edges[:-1] + edges[1:]) / 2
    peak = centers[int(np.argmax(counts))]
    near = zs[np.abs(zs - peak) < 2 * zbin]
    return float(np.median(near)) if len(near) else float(peak)


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
                   min_points=12, pct=97.0, gate_depth=0.0, gate_u=NEAR_GATE_U):
    """Шаг 2: по каждому срезу — боковая граница тоннеля на этой стороне.

    Оценка — ВЫСОКИЙ ПЕРЦЕНТИЛЬ |u| (не максимум и не пик плотности). Максимум
    ловит единичный выброс или залёт луча в нишу; пик плотности ловит самую
    "отражающую" поверхность, которая часто оказывается не стеной, а лотком или
    торцом настила перед ней. Перцентиль же отвечает на нужный вопрос — "докуда
    доходит основная масса точек" — и по построению не боится нескольких
    промахов.

    gate_depth: до этой глубины кандидаты ограничиваются gate_u метрами вбок.
        Смысл: там, где желоб распознан, положение пути известно точно, а тоннель
        вблизи поезда физически узкий — всё, что дальше трёх метров, это не
        стена, а настил платформы, ниша или соседний путь, то есть помеха.

        Ворота МЯГКИЕ: если внутри них опоры нет вовсе, берётся полный коридор.
        Без этого на двухпутном участке, где тоннель действительно раскрыт на
        6 м с самого начала, на глубине ворот возникал бы выдуманный скачок
        ширины — ровно тот артефакт, ради борьбы с которым ворота и ставятся.

    Возвращает (depths, offsets) — по одному значению на срез.
    """
    su = side_sign * u
    band = (su >= u_min) & (su <= u_max)
    depths, offsets = [], []
    for lo, hi in depth_bins:
        sl = band & (d >= lo) & (d < hi)
        if (lo + hi) / 2 <= gate_depth:
            gated = sl & (su <= gate_u)
            if gated.sum() >= min_points:
                sl = gated
        if sl.sum() < min_points:
            continue
        depths.append((lo + hi) / 2)
        offsets.append(float(np.percentile(su[sl], pct)))
    return np.array(depths), np.array(offsets)


def _solve_shape(data, rails, curvature, segs=None, rail_weight=RAIL_WEIGHT):
    """МНК одной общей формы по срезам ОБЕИХ стен И по рельсам сразу.

    Модель: ось тоннеля отклоняется от прямой опоры на s(t) = α·t² + β·t + γ,
    где t = d/DEPTH_SCALE, а боковые границы отстоят от неё на постоянные
    полуширины:
        рельсы  u_рельс(d) = s(d)          — рельсы и есть ось, наблюдаемая прямо
        слева   o_Л(d) = W_Л − s(d)
        справа  o_П(d) = W_П + s(d)
    (o — положительное расстояние вбок, поэтому знак s у сторон разный.)

    Все параметры входят линейно, поэтому это обычный МНК, а не перебор.
    Существенно, что α, β, γ ОБЩИЕ: это запрещает стенам разъезжаться
    "домиком" и позволяет набрать форму по всем свидетельствам сразу.

    Зачем в одну систему с рельсами. Рельсы и стены дополняют друг друга, а не
    дублируют: рельсы точно держат ось вблизи (там их видно надёжно), стены
    дают длинную базу для кривизны (40 м против 20). Раньше рельсы работали
    отдельным предварительным шагом — по ним подгонялась прямая, и её наклон
    неизбежно оказывался наклоном ХОРДЫ, а не касательной, если путь повернут;
    эта ошибка потом целиком уходила в форму стен. Теперь оба свидетельства
    объясняются одной кривой, и такого перекоса не возникает.

    Свободный член γ определим только когда есть рельсы: без них сдвиг оси
    неотличим от согласованного изменения полуширин, поэтому γ исключается.

    Полуширина может быть КУСОЧНО-постоянной: segs задаёт, к какому участку
    относится каждый срез, и на каждый участок заводится своя колонка. Форма
    (α, β, γ) при этом остаётся общей, то есть стены по-прежнему не могут
    разъехаться "домиком" — меняться разрешено только расстоянию до них.
    """
    present = [name for name in ("left", "right") if data[name][2].sum() >= 3]
    if not present:
        return None
    if segs is None:
        segs = {name: np.zeros(len(data[name][0]), dtype=int) for name in ("left", "right")}
    rd, ru, rm = rails
    use_rails = rm.sum() >= 3
    n_shape = (2 if curvature else 1) + (1 if use_rails else 0)   # [α],[β],[γ]

    n_seg, col0 = {}, {}
    col = n_shape
    for name in present:
        n_seg[name] = int(segs[name].max()) + 1 if len(segs[name]) else 1
        col0[name] = col
        col += n_seg[name]
    n_par = col

    def shape_row(t, sgn):
        row = np.zeros(n_par)
        k = 0
        if curvature:
            row[k] = sgn * t * t
            k += 1
        row[k] = sgn * t
        k += 1
        if use_rails:
            row[k] = sgn
        return row

    rows, rhs, weights = [], [], []
    for name, sgn in (("left", -1.0), ("right", +1.0)):
        if name not in present:
            continue
        d, o, m = data[name]
        for di, oi, si in zip(d[m], o[m], segs[name][m]):
            row = shape_row(di / DEPTH_SCALE, sgn)
            row[col0[name] + int(si)] = 1.0
            rows.append(row)
            rhs.append(oi)
            weights.append(1.0)
    if use_rails:
        for di, ui in zip(rd[rm], ru[rm]):
            rows.append(shape_row(di / DEPTH_SCALE, +1.0))
            rhs.append(ui)
            weights.append(rail_weight)
    if len(rows) < n_par + 1:
        return None

    A, b, w = np.array(rows), np.array(rhs), np.sqrt(np.array(weights))
    sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    resid = A @ sol - b
    rms = float(np.sqrt(np.mean(resid ** 2)))

    k = 0
    alpha = float(sol[k]) if curvature else 0.0
    if curvature:
        k += 1
    beta = float(sol[k])
    k += 1
    gamma = float(sol[k]) if use_rails else 0.0
    widths = {name: [float(v) for v in sol[col0[name]:col0[name] + n_seg[name]]]
              for name in present}
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "widths": widths,
            "rms": rms, "used_rails": bool(use_rails)}


def _axis_offset_coeffs(shape):
    """Полином отклонения оси тоннеля от прямой опоры, в СЫРОЙ глубине (м)."""
    return np.array([shape["alpha"] / DEPTH_SCALE ** 2,
                     shape["beta"] / DEPTH_SCALE,
                     shape["gamma"]])


def side_offset(shape, side, depths):
    """Положительное смещение границы o(d) от прямой опоры — в этих же
    координатах считаются метрики. None, если стороны в модели нет."""
    ws = shape["widths"].get(side)
    if ws is None:
        return None
    depths = np.asarray(depths, dtype=float)
    edges = np.asarray(shape.get("edges", {}).get(side, np.zeros(0)), dtype=float)
    idx = np.clip(np.searchsorted(edges, depths), 0, len(ws) - 1)
    sgn = -1.0 if side == "left" else +1.0
    return np.asarray(ws)[idx] + sgn * np.polyval(_axis_offset_coeffs(shape), depths)


def _segment_width(depths, resid, max_breaks=None, min_seg=3,
                   min_step=MIN_WIDTH_STEP, gain=WIDTH_GAIN):
    """Кусочно-постоянная полуширина вдоль кадра.

    Тоннель не обязан быть одной ширины на весь кадр: у станции граница — торец
    платформы, а за её концом — стена позади, и между ними разрыв в полметра;
    перед гермозатвором тоннель раскрывается. Одним числом это не описывается,
    и именно такие кадры и не проходили порог: кривая держалась одного уровня,
    а половина срезов оказывалась совсем на другом.

    Разрыв — это лишние свободные параметры, поэтому разрешается он только
    когда оправдан сразу по трём условиям: участок не короче min_seg срезов,
    скачок не меньше min_step, и разрыв снижает среднюю |невязку| хотя бы в
    1/gain раз. Иначе модель начинает описывать разрывами обычный шум срезов.

    Считается по МЕДИАНЕ и средней |невязке|, а не по МНК: у ступени соседний
    участок — это выброс огромной величины, и квадратичная мера размазала бы
    границу между участками.
    """
    if max_breaks is None:
        max_breaks = MAX_WIDTH_BREAKS
    d = np.asarray(depths, dtype=float)
    r = np.asarray(resid, dtype=float)
    order = np.argsort(d)
    d, r = d[order], r[order]
    n = len(d)

    def evaluate(cuts):
        bounds = [0] + list(cuts) + [n]
        vals, tot = [], 0.0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < min_seg:
                return None
            v = float(np.median(r[a:b]))
            vals.append(v)
            tot += float(np.abs(r[a:b] - v).sum())
        if any(abs(vals[i + 1] - vals[i]) < min_step for i in range(len(vals) - 1)):
            return None
        return tot / n, vals

    flat = evaluate(())
    if flat is None:
        return np.zeros(0), [float(np.median(r))] if n else [0.0]
    best_err, best_vals, best_cuts = flat[0], flat[1], ()
    for k in range(1, max_breaks + 1):
        cand = None
        for cuts in combinations(range(min_seg, n - min_seg + 1), k):
            if any(cuts[i + 1] - cuts[i] < min_seg for i in range(len(cuts) - 1)):
                continue
            got = evaluate(cuts)
            if got is not None and (cand is None or got[0] < cand[0]):
                cand = (got[0], got[1], cuts)
        if cand is not None and cand[0] < gain * best_err:
            best_err, best_vals, best_cuts = cand
    edges = np.array([(d[c - 1] + d[c]) / 2 for c in best_cuts])
    return edges, best_vals


def _fit_parallel_walls(data, rails, window=0.55, prior=None):
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
        seed = _prior_seed(prior, name, o, window)
        if seed is None:
            near = d <= NEAR_DEPTH
            seed = float(np.median(o[near])) if near.sum() >= 2 else float(np.median(o))
        seeded[name] = (d, o, np.abs(o - seed) < window)

    rd, ru = rails
    rail_state = (rd, ru, np.abs(ru - np.median(ru)) < RAIL_WINDOW if len(ru) else
                  np.zeros(0, dtype=bool))
    segs = {name: np.zeros(len(data[name][0]), dtype=int) for name in ("left", "right")}
    edges = {name: np.zeros(0) for name in ("left", "right")}

    shape = None
    # Второй заход нужен потому, что разрывы ширины и форма определяются друг
    # через друга: пока ширина считается одной на весь кадр, ступень у платформы
    # частично утекает в форму, а найдя ступень, форму надо пересчитать уже без
    # неё. Больше двух заходов не нужно — дальше ничего не меняется.
    for _ in range(3):
        shape, seeded, rail_state = _converge(seeded, rail_state, segs, window)
        if shape is None:
            return None, seeded, rail_state
        # Именно здесь, а не после цикла: edges обязаны отвечать тем участкам,
        # по которым только что посчитана модель, иначе ширины и границы
        # разъезжаются на один заход.
        shape["edges"] = edges
        new_segs, new_edges, new_seeded, changed = _resegment(seeded, shape, window)
        if not changed:
            break
        segs, edges, seeded = new_segs, new_edges, new_seeded
    return shape, seeded, rail_state


def _prior_seed(prior, side, offsets, window, min_support=3):
    """Полуширина с прошлого кадра как стартовое приближение — но только если
    точки ТЕКУЩЕГО кадра её подтверждают.

    Проверка обязательна: иначе на въезде в станцию, где тоннель реально
    меняется, прошлое значение утащило бы согласование в пустоту и держало бы
    его там кадр за кадром. Подсказка помогает выбрать правильное скопление,
    когда их несколько, и молча уступает, когда её скопления больше нет.
    """
    if prior is None:
        return None
    w = prior.get("widths", {}).get(side)
    if w is None or not np.isfinite(w):
        return None
    if int(np.sum(np.abs(offsets - w) < window)) < min_support:
        return None
    return float(w)


def _converge(seeded, rail_state, segs, window):
    """Попеременное уточнение "модель -> её инлайеры -> модель" при фиксированной
    разбивке на участки ширины."""
    shape = None
    for _ in range(4):
        straight = _solve_shape(seeded, rail_state, curvature=False, segs=segs)
        if straight is None:
            return None, seeded, rail_state
        arc = _solve_shape(seeded, rail_state, curvature=True, segs=segs)
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
            ws = pick["widths"].get(name)
            if ws is None:
                continue
            sgn = -1.0 if name == "left" else +1.0
            pred = (np.asarray(ws)[np.clip(segs[name], 0, len(ws) - 1)]
                    + sgn * np.polyval(_axis_offset_coeffs(pick), d))
            new_m = np.abs(o - pred) < window
            if not np.array_equal(new_m, m):
                changed = True
            seeded[name] = (d, o, new_m)
        # Рельсовые срезы отбраковываются наравне со стенными: детектор рельс
        # иногда цепляет не тот пик, и такой промах не должен тянуть ось.
        rd, ru, rm = rail_state
        if len(rd):
            new_rm = np.abs(ru - np.polyval(_axis_offset_coeffs(pick), rd)) < RAIL_WINDOW
            if not np.array_equal(new_rm, rm):
                changed = True
            rail_state = (rd, ru, new_rm)
        shape = pick
        if not changed:
            break
    return shape, seeded, rail_state


def _resegment(seeded, shape, window):
    """Ищет разрывы ширины по остаткам "срез минус форма" на каждой стороне.

    Смотрит на ВСЕ срезы, а не только на инлайеры текущей модели. Иначе разрыв
    принципиально не найти: срезы по ту сторону ступени модель с одной шириной
    как раз и объявила выбросами, и по одним инлайерам ступени просто не видно.
    Отбраковку это не отменяет — сама сегментация робастная (медиана и средняя
    |невязка|), а после неё инлайеры пересчитываются уже по участкам.
    """
    segs, edges, out, changed = {}, {}, {}, False
    for name in ("left", "right"):
        d, o, m = seeded[name]
        segs[name] = np.zeros(len(d), dtype=int)
        edges[name] = np.zeros(0)
        out[name] = (d, o, m)
        if len(d) < 8 or shape["widths"].get(name) is None:
            continue
        sgn = -1.0 if name == "left" else +1.0
        resid = o - sgn * np.polyval(_axis_offset_coeffs(shape), d)
        e, vals = _segment_width(d, resid)
        if len(e) == 0:
            continue
        idx = np.searchsorted(e, d)
        # Маски пересеиваются ПО УЧАСТКАМ. Старые остались от модели с одной
        # шириной и как раз отбрасывали всё по ту сторону ступени; если их не
        # обновить, у нового участка не окажется ни одного инлайера, его колонка
        # выродится в ноль и ширина там получится нулевой.
        new_m = np.abs(resid - np.asarray(vals)[idx]) < window
        if new_m.sum() < 3:
            continue
        edges[name], segs[name], out[name] = e, idx, (d, o, new_m)
        changed = True
    return segs, edges, out, changed


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


def _inside_prev_walls(x, y, prior, margin=PREV_WALL_MARGIN):
    """Маска "внутри стен прошлого кадра" в координатах сенсора.

    Границы берутся прямо в координатах сенсора, а не пути: между кадрами поезд
    смещается на единицы метров, так что прошлая кривая стены годится как есть,
    и не нужно строить систему координат текущего кадра до того, как она
    построена. None, если прошлого кадра нет.
    """
    bounds = (prior or {}).get("bounds")
    if not bounds or bounds.get("left") is None or bounds.get("right") is None:
        return None
    d = -y
    lo = np.polyval(bounds["left"], d) - margin
    hi = np.polyval(bounds["right"], d) + margin
    # Дальше, чем прошлый кадр мог видеть, отсечение не применяется
    return (d > bounds["max_depth"]) | ((x >= lo) & (x <= hi))


def _straight_frame(coeffs, floor_coeffs, gauge, rail_records):
    return {
        "axis_fit": {"kind": "straight", "coeffs": coeffs, "radius": None,
                     "residual": None, "inlier_frac": None, "split_depth": None},
        "floor_coeffs": floor_coeffs, "gauge": gauge, "rail_records": rail_records,
    }


def _bootstrap_heading(x, y, z, floor_coeffs, depth_bins, v_lo, v_hi,
                       u_min=1.0, u_max=7.0):
    """Грубый курс по стенам — нужен, когда рельсов нет вовсе.

    Без хоть какой-то оценки курса не обойтись: коридор поиска стены задаётся
    вбок от опоры, а при сильном рыскании (в записях встречается до 4°) стена
    на дальнем конце кадра уезжает за его границы и в коридор не попадает.
    Рельсы такую оценку дают даром, но когда их нет, её можно взять с самих
    стен: коридор здесь нарочно шире рабочего, а модель — только прямая, так
    что от неё требуется лишь грубое направление.
    """
    d = -y
    v = z - np.polyval(floor_coeffs, d)
    band = (v >= v_lo) & (v <= v_hi) & (d > 2) & (d < 45)
    db, xb = d[band], x[band]
    seeded = {}
    for name, sign in (("left", -1.0), ("right", +1.0)):
        dd, oo = _side_envelope(db, xb, sign, depth_bins, u_min=u_min, u_max=u_max)
        seeded[name] = (dd, oo, np.ones(len(dd), dtype=bool))
    no_rails = (np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool))
    sol = _solve_shape(seeded, no_rails, curvature=False)
    if sol is None:
        return np.array([0.0, 0.0])
    return np.array([sol["beta"] / DEPTH_SCALE, 0.0])


RAIL_ROLES = frozenset({"floor", "axis", "rows"})


def fit_tunnel_geometry(points, depth_bins=WALL_DEPTH_BINS, v_lo=V_LO, v_hi=V_HI,
                        use_rails=True, rail_roles=RAIL_ROLES, prior=None,
                        near_gate=None, surface_tol=None, frame=None):
    """Полный проход: опора -> координаты пути -> общая форма тоннеля.

    use_rails=False принудительно отключает рельсы целиком — это режим
    стресс-проверки: на кадрах, где рельсы ЕСТЬ, он позволяет сверить безрельсовый
    путь с рельсовым как с эталоном. Иначе безрельсовую ветку негде проверить:
    в разработочных прогонах рельсы находятся почти всегда, а прогон, где они
    пропадают целыми кусками, отложен как валидационный.

    frame=... подставляет готовую опору вместо поиска рельсов и пола. Нужно при
    накоплении кадров: облако там слито из нескольких, а опора обязана остаться
    от текущего, и заодно детектор рельсов не гоняется дважды.

    Возвращает dict:
        frame  — прямая опора (положение и курс, профиль пола, колея)
        shape  — общая форма: kind ('straight'/'arc'), radius, rms, used_rails
        rails  — (глубины, смещения, маска инлайеров) наблюдений по рельсам
        left/right — по стороне: widths, edges, offset, coverage/leak, сырые срезы
    Положение стены в координатах сенсора считает wall_x().
    """
    x = points['x'].astype(float)
    y = points['y'].astype(float)
    z = points['z'].astype(float)

    roles = frozenset(rail_roles) if use_rails else frozenset()
    if frame is not None:
        # Опора передана готовой — это накопление кадров: облако там слито из
        # нескольких, но рельсы и пол берутся от ТЕКУЩЕГО кадра, где они уже
        # найдены. Повторный прогон детектора дал бы то же самое за те же 35 мс.
        rd, rxc, rail_records, gauge = rail_axis_samples(frame["rail_records"])
        floor_coeffs = frame["floor_coeffs"]
    else:
        # Отсечение по стенам ПРОШЛОГО кадра. Между кадрами поезд проезжает
        # единицы метров, так что стены, найденные на прошлом кадре, — надёжная
        # оценка того, где кончается тоннель сейчас. Всё, что снаружи них,
        # тоннелю не принадлежит: это соседний путь, настил платформы или ниша.
        # Детектору рельсов это важнее всего — он ищет самый глубокий провал
        # профиля в пределах |x| < 4 м, и щель между крайним рельсом и стеной
        # принимал за путь.
        #
        # Запас PREV_WALL_MARGIN оставляет тоннелю право расшириться: за кадр он
        # может стать шире на этот запас, за несколько кадров — на сколько угодно.
        inside = _inside_prev_walls(x, y, prior)
        if inside is not None and inside.sum() < 0.2 * len(x):
            inside = None  # отсечение выбросило почти всё — значит опора устарела
        fit_points = points if inside is None else points[inside]

        if roles:
            rd, rxc, rail_records, gauge = rail_samples(fit_points)
        else:
            rd, rxc, rail_records, gauge = np.zeros(0), np.zeros(0), [], None

        floor_coeffs = floor_profile(fit_points, rail_records if "floor" in roles else [])
        if floor_coeffs is None:
            return None

        if "axis" in roles and len(rd) >= 3:
            near = rd <= NEAR_DEPTH
            dn, xn = (rd[near], rxc[near]) if near.sum() >= 3 else (rd, rxc)
            fit = ransac_poly_fit(dn, xn, 1, 0.15)
            base = fit["coeffs"] if fit is not None else np.array([0.0, float(np.median(xn))])
        else:
            base = _bootstrap_heading(x, y, z, floor_coeffs, depth_bins, v_lo, v_hi)
        frame = _straight_frame(base, floor_coeffs, gauge, rail_records)
    base = frame["axis_fit"]["coeffs"]
    d, u, v = to_track_coords(x, y, z, frame)
    # Полоса доходит до конца ЗОНДИРУЮЩИХ срезов, а не до 45 м, как было.
    # Прежний предел ровно совпадал с серединой первого зондирующего среза
    # (42-48 м), поэтому замер «докуда видно» упирался в 45 м механически, на
    # всех прогонах и при любых данных — и §17 принял этот упор за свойство
    # тоннеля. На подгонку это не влияет: срезы подгонки кончаются на 42 м
    # независимо от полосы.
    d_probe = max(REACH_DEPTH_BINS[-1][1], depth_bins[-1][1])
    band = (v >= v_lo) & (v <= v_hi) & (d > 2) & (d < d_probe)
    # Метрики считаются по ПОЛНОМУ облаку, а подгонка — по отсечённому. Иначе
    # отсечение само себя и аттестует: выброшенные точки перестают попадать в
    # "утечку за стену", и leak обнуляется независимо от того, верна стена или нет.
    # Стены ищутся по ПОЛНОМУ облаку, отсечение на них не распространяется.
    # Замер показал, почему: отсечение работает храповиком. Стоит одному кадру
    # занизить стену — и следующий уже физически не может увидеть дальше неё,
    # потому что точки оттуда выброшены. На двухпутном прогоне, где тоннель
    # раскрыт на 6.5 м, это роняло метрику с 94% до 84%. Детектору рельсов
    # отсечение при этом необходимо, и там оно и остаётся: рельсы ищутся вблизи,
    # где прошлая граница заведомо надёжна.
    db_all, ub_all = d[band], u[band]
    db, ub = db_all, ub_all

    # Отбор по непрерывности (эксперимент 2): в оценку границы идут только точки,
    # лежащие на сплошной поверхности. Метрики при этом по-прежнему считаются по
    # ПОЛНОМУ облаку (db_all/ub_all) — иначе фильтр аттестует сам себя, ровно как
    # это было с отсечением по стенам прошлого кадра (§18).
    if surface_tol:
        # Отрицательный порог — мягкий вариант: точки без соседа (край тени, край
        # сектора обзора — их больше половины) остаются, и фильтр судит только о
        # том, что реально измерено.
        surf = surface_mask(points, step_tol=abs(float(surface_tol)),
                            keep_isolated=surface_tol < 0)
        if surf is not None:
            sb = surf[band]
            if sb.sum() >= 0.1 * max(band.sum(), 1):
                db, ub = db_all[sb], ub_all[sb]

    # Ворота ставятся ровно там, где распознан желоб: глубже путь известен уже
    # только экстраполяцией, и обрезать по нему кандидатов было бы самонадеянно.
    gate_depth = min(float(rd.max()), NEAR_GATE_MAX_DEPTH) if (near_gate and len(rd)) else 0.0
    data = {name: _side_envelope(db, ub, sign, depth_bins, gate_depth=gate_depth,
                                 gate_u=float(near_gate) if near_gate else NEAR_GATE_U)
            for name, sign in (("left", -1.0), ("right", +1.0))}
    ru_all = ((rxc - np.polyval(base, rd)) * np.cos(np.arctan(base[0]))
              if len(rd) else np.zeros(0))
    # В подгонку идут только БЛИЖНИЕ рельсы. Дальше 15 м детектор рельс уже
    # регулярно промахивается, и такие срезы тянут ось на себя; заодно дальние
    # детекции остаются нетронутыми и работают отложенной проверкой оси —
    # предсказание кривизны по стенам можно сверить с тем, где рельсы реально
    # оказались, не используя их при подгонке.
    if len(rd) and "rows" in roles:
        fit_mask = rd <= RAIL_FIT_DEPTH
        rd_fit, ru_fit = rd[fit_mask], ru_all[fit_mask]
    else:
        rd_fit, ru_fit = np.zeros(0), np.zeros(0)
    shape, seeded, rail_state = _fit_parallel_walls(data, (rd_fit, ru_fit), prior=prior)
    if shape is None:
        return None

    sides = {}
    for name, sign in (("left", -1.0), ("right", +1.0)):
        if shape["widths"].get(name) is None:
            sides[name] = None
            continue
        predict = (lambda dd, _n=name: side_offset(shape, _n, dd))
        # Метрика считается ВСЕГДА по одним и тем же срезам, а не по тем, на
        # которых шла подгонка: иначе вариант с продлённой глубиной менялся бы
        # сам и вместе со своей меркой, и сравнивать его было бы не с чем.
        m = wall_metrics(db_all, ub_all, sign, predict, WALL_DEPTH_BINS)
        reach = _probe_reach(db_all, ub_all, sign, predict, REACH_DEPTH_BINS)
        dd, oo, inl = seeded[name]
        sides[name] = {
            "sign": sign, "widths": shape["widths"][name], "reach": reach,
            "edges": shape["edges"].get(name, np.zeros(0)),
            "offset": float(np.median(predict(np.linspace(4, 40, 20)))),
            "depths": dd, "offsets": oo, "inliers": inl,
            **m,
        }
    return {"frame": frame, "shape": shape, "rails": rail_state,
            "rails_all": (rd, ru_all),
            "left": sides["left"], "right": sides["right"]}


def tracked_depth(result, side):
    """До какой глубины граница реально НАБЛЮДАЕТСЯ, а не продолжается моделью.

    Считается по срезам, которые в подгонке участвовали, ПЛЮС по зондирующим
    срезам глубже (REACH_DEPTH_BINS): там проверяется только одно — есть ли
    рядом с предсказанной границей точки. Зондирование не влияет на саму
    подгонку, поэтому ответ "докуда видно" не стоит ничего по качеству.

    Дальше этой глубины кривая — уже экстраполяция, и на графике её честно
    рисовать пунктиром.
    """
    s = result.get(side)
    if s is None:
        return None
    d, inl = np.asarray(s["depths"], dtype=float), np.asarray(s["inliers"], dtype=bool)
    near = float(d[inl].max()) if inl.any() else None
    return max(near, s["reach"]) if (near is not None and s.get("reach")) else (near or s.get("reach"))


def _probe_reach(d, u, side_sign, predict, depth_bins, tol=0.35, min_points=8):
    """Самый дальний зондирующий срез, где у предсказанной границы есть точки."""
    su = side_sign * u
    reach = None
    for lo, hi in depth_bins:
        sl = (d >= lo) & (d < hi) & (su >= U_MIN) & (su <= U_MAX)
        if sl.sum() < min_points:
            continue
        w = float(np.asarray(predict(np.array([(lo + hi) / 2]))).ravel()[0])
        if np.isfinite(w) and int(np.sum(np.abs(su[sl] - w) < tol)) >= min_points // 2:
            reach = (lo + hi) / 2
    return reach


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
    return xc + s["sign"] * side_offset(result["shape"], side, depths) / np.cos(heading)


def tunnel_center_coeffs(result, depths=None):
    """Полином оси ТОННЕЛЯ в координатах сенсора: прямая опора по рельсам плюс
    общая для стен форма. Это и есть итоговая формула трассы в кадре."""
    if depths is None:
        depths = np.linspace(4, 42, 40)
    frame = result["frame"]
    xc = eval_fit(frame["axis_fit"], depths)
    heading = np.arctan(slope_from_fit(frame["axis_fit"], depths))
    s = np.polyval(_axis_offset_coeffs(result["shape"]), depths)
    return np.polyfit(depths, xc + s / np.cos(heading), 2)


def geometry_formula(result, side):
    """Формула боковой границы: смещение от прямой опоры по рельсам, метры."""
    s = result[side]
    if s is None:
        return "нет данных"
    ws, edges = s["widths"], np.asarray(s["edges"])
    if len(edges) == 0:
        return f"полуширина {ws[0]:.2f} м (постоянная)"
    parts, prev = [], 0.0
    for w, e in zip(ws, list(edges) + [None]):
        parts.append(f"{w:.2f} м ({prev:.0f}-{e:.0f} м)" if e is not None
                     else f"{w:.2f} м (от {prev:.0f} м)")
        prev = e if e is not None else prev
    return "полуширина: " + ", ".join(parts)


def axis_formula(result):
    """Формула трассы в кадре: прямая или дуга с радиусом."""
    shape = result["shape"]
    c = tunnel_center_coeffs(result)
    if shape["kind"] == "arc":
        side = "направо" if shape["alpha"] > 0 else "налево"
        return (f"x = {c[2]:.2f} {c[1]:+.3f}·d {c[0]:+.5f}·d² — "
                f"дуга R={shape['radius']:.0f} м ({side})")
    return f"x = {c[2]:.2f} {c[1]:+.3f}·d — прямая"
