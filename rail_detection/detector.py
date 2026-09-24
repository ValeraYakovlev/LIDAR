"""Геометрическая детекция центрального желоба и рельс на срезе кадра лидара.

Метод (см. knowledge.md §13-14): берём тонкий срез по глубине, строим медианный
профиль высоты пола Z(X), находим V-образный желоб (дно + точки выхода на
"полувысоте"), затем ищем локальные пики профиля сразу за краями желоба — это
рельсы. Один срез обрабатывается независимо от других (без сравнения кадров).
"""

import numpy as np
from scipy.signal import find_peaks

# Колея — физическая константа сети: между вершинами головок рельсов получается
# ширина колеи плюс ширина головки, по замерам на всех записях 1.55-1.70 м.
EXPECTED_GAUGE = 1.60
GAUGE_TOL = 0.25
GAUGE_LIMITS = (1.35, 1.85)   # вне этого диапазона "колея" — заведомо промах
# Вес признака «головка рельса темнее пола» в выборе пары пиков. По умолчанию
# ВЫКЛЮЧЕН — замер показал, что он ничего не меняет: на разработочных прогонах
# метрика стен совпадает до кадра, а на участке с плитным основанием (§18, кадры
# 150-200 roundT_squareT_pressureGate_squareT) совпадает и число найденных срезов
# (367), и медиана колеи (1.650). Признак разделяет сам по себе сильно (§19), но
# выраженность пика и близость к колее уже решают те же случаи, и добавить ему
# нечего. Код оставлен: сам знак — измеренный факт (см. _rail_darkness), и
# включается он одним числом, если понадобится на других данных.
DARK_WEIGHT = 0.0

DEFAULT_DEPTH_BINS = [
    (3, 5), (5, 7), (7, 9), (9, 11), (11, 13), (13, 15),
    (15, 18), (18, 21), (21, 24), (24, 27), (27, 30), (30, 35), (35, 40),
]  # надёжная зона по данным ~до 30-40м, дальше плотность точек не позволяет


def _bin_medians(xb, bins, *vals, min_count=3):
    """Медианный профиль по бинам x: центры бинов, где точек не меньше min_count,
    и медиана каждого массива vals в этих бинах.

    Ровно то, что давал цикл `for i: m = idx == i; np.median(v[m])`, но одной
    сортировкой (экспер. 19: цикл — 160 бинов × 13 срезов на кадр). Медиана
    берётся так же, как в np.median: нечётное число — средний элемент, чётное —
    (a + b) / 2 в типе массива. При NaN — прежний цикл (np.median даёт NaN)."""
    idx = np.digitize(xb, bins)
    if any(np.isnan(v).any() for v in vals):
        cols = [[] for _ in vals]
        centers = []
        for i in range(1, len(bins)):
            m = idx == i
            if m.sum() >= min_count:
                centers.append((bins[i - 1] + bins[i]) / 2)
                for c, v in zip(cols, vals):
                    c.append(np.median(v[m]))
        return np.array(centers), [np.array(c) for c in cols]
    counts = np.bincount(idx, minlength=len(bins) + 1)
    start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    keep = np.flatnonzero(counts[1:len(bins)] >= min_count) + 1
    centers = (bins[keep - 1] + bins[keep]) / 2
    n = counts[keep]
    lo = start[keep] + (n - 1) // 2
    hi = start[keep] + n // 2
    out = []
    for v in vals:
        vs = v[np.lexsort((v, idx))]
        a, b = vs[lo], vs[hi]
        out.append(np.where(n % 2 == 1, a, (a + b) / v.dtype.type(2)).astype(v.dtype))
    return centers, out


def find_groove_and_rails(points, depth_lo, depth_hi, x_range=4.0, xbin=0.05, path_coeffs=None):
    """Ищет желоб и рельсы в срезе points на глубине [depth_lo, depth_hi).

    points: structured array с полями x, y, z (ось Y — вперёд, отрицательная).
    path_coeffs: коэффициенты полинома centerline(depth) (см. tracking.
        robust_centerline). Если заданы, точки пересчитываются в систему
        координат вдоль локального курса пути: вычитается ожидаемый центр
        centerline(depth) и вносится поправка cos(курс) — так срез становится
        перпендикулярным реальному изгибу тоннеля, а не фиксированной оси Y
        сенсора (на повороте это разные вещи, см. knowledge.md — обсуждение
        поворотов и виражного превышения).
    Возвращает dict с геометрией или None, если в этом срезе надёжно не нашлось.
    """
    x, y, z = points['x'], points['y'], points['z']
    depth = -y
    if path_coeffs is not None:
        heading = np.arctan(np.polyval(np.polyder(path_coeffs), depth))
        x_center = np.polyval(path_coeffs, depth)
        x = (x - x_center) * np.cos(heading)
    mask = (depth >= depth_lo) & (depth < depth_hi) & (np.abs(x) < x_range)
    if mask.sum() < 200:
        return None
    xr, zr = x[mask], z[mask]

    # адаптивная полоса "пол+желоб": нижние 45% по Z в срезе
    zlo, zhi = np.percentile(zr, [1, 45])
    band = (zr >= zlo) & (zr <= zhi)
    if band.sum() < 100:
        return None
    xb, zb = xr[band], zr[band]

    bins = np.arange(-x_range, x_range + xbin, xbin)
    prof_x, (prof_z,) = _bin_medians(xb, bins, zb)
    if len(prof_x) < 15:
        return None

    # --- желоб: глобальный минимум профиля + точки выхода на полувысоте ---
    search = np.abs(prof_x) < 2.5
    if search.sum() < 8:
        return None
    i_min = np.argmin(prof_z[search])
    x_min, z_min = prof_x[search][i_min], prof_z[search][i_min]

    left_flank = (prof_x > x_min - 0.7) & (prof_x < x_min - 0.2)
    right_flank = (prof_x > x_min + 0.2) & (prof_x < x_min + 0.7)
    if left_flank.sum() < 3 or right_flank.sum() < 3:
        return None
    shoulder_z = (np.median(prof_z[left_flank]) + np.median(prof_z[right_flank])) / 2
    notch_depth = shoulder_z - z_min
    if notch_depth < 0.06:  # <6см — ненадёжно, не считаем это желобом
        return None

    half_level = z_min + 0.5 * notch_depth
    left_idx = np.where((prof_x < x_min) & (prof_z >= half_level))[0]
    right_idx = np.where((prof_x > x_min) & (prof_z >= half_level))[0]
    if len(left_idx) == 0 or len(right_idx) == 0:
        return None
    x_left, x_right = prof_x[left_idx[-1]], prof_x[right_idx[0]]
    groove_width = x_right - x_left
    if not (0.15 < groove_width < 2.0):
        return None

    # --- рельсы: ближайшие локальные пики сразу за краями желоба ---
    peaks_idx, _ = find_peaks(prof_z, prominence=0.02)
    if len(peaks_idx) == 0:
        return None
    peak_x = prof_x[peaks_idx]
    right_candidates = peak_x[(peak_x > x_right) & (peak_x < x_right + 1.0)]
    left_candidates = peak_x[(peak_x < x_left) & (peak_x > x_left - 1.0)]
    if len(right_candidates) == 0 or len(left_candidates) == 0:
        return None
    x_rail_right, x_rail_left = right_candidates.min(), left_candidates.max()
    rail_center = (x_rail_left + x_rail_right) / 2
    rail_gauge = x_rail_right - x_rail_left
    if not (0.3 < rail_gauge < 2.2):
        return None

    return {
        "depth_lo": depth_lo, "depth_hi": depth_hi,
        "x_raw": xb, "z_raw": zb,
        "x_left": x_left, "x_right": x_right,
        "x_rail_left": x_rail_left, "x_rail_right": x_rail_right,
        "rail_center": rail_center, "rail_gauge": rail_gauge,
        "notch_depth": notch_depth, "shoulder_z": shoulder_z,
    }


def _floor_profile_xz(points, depth_lo, depth_hi, x_range=4.0, xbin=0.05):
    """Медианный профиль высоты пола Z(X) в срезе — общая заготовка для обоих
    детекторов рельсов. Заодно профиль ЯРКОСТИ по тем же бинам: головка рельса
    отличается от бетона не только высотой (см. _rail_darkness)."""
    x, y, z = points['x'], points['y'], points['z']
    depth = -y
    mask = (depth >= depth_lo) & (depth < depth_hi) & (np.abs(x) < x_range)
    if mask.sum() < 200:
        return None
    xr, zr = x[mask], z[mask]
    ir = points['intensity'][mask]
    zlo, zhi = np.percentile(zr, [1, 45])
    band = (zr >= zlo) & (zr <= zhi)
    if band.sum() < 100:
        return None
    xb, zb, ib = xr[band], zr[band], ir[band]
    bins = np.arange(-x_range, x_range + xbin, xbin)
    px, (pz, pi) = _bin_medians(xb, bins, zb, ib)
    if len(px) < 15:
        return None
    return px, pz, xb, zb, pi


def _rail_darkness(pi, i_peak):
    """Насколько бин ТЕМНЕЕ окружающего пола, в долях — признак головки рельса.

    Знак здесь обратен тому, которого ждёшь. Предполагалось, что полированный
    металл рельса ЯРЧЕ матового бетона; замер показал ровно наоборот: на 25 821
    точке головки рельса против 466 318 точек пола между рельсами вероятность
    того, что точка рельса ярче точки пола, равна 0.19, то есть рельс СИСТЕМАТИЧЕСКИ
    темнее, и заметно.

    Объяснение физическое: лидар смотрит вдоль пути, и на головку рельса луч
    падает под скользящим углом. Полированная поверхность при таком падении
    работает зеркалом — отражает энергию вперёд, а не обратно в приёмник.
    Матовый бетон рассеивает диффузно и возвращает больше. Чем ровнее металл,
    тем он на таком ракурсе ТЕМНЕЕ.

    0 — не темнее окружения, 1 — полностью чёрный на его фоне.
    """
    base = float(np.median(pi))
    if not np.isfinite(base) or base <= 0:
        return 0.0
    return float(np.clip(1.0 - i_peak / base, 0.0, 1.0))


def find_rails_by_gauge(points, depth_lo, depth_hi, x_range=4.0, xbin=0.05,
                        expected=EXPECTED_GAUGE, tol=GAUGE_TOL, min_rise=0.03,
                        dark_weight=DARK_WEIGHT):
    """Ищет рельсы как ПАРУ ПИКОВ профиля пола на правильном расстоянии друг от
    друга, не требуя выемки между ними.

    Прежний детектор (find_groove_and_rails) опирается на V-образную выемку
    между рельсами и отказывает, если она мельче 6 см. На участках с плитным
    основанием пути пол между рельсами ПЛОСКИЙ — выемка 3-5 см, — и детектор
    молчит на сотнях кадров подряд, хотя сами рельсы видны отлично: два пика по
    10-15 см. Хуже того, единственным достаточно глубоким провалом там
    оказывается щель между крайним рельсом и стеной, и детектор принимает за
    путь её, выдавая колею 0.85-1.40 м.

    Здесь опорой служит не выемка, а колея: расстояние между головками рельсов —
    физическая константа сети, одинаковая на всех записях, тогда как глубина
    выемки зависит от типа основания пути. Пара пиков на нужном расстоянии
    опознаётся однозначно.
    """
    prof = _floor_profile_xz(points, depth_lo, depth_hi, x_range, xbin)
    if prof is None:
        return None
    px, pz, xb, zb, pi = prof

    peaks, props = find_peaks(pz, prominence=min_rise)
    if len(peaks) < 2:
        return None
    cand = [(i, px[i], props["prominences"][k]) for k, i in enumerate(peaks)
            if abs(px[i]) < 2.5]
    best = None
    for a in range(len(cand)):
        for b in range(a + 1, len(cand)):
            sep = cand[b][1] - cand[a][1]
            if abs(sep - expected) > tol:
                continue
            # Из подходящих пар берём самую выраженную И самую близкую к
            # ожидаемой колее. Только по выраженности выбирать нельзя: пара
            # "крайний рельс + выступ у стены" иногда даёт пики не хуже, но
            # расстояние между ними заметно меньше колеи, и это её выдаёт.
            #
            # Третье свидетельство — яркость: обе вершины должны быть ТЕМНЕЕ
            # окружающего пола (см. _rail_darkness — знак измерен, а не
            # угадан). Оно независимо от первых двух и помогает там, где
            # профиль пологий и по выраженности пики неразличимы.
            score = min(cand[a][2], cand[b][2]) - 0.5 * abs(sep - expected)
            if dark_weight:
                score += dark_weight * min(_rail_darkness(pi, pi[cand[a][0]]),
                                           _rail_darkness(pi, pi[cand[b][0]]))
            if best is None or score > best[0]:
                best = (score, cand[a][1], cand[b][1])
    if best is None:
        return None
    _, x_rail_left, x_rail_right = best

    inside = (px > x_rail_left) & (px < x_rail_right)
    if inside.sum() < 3:
        return None
    shoulder_z = float(np.median(pz[inside]))
    return {
        "depth_lo": depth_lo, "depth_hi": depth_hi,
        "x_raw": xb, "z_raw": zb,
        "x_left": x_rail_left + 0.1, "x_right": x_rail_right - 0.1,
        "x_rail_left": float(x_rail_left), "x_rail_right": float(x_rail_right),
        "rail_center": float((x_rail_left + x_rail_right) / 2),
        "rail_gauge": float(x_rail_right - x_rail_left),
        "notch_depth": float(shoulder_z - pz[inside].min()),
        "shoulder_z": shoulder_z,
        "by_gauge": True,
    }


def find_rails(points, depth_lo, depth_hi, **kwargs):
    """Рельсы в срезе: сначала по выемке, при неудаче или неправдоподобной колее
    — по расстоянию между пиками.

    Проверка колеи на правдоподобие обязательна и здесь, а не только при
    согласовании срезов между собой: когда детектор выемки находит "путь"
    шириной 0.85 м, это не шум, который усреднится, а систематический промах на
    щель у стены, и трёх таких срезов хватает, чтобы увести опору на метры.
    """
    r = find_groove_and_rails(points, depth_lo, depth_hi, **kwargs)
    if r is not None and GAUGE_LIMITS[0] <= r["rail_gauge"] <= GAUGE_LIMITS[1]:
        return r
    return find_rails_by_gauge(points, depth_lo, depth_hi)


def analyze_frame(points, depth_bins=DEFAULT_DEPTH_BINS):
    """Прогоняет find_groove_and_rails по списку срезов глубины.

    Возвращает (found, n_skipped): found — список найденных результатов,
    n_skipped — сколько срезов пропущено как ненадёжные.
    """
    found = []
    skipped = 0
    # Один раз за кадр — точки, попадающие хотя бы в один срез (порядок точек
    # сохраняется): каждый срез иначе резал бы маской всё облако, а у
    # doubleT_obstacle это 921 тыс. точек × 13 срезов (экспер. 19). Срез из
    # отобранных — тот же набор в том же порядке: условие среза строже отбора.
    if len(depth_bins):
        d_lo = min(lo for lo, _ in depth_bins)
        d_hi = max(hi for _, hi in depth_bins)
        depth = -points['y']
        points = points[(depth >= d_lo) & (depth < d_hi) & (np.abs(points['x']) < 4.0)]
    for lo, hi in depth_bins:
        r = find_rails(points, lo, hi)
        if r is None:
            skipped += 1
        else:
            found.append(r)
    return found, skipped


def classify_points(record):
    """Делит сырые точки среза на желоб / рельсы / остальное по найденной геометрии."""
    xr = record["x_raw"]
    is_groove = (xr >= record["x_left"]) & (xr <= record["x_right"])
    is_rail = (
        (xr >= record["x_rail_left"] - 0.12) & (xr <= record["x_rail_left"] + 0.12)
    ) | (
        (xr >= record["x_rail_right"] - 0.12) & (xr <= record["x_rail_right"] + 0.12)
    )
    is_rest = ~is_groove & ~is_rail
    return is_groove, is_rail, is_rest
