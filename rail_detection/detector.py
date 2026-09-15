"""Геометрическая детекция центрального желоба и рельс на срезе кадра лидара.

Метод (см. knowledge.md §13-14): берём тонкий срез по глубине, строим медианный
профиль высоты пола Z(X), находим V-образный желоб (дно + точки выхода на
"полувысоте"), затем ищем локальные пики профиля сразу за краями желоба — это
рельсы. Один срез обрабатывается независимо от других (без сравнения кадров).
"""

import numpy as np
from scipy.signal import find_peaks

DEFAULT_DEPTH_BINS = [
    (3, 5), (5, 7), (7, 9), (9, 11), (11, 13), (13, 15),
    (15, 18), (18, 21), (21, 24), (24, 27), (27, 30), (30, 35), (35, 40),
]  # надёжная зона по данным ~до 30-40м, дальше плотность точек не позволяет


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
    idx = np.digitize(xb, bins)
    prof_x, prof_z = [], []
    for i in range(1, len(bins)):
        m = idx == i
        if m.sum() >= 3:
            prof_x.append((bins[i - 1] + bins[i]) / 2)
            prof_z.append(np.median(zb[m]))
    if len(prof_x) < 15:
        return None
    prof_x, prof_z = np.array(prof_x), np.array(prof_z)

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


def analyze_frame(points, depth_bins=DEFAULT_DEPTH_BINS):
    """Прогоняет find_groove_and_rails по списку срезов глубины.

    Возвращает (found, n_skipped): found — список найденных результатов,
    n_skipped — сколько срезов пропущено как ненадёжные.
    """
    found = []
    skipped = 0
    for lo, hi in depth_bins:
        r = find_groove_and_rails(points, lo, hi)
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
