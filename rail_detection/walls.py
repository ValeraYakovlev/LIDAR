"""Оценка положения стен тоннеля по срезам — независимое (от рельс) свидетельство
поворота пути: если стены искривляются в ту же сторону, что и centerline по
рельсам, это подтверждает, что это реальный поворот, а не артефакт детектора.

Позиция стены определяется по НАИБОЛЕЕ ПЛОТНОМУ участку (пик гистограммы X),
а не по крайним/перцентильным точкам — единичные далёкие точки (шум, соседняя
ниша/инфраструктура) не должны перевешивать основную поверхность стены, где
реально сосредоточено большинство отражений лидара.
"""

import numpy as np
from scipy.signal import find_peaks


def find_wall_positions(points, depth_lo, depth_hi, floor_z, height_lo=0.3, height_hi=2.5,
                         x_range=6.0, xbin=0.1, min_points=20, min_peak_count=15):
    """Оценивает X-позиции левой и правой стены в срезе на высоте [floor_z+height_lo,
    floor_z+height_hi] над полом (полоса заведомо выше пола/желоба, но ниже потолка).

    Позиция стены = робастный центр БЛИЖАЙШЕГО К ЦЕНТРУ значимого скопления
    точек (ближайший достаточно "населённый" пик гистограммы по X), а НЕ самое
    плотное скопление вообще и НЕ крайняя точка. Простое "самое плотное" ловит
    посторонние поверхности (напр. платформу — она может отражать плотнее и
    более перпендикулярно, чем сама стена тоннеля под скользящим углом), а
    крайняя точка ловит редкие выбросы — оба варианта проверялись и давали
    физически нереальные скачки.

    floor_z: локальный уровень пола в этом срезе (напр. shoulder_z из
        find_groove_and_rails для того же среза).
    Возвращает dict {left_wall_x, right_wall_x, n_left, n_right} или None,
    если по какой-то стороне точек недостаточно.
    """
    x, y, z = points['x'], points['y'], points['z']
    depth = -y
    mask = (
        (depth >= depth_lo) & (depth < depth_hi)
        & (z > floor_z + height_lo) & (z < floor_z + height_hi)
        & (np.abs(x) < x_range)
    )
    if mask.sum() < min_points:
        return None
    xw = x[mask]

    def nearest_significant_x(vals):
        """Ближайший к 0 пик гистограммы с числом точек >= min_peak_count."""
        if len(vals) < min_points // 2:
            return None
        bins = np.arange(vals.min(), vals.max() + xbin, xbin)
        if len(bins) < 3:
            return float(np.median(vals)) if len(vals) >= min_peak_count else None
        counts, edges = np.histogram(vals, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        peaks_idx, _ = find_peaks(counts)
        # включаем и крайние бины как кандидатов (find_peaks их не видит)
        candidates = list(peaks_idx)
        if len(counts) > 0:
            if counts[0] >= counts[1] if len(counts) > 1 else True:
                candidates.append(0)
            if counts[-1] >= counts[-2] if len(counts) > 1 else True:
                candidates.append(len(counts) - 1)
        candidates = [i for i in set(candidates) if counts[i] >= min_peak_count]
        if not candidates:
            return None
        i_best = min(candidates, key=lambda i: abs(centers[i]))
        peak_center = centers[i_best]
        near_peak = vals[np.abs(vals - peak_center) < 3 * xbin]
        return float(np.median(near_peak)) if len(near_peak) else float(peak_center)

    right_pts = xw[xw > 0]
    left_pts = xw[xw < 0]
    right_wall = nearest_significant_x(right_pts)
    left_wall = nearest_significant_x(left_pts)
    if right_wall is None or left_wall is None:
        return None
    return {
        "depth_lo": depth_lo, "depth_hi": depth_hi,
        "left_wall_x": left_wall, "right_wall_x": right_wall,
        "n_left": int(len(left_pts)), "n_right": int(len(right_pts)),
    }


def analyze_walls(points, records_by_depth, depth_bins, **kwargs):
    """Прогоняет find_wall_positions по срезам, используя локальный floor_z из
    уже найденных groove/rail записей (shoulder_z) в качестве опорного уровня.

    records_by_depth: {(depth_lo, depth_hi): record} — результаты
        find_groove_and_rails для тех же срезов (нужен shoulder_z).
    Возвращает список найденных wall-записей.
    """
    found = []
    for lo, hi in depth_bins:
        rec = records_by_depth.get((lo, hi))
        if rec is None:
            continue
        w = find_wall_positions(points, lo, hi, floor_z=rec["shoulder_z"], **kwargs)
        if w is not None:
            found.append(w)
    return found


def fit_wall(wall_records, side, degree=2):
    """(Устаревшее, оставлено для обратной совместимости с простыми графиками.)
    Подгоняет обычный полином wall_x(depth). Для физически осмысленной
    прямая/дуга-подгонки используйте curvature.fit_straight_or_arc."""
    key = f"{side}_wall_x"
    depths = np.array([(w["depth_lo"] + w["depth_hi"]) / 2 for w in wall_records])
    xs = np.array([w[key] for w in wall_records])
    if len(depths) < degree + 2:
        return None
    order = np.argsort(depths)
    return np.polyfit(depths[order], xs[order], degree)
