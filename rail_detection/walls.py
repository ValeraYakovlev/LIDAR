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

from .curvature import fit_straight_or_arc


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


def height_band_mask(x, y, z, floor_z, height_lo=0.3, height_hi=2.5, depth_max=40.0, x_range=6.0):
    depth = -y
    return (
        (depth > 0) & (depth < depth_max)
        & (z > floor_z + height_lo) & (z < floor_z + height_hi)
        & (np.abs(x) < x_range)
    )


def _candidates_density(x, depth, band_idx, depth_bins, xbin=0.1, min_points=20, min_peak_count=15):
    """Кандидаты по методу 'плотность' (см. find_wall_positions), но
    возвращает МНОЖЕСТВО точек вокруг найденного пика на каждой стороне/срезе,
    а не одно число — для объединения с другими методами."""
    kept = np.zeros(len(band_idx), dtype=bool)
    for lo, hi in depth_bins:
        sl = (depth[band_idx] >= lo) & (depth[band_idx] < hi)
        idx_sl = band_idx[sl]
        if len(idx_sl) < 20:
            continue
        xs = x[idx_sl]
        for side in (xs > 0, xs < 0):
            vals = xs[side]
            if len(vals) < min_points // 2:
                continue
            bins = np.arange(vals.min(), vals.max() + xbin, xbin)
            if len(bins) < 3:
                continue
            counts, edges = np.histogram(vals, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            peaks_idx, _ = find_peaks(counts)
            cands = list(peaks_idx)
            if len(counts) > 1:
                if counts[0] >= counts[1]:
                    cands.append(0)
                if counts[-1] >= counts[-2]:
                    cands.append(len(counts) - 1)
            cands = [i for i in set(cands) if counts[i] >= min_peak_count]
            if not cands:
                continue
            i_best = min(cands, key=lambda i: abs(centers[i]))
            peak_x = centers[i_best]
            local_idx = idx_sl[side]
            kept[np.isin(band_idx, local_idx[np.abs(vals - peak_x) < 0.15])] = True
    return kept


def _candidates_envelope(x, depth, band_idx, depth_bins, support_radius=0.3, min_support=6):
    """Кандидаты по методу 'внешняя граница с поддержкой'."""
    kept = np.zeros(len(band_idx), dtype=bool)
    for lo, hi in depth_bins:
        sl = (depth[band_idx] >= lo) & (depth[band_idx] < hi)
        idx_sl = band_idx[sl]
        if len(idx_sl) < 20:
            continue
        xs = x[idx_sl]
        for side in (xs > 0, xs < 0):
            vals = xs[side]
            if len(vals) < min_support:
                continue
            order = np.argsort(-np.abs(vals))
            found = None
            for v in vals[order]:
                if np.sum(np.abs(vals - v) < support_radius) >= min_support:
                    found = v
                    break
            if found is None:
                continue
            local_idx = idx_sl[side]
            kept[np.isin(band_idx, local_idx[np.abs(vals - found) < 0.15])] = True
    return kept


def _candidates_polar(x, y, band_idx, angle_bin_deg=0.5, max_range=15.0,
                       support_bins=3, support_tol=0.5, min_neighbors=3):
    """Кандидаты по методу 'полярный, дальний луч + поддержка соседних лучей'."""
    xb, yb = x[band_idx], y[band_idx]
    angle = np.degrees(np.arctan2(xb, -yb))
    rng = np.sqrt(xb ** 2 + yb ** 2)
    kept = np.zeros(len(band_idx), dtype=bool)
    valid = rng < max_range
    if valid.sum() == 0:
        return kept
    bins = np.arange(angle.min(), angle.max() + angle_bin_deg, angle_bin_deg)
    idx_bin = np.digitize(angle, bins)
    n_bins = len(bins)
    far_per_bin = np.full(n_bins + 1, np.nan)
    for i in range(1, n_bins):
        sl = (idx_bin == i) & valid
        if sl.sum() > 0:
            far_per_bin[i] = rng[sl].max()
    for i in range(1, n_bins):
        if np.isnan(far_per_bin[i]):
            continue
        neigh = far_per_bin[max(1, i - support_bins):min(n_bins, i + support_bins + 1)]
        neigh = neigh[~np.isnan(neigh)]
        if np.sum(np.abs(neigh - far_per_bin[i]) < support_tol) < min_neighbors:
            continue
        sl = (idx_bin == i) & valid
        local = np.where(sl)[0]
        kept[local[np.abs(rng[local] - far_per_bin[i]) < 0.15]] = True
    return kept


def combined_wall_fit(points, floor_z, depth_bins, straight_resid_thresh=0.08):
    """Объединяет кандидатов от ТРЁХ методов (density, envelope, polar) —
    объединение множеств (union), не голосование — и подгоняет через них ОДНУ
    гладкую функцию (прямая/дуга/переход, RANSAC) для левой и правой стены.

    Идея: методы ошибаются по-разному и в разных местах; если бы один метод
    справлялся идеально, объединять было бы не нужно. Но реальная стена
    физически ОДНА гладкая линия — поэтому вместо того, чтобы верить одному
    методу целиком, отдаём RANSAC-подгонке (curvature.fit_straight_or_arc)
    право решить, какое подмножество кандидатов (от любого из методов)
    складывается в одну согласованную кривую, а какое — шум/ошибки метода и
    отбрасывается как выбросы.

    Возвращает dict {left_fit, right_fit, left_candidates, right_candidates}
    (candidates — (depth, x) массивы всех кандидатов до подгонки, для
    визуализации/диагностики)."""
    x, y, z = points['x'], points['y'], points['z']
    depth = -y
    band = height_band_mask(x, y, z, floor_z)
    band_idx = np.where(band)[0]
    if len(band_idx) < 50:
        return None

    kept_density = _candidates_density(x, depth, band_idx, depth_bins)
    kept_envelope = _candidates_envelope(x, depth, band_idx, depth_bins)
    kept_polar = _candidates_polar(x, y, band_idx)
    union = kept_density | kept_envelope | kept_polar

    cand_idx = band_idx[union]
    cx, cdepth = x[cand_idx], depth[cand_idx]

    # ВАЖНО: fit_straight_or_arc (RANSAC + перебор точки перехода) рассчитан на
    # ~1 значение на срез глубины (как раньше), а не на тысячи сырых точек —
    # переход прямая->дуга перебирает точки как кандидатов разреза, и на сырых
    # точках это O(n^2) и попросту не отрабатывает за разумное время. Поэтому
    # агрегируем кандидатов ПО СРЕЗАМ ГЛУБИНЫ (медиана X всех кандидатов трёх
    # методов в срезе) — это и есть точка объединения методов, а дальше подгонка
    # работает как раньше, на компактном наборе точек.
    results = {}
    for side_name, cmp_fn in [("left", np.less), ("right", np.greater)]:
        side_mask = cmp_fn(cx, 0)
        sx_raw, sd_raw = cx[side_mask], cdepth[side_mask]
        bin_depths, bin_xs = [], []
        for lo, hi in depth_bins:
            sl = (sd_raw >= lo) & (sd_raw < hi)
            if sl.sum() == 0:
                continue
            bin_depths.append((lo + hi) / 2)
            bin_xs.append(float(np.median(sx_raw[sl])))
        fit = fit_straight_or_arc(bin_depths, bin_xs, straight_resid_thresh=straight_resid_thresh)
        results[f"{side_name}_fit"] = fit
        results[f"{side_name}_candidates"] = (sd_raw, sx_raw)  # сырые точки — для наглядности на графике
        results[f"{side_name}_bins"] = (bin_depths, bin_xs)    # агрегированные — то, что реально подгонялось
    return results


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
