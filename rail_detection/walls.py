"""Оценка положения стен тоннеля по срезам — независимое (от рельс) свидетельство
поворота пути: если стены искривляются в ту же сторону, что и centerline по
рельсам, это подтверждает, что это реальный поворот, а не артефакт детектора.
"""

import numpy as np


def find_wall_positions(points, depth_lo, depth_hi, floor_z, height_lo=0.3, height_hi=2.5,
                         x_range=6.0, pctl=2):
    """Оценивает X-позиции левой и правой стены в срезе на высоте [floor_z+height_lo,
    floor_z+height_hi] над полом (полоса заведомо выше пола/желоба, но ниже потолка).

    floor_z: локальный уровень пола в этом срезе (напр. shoulder_z из
        find_groove_and_rails для того же среза).
    pctl: перцентиль для устойчивой оценки крайней точки (не абсолютный max/min,
        чтобы не ловить редкие выбросы).

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
    if mask.sum() < 30:
        return None
    xw = x[mask]
    right_pts = xw[xw > 0]
    left_pts = xw[xw < 0]
    if len(right_pts) < 10 or len(left_pts) < 10:
        return None
    right_wall = float(np.percentile(right_pts, 100 - pctl))
    left_wall = float(np.percentile(left_pts, pctl))
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
    """Подгоняет гладкую кривую wall_x(depth) для 'left_wall_x' или 'right_wall_x'.
    Возвращает coeffs или None, если точек недостаточно."""
    key = f"{side}_wall_x"
    depths = np.array([(w["depth_lo"] + w["depth_hi"]) / 2 for w in wall_records])
    xs = np.array([w[key] for w in wall_records])
    if len(depths) < degree + 2:
        return None
    order = np.argsort(depths)
    return np.polyfit(depths[order], xs[order], degree)
