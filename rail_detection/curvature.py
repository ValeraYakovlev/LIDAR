"""Учёт реальной кривизны пути: перпендикулярные изгибу срезы (а не срез по
фиксированной оси Y сенсора) и разделение срезов на прямые/поворотные участки.

На повороте срез по константной глубине Y пересекает рельсы не строго
поперёк, а "наискосок" — это завышает измеренное расстояние между рельсами
(геометрия проекции), помимо возможного настоящего уширения колеи. Здесь это
корректируется: сначала грубо оцениваем centerline (слепой проход), подгоняем
по ней гладкую кривую пути, затем пересчитываем точки в систему координат
вдоль локального курса этой кривой перед повторной детекцией.
"""

import numpy as np

from .tracking import robust_centerline


def fit_path(records):
    """Подгоняет гладкую кривую centerline(depth) по результатам детекции
    (обычно — по "слепому" первому проходу). Возвращает коэффициенты полинома
    (numpy.polyfit) или None, если точек недостаточно для устойчивой подгонки."""
    _, _, _, coeffs = robust_centerline(records)
    return coeffs


def local_heading_deg(coeffs, depth):
    """Угол локального курса пути (град.) относительно оси сенсора на глубине depth."""
    if coeffs is None:
        return 0.0
    slope = np.polyval(np.polyder(coeffs), depth)
    return float(np.degrees(np.arctan(slope)))


def classify_section(coeffs, depth, thresh_deg=1.5):
    """'straight' или 'curve' — по модулю локального угла курса на глубине depth."""
    return "curve" if abs(local_heading_deg(coeffs, depth)) >= thresh_deg else "straight"


def describe_path(coeffs, depths, thresh_deg=1.5):
    """Сводка по курсу пути на заданных глубинах: схлопывает подряд идущие
    срезы одного направления в сегменты и возвращает строку вида
    'прямо (3-20м) -> направо (20-38м)'. Использует turn_direction на каждой
    глубине."""
    if coeffs is None or len(depths) == 0:
        return "недостаточно данных для оценки курса"
    depths = sorted(depths)
    labels_ru = {"straight": "прямо", "left": "налево", "right": "направо"}
    segments = []
    cur_label = turn_direction(coeffs, depths[0], thresh_deg)
    seg_start = depths[0]
    for d in depths[1:]:
        lbl = turn_direction(coeffs, d, thresh_deg)
        if lbl != cur_label:
            segments.append((cur_label, seg_start, d))
            cur_label, seg_start = lbl, d
    segments.append((cur_label, seg_start, depths[-1]))
    return " -> ".join(f"{labels_ru[l]} ({a:.0f}-{b:.0f}м)" for l, a, b in segments)


def turn_direction(coeffs, depth, thresh_deg=1.5):
    """'straight' / 'left' / 'right' по знаку локального курса на глубине depth.

    Соглашение (то же, что и на графиках вида сверху — plot_topdown): смотрим на
    тоннель сверху, глубина идёт вверх по графику, ось X — как на графике (вправо
    по странице = положительный X). Путь, уходящий в сторону +X, — поворот
    направо (от лица поезда, смотрящего вперёд, тоже направо: обе точки зрения
    совпадают при виде сверху с "вперёд" от наблюдателя)."""
    heading = local_heading_deg(coeffs, depth)
    if abs(heading) < thresh_deg:
        return "straight"
    return "right" if heading > 0 else "left"
