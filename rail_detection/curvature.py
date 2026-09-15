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
