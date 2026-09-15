"""Устойчивое построение centerline по срезам одного кадра.

find_groove_and_rails детектирует рельсы независимо в каждом срезе — иногда он
промахивается (цепляет не тот пик) и даёт одиночный резкий скачок centerline,
который физически невозможен (путь не может дёрнуться на метр и тут же
вернуться). Здесь это исправляется: путь должен быть гладким по построению
(минимальный радиус кривизны у реального пути), поэтому одиночные выбросы
относительно устойчивой (робастной) подгонки гладкой кривой можно уверенно
отбраковывать и заменять значением по тренду соседних срезов.
"""

import numpy as np


def robust_centerline(records, degree=2, outlier_thresh=0.25, max_iter=5):
    """Итеративная робастная полиномиальная подгонка centerline(depth) с
    отбраковкой выбросов (аналог sigma-clipping / RANSAC-подобной идеи).

    records: список результатов find_groove_and_rails (для одного кадра).
    degree: степень полинома (2 — гладкая кривая, годится для типичного
        поворота пути; при малом числе срезов автоматически понижается).
    outlier_thresh: порог отклонения от подгонки, метры — точка дальше этого
        считается выбросом (промах детектора, не реальный путь).

    Возвращает (records_sorted, is_outlier, corrected_centers, coeffs):
        records_sorted — записи, отсортированные по глубине;
        is_outlier — булев массив той же длины (True — точка отброшена);
        corrected_centers — centerline той же длины: для inlier-точек — то,
            что нашёл детектор, для выбросов — значение по сглаженной кривой;
        coeffs — коэффициенты финального полинома (или None, если точек мало).
    """
    n = len(records)
    if n == 0:
        return [], np.array([], dtype=bool), np.array([]), None

    depths = np.array([(r["depth_lo"] + r["depth_hi"]) / 2 for r in records])
    centers = np.array([r["rail_center"] for r in records])
    order = np.argsort(depths)
    depths, centers = depths[order], centers[order]
    records_sorted = [records[i] for i in order]

    eff_degree = min(degree, max(0, n - 2))
    if eff_degree < 1 or n < 4:
        # слишком мало точек для устойчивой подгонки — ничего не отбрасываем
        return records_sorted, np.zeros(n, dtype=bool), centers.copy(), None

    keep = np.ones(n, dtype=bool)
    coeffs = np.polyfit(depths, centers, eff_degree)
    for _ in range(max_iter):
        if keep.sum() < eff_degree + 2:
            break
        coeffs = np.polyfit(depths[keep], centers[keep], eff_degree)
        fitted_all = np.polyval(coeffs, depths)
        residual = np.abs(centers - fitted_all)
        new_keep = residual < outlier_thresh
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    is_outlier = ~keep
    fitted_all = np.polyval(coeffs, depths)
    corrected_centers = np.where(is_outlier, fitted_all, centers)

    return records_sorted, is_outlier, corrected_centers, coeffs
