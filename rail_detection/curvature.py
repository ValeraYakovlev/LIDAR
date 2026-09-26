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


def _best_hypothesis(depths, xs, idxs, degree, threshold):
    """Та же гипотеза, что выбирал цикл `np.polyfit` → `np.polyval` → подсчёт
    согласных (первая с наибольшим числом), но для всех выборок разом
    (экспер. 19). Повторяет np.polyfit по шагам: матрица Вандермонда
    накопленным произведением, масштаб столбцов, тот же lstsq с
    rcond = len(x)·eps; np.polyval — схемой Горнера."""
    from .fastops import lstsq_batch

    order = degree + 1
    X = depths[idxs] + 0.0
    Y = xs[idxs] + 0.0
    k, m = X.shape
    V = np.empty((k, m, order))
    tmp = V[:, :, ::-1]
    tmp[:, :, 0] = 1
    if order > 1:
        tmp[:, :, 1:] = X[:, :, None]
        np.multiply.accumulate(tmp[:, :, 1:], out=tmp[:, :, 1:], axis=2)
    scale = np.sqrt((V * V).sum(axis=1))
    V /= scale[:, None, :]
    C = lstsq_batch(V, Y, rcond=m * np.finfo(float).eps) / scale
    P = np.zeros((k, len(depths)))
    for j in range(order):
        P = P * depths[None, :] + C[:, j:j + 1]
    inl = np.abs(P - xs[None, :]) < threshold
    return inl[int(np.argmax(inl.sum(axis=1)))]


def ransac_poly_fit(depths, xs, degree, threshold, n_iter=300, min_inlier_frac=0.6, seed=0):
    """RANSAC-подгонка полинома степени degree к (depths, xs), устойчивая к
    выбросам среди отдельных срезов (один плохой срез не должен портить всю
    форму). Возвращает dict {coeffs, inliers (bool-маска), residual (RMS по
    инлайерам), inlier_frac} или None, если надёжно подогнать не удалось.

    coeffs — подогнаны по ВСЕМ инлайерам финальным МНК (не только по
    минимальной выборке), это стандартная схема RANSAC (минимальная выборка —
    только для голосования, финальная модель — по консенсусу)."""
    depths = np.asarray(depths, dtype=float)
    xs = np.asarray(xs, dtype=float)
    n = len(depths)
    min_sample = degree + 1
    if n < min_sample:
        return None
    rng = np.random.default_rng(seed)
    idxs = np.array([rng.choice(n, size=min_sample, replace=False) for _ in range(n_iter)])
    try:
        best_inliers = _best_hypothesis(depths, xs, idxs, degree, threshold)
    except np.linalg.LinAlgError:
        best_inliers, best_count = None, -1
        for idx in idxs:
            try:
                c = np.polyfit(depths[idx], xs[idx], degree)
            except np.linalg.LinAlgError:
                continue
            resid = np.abs(np.polyval(c, depths) - xs)
            inliers = resid < threshold
            count = int(inliers.sum())
            if count > best_count:
                best_count, best_inliers = count, inliers
    if best_inliers is None or best_inliers.sum() < max(min_sample, min_inlier_frac * n):
        return None
    coeffs = np.polyfit(depths[best_inliers], xs[best_inliers], degree)
    resid_rms = float(np.sqrt(np.mean((np.polyval(coeffs, depths[best_inliers]) - xs[best_inliers]) ** 2)))
    return {"coeffs": coeffs, "inliers": best_inliers, "residual": resid_rms,
            "inlier_frac": float(best_inliers.sum() / n)}


def fit_straight_or_arc(depths, xs, straight_resid_thresh=0.08, min_points=4,
                         min_inlier_frac=0.6, n_iter=300, seed=0):
    """Физически осмысленная подгонка пути: ЛИБО прямая (нулевая кривизна),
    ЛИБО дуга окружности (постоянная кривизна) — так реально устроены пути
    (прямые участки соединяются круговыми кривыми), в отличие от произвольного
    полинома, который может "вилять" нефизично. Подгонка — через RANSAC
    (ransac_poly_fit), а не обычный МНК: несколько шумных/ошибочных срезов
    (единичные промахи детектора стены) не должны портить всю форму — они
    просто попадут в выбросы (inliers=False) вместо того чтобы утащить кривую.

    Для мягкой кривизны дуга окружности аппроксимируется параболой:
    x(s) = s²/(2R) + b·s + c, откуда радиус R = 1/(2|a|) по старшему
    коэффициенту a (малоугловое приближение).

    Если ни прямая, ни дуга целиком не описывают данные — ищет точку перехода
    прямая->дуга (реальный участок "было прямо, начался поворот" в кадре).

    Возвращает dict: kind ('straight'/'arc'/'transition'/'unclear'/None),
    coeffs, radius, residual (RMS по инлайерам), inlier_frac, split_depth."""
    depths = np.asarray(depths, dtype=float)
    xs = np.asarray(xs, dtype=float)
    order = np.argsort(depths)
    depths, xs = depths[order], xs[order]
    n = len(depths)
    empty = {"kind": None, "coeffs": None, "radius": None, "residual": None,
             "inlier_frac": None, "split_depth": None}
    if n < min_points:
        return empty

    fit1 = ransac_poly_fit(depths, xs, 1, straight_resid_thresh, n_iter, min_inlier_frac, seed)
    if fit1 is not None and fit1["residual"] < straight_resid_thresh:
        return {"kind": "straight", "coeffs": fit1["coeffs"], "radius": None,
                "residual": fit1["residual"], "inlier_frac": fit1["inlier_frac"], "split_depth": None}

    if n < min_points + 1:
        if fit1 is not None:
            return {"kind": "unclear", "coeffs": fit1["coeffs"], "radius": None,
                    "residual": fit1["residual"], "inlier_frac": fit1["inlier_frac"], "split_depth": None}
        return empty

    fit2 = ransac_poly_fit(depths, xs, 2, straight_resid_thresh, n_iter, min_inlier_frac, seed)
    if fit2 is not None:
        a = fit2["coeffs"][0]
        radius = float(1 / (2 * abs(a))) if abs(a) > 1e-6 else float("inf")
        if fit2["residual"] < straight_resid_thresh:
            return {"kind": "arc", "coeffs": fit2["coeffs"], "radius": radius,
                    "residual": fit2["residual"], "inlier_frac": fit2["inlier_frac"], "split_depth": None}

    # ни прямая, ни дуга не описали >= min_inlier_frac точек -> ищем переход
    best = None
    for i in range(min_points, n - min_points):
        split = depths[i]
        d1, x1 = depths[:i], xs[:i]
        d2, x2 = depths[i:], xs[i:]
        try:
            cc1 = np.polyfit(d1, x1, 1)
            cc2 = np.polyfit(d2, x2, min(2, len(d2) - 1))
        except (np.linalg.LinAlgError, ValueError):
            continue
        r1 = np.polyval(cc1, d1) - x1
        r2 = np.polyval(cc2, d2) - x2
        total_resid = float(np.sqrt(np.mean(np.concatenate([r1, r2]) ** 2)))
        if best is None or total_resid < best[0]:
            best = (total_resid, split, cc1, cc2)

    resid1 = fit1["residual"] if fit1 is not None else float("inf")
    resid2 = fit2["residual"] if fit2 is not None else float("inf")
    if best is not None and best[0] < min(resid1, resid2):
        total_resid, split, cc1, cc2 = best
        return {"kind": "transition", "coeffs": (cc1, cc2), "radius": None,
                "residual": total_resid, "inlier_frac": None, "split_depth": float(split)}

    if fit2 is not None:
        a = fit2["coeffs"][0]
        radius = float(1 / (2 * abs(a))) if abs(a) > 1e-6 else float("inf")
        return {"kind": "unclear", "coeffs": fit2["coeffs"], "radius": radius,
                "residual": fit2["residual"], "inlier_frac": fit2["inlier_frac"], "split_depth": None}
    return empty


def eval_fit(fit, depths):
    """x(depth) по результату fit_straight_or_arc — единообразно для всех kind,
    включая кусочный 'transition'. NaN там, где модели нет."""
    depths = np.asarray(depths, dtype=float)
    kind = fit.get("kind") if fit else None
    if kind in ("straight", "arc", "unclear"):
        return np.polyval(fit["coeffs"], depths)
    if kind == "transition":
        cc1, cc2 = fit["coeffs"]
        return np.where(depths <= fit["split_depth"],
                        np.polyval(cc1, depths), np.polyval(cc2, depths))
    return np.full(depths.shape, np.nan)


def slope_from_fit(fit, depths):
    """dx/d(depth) по результату fit_straight_or_arc (для локального курса)."""
    depths = np.asarray(depths, dtype=float)
    kind = fit.get("kind") if fit else None
    if kind in ("straight", "arc", "unclear"):
        return np.polyval(np.polyder(fit["coeffs"]), depths)
    if kind == "transition":
        cc1, cc2 = fit["coeffs"]
        return np.where(depths <= fit["split_depth"],
                        np.polyval(np.polyder(cc1), depths),
                        np.polyval(np.polyder(cc2), depths))
    return np.zeros(depths.shape)


def _heading_deg_from_fit(fit, depth):
    """Локальный курс (град.) по результату fit_straight_or_arc в точке depth —
    работает единообразно для 'straight'/'arc'/'unclear' (один полином) и
    'transition' (кусочно, по своей стороне от split_depth)."""
    if fit.get("kind") is None:
        return None
    return float(np.degrees(np.arctan(float(slope_from_fit(fit, depth)))))


def walls_consistent(left_fit, right_fit, depths, heading_tol_deg=3.0):
    """Проверяет, согласуются ли курс левой и правой стены на общем диапазоне
    depths — НЕПРЕРЫВНОЕ сравнение фактического угла курса в нескольких точках,
    а не по дискретной метке kind (straight/arc). Дискретное сравнение ложно
    бракует пары вида "почти прямая дуга огромного радиуса" + "прямая" — они
    физически совпадают, просто по разные стороны порога классификации.
    Настоящая несогласованность (разный знак поворота, сильно разная кривизна)
    по-прежнему ловится, потому что тогда углы курса реально разойдутся."""
    if left_fit.get("kind") is None or right_fit.get("kind") is None:
        return False
    depths = np.asarray(depths, dtype=float)
    if len(depths) == 0:
        return False
    for d in np.linspace(depths.min(), depths.max(), 5):
        hl = _heading_deg_from_fit(left_fit, float(d))
        hr = _heading_deg_from_fit(right_fit, float(d))
        if hl is None or hr is None or abs(hl - hr) > heading_tol_deg:
            return False
    return True


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
