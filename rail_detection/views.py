"""Геометрия тоннеля по виду сверху — три взгляда на одну и ту же сетку.

Эксперимент 14 (ветка `feature/topdown-floor-ceiling`). Облако кадра
раскладывается на сетку вида сверху (x — вбок, d = -y — вглубь). В каждой клетке
помнится, сколько в ней точек, самая низкая и самая высокая точка. Дальше три
подхода отличаются только тем, КАКУЮ маску строят на этой сетке:

  silhouette — «смотреть с большой высоты». Высота отбрасывается целиком,
               остаётся силуэт тоннеля. Плотность падает с глубиной примерно
               как d^-2.5, поэтому яркость нормируется по глубине — это и есть
               «сделать контрастным»: дальний тоннель становится таким же ярким,
               как ближний, а пустота вокруг остаётся пустотой. Граница тоннеля
               — граница контраста.
  floor      — пол: клетки, чья самая НИЗКАЯ точка лежит на уровне пола. Уровень
               пола меряется по самому облаку (вдоль глубины, с тангажом).
  ceiling    — свод: клетки, чья самая ВЫСОКАЯ точка лежит у верха тоннеля.
               Уровень свода меряется так же.

Общая часть у всех трёх: по маске в каждой полосе глубины (длина полосы
растёт как d², см. BAND_K) берутся левый и правый край, и обе кромки подгоняются ОДНОЙ формой со своими
смещениями (прямая, дуга или кубика). Ось — середина между ними. Ни один подход
не использует детектор рельсов: рельсы нужны только как независимая проверка.
"""

import numpy as np
from scipy import ndimage

from .fastops import group_percentile, lstsq_batch

CELL_X = 0.10       # м, клетка сетки поперёк
CELL_D = 0.50       # м, клетка сетки вдоль
X_HALF = 16.0       # м, сетка по x: [-X_HALF, X_HALF]
D_MIN = 2.0
D_MAX = 150.0
FOV_UP_DEG = 14.4   # верхний луч лидара: замер по кольцу 0 на кадрах трёх
                    # записей (doubleT_obstacle, doubleT_platform, roundT_doubleT),
                    # во всех трёх совпадает до сотых
DEPTH_SCALE = 50.0  # нормировка глубины в модели формы (обусловленность)

# Окно поиска кромок вокруг предсказанной оси: шире двухпутного тоннеля
# (±4.5 м), но уже станции целиком — станционный зал всё равно не «тоннель».
SEARCH_HALF = 7.5

METHODS = ("silhouette", "floor", "ceiling")


def _center_fn(prior):
    """Ожидаемая ось x(d) из прошлого кадра (или x=0, если его нет)."""
    if prior is None:
        return lambda d: 0.0
    return lambda d: float(shape_x(prior, d)) if np.isfinite(shape_x(prior, d)) else 0.0


# ---------------------------------------------------------------- сетка

def rasterize(points, cell_x=CELL_X, cell_d=CELL_D, x_half=X_HALF, d_min=D_MIN,
              d_max=D_MAX):
    """Вид сверху: счётчик точек, z_min и z_max по клеткам.

    Нулевые точки (лидар не получил отражения — таких 40-50% в кадре) выкидываются:
    они лежат в начале координат и дали бы ложный пол под сенсором.
    """
    x = points['x'].astype(np.float64)
    d = -points['y'].astype(np.float64)
    z = points['z'].astype(np.float64)
    ok = ((np.abs(x) + np.abs(d) + np.abs(z)) > 0.1) & (d >= d_min) & (d < d_max) \
        & (np.abs(x) < x_half)
    x, d, z = x[ok], d[ok], z[ok]
    nx = int(round(2 * x_half / cell_x))
    nd = int(round((d_max - d_min) / cell_d))
    ix = np.clip(((x + x_half) / cell_x).astype(int), 0, nx - 1)
    idd = np.clip(((d - d_min) / cell_d).astype(int), 0, nd - 1)
    flat = idd * nx + ix
    count = np.bincount(flat, minlength=nd * nx).reshape(nd, nx)
    zmin = np.full(nd * nx, np.inf)
    zmax = np.full(nd * nx, -np.inf)
    np.minimum.at(zmin, flat, z)
    np.maximum.at(zmax, flat, z)
    zmin = zmin.reshape(nd, nx)
    zmax = zmax.reshape(nd, nx)
    zmin[count == 0] = np.nan
    zmax[count == 0] = np.nan
    xc = -x_half + (np.arange(nx) + 0.5) * cell_x
    dc = d_min + (np.arange(nd) + 0.5) * cell_d
    return {"count": count, "zmin": zmin, "zmax": zmax, "xc": xc, "dc": dc,
            "cell_x": cell_x, "cell_d": cell_d}


def _row_windows(dc, near=1.0, mid=2.0, far=5.0):
    """Полосы глубины для оценки уровня пола/свода: вблизи узкие (точек много,
    уровень меняется быстро с тангажом), вдали широкие (точек мало)."""
    edges = list(np.arange(dc[0], 30, near)) + list(np.arange(30, 60, mid)) \
        + list(np.arange(60, dc[-1] + far, far))
    return np.array(edges)


# ---------------------------------------------------------------- силуэт

def contrast_image(grid, ref_pct=75.0, ref_window=4.0):
    """Нормированная по глубине яркость: log(1+n) / log(1+n_ref(d)).

    n_ref(d) — типичная (75-й перцентиль) населённость занятой клетки на этой
    глубине. Вблизи клетка стены набирает сотни точек, на 80 м — единицы, и
    без нормировки дальний тоннель на картинке просто исчезает. После неё
    стена на 100 м так же ярка, как на 10 м, а редкая точка, случайно
    пролетевшая в проём, тусклая на любой глубине.
    """
    count = grid["count"].astype(float)
    nd = count.shape[0]
    half = max(1, int(round(ref_window / 2 / grid["cell_d"])))
    ref = np.ones(nd)
    # Окно строки i — строки [i − half, i + half]; занятая клетка строки r входит
    # в окна r − half … r + half. Перцентили всех окон — одним проходом
    # (экспер. 19: раньше np.percentile на каждую строку), ответ тот же.
    rows, cols = np.nonzero(count > 0)
    vals = count[rows, cols]
    off = np.arange(-half, half + 1)
    win = (rows[None, :] + off[:, None]).ravel()
    vv = np.broadcast_to(vals, (len(off), len(vals))).ravel()
    ok = (win >= 0) & (win < nd)
    pct = group_percentile(vv[ok], win[ok], nd, ref_pct)
    has = ~np.isnan(pct)
    ref[has] = np.maximum(pct[has], 1.0)
    img = np.log1p(count) / np.log1p(ref)[:, None]
    return np.clip(img, 0.0, 1.0)


def silhouette_mask(grid, thresh=0.25):
    """Контраст -> порог. Порог отсекает тусклые клетки: одиночные точки на
    фоне плотной стены, пылинки, отражения через проём. Связность с поездом
    обеспечивает ведение оси в band_edges (окно ±SEARCH_HALF вокруг неё), а
    не разметка компонент: дальняя стена зондируется раз в 1-3 м, и любое
    замыкание, которое бы её сшило, сшило бы заодно и соседний тоннель."""
    img = contrast_image(grid)
    return img, img >= thresh


# ---------------------------------------------------------------- пол и свод

def _level_samples(grid, values, pick, center_fn, halfwin=SEARCH_HALF, min_cells=6):
    """Уровень поверхности по полосам глубины: (глубины, уровни).

    pick(vals) -> уровень по значениям z_min/z_max занятых клеток полосы.
    center_fn(d) -> ожидаемая ось (чтобы на повороте полоса смотрела на тоннель,
    а не в сторону). Окно ±halfwin вокруг неё.
    """
    dc, xc = grid["dc"], grid["xc"]
    edges = _row_windows(dc)
    ds, lv = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        rows = (dc >= lo) & (dc < hi)
        if not rows.any():
            continue
        mid = 0.5 * (lo + hi)
        cols = np.abs(xc - center_fn(mid)) < halfwin
        vals = values[np.ix_(rows, cols)]
        vals = vals[np.isfinite(vals)]
        if len(vals) < min_cells:
            continue
        ds.append(mid)
        lv.append(pick(vals))
    return np.array(ds), np.array(lv)


def _robust_line(d, z, tol, n_iter=200, seed=0):
    """RANSAC-прямая z(d) + доуточнение МНК по согласным. None, если согласных мало."""
    if len(d) < 4:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(n_iter):
        i, j = rng.choice(len(d), 2, replace=False)
        if abs(d[i] - d[j]) < 3.0:
            continue
        k = (z[j] - z[i]) / (d[j] - d[i])
        inl = np.abs(z[i] + k * (d - d[i]) - z) < tol
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None or best_n < max(4, 0.4 * len(d)):
        return None
    return np.polyfit(d[best], z[best], 1)


def _floor_pick(vals):
    """Пол — самое населённое плато среди нижних клеток полосы.

    Низкий перцентиль сам по себе смещён вниз на глубину желоба и лотков;
    мода гистограммы нижней трети — нет: пол широкий, желоб узкий.
    """
    lo_part = vals[vals <= np.percentile(vals, 40)]
    if len(lo_part) < 3:
        return float(np.median(vals))
    h, e = np.histogram(lo_part, bins=np.arange(lo_part.min(), lo_part.max() + 0.05, 0.05))
    if len(h) == 0 or h.sum() == 0:
        return float(np.median(lo_part))
    k = int(np.argmax(h))
    return float(0.5 * (e[k] + e[k + 1]))


def floor_level(grid, center_fn=lambda d: 0.0):
    """Уровень пола вдоль глубины: коэффициенты прямой z_f(d) (тангаж + уклон).

    Прямая, а не произвольная кривая: на 150 м вертикальная кривая пути даёт
    сантиметры, а редкие дальние полосы без неё начинают гулять.
    """
    d, lv = _level_samples(grid, grid["zmin"], _floor_pick, center_fn)
    coeffs = _robust_line(d, lv, tol=0.15)
    return coeffs, d, lv


def floor_mask(grid, level, below=0.35, above=0.25):
    """Клетки, чья НИЖНЯЯ точка на уровне пола: [-below, +above] от z_f(d).

    Ниже пола на 0.35 м — желоб и лотки (их глубина 18-32 см, §15), выше на
    0.25 м — головки рельсов и шпалы. Кромка пола — там, где стена поднялась
    выше четверти метра над полом.
    """
    zf = np.polyval(level, grid["dc"])[:, None]
    zm = grid["zmin"]
    with np.errstate(invalid="ignore"):
        return (zm >= zf - below) & (zm <= zf + above)


def _ceiling_pick(vals):
    return float(np.percentile(vals, 97))


def ceiling_level(grid, center_fn=lambda d: 0.0, fov_margin=0.25):
    """Уровень свода по полосам: 97-й перцентиль z_max у оси, со скользящей медианой.

    Не прямая, как у пола: тип тоннеля меняется по ходу (круглый свод 3.2 м над
    сенсором, плоский потолок 2.8 м, станционный зал 6 м), и глобальная прямая
    срезала бы один из них. Полосы, где свод физически не виден — верхний луч
    лидара (+14.4°) ещё не достаёт до потолка, — помечаются: там «верх» это
    обрез поля зрения по стене, а не свод.
    """
    d, lv = _level_samples(grid, grid["zmax"], _ceiling_pick, center_fn)
    if len(d) == 0:
        return d, lv, np.zeros(0, bool)
    fov_top = d * np.tan(np.radians(FOV_UP_DEG))
    seen = lv < fov_top - fov_margin
    smooth = lv.copy()
    for i in range(len(lv)):
        blk = lv[max(0, i - 2):i + 3][seen[max(0, i - 2):i + 3]]
        if len(blk):
            smooth[i] = float(np.median(blk))
    return d, smooth, seen


def ceiling_mask(grid, level_d, level_z, level_seen, drop=0.5):
    """Клетки, чья ВЕРХНЯЯ точка не ниже уровня свода больше чем на drop.

    На круглом своде радиуса 2.75 м это полоса шириной ~3.2 м вокруг шелыги,
    на плоском потолке — весь потолок. Строки, где свод не виден, пустые.
    """
    zc = grid["zmax"]
    mask = np.zeros(zc.shape, bool)
    if len(level_d) < 2 or not level_seen.any():
        return mask
    dc = grid["dc"]
    lz = np.interp(dc, level_d[level_seen], level_z[level_seen])
    ok_row = dc >= level_d[level_seen].min() - 1.0
    with np.errstate(invalid="ignore"):
        mask = (zc >= (lz - drop)[:, None]) & ok_row[:, None]
    return mask


# ---------------------------------------------------------------- кромки

# Длина полосы по глубине, в которой собираются кромки: L(d) = k * d^2, но не
# меньше строки сетки и не больше потолка. Квадрат — из геометрии развёртки:
# соседние кольца (шаг 0.125° у горизонта) ложатся на поверхность, отстоящую от
# сенсора по высоте на h, с шагом d^2 * 0.0022 / h. Пол (h ≈ 1.35 м) на 70 м
# зондируется раз в ~8 м, и одна строка сетки 0.5 м почти всегда пуста — кромку
# брать не из чего. Полоса длиной в шаг колец гарантирует хотя бы одну дугу.
BAND_K = {"silhouette": 0.0005, "floor": 0.0021, "ceiling": 0.0020}
BAND_MAX = {"silhouette": 6.0, "floor": 12.0, "ceiling": 12.0}


def _predict(hist_d, hist_c, d, fallback):
    """Ожидаемая ось на глубине d: прямая по середине полос за последние 20 м."""
    if len(hist_d) >= 3:
        hd, hc = np.asarray(hist_d), np.asarray(hist_c)
        sel = hd > hd[-1] - 20.0
        if sel.sum() >= 3 and np.ptp(hd[sel]) > 3.0:
            k, b = np.polyfit(hd[sel], hc[sel], 1)
            return k * d + b
    if hist_c:
        return hist_c[-1]
    return fallback(d)


def band_edges(mask, grid, method, weight=None, gap=0.6, min_width=0.0,
               halfwin=SEARCH_HALF, mode="run", d_from=0.0, prior=None, mass=0.02):
    """Левый и правый край маски по полосам глубины — с ведением оси от ближних
    полос к дальним.

    Каждая строка полосы сдвигается на предсказанную ось в своей глубине, и
    только потом строки складываются: на повороте с курсом 8° полоса в 10 м
    иначе размазала бы кромку на 1.4 м поперёк.

    mode="run": связный кусок (разрывы до gap м сшиваются), в котором стоит
        предсказанная ось или ближайший к ней. Рельсы и желоб не рвут пол, а
        соседний путь за стенкой не присоединяется.
    mode="mass": граница контраста — там, где набирается доля mass всей
        яркости полосы с каждого края. Для силуэта: середина дальней полосы
        часто пуста (кольца легли на стены, а пол между ними не задели), так
        что «связного куска» там нет, а края есть.
    Возвращает (d, left, right, band_len) по полосам, где маска нашлась.
    """
    xc, dc = grid["xc"], grid["dc"]
    cx = grid["cell_x"]
    nx = len(xc)
    gap_cells = max(1, int(round(gap / cx)))
    fallback = _center_fn(prior)
    k, lmax = BAND_K[method], BAND_MAX[method]
    offs = (np.arange(nx) - nx // 2) * cx          # сетка смещений от оси
    ds, ls, rs, lens = [], [], [], []
    hist_d, hist_c = [], []
    i = int(np.searchsorted(dc, d_from))
    while i < len(dc):
        L = float(np.clip(k * dc[i] ** 2, grid["cell_d"], lmax))
        j = int(np.searchsorted(dc, dc[i] + L))
        j = max(j, i + 1)
        prof = np.zeros(nx)
        dsum, wsum = 0.0, 0.0
        for row in range(i, j):
            pred = _predict(hist_d, hist_c, dc[row], fallback)
            cols = np.where(mask[row] & (np.abs(xc - pred) < halfwin))[0]
            if len(cols) == 0:
                continue
            w = weight[row, cols] if weight is not None else np.ones(len(cols))
            o = np.clip(np.round((xc[cols] - pred) / cx).astype(int) + nx // 2, 0, nx - 1)
            np.add.at(prof, o, w)
            dsum += dc[row] * w.sum()
            wsum += w.sum()
        if wsum > 0:
            d_band = dsum / wsum
            pred = _predict(hist_d, hist_c, d_band, fallback)
            occ = np.where(prof > 0)[0]
            if mode == "mass":
                cum = np.cumsum(prof) / prof.sum()
                lo = int(np.searchsorted(cum, mass))
                hi = int(np.searchsorted(cum, 1 - mass))
            else:
                breaks = np.where(np.diff(occ) > gap_cells)[0]
                starts = np.r_[occ[0], occ[breaks + 1]]
                ends = np.r_[occ[breaks], occ[-1]]
                zero = nx // 2
                inside = (starts <= zero) & (ends >= zero)
                sel = int(np.where(inside)[0][0]) if inside.any() else \
                    int(np.argmin(np.minimum(np.abs(starts - zero), np.abs(ends - zero))))
                lo, hi = starts[sel], ends[sel]
            left = pred + offs[lo] - cx / 2
            right = pred + offs[min(hi, nx - 1)] + cx / 2
            if right - left >= min_width:
                ds.append(d_band)
                ls.append(left)
                rs.append(right)
                lens.append(L)
                hist_d.append(d_band)
                hist_c.append(0.5 * (left + right))
        i = j
    return np.array(ds), np.array(ls), np.array(rs), np.array(lens)


# ---------------------------------------------------------------- форма

VIS_GRID = np.arange(D_MIN, D_MAX + 0.5, 0.5)   # сетка глубин для расчёта видимости
HIDDEN_TOL = 0.05   # м: линия взгляда должна отойти от стены хотя бы на столько


def _design(d, set_id, n_sets, deg):
    t = d / DEPTH_SCALE
    cols = [(set_id == k).astype(float) for k in range(n_sets)]
    cols += [t ** j for j in range(1, deg + 1)]
    return np.column_stack(cols)


def _observed_edge(coef, n_sets, set_ids, d, side_of_set):
    """Какую кромку лидар ДОЛЖЕН увидеть при данной форме — с учётом заслона.

    На повороте внутренняя стена дальше точки касания не видна: луч, скользящий
    по ней, уходит в противоположную стену. Поэтому наблюдаемый левый край на
    глубине d — это не стена x_L(d), а d·tan(Φ(d)), где Φ(d) — наибольший угол
    на левую стену среди всех глубин до d (бегущий максимум). Пока стена видна,
    это она сама; дальше — прямая линия взгляда через точку касания. Справа то
    же с бегущим минимумом. На прямом тоннеле угол на стену монотонен, и
    формула совпадает со стеной на всей глубине.

    coef: (H, p) — сразу пачка гипотез. Возвращает (pred (H, n), hidden (H, n)).
    """
    coef = np.atleast_2d(coef)
    tg = VIS_GRID / DEPTH_SCALE
    base_g = sum(coef[:, n_sets + j:n_sets + j + 1] * tg[None, :] ** (j + 1)
                 for j in range(coef.shape[1] - n_sets))
    if np.isscalar(base_g):
        base_g = np.zeros((coef.shape[0], len(VIS_GRID)))
    td = d / DEPTH_SCALE
    base_d = sum(coef[:, n_sets + j:n_sets + j + 1] * td[None, :] ** (j + 1)
                 for j in range(coef.shape[1] - n_sets))
    if np.isscalar(base_d):
        base_d = np.zeros((coef.shape[0], len(d)))
    pred = np.empty((coef.shape[0], len(d)))
    hidden = np.zeros((coef.shape[0], len(d)), bool)
    for k in range(n_sets):
        m = set_ids == k
        if not m.any():
            continue
        wall_g = coef[:, k:k + 1] + base_g
        ang = np.arctan2(wall_g, VIS_GRID[None, :])
        if side_of_set[k] == "left":
            env = np.maximum.accumulate(ang, axis=1)
        else:
            env = np.minimum.accumulate(ang, axis=1)
        env_d = np.stack([np.interp(d[m], VIS_GRID, e) for e in env])
        wall_d = coef[:, k:k + 1] + base_d[:, m]
        ang_d = np.arctan2(wall_d, d[m][None, :])
        sight = d[m][None, :] * np.tan(env_d)
        # сравнение в метрах поперёк, а не в радианах: угол у самого сенсора
        # меняется круто, и интерполяция по сетке сама по себе дала бы «заслон»
        gap = (sight - wall_d) if side_of_set[k] == "left" else (wall_d - sight)
        hid = gap > HIDDEN_TOL
        pred[:, m] = np.where(hid, sight, wall_d)
        hidden[:, m] = hid
    return pred, hidden


def _msac(resid, w, tol):
    return float(np.sum(w * np.minimum(resid ** 2, tol ** 2)) / max(w.sum(), 1e-9))


def _choice_no_replace(p, k, take):
    """rng.choice(len(p), k, replace=False, p=p) — тот же алгоритм numpy, что и
    в Generator.choice (отбор с повторной выборкой совпавших), на равномерных
    числах take(m) из того же потока. Возвращает индексы."""
    p = p.copy()
    found = np.zeros(k, np.int64)
    n_uniq = 0
    while n_uniq < k:
        u = take(k - n_uniq)
        if n_uniq > 0:
            p[found[:n_uniq]] = 0
        cdf = np.cumsum(p)
        cdf /= cdf[-1]
        new = cdf.searchsorted(u, side="right")
        _, first = np.unique(new, return_index=True)
        first.sort()
        new = new.take(first)
        found[n_uniq:n_uniq + new.size] = new
        n_uniq += new.size
    return found


def _ransac_draws(rng, prob, idx_by, k_rest, n_hyp):
    """Случайные выборки всех гипотез RANSAC разом — те же, что давал цикл

        pick = [rng.choice(ix, p=prob[ix] / prob[ix].sum()) for ix in idx_by]
        rest = rng.choice(len(prob), k_rest, replace=False, p=prob)

    (экспер. 19: цикл занимал ~30 мс на степень). Generator.choice с весами
    берёт равномерное число из потока и ищет его в накопленной сумме весов, а
    без возвращения — повторяет выборку для совпавших. Поток читается пачкой
    (random(N) даёт ровно те же числа, что N вызовов подряд), гипотезы без
    совпадений внутри «остатка» считаются массивом, а редкие с совпадением —
    точным повтором алгоритма numpy, со сдвигом дальнейшего потока.
    """
    n_sets = len(idx_by)
    per = n_sets + k_rest
    cdf_set = []
    for ix in idx_by:
        q = prob[ix] / prob[ix].sum()
        c = q.cumsum()
        c /= c[-1]
        cdf_set.append(c)
    cdf_all = np.cumsum(prob)
    cdf_all /= cdf_all[-1]
    buf = [rng.random(n_hyp * per + 64)]
    pos = 0

    def take(m):
        nonlocal pos
        while pos + m > len(buf[0]):
            buf[0] = np.concatenate([buf[0], rng.random(max(m, 64))])
        out = buf[0][pos:pos + m]
        pos += m
        return out

    picks = np.zeros((n_hyp, n_sets), np.int64)
    rests = np.zeros((n_hyp, k_rest), np.int64)
    h = 0
    while h < n_hyp:
        m = n_hyp - h
        blk = take(m * per).reshape(m, per)
        P = np.column_stack([ix[c.searchsorted(blk[:, j], side="right")]
                             for j, (ix, c) in enumerate(zip(idx_by, cdf_set))])
        R = cdf_all.searchsorted(blk[:, n_sets:], side="right")
        Rs = np.sort(R, axis=1)
        dup = np.flatnonzero(np.any(Rs[:, 1:] == Rs[:, :-1], axis=1)) if k_rest > 1 else []
        ok = m if len(dup) == 0 else int(dup[0])
        picks[h:h + ok] = P[:ok]
        rests[h:h + ok] = R[:ok]
        # вернуть в поток всё, что взято сверх принятых гипотез
        pos -= (m - ok) * per
        h += ok
        if h < n_hyp:
            picks[h] = P[ok]
            pos += n_sets
            rests[h] = _choice_no_replace(prob, k_rest, take)
            h += 1
    return picks, rests


def _ransac(d, x, w, sid, n_sets, sides, deg, tol, n_hyp=400, seed=0):
    """Старт подгонки: гипотезы по минимальным выборкам, у каждой — наблюдаемая
    кромка с учётом заслона, побеждает наименьшая MSAC-цена. Нужен потому, что
    МНК-старт на станции, стрелке или переходе «прямая -> дуга» уже стоит между
    двумя структурами, и Тьюки от него сходится к компромиссу, а не к тоннелю."""
    rng = np.random.default_rng(seed)
    A = _design(d, sid, n_sets, deg)
    p = A.shape[1]
    # Выборка пропорционально весу, то есть длине полосы: иначе ближние 30 м,
    # нарезанные по 0.5 м, дают две трети кромок, и гипотезы, задевающие дальний
    # поворот, почти не выпадают.
    prob = w / w.sum()
    idx_by = [np.where(sid == k)[0] for k in range(n_sets)]
    picks, rests = _ransac_draws(rng, prob, idx_by, p - n_sets, n_hyp)
    S = np.sort(np.column_stack([picks, rests]), axis=1)
    distinct = np.all(S[:, 1:] != S[:, :-1], axis=1)
    dd = d[S]
    # по глубине выборка должна быть разнесена, иначе кривизна — шум
    spread = (dd.max(axis=1) - dd.min(axis=1)) >= 10.0
    good = S[distinct & spread]
    if not len(good):
        return None
    H = lstsq_batch(A[good], x[good])
    pred, _ = _observed_edge(H, n_sets, sid, d, sides)
    cost = np.sum(w[None, :] * np.minimum((x[None, :] - pred) ** 2, tol ** 2), axis=1)
    return H[int(np.argmin(cost))]


def _refine(d, x, w, sid, n_sets, sides, coef, scales=(0.4, 0.15), floor=0.05, n_iter=5):
    """Тьюки с убывающим масштабом по ВИДИМЫМ кромкам; скрытые заслоном кромки
    в решение не идут (они говорят о линии взгляда, а не о стене), но набор
    скрытых пересчитывается на каждом шаге — он зависит от формы."""
    A = _design(d, sid, n_sets, len(coef) - n_sets)
    for c in scales:
        for _ in range(n_iter):
            pred, hid = _observed_edge(coef, n_sets, sid, d, sides)
            r = x - pred[0]
            s = max(c, floor)
            u = r / (4.685 * s)
            tw = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0) * (~hid[0])
            ww = w * tw
            if ww.sum() < A.shape[1] + 2:
                return coef
            coef = np.linalg.lstsq(A * np.sqrt(ww)[:, None], x * np.sqrt(ww), rcond=None)[0]
    return coef


def _chain_reach(depths, support=2):
    """До какой глубины кромка НАБЛЮДАЕТСЯ: самая дальняя согласная кромка, у
    которой перед ней, в окне max(15 м, 0.3·d), есть ещё хотя бы support
    согласных. Одинокая согласная кромка далеко впереди — совпадение, а не
    наблюдение; окно растёт с глубиной вместе с длиной полосы (BAND_K).

    Не «непрерывно от сенсора»: на станции ближние 30-50 м кромки лежат на
    платформе и законно отброшены, а стены тоннеля за станцией видны и ложатся
    на форму до 130 м. Цепочка от сенсора обрывалась бы на платформе.
    """
    dd = np.sort(np.asarray(depths, float))
    for k in range(len(dd) - 1, support - 1, -1):
        win = max(15.0, 0.3 * dd[k])
        if np.sum((dd[:k] >= dd[k] - win)) >= support:
            return float(dd[k])
    return None


def fit_shared_shape(d_left, x_left, d_right, x_right, w_left=None, w_right=None,
                     max_deg=3, gain=0.85, tol=0.25, min_samples=10):
    """Обе кромки — одной формой: x = смещение_стороны + Σ a_j (d/50)^j.

    Степень выбирается как в §15: следующая степень берётся, только если
    сбивает цену хотя бы до gain от прежней. Кубика нужна не для красоты: на
    100+ м в кадр попадает переход «прямая -> дуга» и S-кривая целиком.

    Цена — MSAC по НАБЛЮДАЕМОЙ кромке (_observed_edge): внутренняя кромка
    поворота дальше точки касания обязана лечь на линию взгляда, и форма,
    которая это предсказывает, за неё не штрафуется. Без этого дальняя
    внутренняя кромка тянет дугу к прямой.

    Вес кромки — длина её полосы (в строках сетки, не больше 4).

    Дальность наблюдения — по ДАЛЬНЕЙ видимой кромке (_chain_reach): форма общая, и если
    правая стена видна до 120 м, ось до 120 м определена, даже когда левая
    спряталась за поворотом раньше.
    """
    sides_all = ("left", "right")
    d = np.r_[d_left, d_right].astype(float)
    x = np.r_[x_left, x_right].astype(float)
    w = np.r_[np.ones(len(d_left)) if w_left is None else w_left,
              np.ones(len(d_right)) if w_right is None else w_right].astype(float)
    sid = np.r_[np.zeros(len(d_left), int), np.ones(len(d_right), int)]
    n_sets, sides = 2, sides_all
    if len(d_left) < 3 or len(d_right) < 3:
        # одна кромка не наблюдается — форма по другой, смещение пропавшей неизвестно
        keep = 0 if len(d_left) >= len(d_right) else 1
        m = sid == keep
        d, x, w, sid = d[m], x[m], w[m], np.zeros(m.sum(), int)
        n_sets, sides = 1, (sides_all[keep],)
    if len(d) < min_samples:
        return None
    best = None
    for deg in range(1, max_deg + 1):
        # кубике нужна длинная база: иначе она объясняет шум, а не S-кривую
        if deg == 3 and np.percentile(d, 90) < 60:
            break
        coef = _ransac(d, x, w, sid, n_sets, sides, deg, tol)
        if coef is None:
            continue
        coef = _refine(d, x, w, sid, n_sets, sides, coef)
        pred, hid = _observed_edge(coef, n_sets, sid, d, sides)
        r = x - pred[0]
        cost = _msac(r, w, tol)
        cand = {"deg": deg, "coef": coef, "resid": r, "hidden": hid[0], "cost": cost}
        # Без досрочной остановки: «прямо вдоль платформы, потом поворот» дуга
        # описывает не лучше прямой, а кубика — хорошо. Если остановиться на
        # первой неудачной степени, до кубики дело не дойдёт.
        if best is None or cost < gain * best["cost"]:
            best = cand
    if best is None:
        return None
    coef, r, hid = best["coef"], best["resid"], best["hidden"]
    inl = np.abs(r) < tol
    if (inl & ~hid).sum() < min_samples:
        return None
    offsets = list(coef[:n_sets])
    if n_sets == 1:
        offsets = [offsets[0], np.nan] if sides[0] == "left" else [np.nan, offsets[0]]
    side_lbl = np.array([sides[k] for k in sid])
    reach = {name: _chain_reach(d[inl & ~hid & (side_lbl == name)]) for name in sides_all}
    both = [v for v in reach.values() if v is not None]
    vis = inl & ~hid
    return {
        "deg": best["deg"], "shape": coef[n_sets:], "offsets": offsets,
        "rms": float(np.sqrt(np.average(r[vis] ** 2, weights=w[vis]))),
        "cost": best["cost"], "inlier_frac": float(np.average(inl, weights=w)),
        "reach_side": reach, "reach": float(max(both)) if both else None,
        "samples": {"d": d, "x": x, "side": side_lbl, "inlier": inl, "hidden": hid},
    }


def shape_x(fit, depths, side="center"):
    """x(d) кромки или оси по результату fit_shared_shape."""
    depths = np.asarray(depths, float)
    t = depths / DEPTH_SCALE
    base = sum(a * t ** (j + 1) for j, a in enumerate(fit["shape"]))
    ol, orr = fit["offsets"]
    if side == "left":
        return ol + base
    if side == "right":
        return orr + base
    if np.isfinite(ol) and np.isfinite(orr):
        return 0.5 * (ol + orr) + base
    return np.full(depths.shape, np.nan)


def half_width(fit):
    ol, orr = fit["offsets"]
    return 0.5 * (orr - ol) if np.isfinite(ol) and np.isfinite(orr) else np.nan


def curvature(fit, depth):
    """Кривизна оси, 1/м, со знаком: плюс — поворот направо (x растёт)."""
    t = depth / DEPTH_SCALE
    s = fit["shape"]
    d1 = sum((j + 1) * a * t ** j for j, a in enumerate(s)) / DEPTH_SCALE
    d2 = sum((j + 1) * j * a * t ** (j - 1) for j, a in enumerate(s) if j >= 1) / DEPTH_SCALE ** 2
    return float(d2 / (1 + d1 ** 2) ** 1.5)


# ---------------------------------------------------------------- три подхода

def _edges_fit(mask, grid, method, weight=None, **kw):
    d, l, r, L = band_edges(mask, grid, method, weight=weight, **kw)
    wt = np.clip(L / grid["cell_d"], 1.0, 4.0)
    fit = fit_shared_shape(d, l, d, r, wt, wt)
    return (d, l, r, L), fit


def run_silhouette(grid, prior=None):
    img, mask = silhouette_mask(grid)
    edges, fit = _edges_fit(mask, grid, "silhouette", weight=img, mode="mass", prior=prior)
    return {"method": "silhouette", "image": img, "mask": mask, "edges": edges, "fit": fit}


def run_floor(grid, prior=None):
    level, ld, lz = floor_level(grid, _center_fn(prior))
    if level is None:
        return {"method": "floor", "image": grid["zmin"], "mask": None, "edges": None,
                "fit": None, "level": None, "level_samples": (ld, lz)}
    mask = floor_mask(grid, level)
    # Кусок пола уже метра — это не пол тоннеля, а верх шпалы или лотка,
    # случайно попавший в полосу высот.
    edges, fit = _edges_fit(mask, grid, "floor", gap=0.6, min_width=1.0, prior=prior)
    return {"method": "floor", "image": grid["zmin"], "mask": mask, "edges": edges,
            "fit": fit, "level": level, "level_samples": (ld, lz)}


def run_ceiling(grid, prior=None):
    ld, lz, seen = ceiling_level(grid, _center_fn(prior))
    mask = ceiling_mask(grid, ld, lz, seen)
    d_from = float(ld[seen].min()) if seen.any() else D_MAX
    edges, fit = _edges_fit(mask, grid, "ceiling", gap=1.0, min_width=0.8, d_from=d_from,
                            prior=prior)
    return {"method": "ceiling", "image": grid["zmax"], "mask": mask, "edges": edges,
            "fit": fit, "level_samples": (ld, lz, seen)}


RUNNERS = {"silhouette": run_silhouette, "floor": run_floor, "ceiling": run_ceiling}


def fit_views(points, priors=None, methods=METHODS):
    """Все три подхода на одном кадре. priors: {метод: fit прошлого кадра}."""
    grid = rasterize(points)
    priors = priors or {}
    return grid, {m: RUNNERS[m](grid, priors.get(m)) for m in methods}
