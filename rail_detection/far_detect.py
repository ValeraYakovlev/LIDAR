"""Эксперимент 18: габарит, скопления и подтверждение — поверх пути §31.

Работает на точках в координатах габарита (s, u, v): s — длина вдоль пути,
u — поперёк от середины между головками рельсов, v — над плоскостью головок
(с поворотом на крен), ровно как их считает `ParallelGauge`. Путь и стены не
трогаются.

Параметры варианта — словарь; `BASE` воспроизводит эталон wall-parallel-v1.
"""

import numpy as np

from .contrast_gauge import _voxel
from .gauge import ObstacleWatch

BASE = dict(half=1.10, top=3.30, bottom=0.25, low_half=None, v_step=None, near=2.0,
            eps=0.35, min_samples=12, min_points=25, min_extent=0.25,
            watch="nearest", confirm=3, tol=1.5)

# Кузов 81-717/714 по паспорту (§24): 2670 мм в ширину, 3650 мм над головками.
TRAIN = dict(half=1.335, top=3.65)
# Он же снизу: ниже 0.5 м — прежние ±1.1 м (контактный рельс стоит на u ≈ 1.25–1.5
# и до 0.45 м над головками, замер экспер. 18 на реальной и синтетической
# записи), низ — 0.10 м над головками (полоса 0.10–0.25 между рельсами пуста в
# 99% кадров чистых прогонов на всех дальностях, замер экспер. 18).
TRAIN_LOW = dict(half=1.335, top=3.65, low_half=1.10, v_step=0.50, bottom=0.10)
# Скопления с порогом по дальности и подтверждение каждого скопления.
FAR = dict(far=True, r0=20.0, eps=0.35, floor=3, frac=0.5, w_min=0.25, h_min=0.25,
           watch="multi", M=3, N=5)

VARIANTS = {
    "base": dict(BASE),
    "train": {**BASE, **TRAIN},
    "far": {**BASE, **FAR},
    "train_far": {**BASE, **TRAIN, **FAR},
    "tl_far": {**BASE, **TRAIN_LOW, **FAR},
}
for _k in (0.003, 0.005, 0.008, 0.012):
    VARIANTS[f"tl_far_m{int(_k * 1000)}"] = {**BASE, **TRAIN_LOW, **FAR, "m_slope": _k}
    VARIANTS[f"far_m{int(_k * 1000)}"] = {**BASE, **FAR, "m_slope": _k}
for _k in (0.005, 0.008):
    VARIANTS[f"tlv_m{int(_k * 1000)}"] = {**BASE, **TRAIN_LOW, **FAR, "m_slope": _k, "vprof": True}
    for _alt in (0.3, 0.5):
        VARIANTS[f"tlv_m{int(_k * 1000)}_a{int(_alt * 10)}"] = {
            **BASE, **TRAIN_LOW, **FAR, "m_slope": _k, "vprof": True, "alt_max": _alt}
        VARIANTS[f"tl_m{int(_k * 1000)}_a{int(_alt * 10)}"] = {
            **BASE, **TRAIN_LOW, **FAR, "m_slope": _k, "alt_max": _alt}
SLAB = (3.0, 1.0, 1.5)   # м: длиннее вдоль пути, уже поперёк, выше
VARIANTS["tlv_m8_a3_slab"] = {**VARIANTS["tlv_m8_a3"], "slab": SLAB}
VARIANTS["tlv_m8_a3_slab_c1"] = {**VARIANTS["tlv_m8_a3"], "slab": SLAB, "coast": 1}
VARIANTS["tlv_m8_a3_slab_m24"] = {**VARIANTS["tlv_m8_a3"], "slab": SLAB, "M": 2, "N": 4}
VARIANTS["tlv_m8_a3_slab_m35c1"] = {**VARIANTS["tlv_m8_a3"], "slab": SLAB, "coast": 1}
VARIANTS["tlv_m8_a3_slab_m46"] = {**VARIANTS["tlv_m8_a3"], "slab": SLAB, "M": 4, "N": 6}
for _d0 in (25, 30):
    VARIANTS[f"tlv_m8_a3_d{_d0}"] = {**VARIANTS["tlv_m8_a3"], "m_d0": float(_d0)}
    VARIANTS[f"tlv_m8_a3_d{_d0}_m46"] = {**VARIANTS["tlv_m8_a3"], "m_d0": float(_d0), "M": 4, "N": 6}
    VARIANTS[f"tlv_m6_a3_d{_d0}"] = {**VARIANTS["tlv_m8_a3"], "m_d0": float(_d0), "m_slope": 0.006}
# Итог эксперимента 18 (выбран на разработке, до отложенного замера):
# габарит кузова с вырезом под контактный рельс и низом 0.10 м; запас 0.008·(D − 40)
# — 90-й перцентиль расхождения гипотез пути на разработке; профиль по высоте из
# вида сбоку; предел пути там, где гипотезы разошлись больше 0.3 м; скопления в
# угловых координатах (eps 0.35 м — два шага колец по высоте на 80 м); подтверждение
# 3 из 5 кадров для каждого скопления.
FINAL = {**BASE, **TRAIN_LOW, **FAR, "r0": 80.0, "m_slope": 0.008, "m_d0": 40.0,
         "vprof": True, "alt_max": 0.3}
VARIANTS["final"] = FINAL
# Экспер. 18б (после отложенного замера, им не проверено): запас снизу отдельно.
for _kb in (0.0, 0.002, 0.004):
    VARIANTS[f"final_b{int(_kb * 1000)}"] = {**FINAL, "m_bottom_slope": _kb}
for _r0 in (50, 100):
    for _k in (0.005, 0.008):
        VARIANTS[f"tl_r{_r0}_m{int(_k * 1000)}"] = {**BASE, **TRAIN_LOW, **FAR, "r0": float(_r0),
                                                  "m_slope": _k}


# ---------------------------------------------------------------- вид сбоку

# Профиль пути по высоте — из вида сбоку (s, v) полосы над путём. Вблизи высоту
# держат головки рельсов (крен и уровень меряются до 34 м, §25), а дальше она
# продолжена прямой, проведённой по ближнему полу. Когда путь впереди уходит
# под уклон или на подъём, габарит эту прямую и продолжает: на стрелочном
# прогоне (кадры 815–851) свод на 150 м оказался на 1.5 м ниже, чем должен. Пол
# вдали почти не виден (скользящий взгляд), свод виден до 150 м — и идёт
# параллельно пути, как стены в плане (§31). Отсюда профиль: пол там, где он
# виден, и свод минус его высота над головками, измеренная вблизи.
VP_S0 = 35.0          # м: ближе профиль держат рельсы, поправка ноль
VP_BIN = 5.0          # м: полоса, по которой берётся уровень пола и свода
VP_MIN_PTS = 15       # точек в полосе, чтобы уровень считался измеренным
VP_REF = (15.0, 35.0) # м: где мерится высота свода над головками
VP_SIG_F, VP_SIG_C = 0.08, 0.25   # м: разброс уровня пола и свода по полосам
VP_MAX_GRADE = 0.06   # предел смены уклона (60 ‰: 40 ‰ — предел уклона в метро)
VP_MIN_RADIUS = 1500.0  # м: вертикальная кривая не круче
# Мягкий приор к «прямо»: смена уклона ~15 ‰, вертикальная кривая R ~ 4000 м.
# Без него парабола по трём-четырём полосам свода (синтетика видит свод до
# 50–70 м) улетала за пределами данных на 1–2 м (замер экспер. 18).
VP_PRIOR_A, VP_PRIOR_B = 0.015, 1.0 / (2 * 4000.0)


def _wpct(vals, w, q):
    c = np.cumsum(w)
    return float(vals[np.searchsorted(c, q / 100.0 * c[-1])])


def side_levels(H, s_edges, v_edges, bin_len=VP_BIN):
    """Уровни пола и свода по полосам длины bin_len из гистограммы вида сбоку.

    Пол — 90-й перцентиль точек ниже 0.8 м над головками (верх полотна —
    головки рельсов; шпалы и балласт ниже). Свод — 90-й перцентиль точек выше
    2.2 м (верхняя огибающая: вдали лидар видит снизу подвесное оборудование,
    нижняя кромка «свода» из-за этого опускается сама по себе, §34)."""
    vc = 0.5 * (v_edges[1:] + v_edges[:-1])
    lo_f = (vc >= -1.5) & (vc <= 0.8)
    hi_c = vc >= 2.2
    starts = np.arange(s_edges[0], s_edges[-1] - bin_len + 1e-6, bin_len)
    out = []
    for a in starts:
        i0, i1 = np.searchsorted(s_edges, [a, a + bin_len])
        col = H[i0:i1].sum(axis=0)
        f = c = np.nan
        if col[lo_f].sum() >= VP_MIN_PTS:
            f = _wpct(vc[lo_f], col[lo_f], 90)
        if col[hi_c].sum() >= VP_MIN_PTS:
            c = _wpct(vc[hi_c], col[hi_c], 90)
        out.append((a + bin_len / 2, f, c))
    return np.array(out)


def vertical_profile(H, s_edges, v_edges, limit):
    """Смещение пути по высоте δ(s) относительно прямой ближнего пола.

    δ = 0 до VP_S0; дальше a·(s − s0) + b·(s − s0)² — смена уклона и
    вертикальная кривая, с пределами на то и другое. Наблюдения: уровень пола
    (где виден) и уровень свода минус его высота над головками на VP_REF;
    робастно (Тьюки), чтобы ступенька свода (смена типа тоннеля, станция) и
    предмет на пути остались выбросами. Возвращает функцию s -> δ и сведения
    для рисования."""
    L = side_levels(H, s_edges, v_edges)
    sc, fl, ce = L[:, 0], L[:, 1], L[:, 2]
    ref = (sc >= VP_REF[0]) & (sc <= VP_REF[1]) & np.isfinite(ce)
    info = {"levels": L, "h_ceil": None, "a": 0.0, "b": 0.0}
    if ref.sum() < 2:
        return (lambda s: np.zeros_like(np.asarray(s, float))), info
    h_c = float(np.median(ce[ref] - np.where(np.isfinite(fl[ref]), fl[ref], 0.0)))
    info["h_ceil"] = h_c
    far = (sc > VP_S0) & (sc <= limit)
    xs, ys, sg = [], [], []
    for keep, val, sig in ((far & np.isfinite(fl), fl, VP_SIG_F),
                           (far & np.isfinite(ce), ce - h_c, VP_SIG_C)):
        xs.append(sc[keep] - VP_S0)
        ys.append(val[keep])
        sg.append(np.full(keep.sum(), sig))
    x, y, sig = np.concatenate(xs), np.concatenate(ys), np.concatenate(sg)
    if len(x) < 3:
        return (lambda s: np.zeros_like(np.asarray(s, float))), info
    A = np.column_stack([x, x ** 2])
    w = 1.0 / sig ** 2
    coef = np.zeros(2)
    bmax = 1.0 / (2.0 * VP_MIN_RADIUS)
    P = np.diag([1.0 / VP_PRIOR_A ** 2, 1.0 / VP_PRIOR_B ** 2])
    for _ in range(8):
        Aw = A * w[:, None]
        coef = np.linalg.solve(Aw.T @ A + P, Aw.T @ y)
        coef = np.array([np.clip(coef[0], -VP_MAX_GRADE, VP_MAX_GRADE),
                         np.clip(coef[1], -bmax, bmax)])
        r = (y - A @ coef) / sig
        w = np.where(np.abs(r) < 4.685, (1 - (r / 4.685) ** 2) ** 2, 0.0) / sig ** 2
        if w.sum() == 0:
            break
    a, b = float(coef[0]), float(coef[1])
    # за последней полосой, которая голосовала, профиль не продолжается —
    # держится на последнем значении (как путь за пределом стен, §31)
    t_max = float(x[w > 0].max()) if (w > 0).any() else 0.0
    info.update({"a": a, "b": b, "s_max": VP_S0 + t_max})

    def delta(s):
        t = np.clip(np.asarray(s, float) - VP_S0, 0.0, t_max)
        return a * t + b * t ** 2

    return delta, info


def margin(s, p):
    """Запас на погрешность пути и отсчёта высоты: ноль до D0 (докуда путь и
    крен держат рельсы), дальше растёт пропорционально дальности — ошибка
    курса, кривизны и тангажа даёт смещение, пропорциональное плечу."""
    k = p.get("m_slope", 0.0)
    return k * np.maximum(0.0, s - p.get("m_d0", 40.0)) if k else 0.0


def margin_bottom(s, p):
    """Запас снизу — свой (экспер. 18б): по умолчанию тот же, что с боков и сверху."""
    if "m_bottom_slope" not in p:
        return margin(s, p)
    k = p["m_bottom_slope"]
    return k * np.maximum(0.0, s - p.get("m_d0", 40.0)) if k else 0.0


def gauge_mask(s, u, v, limit, p):
    """Точки внутри габарита: прямоугольник кузова, снизу — по желанию уже
    (вырез под контактный рельс). Вдали габарит сжат на запас margin(s) со
    всех сторон: находкой считается только то, что зашло в габарит глубже
    погрешности, с которой мы знаем, где он проходит."""
    au = np.abs(u)
    mg = margin(s, p)
    m = (v >= p["bottom"] + margin_bottom(s, p)) & (v <= p["top"] - mg) & (s > p["near"]) & \
        (s <= limit)
    if p.get("low_half") is None:
        return m & (au <= p["half"] - mg)
    low = v < p["v_step"]
    return m & np.where(low, au <= p["low_half"] - mg, au <= p["half"] - mg)


def clusters(s, u, v, p):
    """Скопления DBSCAN в габарите, в координатах (s, u, v)."""
    from sklearn.cluster import DBSCAN

    pts = np.column_stack([s, u, v])
    if len(pts) < p["min_samples"]:
        return []
    small = _voxel(pts)
    if len(small) < p["min_samples"]:
        return []
    lab = DBSCAN(eps=p["eps"], min_samples=p["min_samples"]).fit_predict(small)
    out = []
    for L in np.unique(lab[lab >= 0]):
        m = lab == L
        if m.sum() < p["min_points"]:
            continue
        c = small[m]
        size = c.max(axis=0) - c.min(axis=0)
        if size.max() < p["min_extent"]:
            continue
        out.append({"dist": float(c[:, 0].min()), "n": int(m.sum()),
                    "size": tuple(float(t) for t in size),
                    "u": float(np.median(c[:, 1])), "v": float(np.median(c[:, 2]))})
    out.sort(key=lambda q: q["dist"])
    return out


# Разрешение лидара (§29, замер эксперимента 18): 1200 столбцов на 100° по
# азимуту и ~0.125° между каналами у горизонта. Шаг между лучами на дальности D:
AZ_STEP = np.radians(100.0 / 1200)      # 1.45 мрад: 0.145 м на 100 м
EL_STEP = np.radians(0.125)             # 2.2 мрад: 0.22 м на 100 м


def angular_coords(s, u, v, r0):
    """Координаты, в которых шаг между лучами не растёт с дальностью.

    Ближе r0 — как есть (там шаг лучей мельче вокселя 5 см). Дальше поперечные
    координаты сжимаются в r0/s раз, а дальность идёт по логарифму с той же
    производной r0/s: локально это равномерное сжатие, и DBSCAN с постоянным
    eps в этих координатах связывает соседние лучи на любой дальности, как
    связывает их вблизи.
    """
    c = np.minimum(1.0, r0 / np.maximum(s, 1e-3))
    sa = np.where(s <= r0, s, r0 + r0 * np.log(np.maximum(s, r0) / r0))
    return np.column_stack([sa, u * c, v * c])


def expected_hits(D, w, h):
    """Сколько лучей ложится на предмет w × h (м) на дальности D (не меньше
    одного по каждой оси — луч либо попал, либо нет)."""
    return np.maximum(1.0, w / (D * AZ_STEP)) * np.maximum(1.0, h / (D * EL_STEP))


def clusters_far(s, u, v, p):
    """Скопления с порогом по дальности: DBSCAN в угловых координатах, порог
    числа вокселей — доля от ожидаемого числа лучей на самый маленький предмет,
    который мы обязуемся видеть, но не больше порога эталона."""
    from sklearn.cluster import DBSCAN

    if len(s) < p["floor"]:
        return []
    small = _voxel(np.column_stack([s, u, v]))
    if len(small) < p["floor"]:
        return []
    A = angular_coords(small[:, 0], small[:, 1], small[:, 2], p["r0"])
    lab = DBSCAN(eps=p["eps"], min_samples=p["floor"]).fit_predict(A)
    out = []
    for L in np.unique(lab[lab >= 0]):
        m = lab == L
        c = small[m]
        D = float(c[:, 0].min())
        need = min(p["min_points"], max(p["floor"],
                                        p["frac"] * expected_hits(D, p["w_min"], p["h_min"])))
        if m.sum() < need:
            continue
        size = c.max(axis=0) - c.min(axis=0)
        # предмет имеет протяжённость поперёк луча хотя бы в один шаг лучей;
        # плоский «налёт» вдоль стены — много точек на тонком слое (§24)
        if p.get("min_extent") and m.sum() >= p["min_points"] and size.max() < p["min_extent"]:
            continue
        # Стена, срезающая габарит наискось, где путь вдали ошибся (раструб
        # двухпутного): вертикальная плоскость вдоль пути — длинная по s, высокая
        # и узкая поперёк. Предмет на пути обращён к поезду лицом: вдоль пути он
        # короткий, поперёк — сколько сам шириной.
        if p.get("slab") and size[0] > p["slab"][0] and size[1] < p["slab"][1] \
                and size[2] > p["slab"][2]:
            continue
        out.append({"dist": D, "n": int(m.sum()), "need": float(need),
                    "size": tuple(float(t) for t in size),
                    "u": float(np.median(c[:, 1])), "v": float(np.median(c[:, 2]))})
    out.sort(key=lambda q: q["dist"])
    return out


class TrackWatch:
    """Подтверждение КАЖДОГО скопления, а не только ближайшего.

    Правило §24 то же: неподвижный предмет приближается ровно на Δs за кадр.
    Скопление ведётся в координатах тоннеля: S — пройденный путь (сумма Δs),
    положение предмета S + dist, поперёк u, по высоте v. Подтверждено, если в
    последних N кадрах скопление найдено на своём месте хотя бы M раз: вдали на
    предмет падает 1–4 луча, и в части кадров он выпадает просто по фазе
    развёртки, — правило «три кадра подряд» такой предмет не подтверждает никогда.
    """

    def __init__(self, M=3, N=5, tol_s=1.5, tol_s_rel=0.03, tol_uv=0.6, tol_uv_rel=0.004,
                 coast=0):
        self.M, self.N = M, N
        self.tol_s, self.tol_s_rel = tol_s, tol_s_rel
        self.tol_uv, self.tol_uv_rel = tol_uv, tol_uv_rel
        self.coast = coast
        self.reset()

    def reset(self):
        self.S = 0.0
        self.tracks = []

    def update(self, dets, ds):
        """dets — скопления кадра (dist, u, v); ds — Δs. Возвращает список
        подтверждённых дальностей на этом кадре."""
        if ds is not None and np.isfinite(ds):
            self.S += float(ds)
        for t in self.tracks:
            t["hits"].append(False)
        free = list(self.tracks)
        for q in sorted(dets, key=lambda q: q["dist"]):
            g = self.S + q["dist"]
            ts = max(self.tol_s, self.tol_s_rel * q["dist"])
            tuv = self.tol_uv + self.tol_uv_rel * q["dist"]
            best, bd = None, np.inf
            for t in free:
                ds_ = abs(g - t["g"])
                if ds_ < ts and abs(q["u"] - t["u"]) < tuv and abs(q["v"] - t["v"]) < tuv:
                    d = ds_ / ts + abs(q["u"] - t["u"]) / tuv
                    if d < bd:
                        best, bd = t, d
            if best is None:
                self.tracks.append({"g": g, "u": q["u"], "v": q["v"], "hits": [True]})
            else:
                free.remove(best)
                best["g"], best["u"], best["v"] = g, q["u"], q["v"]
                best["hits"][-1] = True
        for t in self.tracks:
            t["hits"] = t["hits"][-self.N:]
        self.tracks = [t for t in self.tracks if any(t["hits"])]
        out = []
        for t in self.tracks:
            if sum(t["hits"]) < self.M:
                continue
            last = max(i for i, h in enumerate(t["hits"]) if h)
            if len(t["hits"]) - 1 - last <= self.coast:
                out.append(t["g"] - self.S)
        return sorted(out)


ALT_D = np.arange(10.0, 151.0, 10.0)   # глубины, на которых кэш хранит расхождение гипотез


def alt_limit(alt_dx, lim):
    """Предел, дальше которого путь не определён: первая глубина, где «память» и
    «заново» разошлись поперёк больше lim (м). Нет второй гипотезы — без предела."""
    if alt_dx is None or not np.isfinite(alt_dx).any():
        return np.inf
    bad = np.flatnonzero(np.abs(np.nan_to_num(alt_dx)) > lim)
    return float(ALT_D[bad[0]] - 10.0) if len(bad) else np.inf


def run_sequence(p, n, getpts, limit, ds, getside=None, alt_dx=None):
    """Проход по записи: на кадр — (сырые дальности, подтверждённые дальности).

    getpts(k) -> (s, u, v) или None (кадр без геометрии); limit[k] — предел пути;
    ds[k] — Δs, которым трекер перенёс состояние; getside(k) -> (H, s_edges,
    v_edges) — вид сбоку для профиля по высоте; alt_dx[k] — расхождение гипотез
    пути на глубинах ALT_D.
    """
    if p["watch"] == "nearest":
        watch = ObstacleWatch(confirm=p["confirm"], tol=p["tol"])
    else:
        watch = TrackWatch(M=p["M"], N=p["N"], coast=p.get("coast", 0))
    out = []
    for k in range(n):
        P = getpts(k)
        if P is None:
            watch.reset()
            out.append(([], []))
            continue
        s, u, v = P
        lim = limit[k]
        if p.get("alt_max") and alt_dx is not None:
            lim = min(lim, alt_limit(alt_dx[k], p["alt_max"]))
        if p.get("vprof") and getside is not None:
            H, se, ve = getside(k)
            delta, _ = vertical_profile(H, se, ve, lim)
            v = v - delta(s)
        m = gauge_mask(s, u, v, lim, p)
        if p.get("far"):
            cl = clusters_far(s[m], u[m], v[m], p)
        else:
            cl = clusters(s[m], u[m], v[m], p)
        if p["watch"] == "nearest":
            conf = watch.update(cl[0]["dist"] if cl else None, ds[k])
            conf = [conf] if conf else []
        else:
            conf = watch.update(cl, ds[k])
        out.append(([c["dist"] for c in cl], conf))
    return out


# ---------------------------------------------------------------- на живом потоке

# Коробка и вид сбоку — ровно как в кэше эксперимента 18 (exp_far_cache.py):
# сначала прореживание широкой коробки вокруг пути, потом габарит. Так находки
# на живом потоке совпадают с замером на кэше кадр в кадр.
BOX_U, BOX_V, BOX_S = 2.2, (-0.15, 4.5), (1.0, 155.0)
SIDE_U = 1.0
SIDE_S = np.arange(0.0, 156.0, 1.0)
SIDE_V = np.arange(-2.0, 6.55, 0.05)


class FarDetector:
    """Габарит, скопления и подтверждение эксперимента 18 поверх `ParallelGauge`.

    update(res) берёт результат `ParallelGauge.update` (точки в координатах
    габарита, предел пути, Δs, проигравшую гипотезу пути) и возвращает находки
    кадра: сырые скопления, подтверждённые дальности и всё, что нужно рисовать.
    """

    def __init__(self, p):
        self.p = p
        self.watch = TrackWatch(M=p["M"], N=p["N"], coast=p.get("coast", 0))

    def reset(self):
        self.watch.reset()

    def update(self, res):
        from .contrast_gauge import PATH_GRID
        from .parallel_path import to_path_dict

        p = self.p
        if res is None:
            self.reset()
            return None
        s, ug, vg = res["s"], res["ug"], res["vg"]
        tr = res["track"]
        lim = res["limit"]
        alt_dx = None
        if tr.get("alt") is not None:
            a = to_path_dict(tr["alt"]["curve"], PATH_GRID, "")
            alt_dx = (np.interp(ALT_D, a["d"], a["x"])
                      - np.interp(ALT_D, res["path"]["d"], res["path"]["x"]))
        if p.get("alt_max"):
            lim = min(lim, alt_limit(alt_dx, p["alt_max"]))
        ms = np.abs(ug) <= SIDE_U
        H, _, _ = np.histogram2d(s[ms], vg[ms], bins=[SIDE_S, SIDE_V])
        delta, vinfo = vertical_profile(H, SIDE_S, SIDE_V, lim)
        m = (np.abs(ug) <= BOX_U) & (vg >= BOX_V[0]) & (vg <= BOX_V[1]) & \
            (s > BOX_S[0]) & (s <= BOX_S[1])
        vox = _voxel(np.column_stack([s[m], ug[m], vg[m]]))
        # те же float16, что в кэше: иначе на границе габарита расходятся точки
        S0 = vox[:, 0].astype(np.float32)
        U0 = vox[:, 1].astype(np.float16).astype(np.float32)
        V0 = vox[:, 2].astype(np.float16).astype(np.float32)
        dv = delta(S0) if p.get("vprof") else 0.0
        V1 = V0 - dv
        g = gauge_mask(S0, U0, V1, lim, p)
        cl = clusters_far(S0[g], U0[g], V1[g], p)
        conf = self.watch.update(cl, tr["ds"])
        return {"clusters": cl, "confirmed": conf, "limit": lim, "alt_dx": alt_dx,
                "delta": delta if p.get("vprof") else None, "vinfo": vinfo,
                "side": H, "gauge_pts": np.column_stack([S0[g], U0[g], V1[g]])}
