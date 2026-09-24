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
# Скопления с порогом по дальности и подтверждение каждого скопления.
FAR = dict(far=True, r0=20.0, eps=0.35, floor=3, frac=0.5, w_min=0.25, h_min=0.25,
           watch="multi", M=3, N=5)

VARIANTS = {
    "base": dict(BASE),
    "train": {**BASE, **TRAIN},
    "far": {**BASE, **FAR},
    "train_far": {**BASE, **TRAIN, **FAR},
}


def gauge_mask(s, u, v, limit, p):
    """Точки внутри габарита: прямоугольник кузова, снизу — по желанию уже
    (вырез под контактный рельс)."""
    au = np.abs(u)
    m = (v >= p["bottom"]) & (v <= p["top"]) & (s > p["near"]) & (s <= limit)
    if p.get("low_half") is None:
        return m & (au <= p["half"])
    low = v < p["v_step"]
    return m & np.where(low, au <= p["low_half"], au <= p["half"])


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


def run_sequence(p, n, getpts, limit, ds):
    """Проход по записи: на кадр — (сырые дальности, подтверждённые дальности).

    getpts(k) -> (s, u, v) или None (кадр без геометрии); limit[k] — предел пути;
    ds[k] — Δs, которым трекер перенёс состояние.
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
        m = gauge_mask(s, u, v, limit[k], p)
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
