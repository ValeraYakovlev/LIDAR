"""Путь и стены — одна кривая: стены на постоянном расстоянии от пути.

Эксперимент 17 (ветка `feature/wall-parallel-path`). Две претензии к §27–§28,
обе видны на GIF глазами:

* **стены прыгают.** Каждый кадр подгоняется с нуля: степень формы (прямая,
  дуга, кубика) выбирается заново, дальность наблюдения скачет 78 → 127 → 85 м,
  и зелёная линия на соседних кадрах — разные кривые;
* **путь смотрит в стену.** Путь склеен из прямой по рельсам до 25 м и изгиба
  контраста дальше. Наклон прямой по 20 м рельсов и наклон оси тоннеля в точке
  склейки не обязаны совпадать, и их разница умножается на плечо: 0.5° дают
  0.9 м на 100 м — ровно столько, чтобы коридор въехал в стену.

## Модель

Путь — плоская кривая, заданная так, как её строят путейцы: профилем кривизны
по длине пути κ(s) (кусочно-линейный, узлы через 10 м — прямые, переходные
кривые и дуги им описываются) и позой вагона относительно пути (боковое
смещение y0 и угол ψ0 в точке под сенсором). Стены — не отдельные кривые, а
ОТСТУПЫ от пути ПО НОРМАЛИ: путь идёт на одном и том же расстоянии от каждой
стены и смотреть в стену не может по построению.

Отступ каждой стены постоянный, с не более чем двумя ступеньками (MAX_STEPS):
смена сечения — станция кончилась, круглый тоннель стал двухпутным, камера
съезда — описывается ступенькой, а плавное расхождение путь-стена, растущее с
расстоянием, — только кривизной. Ближе W_MIN = 1.3 м к пути стены не бывает.
Кромки, не согласные с отступом (ниша, ответвление стрелки, раструб), отбрасы-
ваются Тьюки как выбросы.

Рельсы — это точки с отступом 0: они держат путь вблизи (y0, ψ0 и кривизну
первых 25 м), стены — кривизну до 150 м.

Почему не произвольный кусочный отступ — это замер (knowledge.md §31): ошибка
кривизны на дальнем участке и одинаковый сдвиг отступов обеих стен дают почти
одну и ту же картину кромок, и подгонка находила самосогласованные неверные
состояния — путь загнут, стены «лесенкой» под него. Одна-две ступеньки высотой
от полуметра плавную ошибку кривизны изобразить не могут.

## Связь кадров

Узлы профиля кривизны привязаны к ТОННЕЛЮ, а не к кадру: за кадр поезд
проходит Δs, и узлы сдвигаются на −Δs. Перенос точный — без поворотов облака
или готовой кривой и без накопления ошибки курса. Поза вагона относительно пути
за кадр почти не меняется, отступы стен — тоже.

Δs, противоречащий инерции поезда, не принимается (DS_JUMP). Перенесённое
прошлое состояние входит в подгонку мягким приором ФИКСИРОВАННОЙ
силы — это память на кадр-два, а не накопленная уверенность: фильтр Калмана с
растущей уверенностью в этой задаче запирался в неверных состояниях (ошибки
кромок систематические, а считаются независимыми, и фильтр становится
самоуверенным).

Каждый кадр подгонка стартует дважды: из перенесённого прошлого состояния и
заново — из пути этого кадра, построенного без памяти (§27), с поиском
ступенек. Берётся решение с меньшей ценой, а расхождение с памятью в цене
ограничено сверху (PRIOR_CAP, он же гистерезис смены гипотезы). Так прошлое
сглаживает дрожание, но запереть трекер не может: если оно неверно, свежий
старт даёт меньшую цену и побеждает. Ступенька привязана к тоннелю и
подъезжает к поезду на Δs за кадр; память её только уточняет, находит новую
только свежий старт.
"""

import numpy as np

from .curvature import ransac_poly_fit
from .views import HIDDEN_TOL, _chain_reach

S_MAX = 165.0          # м длины пути в состоянии: хватает на D_MAX = 150 м глубины
DS = 0.25              # м, шаг численного интегрирования
SG = np.arange(0.0, S_MAX + DS / 2, DS)
KAPPA_STEP = 10.0

# Точности измерений (1σ). Кромка контраста — клетка 0.1 м и полоса строк,
# сдвинутых на предсказанную ось; центр колеи по рельсам — сантиметры (§18).
SIGMA_EDGE = 0.15
SIGMA_RAIL = 0.03
TUKEY = 4.685          # отсечка Тьюки в σ: кромка дальше 0.7 м от стены модели не голосует
RAIL_MAX = 25.0        # м: дальше детектор рельсов хватает чужую пару (§25)
# Рельсы отбрасываются, только если дальше 0.8 м от пути: так ошибается
# детектор, схвативший чужую пару (соседний путь — в 3.5-4 м). Отсечка по
# σ = 3 см выбросила бы и свои рельсы, стоит пути уйти на 15 см, — и путь,
# потеряв опору, дальше шёл бы только по стенам.
RAIL_GATE = 0.8
# До подгонки рельсовые срезы чистятся между собой: RANSAC-парабола с допуском
# 8 см, как прямая у §27 (парабола — потому что на кривой R = 400 м прямая на
# 21 м расходится с дугой на 14 см). Иначе срез на 22.5 м, через кадр
# прыгающий на соседний пик (−0.6 ↔ −1.1 м на стоящем поезде), проходит в
# допуск 0.8 м и качает курс пути.
RAIL_TOL = 0.08
# Вес кромки — в метрах стены, а не в полосах: вблизи полоса 0.5 м, и сотня
# ближних кромок, посчитанных независимыми, перевешивала бы рельсы, хотя их
# ошибки (неровная стена, кабели, ниши) коррелированы на метры вдоль стены.
EDGE_CORR = 2.0        # м: длина, на которой ошибки кромок считаются независимыми
EDGE_W_MIN, EDGE_W_MAX = 0.25, 3.0

# Гладкость кривизны (1σ на разность соседних узлов через 10 м). Переходная
# кривая набирает 1/R = 2.5e-3 (R = 400 м) за 40-60 м — это 4-6e-4 за 10 м;
# вторая разность — излом профиля, он бывает только на стыках элементов.
SIGMA_DK1 = 5.0e-4
SIGMA_DK2 = 2.0e-4

# Приор из прошлого кадра (1σ), фиксированной силы. Кривизна узла, впервые
# попавшего в поле зрения, неизвестна (PRIOR_NEW_KAPPA).
PRIOR_Y0 = 0.05
PRIOR_PSI0 = 3.0e-3
PRIOR_KAPPA = 2.0e-4
PRIOR_NEW_KAPPA = 3.0e-3
PRIOR_WIDTH = 0.15
# Предел штрафа за расхождение с памятью при выборе старта — он же гистерезис
# смены гипотезы. В камере съезда гипотезы «стена у поезда 7.5 м» и «2.0 м»
# различались по цене на 7-17 единиц (уровень шума кадра), и при пределе 30 выбор
# перескакивал: 18 необъяснённых скачков стены у поезда на 1943 кадрах
# разработки, при 60 — 8 (у базы §28 — 160). Настоящая смена сечения даёт
# выигрыш ~190 (двухпутный прогон) и проходит при любом из двух.
PRIOR_CAP = 60.0

REACH_HISTORY = 5      # кадров в медиане дальности наблюдения стены

# Смена сечения: у каждой стены не больше MAX_STEPS ступенек отступа. Ступенька
# не может изобразить плавную ошибку кривизны — расхождение, растущее с
# расстоянием, — а смену сечения (станция кончилась, круглый тоннель стал
# двухпутным) описывает. Произвольный кусочный отступ вырожден: «лесенка» под
# загнутый путь (§31). Двух хватает на камеру съезда: у поезда стена в 2 м,
# дальше камера в 7.5 м, за ней снова тоннель; с одной ступенькой две неполные
# гипотезы чередовались, и стена у поезда мигала 2 ↔ 7.5 м. Ступенька привязана
# к тоннелю и едет к поезду на Δs за кадр; память её только уточняет, найти
# новую может только свежий старт. Цена — как у ~4 единиц веса кромок, ставших
# из выбросов согласными (≈ 8 м ближней стены или 2 дальние полосы); высота —
# от STEP_MIN.
MAX_STEPS = 2
STEP_COST = 15.0
STEP_MIN = 0.5
STEP_GRID = 2.5        # м, шаг перебора положения ступеньки
# Ближе W_MIN к оси пути стены не бывает: полуширина габарита 1.1 м плюс
# запас. Без этого ограничения модель подгоняла к разбросанным кромкам раструба
# двухпутного тоннеля «ступеньку внутрь» — стену в 0.2 м от пути. И это же
# условие не даёт предмету в габарите «объясниться» стеной.
W_MIN = 1.3
FRESH_WIDTH_SIGMA = 2.0   # м: слабый приор отступа у свежего старта (сторона без
                          # кромок иначе не определена; замер: −749 м на стрелке)

# Δs, противоречащий инерции поезда, не принимается: за 0.1 с скорость меняется
# не больше чем на 0.15 м/с (1.5 м/с²), то есть Δs — на 0.015 м. В камере съезда
# измеритель §20 давал 0.1 м через кадр при настоящих 1.1 м — перенос состояния
# на такой Δs уводит ступеньки и кривизну на метр за кадр.
DS_JUMP = 0.3
DS_HISTORY = 5


# ---------------------------------------------------------------- базис

def _hat(grid, knots):
    """Матрица кусочно-линейной интерполяции: значения в узлах -> на сетке.
    За крайними узлами значение держится постоянным."""
    grid = np.asarray(grid, float)
    n = len(knots)
    idx = np.clip(np.searchsorted(knots, grid, side="right") - 1, 0, n - 2)
    t = np.clip((grid - knots[idx]) / (knots[idx + 1] - knots[idx]), 0.0, 1.0)
    B = np.zeros((len(grid), n))
    rows = np.arange(len(grid))
    B[rows, idx] = 1 - t
    B[rows, idx + 1] += t
    return B


def _cumint(f):
    """Накопленный интеграл по SG методом трапеций (по первой оси)."""
    out = np.zeros_like(f, dtype=float)
    out[1:] = np.cumsum(0.5 * (f[1:] + f[:-1]) * DS, axis=0)
    return out


def _gather(table, s):
    """Линейная интерполяция столбцов таблицы на сетке SG в точках s."""
    s = np.clip(np.asarray(s, float), 0.0, SG[-1])
    i = np.minimum((s / DS).astype(int), len(SG) - 2)
    t = (s - SG[i]) / DS
    if table.ndim == 1:
        return table[i] * (1 - t) + table[i + 1] * t
    return table[i] * (1 - t)[:, None] + table[i + 1] * t[:, None]


def _diff(n, order):
    D = np.eye(n)
    for _ in range(order):
        D = D[1:] - D[:-1]
    return D


# ---------------------------------------------------------------- состояние

class TrackState:
    """Параметры пути и стен: m = [y0, ψ0, κ(узлы sk), отступы].

    Узлы sk — длина пути ОТ ТЕКУЩЕГО положения сенсора; они сдвигаются вместе
    с поездом. new — узлы, впервые попавшие в состояние при последнем переносе.
    Отступы — по (MAX_STEPS + 1) на сторону: отрезок 0 у поезда, дальше — за
    каждой ступенькой; sb[сторона] — где ступеньки (по возрастанию, inf — нет;
    отступ неиспользуемого отрезка равен последнему используемому).
    """

    NW = MAX_STEPS + 1

    def __init__(self, m, sk, new=None, sb=None):
        self.m = np.asarray(m, float)
        self.sk = np.asarray(sk, float)
        self.new = np.zeros(len(sk), bool) if new is None else np.asarray(new, bool)
        self.sb = (np.full((2, MAX_STEPS), np.inf) if sb is None
                   else np.array(sb, float).reshape(2, MAX_STEPS))

    @property
    def nk(self):
        return len(self.sk)

    @property
    def ik(self):
        return slice(2, 2 + self.nk)

    @property
    def npath(self):
        return 2 + self.nk

    def iw(self, side, seg=0):
        return 2 + self.nk + side * self.NW + seg

    def widths(self, m=None):
        m = self.m if m is None else m
        return m[self.npath:].reshape(2, self.NW)

    def nsteps(self, side):
        return int(np.isfinite(self.sb[side]).sum())

    def segment(self, side, s):
        return np.searchsorted(self.sb[side], s, side="right")

    def with_m(self, m, sb=None):
        return TrackState(m, self.sk, self.new, self.sb if sb is None else sb)

    @staticmethod
    def fresh(y0=0.0, psi0=0.0, wl=-2.5, wr=2.5):
        sk = np.arange(0.0, S_MAX + KAPPA_STEP - 1e-6, KAPPA_STEP)
        w = np.r_[np.full(TrackState.NW, wl), np.full(TrackState.NW, wr)]
        return TrackState(np.r_[y0, psi0, np.zeros(len(sk)), w], sk)

    def advance(self, ds):
        """Перенос на Δs вперёд: узлы сдвигаются, ушедшие за спину выпадают
        (один остаётся — для интерполяции в s = 0), впереди добавляются новые
        со значением последнего. Ступенька, доехавшая до поезда, исчезает, и
        отступ за ней становится отступом у поезда."""
        sk = self.sk - ds
        keep = np.r_[sk[1:] > 0, True]
        k = self.m[self.ik][keep]
        sk = sk[keep]
        add = np.arange(sk[-1] + KAPPA_STEP, S_MAX + KAPPA_STEP - 1e-6, KAPPA_STEP)
        W = self.widths().copy()
        sb = self.sb - ds
        for j in (0, 1):
            while sb[j][0] <= 0:
                W[j] = np.r_[W[j][1:], W[j][-1]]
                sb[j] = np.r_[sb[j][1:], np.inf]
        m = np.r_[self.m[:2], k, np.full(len(add), k[-1]), W.ravel()]
        new = np.r_[np.zeros(len(sk), bool), np.ones(len(add), bool)]
        return TrackState(m, np.r_[sk, add], new, sb)

    def prior_sigma(self):
        return np.r_[PRIOR_Y0, PRIOR_PSI0,
                     np.where(self.new, PRIOR_NEW_KAPPA, PRIOR_KAPPA),
                     np.full(2 * self.NW, PRIOR_WIDTH)]


def _tie_widths(st, m):
    """Неиспользуемые отрезки — равны последнему используемому; стена не ближе
    W_MIN к пути."""
    W = st.widths(m).copy()
    for j in (0, 1):
        n = st.nsteps(j)
        W[j][n + 1:] = W[j][n]
        W[j] = np.minimum(W[j], -W_MIN) if j == 0 else np.maximum(W[j], W_MIN)
    m = m.copy()
    m[st.npath:] = W.ravel()
    return m


# ---------------------------------------------------------------- геометрия

def track_curve(st, m=None):
    """Путь по состоянию: точки (x, d), курс ψ, длина s на сетке SG.

    ψ(s) = ψ0 + ∫κ,  x = y0 + ∫sin ψ,  d = ∫cos ψ — точное интегрирование, без
    приближения малых углов (на 130 м при курсе 10° оно ошибалось бы на 0.1 м).
    """
    m = st.m if m is None else m
    BK = _hat(SG, st.sk)
    kap = BK @ m[st.ik]
    psi = m[1] + _cumint(kap)
    x = m[0] + _cumint(np.sin(psi))
    d = _cumint(np.cos(psi))
    W = st.widths(m)
    steps = [[(float(st.sb[j][g]), float(W[j][g + 1])) for g in range(st.nsteps(j))]
             for j in (0, 1)]
    return {"s": SG, "x": x, "d": d, "psi": psi, "kappa": kap, "BK": BK,
            "wl": W[0][st.segment(0, SG)], "wr": W[1][st.segment(1, SG)],
            "w0": (float(W[0][0]), float(W[1][0])), "steps": steps,
            "sb": st.sb.copy(), "y0": m[0]}


def project(curve, x, d):
    """(s, u) точек: длина вдоль пути до основания нормали и отступ по нормали
    (плюс — вправо). Касательная в точке с той же глубиной — основание нормали
    в первом приближении; кривизна добавляет u²·κ/2 — миллиметры."""
    xp = np.interp(d, curve["d"], curve["x"])
    ps = np.interp(d, curve["d"], curve["psi"])
    sp = np.interp(d, curve["d"], curve["s"])
    dx = x - xp
    return sp + dx * np.sin(ps), dx * np.cos(ps)


def offset_curve(curve, w):
    """Кривая на отступе w по нормали: (x, d)."""
    return (curve["x"] + w * np.cos(curve["psi"]),
            curve["d"] - w * np.sin(curve["psi"]))


def path_jacobian(curve, s):
    """Производная смещения точки пути ПО НОРМАЛИ в s по (y0, ψ0, κ_k).

    Возмущение курса δψ(s') сдвигает точку s на ∫_0^s δψ(s')·N(s') ds', а по
    нормали N(s) это ∫ δψ(s')·cos(ψ(s) − ψ(s')) ds'. Для ψ0 интеграл берётся в
    замкнутом виде, для κ_k — по накопленным таблицам. Сверено с конечными
    разностями до 6e-4 при значениях до 1.4e3.
    """
    c, sn = np.cos(curve["psi"]), np.sin(curve["psi"])
    bint = _cumint(curve["BK"])
    A = _cumint(bint * c[:, None])
    B = _cumint(bint * sn[:, None])
    ci, si = _gather(c, s), _gather(sn, s)
    xi, di = _gather(curve["x"], s), _gather(curve["d"], s)
    J = np.empty((len(s), 2 + curve["BK"].shape[1]))
    J[:, 0] = ci
    J[:, 1] = ci * di + si * (xi - curve["y0"])
    J[:, 2:] = ci[:, None] * _gather(A, s) + si[:, None] * _gather(B, s)
    return J


def hidden_edges(curve, d, side, x=None):
    """Кромки внутренней стены поворота за точкой касания: там лидар видит
    линию взгляда, а не стену (views._observed_edge — то же в других
    координатах). side: 0 — левая, 1 — правая. Возвращает маску скрытых, а
    при заданном x — ещё и боковую невязку кромки относительно линии взгляда
    (для скрытых: скрытая кромка тоже обязана лежать там, где её предсказывает
    модель)."""
    hid = np.zeros(len(d), bool)
    r_sight = np.zeros(len(d))
    for k, (w, acc) in enumerate(((curve["wl"], np.maximum), (curve["wr"], np.minimum))):
        m = side == k
        if not m.any():
            continue
        xw, dw = offset_curve(curve, w)
        ok = dw > 1.0
        if ok.sum() < 3:
            continue
        xw, dw = xw[ok], dw[ok]
        order = np.argsort(dw)
        xw, dw = xw[order], dw[order]
        env = acc.accumulate(np.arctan2(xw, dw))
        sight = d[m] * np.tan(np.interp(d[m], dw, env))
        wall = np.interp(d[m], dw, xw)
        gap = (sight - wall) if k == 0 else (wall - sight)
        hid[m] = gap > HIDDEN_TOL
        if x is not None:
            r_sight[m] = x[m] - sight
    return hid if x is None else (hid, r_sight)


# ---------------------------------------------------------------- подгонка

def _tukey_w(r, c):
    z = r / c
    return np.where(np.abs(z) < 1, (1 - z ** 2) ** 2, 0.0)


def _tukey_rho(r, c):
    z = np.minimum(np.abs(r / c), 1.0)
    return c ** 2 / 6 * (1 - (1 - z ** 2) ** 3)


def _wall_at(st, m, s, side):
    """Отступ стены своей стороны у кромок (s — длина вдоль пути)."""
    W = st.widths(m)
    out = np.empty(len(s))
    for j in (0, 1):
        on = side == j
        out[on] = W[j][st.segment(j, s[on])]
    return out


def _evaluate(st, m, edges, rails, with_sight=False):
    curve = track_curve(st, m)
    ed, ex, es, _ = edges
    s_e, u_e = project(curve, ex, ed)
    hid, r_sight = hidden_edges(curve, ed, es, ex)
    r_e = u_e - _wall_at(st, m, s_e, es)
    ok_e = (~hid) & (s_e > 0) & (s_e < S_MAX)
    rd, rx = rails
    s_r, u_r = project(curve, rx, rd) if len(rd) else (np.zeros(0), np.zeros(0))
    if with_sight:
        return curve, s_e, r_e, ok_e, s_r, u_r, np.where(hid, r_sight, np.nan)
    return curve, s_e, r_e, ok_e, s_r, u_r


def _regularizer(st):
    """Гладкость кривизны: матрица квадратичной формы на векторе κ."""
    D1, D2 = _diff(st.nk, 1), _diff(st.nk, 2)
    return D1.T @ D1 / SIGMA_DK1 ** 2 + D2.T @ D2 / SIGMA_DK2 ** 2


def total_cost(st, m, edges, rails, prior_m, prior_sig):
    """Полная цена решения: робастная невязка кромок и рельсов + гладкость +
    приор. По ней выбирается один из двух стартов.

    Скрытая за поворотом кромка не бесплатна: она платит за расстояние до
    линии взгляда, которую предсказывает модель. Иначе выигрывала бы модель,
    объявившая больше кромок скрытыми: замер на двухпутном прогоне — память
    загнула путь сильнее тоннеля, дальние левые кромки у неё «скрыты», правые —
    выбросы, и свежий старт с верной кривизной проигрывал выбор."""
    _, s_e, r_e, ok_e, _, u_r, r_sight = _evaluate(st, m, edges, rails, with_sight=True)
    ew = edges[3]
    c_e = TUKEY * SIGMA_EDGE
    cost = np.sum(ew * ok_e * _tukey_rho(r_e, c_e)) / SIGMA_EDGE ** 2
    hid = np.isfinite(r_sight) & (s_e > 0) & (s_e < S_MAX)
    cost += np.sum(ew[hid] * _tukey_rho(r_sight[hid], c_e)) / SIGMA_EDGE ** 2
    cost += np.sum(_tukey_rho(u_r, RAIL_GATE)) / SIGMA_RAIL ** 2
    k = m[st.ik]
    cost += float(k @ _regularizer(st) @ k)
    cost += STEP_COST * (st.nsteps(0) + st.nsteps(1))
    if prior_m is not None:
        # приор с потолком: смена режима (кончилась станция, тоннель стал
        # двухпутным) стоит конечную цену, иначе свежий старт не мог бы победить
        # память, как бы плохо та ни объясняла кадр. Потолок — на СУММУ, а не на
        # каждый параметр: узлов кривизны ~20, и даже равноценный независимый
        # старт набирает по ним сотни единиц. Замер: двухпутный прогон, память
        # отбросила всю правую стену (1 согласная кромка против 93 у свежего
        # старта, данные 232 против 39), а по попараметрному потолку свежий
        # старт платил 320 и проигрывал.
        z2 = ((m - prior_m) / prior_sig) ** 2
        cost += float(min(np.sum(z2), PRIOR_CAP))
    return float(cost)


def refine(st, m, edges, rails, prior_m=None, prior_sig=None, scales=(1.0,)):
    """Робастный Гаусс-Ньютон по всем параметрам сразу: поза, кривизна и оба
    отступа. scales — множитель отсечки Тьюки по итерациям (на старте шире:
    модель ещё далеко от данных)."""
    es, ew = edges[2], edges[3]
    n_all = len(m)
    Rk = _regularizer(st)
    m_start = m.copy()
    info = {}
    for sc in scales:
        curve, s_e, r_e, ok_e, s_r, u_r = _evaluate(st, m, edges, rails)
        W_e = ew / SIGMA_EDGE ** 2 * _tukey_w(r_e, TUKEY * SIGMA_EDGE * sc) * ok_e
        W_r = _tukey_w(u_r, RAIL_GATE * sc) / SIGMA_RAIL ** 2 * (s_r > 0)
        # r(m + δ) ≈ r − G δ: смещение пути по нормали уменьшает u, рост
        # отступа — невязку кромки своей стороны
        Ge = np.zeros((len(s_e), n_all))
        Ge[:, :st.npath] = path_jacobian(curve, s_e)
        for side in (0, 1):
            on = np.flatnonzero(es == side)
            seg = st.segment(side, s_e[on])
            Ge[on, st.iw(side, 0) + seg] = 1.0
        Gr = np.zeros((len(s_r), n_all))
        if len(s_r):
            Gr[:, :st.npath] = path_jacobian(curve, s_r)
        N = (Ge.T * W_e) @ Ge + (Gr.T * W_r) @ Gr
        b = Ge.T @ (W_e * r_e) + Gr.T @ (W_r * u_r)
        N[st.ik, st.ik] += Rk
        b[st.ik] -= Rk @ m[st.ik]
        if prior_m is not None:
            L = 1.0 / prior_sig ** 2
            N[np.diag_indices(n_all)] += L
            b -= L * (m - prior_m)
        else:
            # без приора отступ стороны без единой кромки не определён: слабая
            # привязка к стартовому значению
            N[np.diag_indices(n_all)] += 1e-6
            iw = np.arange(st.npath, n_all)
            N[iw, iw] += 1.0 / FRESH_WIDTH_SIGMA ** 2
            b[iw] -= (m[iw] - m_start[iw]) / FRESH_WIDTH_SIGMA ** 2
        m = _tie_widths(st, m + np.linalg.solve(N, b))
        info = {"W_e": W_e, "W_r": W_r, "r_e": r_e, "s_e": s_e, "hid": ~ok_e}
    return m, info


def state_from_path(path_d, path_x, edges, walls_fn=None):
    """Состояние по готовому пути x(d) (путь §27 этого кадра) и кромкам: поза и
    кривизна — подгонкой к пути, отступы — медианой кромок, согласных с
    подгонкой контраста этого кадра (walls_fn(d, side) -> x стены).

    Именно согласных с контрастом, а не ближних: в конце станции ближние 40 м —
    стены зала, а дальний тоннель уже другого сечения. Отступ по залу делает все
    дальние кромки выбросами, и робастная подгонка находит «выход» — загнуть
    путь к одной из дальних стен (замер: −3.5 м на 100 м при прямых рельсах).
    Контраст §26 рельсами не привязан и берёт консенсус по всей длине."""
    st = TrackState.fresh()
    m = st.m.copy()
    slope = np.gradient(path_x, path_d)
    m[0] = float(np.interp(0.0, path_d, path_x))
    m[1] = float(np.arctan(slope[0]))
    sel = path_d <= 150.0
    pseudo = (path_d[sel], path_x[sel])
    no_edges = (np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0))
    for _ in range(4):
        m, _ = refine(st, m, no_edges, pseudo, scales=(30.0,))
    curve = track_curve(st, m)
    ed, ex, es, _ = edges
    s_e, u_e = project(curve, ex, ed)
    for k, dflt, side in ((0, -2.5, "left"), (1, 2.5, "right")):
        sel = (es == k) & (s_e < 40)
        if walls_fn is not None:
            agree = (es == k) & (np.abs(ex - walls_fn(ed, side)) < 0.25)
            if agree.sum() >= 5:
                sel = agree
        v = float(np.median(u_e[sel])) if sel.sum() >= 3 else dflt
        m[st.iw(k, 0):st.iw(k, 0) + st.NW] = v
    return st.with_m(_tie_widths(st, m))


def _chain_start(depths, window=10.0, support=3):
    """С какой глубины стена наблюдается на этом отступе: первая согласная
    кромка, за которой в окне window есть ещё support-1 согласных. Нужна для
    рисования: при смене сечения ближняя часть стены — выброс, и линия на
    отступе дальнего тоннеля у самого поезда прошла бы сквозь пустоту."""
    dd = np.sort(np.asarray(depths, float))
    for k in range(len(dd)):
        if np.sum((dd >= dd[k]) & (dd <= dd[k] + window)) >= support:
            return float(dd[k])
    return None


def clean_rails(d, x):
    """Рельсовые срезы, согласные между собой (RAIL_TOL). Меньше трёх — нет опоры."""
    d, x = np.asarray(d, float), np.asarray(x, float)
    if len(d) < 3:
        return d[:0], x[:0]
    fit = ransac_poly_fit(d, x, 2 if len(d) >= 5 else 1, RAIL_TOL)
    if fit is None:
        return d[:0], x[:0]
    return d[fit["inliers"]], x[fit["inliers"]]


def _wmedian(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1])])


def _split_cost(s, u, w, bounds, c):
    """Цена разбиения кромок одной стороны по границам bounds: отступы отрезков
    — взвешенные медианы. None, если разбиение недопустимо (мало кромок в
    отрезке, ступенька ниже STEP_MIN, стена ближе W_MIN, стена по ту сторону)."""
    edges = np.r_[-np.inf, bounds, np.inf]
    vals, cost = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = (s >= lo) & (s < hi)
        if g.sum() < 3 or w[g].sum() < 2:
            return None
        v = _wmedian(u[g], w[g])
        vals.append(v)
        cost += float(np.sum(w[g] * _tukey_rho(u[g] - v, c))) / SIGMA_EDGE ** 2
    vals = np.array(vals)
    if (np.abs(vals) < W_MIN).any() or len(set(np.sign(vals))) > 1:
        return None
    if len(vals) > 1 and np.min(np.abs(np.diff(vals))) < STEP_MIN:
        return None
    return cost + STEP_COST * len(bounds), vals


def search_steps(st, m, edges, rails):
    """Для каждой стены: сколько ступенек отступа (0..MAX_STEPS) и где.

    Жадно: лучшая одна ступенька (перебор по STEP_GRID), затем лучшая вторая
    внутри получившихся отрезков — каждая принимается, только если окупает
    STEP_COST. Без ступенек — лучший из текущего отступа и медианы всех кромок.
    """
    curve, s_e, r_e, ok_e, _, _ = _evaluate(st, m, edges, rails)
    es, ew = edges[2], edges[3]
    u_e = r_e + _wall_at(st, m, s_e, es)
    c = TUKEY * SIGMA_EDGE
    W = st.widths(m).copy()
    sb = np.full((2, MAX_STEPS), np.inf)
    for side in (0, 1):
        sel = (es == side) & ok_e
        s, u, w = s_e[sel], u_e[sel], ew[sel]
        if len(s) < 6:
            continue
        base = min((float(np.sum(w * _tukey_rho(u - v, c))) / SIGMA_EDGE ** 2, v)
                   for v in (W[side][0], _wmedian(u, w)))
        best = (base[0], [], np.array([base[1]]))
        grid = np.arange(5.0, s.max() - 5.0, STEP_GRID)
        for _ in range(MAX_STEPS):
            cur = best
            for b in grid:
                if any(abs(b - x) < 5.0 for x in cur[1]):
                    continue
                bounds = sorted(cur[1] + [b])
                r = _split_cost(s, u, w, bounds, c)
                if r is not None and r[0] < best[0]:
                    best = (r[0], bounds, r[1])
            if best is cur:
                break
        n = len(best[1])
        sb[side][:n] = best[1]
        W[side][:n + 1] = best[2]
        W[side][n + 1:] = best[2][-1]
    st = st.with_m(m, sb)
    m = m.copy()
    m[st.npath:] = W.ravel()
    return st, _tie_widths(st, m)


def refine_with_steps(st, m, edges, rails, scales):
    """Свежий старт: перебор ступенек и Гаусс-Ньютон по очереди, без приора.
    Перебор — на первой, средней и последней итерации: между ними путь
    сдвигается мало, а перебор стоит половину времени кадра."""
    search_at = {0, len(scales) // 2, len(scales) - 1}
    for i, sc in enumerate(scales):
        if i in search_at:
            st, m = search_steps(st, m, edges, rails)
        m, info = refine(st, m, edges, rails, scales=(sc,))
    st, m = merge_small_steps(st, m)
    return st, m, info


def merge_small_steps(st, m):
    """Ступенька, выродившаяся после уточнения ниже STEP_MIN, снимается:
    иначе память несёт «ступеньку» в 0.14 м, которую никто не ставил бы."""
    sb = st.sb.copy()
    W = st.widths(m).copy()
    for j in (0, 1):
        n = st.nsteps(j)
        g = 0
        while g < n:
            if abs(W[j][g + 1] - W[j][g]) < STEP_MIN:
                W[j][g] = 0.5 * (W[j][g] + W[j][g + 1])
                W[j] = np.r_[W[j][:g + 1], W[j][g + 2:], W[j][-1]]
                sb[j] = np.r_[sb[j][:g], sb[j][g + 1:], np.inf]
                n -= 1
            else:
                g += 1
    st = st.with_m(m, sb)
    m = m.copy()
    m[st.npath:] = W.ravel()
    return st, _tie_widths(st, m)


def _resample(st, like):
    """Перевести профиль кривизны st на узлы like (интерполяцией)."""
    k = np.interp(like.sk, st.sk, st.m[st.ik])
    m = np.r_[st.m[:2], k, st.m[st.npath:]]
    return TrackState(m, like.sk, like.new, st.sb)


# ---------------------------------------------------------------- трекер

def to_path_dict(curve, grid, mode, offset=0.0):
    """Путь в формате contrast_gauge (x, ψ, длина дуги на сетке глубин)."""
    return {"d": grid, "x": np.interp(grid, curve["d"], curve["x"]),
            "psi": np.interp(grid, curve["d"], curve["psi"]),
            "arc": np.interp(grid, curve["d"], curve["s"]),
            "mode": mode, "d0": None, "offset": offset}


def edges_from_bands(d, left, right, band_len):
    """Кромки силуэта по полосам (views.band_edges) в виде (d, x, сторона, вес)."""
    wt = np.clip(np.asarray(band_len, float) / EDGE_CORR, EDGE_W_MIN, EDGE_W_MAX)
    d = np.asarray(d, float)
    return (np.r_[d, d], np.r_[left, right],
            np.r_[np.zeros(len(d), int), np.ones(len(d), int)], np.r_[wt, wt])


class WallParallelTracker:
    """Путь и стены между кадрами: два старта на кадр, приор из прошлого кадра."""

    FRESH_SCALES = (3.0, 2.0, 1.5, 1.0, 1.0, 1.0)
    TRACK_SCALES = (1.5, 1.0, 1.0, 1.0)

    def __init__(self):
        self.st = None
        self.last_ds = None
        self.ds_hist = []
        self.reach_hist = []

    def reset(self):
        self.st = None
        self.reach_hist = []

    def step(self, edges, rail_d, rail_x, ds, fresh_path=None, fresh_walls=None):
        """Обработать кадр.

        edges — кромки кадра (edges_from_bands), rail_d/rail_x — центры колеи,
        ds — измеренное Δs (None — не измерено), fresh_path — (d, x) пути этого
        кадра, построенного без памяти (§27), для свежего старта; fresh_walls —
        стены подгонки контраста этого кадра, (d, side) -> x.

        Свежий старт подгоняется БЕЗ приора: он и нужен для того, чтобы найти
        решение, которого память не допускает. Приор входит только в цену выбора
        (с потолком PRIOR_CAP).
        """
        sel = rail_d <= RAIL_MAX
        rails = clean_rails(rail_d[sel], rail_x[sel])
        if len(edges[0]) < 10:
            self.reset()
            return None
        ds_ok = ds is not None and np.isfinite(ds)
        if ds_ok and self.ds_hist and abs(ds - np.median(self.ds_hist)) > DS_JUMP:
            ds_ok = False       # противоречит инерции поезда (DS_JUMP)
        if ds_ok:
            ds_use = self.last_ds = float(ds)
            self.ds_hist = (self.ds_hist + [ds_use])[-DS_HISTORY:]
        else:
            ds_use = self.last_ds if self.last_ds is not None else 0.0

        from . import parallel as par

        cands = []
        prior_m = prior_sig = None
        pred = None
        mem_job = None
        if self.st is not None:
            pred = self.st.advance(ds_use)
            prior_m, prior_sig = pred.m, pred.prior_sigma()
            # «память» и «заново» независимы: при нескольких потоках «память»
            # считается в пуле, пока здесь считается «заново» (экспер. 19)
            mem_job = par.submit(refine, pred, pred.m.copy(), edges, rails, prior_m, prior_sig,
                                 self.TRACK_SCALES)
        if mem_job is not None and par.workers() <= 1:
            m, info = mem_job.result()
            cands.append(("память", pred, m, info))
            mem_job = None
        fresh = None
        if fresh_path is not None:
            st0 = state_from_path(fresh_path[0], fresh_path[1], edges, fresh_walls)
            if pred is not None:
                # те же узлы, что у прогноза: иначе приор не с чем сравнивать
                st0 = _resample(st0, pred)
            st0, m, info = refine_with_steps(st0, st0.m.copy(), edges, rails,
                                             self.FRESH_SCALES)
            fresh = ("заново", st0, m, info)
        if mem_job is not None:
            m, info = mem_job.result()
            cands.append(("память", pred, m, info))
        if fresh is not None:
            cands.append(fresh)
        if not cands:
            return None
        scored = [(total_cost(st, m, edges, rails, prior_m, prior_sig), name, st, m, info)
                  for name, st, m, info in cands]
        scored.sort(key=lambda t: t[0])
        cost, origin, st, m, info = scored[0]
        st, m = merge_small_steps(st, m)
        self.st = st.with_m(m)
        curve = track_curve(self.st)

        ed, _, es, _ = edges
        vis = info["W_e"] > 0
        now = {name: _chain_reach(ed[vis & (es == k)]) for k, name in ((0, "left"), (1, "right"))}
        start_now = {name: _chain_start(ed[vis & (es == k)])
                     for k, name in ((0, "left"), (1, "right"))}
        # Дальность каждой стены — медиана по REACH_HISTORY кадрам, перенесённым на
        # пройденный путь: одиночная дальняя кромка то попадает в цепочку, то нет,
        # и без этого нарисованная стена на соседних кадрах то 108, то 147 м при
        # той же геометрии.
        self.reach_hist = [{k: (v - ds_use if v is not None else None) for k, v in h.items()}
                           for h in self.reach_hist][-(REACH_HISTORY - 1):]
        self.reach_hist.append({**now, **{"from_" + k: v for k, v in start_now.items()}})
        reach_side, wall_from = {}, {}
        for name in ("left", "right"):
            vals = [h[name] for h in self.reach_hist if h[name] is not None]
            reach_side[name] = float(np.median(vals)) if vals else None
            vals = [h["from_" + name] for h in self.reach_hist if h["from_" + name] is not None]
            wall_from[name] = max(0.0, float(np.median(vals))) if vals else None
        both = [v for v in reach_side.values() if v is not None]
        reach = float(max(both)) if both else None
        reach_now = max([v for v in now.values() if v is not None], default=None)
        # Проигравшая гипотеза — только для сведения (эксперимент 18: где «память»
        # и «заново» расходятся, путь вдали не определён); на выбор не влияет.
        alt = None
        if len(scored) > 1:
            _, alt_origin, alt_st, alt_m, _ = scored[1]
            alt = {"origin": alt_origin, "cost": scored[1][0],
                   "curve": track_curve(alt_st.with_m(alt_m))}
        return {"state": self.st, "curve": curve, "edges": edges, "info": info,
                "reach_side": reach_side, "reach": reach, "reach_now": reach_now,
                "wall_from": wall_from, "ds": ds_use, "ds_measured": ds_ok,
                "origin": origin, "cost": cost, "n_rails": int(len(rails[0])), "alt": alt}


# ---------------------------------------------------------------- габарит

def _xyz(px, py, pz):
    """Точки кадра с откликом (нулевые — лидар не получил отражения), float64."""
    x = px.astype(float)
    y = py.astype(float)
    z = pz.astype(float)
    keep = (np.abs(x) + np.abs(y) + np.abs(z)) > 0.1
    return x[keep], y[keep], z[keep]


def _gauge_frame(s, u, v, pgrid, theta, uc, vc, limit):
    """Координаты габарита (поворот на крен своего сечения, от середины между
    головками) и маска коробки эталона §27 — поэлементно по точкам."""
    from . import contrast_gauge as cg

    th_p = np.interp(s, pgrid, theta)
    du, dv = u - np.interp(s, pgrid, uc), v - np.interp(s, pgrid, vc)
    ug = du * np.cos(th_p) + dv * np.sin(th_p)
    vg = -du * np.sin(th_p) + dv * np.cos(th_p)
    inside = ((np.abs(ug) <= cg.HALF_WIDTH) & (vg >= cg.GAUGE_BOTTOM) & (vg <= cg.GAUGE_H)
              & (s > cg.NEAR_LIMIT) & (s <= limit))
    return ug, vg, inside


class ParallelGauge:
    """Последовательный проход по записи: путь и стены — WallParallelTracker,
    габарит и находки — ровно как в §27 (contrast_gauge.ContrastGauge), только
    вдоль нового пути. Всё, что не касается пути, намеренно не менялось:
    находки двух методов должны отличаться только из-за пути."""

    def __init__(self, confirm=3):
        from .gauge import ObstacleWatch
        self.tracker = WallParallelTracker()
        self.prior = None
        self.offset = 0.0
        self.prev_band = None
        self.watch = ObstacleWatch(confirm=confirm)

    def update(self, points, steps=1):
        from . import contrast_gauge as cg
        from .roll import rail_pose_track
        from .shift import estimate_shift
        from .tunnel_frame import rail_samples
        from .views import D_MAX, floor_level, rasterize, run_silhouette, shape_x

        from . import parallel as par

        # Рельсы от вида сверху не зависят: при нескольких потоках они считаются,
        # пока строится вид сверху (экспер. 19). При одном — сразу, как раньше.
        rails_job = par.submit(rail_samples, points)
        x, y, z = par.chunked(_xyz, (points['x'], points['y'], points['z']))

        grid = rasterize(points)
        sil = run_silhouette(grid, self.prior)
        fit = sil["fit"]
        self.prior = fit
        d_e, l_e, r_e, L_e = sil["edges"]
        edges = edges_from_bands(d_e, l_e, r_e, L_e)

        rail_d, rail_x, _, gauge = rails_job.result()
        gauge = gauge or 1.60
        fresh = None
        if fit is not None and fit.get("reach"):
            fresh = cg.build_path(fit, rail_d, rail_x, self.offset)
            if fresh["mode"] == "rails+contrast":
                self.offset = fresh["offset"]

        # Δs меряется по пути этого кадра без памяти (как в §27-§28): продольное
        # смещение к поперечным поправкам нечувствительно, а трекеру Δs нужен
        # раньше, чем он построит свой путь.
        ref = fresh
        if ref is None and self.tracker.st is not None:
            ref = to_path_dict(track_curve(self.tracker.st), cg.PATH_GRID, "")
        if ref is None:
            self.prev_band = None
            self.watch.reset()
            self.tracker.reset()
            return None
        floor, _, _ = floor_level(grid, lambda d: float(np.interp(d, ref["d"], ref["x"])))
        if floor is None:
            floor = np.array([0.0, float(np.percentile(z, 2))])
        s0, u0, v0 = par.chunked(cg.to_path_coords, (x, y, z), ref, floor)
        band_sel = (v0 >= cg.BAND_LO) & (v0 <= cg.BAND_HI) & (s0 > 2) & (s0 < 60)
        band = {"d": s0[band_sel], "u": u0[band_sel], "v": v0[band_sel]}
        shift = None
        if self.prev_band is not None and len(band["d"]) > 500:
            est = estimate_shift(self.prev_band, band, max_shift=2.2 * max(steps, 1))
            if est.get("ok"):
                shift = float(est["shift"])
        self.prev_band = band if len(band["d"]) > 500 else None

        tr = self.tracker.step(edges, rail_d, rail_x, shift,
                               fresh_path=(fresh["d"], fresh["x"]) if fresh else None,
                               fresh_walls=(lambda d, side: shape_x(fit, d, side)) if fresh
                               else None)
        if tr is None or not tr["reach"]:
            self.watch.reset()
            return None
        mode = "rails+walls" if tr["n_rails"] >= 3 else "walls"
        path = to_path_dict(tr["curve"], cg.PATH_GRID, mode)

        s, u, v = par.chunked(cg.to_path_coords, (x, y, z), path, floor)
        pose = cg._clean_pose(rail_pose_track(s, u, v, gauge, cg.POSE_DEPTHS,
                                              lambda D: max(1.0, cg.half_thick(D))))
        pgrid = np.arange(0.0, D_MAX + 1.0, 1.0)
        theta, uc, vc, has_pose = cg._pose_curves(pose, pgrid)
        limit = min(float(tr["reach"]), D_MAX)
        ug, vg, inside = par.chunked(_gauge_frame, (s, u, v), pgrid, theta, uc, vc, limit)
        clusters = cg.cluster(s[inside], ug[inside], vg[inside])
        # подтверждение — по тому Δs, которым перенесено состояние: отброшенный
        # как противоречащий инерции замер заменён прежней скоростью
        confirmed = self.watch.update(clusters[0]["dist"] if clusters else None, tr["ds"])

        return {
            "fit": fit, "grid": grid, "silhouette": sil, "offset": self.offset,
            "path": path, "track": tr,
            "floor": floor, "pose": pose, "has_pose": has_pose,
            "pose_curves": (pgrid, theta, uc, vc), "gauge_value": gauge,
            "s": s, "u": u, "v": v, "ug": ug, "vg": vg, "xyz": (x, y, z),
            "inside": inside, "clusters": clusters, "confirmed": confirmed,
            "shift": shift, "limit": limit, "reach": float(tr["reach"]),
        }
