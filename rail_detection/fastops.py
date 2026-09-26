"""Пакетные версии мелких вычислений — тот же ответ, бит в бит (эксперимент 19).

Конвейер тратил большую часть кадра не на арифметику, а на тысячи мелких
вызовов numpy из циклов Python: lstsq на гипотезу RANSAC, np.percentile на бин
профиля. Здесь те же вычисления делаются для всей пачки сразу, и каждая функция
повторяет numpy до последнего бита — это проверено на реальных кадрах против
прежнего цикла (exp_speed.py и прямые сравнения в experiments/19-speed.md).
"""

import numpy as np

try:        # тот же LAPACK-вызов, которым np.linalg.lstsq решает одну систему
    from numpy.linalg import _umath_linalg as _ul
    _HAVE_GUFUNC = hasattr(_ul, "lstsq_m") and hasattr(_ul, "lstsq_n")
except ImportError:         # pragma: no cover
    _HAVE_GUFUNC = False


def lstsq_batch(A, b, rcond=None):
    """Решения np.linalg.lstsq(A[i], b[i], rcond)[0] для пачки систем.

    A: (k, m, n), b: (k, m). Вызывается тот же обобщённый ufunc, что внутри
    np.linalg.lstsq (он и так умеет пачки — обёртка их просто не пропускает),
    с тем же rcond; ответ совпадает бит в бит. Без него — прежний цикл.
    """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    k, m, n = A.shape
    if k == 0:
        return np.zeros((0, n))
    if not _HAVE_GUFUNC:
        return np.array([np.linalg.lstsq(A[i], b[i], rcond=rcond)[0] for i in range(k)])
    if rcond is None:
        rcond = np.finfo(np.float64).eps * max(n, m)
    g = _ul.lstsq_m if m <= n else _ul.lstsq_n
    x, _, _, _ = g(A, b[..., None], rcond, signature="ddd->ddid")
    return x[..., 0]


def group_percentile(v, group, n_groups, pct):
    """np.percentile(v[group == g], pct) для каждой группы g (метод linear).

    Повторяет numpy 2 до бита: q = pct / 100 в типе данных, виртуальный индекс
    метода linear — (n − 1)·q (не общая формула n·q + (1 − q) − 1: при q = 0.9
    они расходятся в последнем бите), соседи с правилами на краях, интерполяция
    _lerp (две формулы — по a и по b — в зависимости от доли ≥ 0.5). Пустая
    группа — NaN.
    v без NaN (иначе np.percentile вернул бы NaN, а здесь — нет).
    """
    v = np.asarray(v)
    out = np.full(n_groups, np.nan)
    if len(v) == 0:
        return out
    order = np.lexsort((v, group))
    vs = v[order]
    counts = np.bincount(group, minlength=n_groups)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    has = np.flatnonzero(counts > 0)
    n = counts[has]
    q = np.true_divide(pct, v.dtype.type(100) if v.dtype.kind == "f" else 100)
    virt = (n - 1) * q
    prev = np.floor(virt)
    nxt = prev + 1
    above = virt >= n - 1
    prev = np.where(above, -1, prev)
    nxt = np.where(above, -1, nxt)
    below = virt < 0
    prev = np.where(below, 0, prev)
    nxt = np.where(below, 0, nxt)
    gamma = virt - prev
    pi = prev.astype(np.intp)
    ni = nxt.astype(np.intp)
    a = vs[starts[has] + np.where(pi < 0, n + pi, pi)]
    b = vs[starts[has] + np.where(ni < 0, n + ni, ni)]
    diff = b - a
    res = a + diff * gamma
    hi = gamma >= 0.5
    res[hi] = (b - diff * (1 - gamma))[hi]
    out[has] = res
    return out
