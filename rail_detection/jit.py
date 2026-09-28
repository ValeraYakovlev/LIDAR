"""Эксперимент 23: горячие циклы в numba — те же вычисления, что в numpy, бит в бит.

Что повторяется дословно:

* `pairwise_sum` — сложение `np.sum` для float64: меньше 8 элементов — подряд
  от нуля; до 128 — восемь накопителей по шагу 8, их сумма
  ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7)), затем остаток подряд;
  больше 128 — пополам (граница кратна 8), половины — тем же правилом.
* `tukey_rho` — та же последовательность операций, что в
  `parallel_path._tukey_rho`; куб — через `pow`, как `np.power(x, 3)`
  (у numpy быстрый путь только для степеней −1, 0, 0.5, 1, 2).
* `wmedian` — `argsort`, накопленная сумма, `searchsorted` слева. Сортировка
  в numba другая — при равных значениях порядок весов может отличаться.

Выключается переменной окружения RAIL_JIT=0 (и сам, если numba нет): тогда
вызывающий код идёт прежним путём на numpy.
"""

import math
import os

import numpy as np

ENABLED = os.environ.get("RAIL_JIT", "1") == "1"
try:
    from numba import njit
except ImportError:          # без numba — прежний код
    ENABLED = False

    def njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]

_jit = njit(cache=True, error_model="numpy")

PW_BLOCK = 128               # numpy: PW_BLOCKSIZE


@_jit
def _block_sum(a, lo, n):
    """Лист попарного сложения numpy: n ≤ PW_BLOCK."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[lo + i]
        return res
    r0, r1, r2, r3 = a[lo], a[lo + 1], a[lo + 2], a[lo + 3]
    r4, r5, r6, r7 = a[lo + 4], a[lo + 5], a[lo + 6], a[lo + 7]
    m = n - n % 8
    i = 8
    while i < m:
        j = lo + i
        r0 += a[j]
        r1 += a[j + 1]
        r2 += a[j + 2]
        r3 += a[j + 3]
        r4 += a[j + 4]
        r5 += a[j + 5]
        r6 += a[j + 6]
        r7 += a[j + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[lo + i]
        i += 1
    return res


@_jit
def pairwise_sum(a, lo, n):
    """Сумма a[lo:lo + n] ровно так, как её считает np.sum (float64).

    Свыше PW_BLOCK numpy делит пополам (граница кратна 8) и складывает
    сумму левой половины с суммой правой. Здесь то же дерево обходится явным
    стеком: рекурсивные функции numba при загрузке из кэша (cache=True)
    роняют процесс."""
    if n <= PW_BLOCK:
        return _block_sum(a, lo, n)
    s_lo = np.empty(64, np.int64)
    s_n = np.empty(64, np.int64)
    s_left = np.empty(64)
    s_st = np.empty(64, np.int64)     # 0 — нужна левая, 1 — нужна правая, 2 — ждём правую
    sp = 0
    s_lo[0], s_n[0], s_st[0] = lo, n, 0
    ret = 0.0
    while True:
        l, nn = s_lo[sp], s_n[sp]
        n2 = nn // 2
        n2 -= n2 % 8
        st = s_st[sp]
        if st == 0:
            if n2 > PW_BLOCK:
                sp += 1
                s_lo[sp], s_n[sp], s_st[sp] = l, n2, 0
                continue
            s_left[sp] = _block_sum(a, l, n2)
            s_st[sp] = 1
            st = 1
        if st == 1:
            rn = nn - n2
            if rn > PW_BLOCK:
                s_st[sp] = 2
                sp += 1
                s_lo[sp], s_n[sp], s_st[sp] = l + n2, rn, 0
                continue
            val = s_left[sp] + _block_sum(a, l + n2, rn)
        else:
            val = s_left[sp] + ret
        if sp == 0:
            return val
        sp -= 1
        if s_st[sp] == 0:             # готова левая половина родителя
            s_left[sp] = val
            s_st[sp] = 1
        else:                         # готова правая
            ret = val


@_jit
def tukey_rho(r, c, k):
    """ρ Тьюки одного остатка; k = c ** 2 / 6 (считается снаружи, как в numpy)."""
    z = abs(r / c)
    if not z < 1.0:          # np.minimum(z, 1.0); NaN не бывает
        z = 1.0
    return k * (1.0 - math.pow(1.0 - z * z, 3.0))


@_jit
def wmedian(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return v[o][np.searchsorted(c, 0.5 * c[-1])]


@_jit
def _weighted_rho_sum(u, w, v, c, k):
    t = np.empty(len(u))
    for i in range(len(u)):
        t[i] = w[i] * tukey_rho(u[i] - v, c, k)
    return pairwise_sum(t, 0, len(t))


@_jit
def _segment(s, u, w, lo, hi, c, k, sig2):
    """Отрезок кромок [lo, hi): (годен, отступ, вклад в цену) — как
    parallel_path._split_cost для одного отрезка."""
    n = 0
    for i in range(len(s)):
        if s[i] >= lo and s[i] < hi:
            n += 1
    ug = np.empty(n)
    wg = np.empty(n)
    j = 0
    for i in range(len(s)):
        if s[i] >= lo and s[i] < hi:
            ug[j] = u[i]
            wg[j] = w[i]
            j += 1
    if n < 3 or pairwise_sum(wg, 0, n) < 2:
        return False, 0.0, 0.0
    v = wmedian(ug, wg)
    return True, v, _weighted_rho_sum(ug, wg, v, c, k) / sig2


@_jit
def _split_cost(s, u, w, bounds, c, k, sig2, w_min, step_min, step_cost):
    nb = len(bounds)
    vals = np.empty(nb + 1)
    cost = 0.0
    for q in range(nb + 1):
        lo = -np.inf if q == 0 else bounds[q - 1]
        hi = np.inf if q == nb else bounds[q]
        ok, v, part = _segment(s, u, w, lo, hi, c, k, sig2)
        if not ok:
            return False, 0.0, vals
        vals[q] = v
        cost += part
    sg = np.sign(vals[0])
    for q in range(nb + 1):
        if abs(vals[q]) < w_min or np.sign(vals[q]) != sg:
            return False, 0.0, vals
    if nb > 0:
        dmin = np.inf
        for q in range(nb):
            dmin = min(dmin, abs(vals[q + 1] - vals[q]))
        if dmin < step_min:
            return False, 0.0, vals
    return True, cost + step_cost * nb, vals


@_jit
def search_side(s, u, w, w0, c, k, sig2, grid, max_steps, w_min, step_min, step_cost,
                min_gap):
    """parallel_path._search_side: (цена, границы, отступы отрезков)."""
    cost0 = _weighted_rho_sum(u, w, w0, c, k) / sig2
    vm = wmedian(u, w)
    costm = _weighted_rho_sum(u, w, vm, c, k) / sig2
    # min((цена, отступ), ...) — по цене, при равной — по отступу, первый при равных
    if costm < cost0 or (costm == cost0 and vm < w0):
        best_cost, base_v = costm, vm
    else:
        best_cost, base_v = cost0, w0
    best_b = np.empty(max_steps)
    best_nb = 0
    best_vals = np.array([base_v])
    for _ in range(max_steps):
        cur_nb = best_nb
        cur_b = best_b[:cur_nb].copy()
        improved = False
        for b in grid:
            skip = False
            for x in cur_b:
                if abs(b - x) < min_gap:
                    skip = True
                    break
            if skip:
                continue
            bounds = np.sort(np.append(cur_b, b))
            ok, cost, vals = _split_cost(s, u, w, bounds, c, k, sig2, w_min, step_min,
                                         step_cost)
            if ok and cost < best_cost:
                best_cost = cost
                best_nb = cur_nb + 1
                best_b[:best_nb] = bounds
                best_vals = vals
                improved = True
        if not improved:
            break
    return best_cost, best_b[:best_nb].copy(), best_vals


@_jit
def band_rows(mask, weight, has_weight, xc, dc, preds, i, j, cx, halfwin):
    """Строки i..j-1 одной полосы views.band_edges: профиль поперёк пути,
    сдвинутый на предсказанную ось строки, и суммы для глубины полосы.

    Порядок тот же, что у numpy-цикла: колонки строки слева направо
    (np.where), каждая — в свою ячейку профиля (np.add.at по порядку), сумма
    весов строки — попарная, как w.sum()."""
    nx = len(xc)
    half = nx // 2
    prof = np.zeros(nx)
    wbuf = np.empty(nx)
    dsum = 0.0
    wsum = 0.0
    for row in range(i, j):
        pred = preds[row - i]
        cnt = 0
        for col in range(nx):
            if mask[row, col] and abs(xc[col] - pred) < halfwin:
                wv = weight[row, col] if has_weight else 1.0
                o = int(np.rint((xc[col] - pred) / cx)) + half
                if o < 0:
                    o = 0
                elif o > nx - 1:
                    o = nx - 1
                prof[o] += wv
                wbuf[cnt] = wv
                cnt += 1
        if cnt == 0:
            continue
        ws = pairwise_sum(wbuf, 0, cnt)
        dsum += dc[row] * ws
        wsum += ws
    return prof, dsum, wsum
