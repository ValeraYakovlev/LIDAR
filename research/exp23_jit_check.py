#!/usr/bin/env python3
"""Эксперимент 23, шаг 1: numba-функции против numpy — бит в бит.

    python exp23_jit_check.py            # случайные массивы + настоящие входы перебора ступенек

1. pairwise_sum против np.sum, tukey_rho против parallel_path._tukey_rho,
   wmedian против parallel_path._wmedian — случайные массивы длиной 0–5000.
2. search_side против parallel_path._search_side (прежний код) — на входах,
   снятых с кадров записей разработки: цена, границы, отступы бит в бит.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from rail_detection import jit                              # noqa: E402
from rail_detection import parallel_path as pp              # noqa: E402

BAGS = [("/Volumes/T7/Dataset", "doubleT_obstacle"),
        ("/Volumes/T7/Dataset", "squareT_platform_squareT_switch"),
        ("/Volumes/T7/Synthetic_data", "human_smashed")]


def same(a, b):
    a, b = np.atleast_1d(np.asarray(a, np.float64)), np.atleast_1d(np.asarray(b, np.float64))
    return a.shape == b.shape and np.array_equal(a.view(np.int64), b.view(np.int64))


def random_checks(n_probe=10000, seed=0):
    rng = np.random.default_rng(seed)
    bad = {"pairwise_sum": 0, "tukey_rho": 0, "wmedian": 0}
    for t in range(n_probe):
        n = int(rng.integers(0, 300)) if t % 4 else int(rng.integers(300, 5000))
        a = rng.normal(0, 10 ** rng.uniform(-3, 3), n) * rng.choice([1, -1], n)
        if not same(jit.pairwise_sum(a, 0, n), np.sum(a)):
            bad["pairwise_sum"] += 1
        c = float(rng.uniform(0.2, 1.0))
        r = rng.normal(0, 1.0, max(n, 1))
        k = c ** 2 / 6
        mine = np.array([jit.tukey_rho(x, c, k) for x in r[:64]])
        if not same(mine, pp._tukey_rho(r[:64], c)):
            bad["tukey_rho"] += 1
        if n >= 1:
            v = np.round(rng.normal(0, 3, n), 1 if t % 3 == 0 else 6)    # треть — с равными
            w = rng.uniform(0.1, 4.0, n)
            if not same(jit.wmedian(v, w), pp._wmedian(v, w)):
                bad["wmedian"] += 1
    return bad


def real_checks(max_frames=40):
    """Перебор ступенек: прежний код и numba на одних и тех же входах."""
    from rail_detection import iter_frames

    calls, bad = 0, []
    py = pp._search_side_py
    grid_step, c_edge = pp.STEP_GRID, pp.SIGMA_EDGE ** 2

    def both(s, u, w, w0, c):
        nonlocal calls
        calls += 1
        a = py(s, u, w, w0, c)
        grid = np.arange(5.0, s.max() - 5.0, grid_step)
        b = jit.search_side(s, u, w, w0, c, c ** 2 / 6, c_edge, grid, pp.MAX_STEPS,
                            pp.W_MIN, pp.STEP_MIN, pp.STEP_COST, 5.0)
        if not (same(a[0], b[0]) and same(a[1], b[1]) and same(a[2], b[2])):
            bad.append((calls, a, b))
        return a

    orig = pp._search_side
    pp._search_side = both
    try:
        for dataset, bag in BAGS:
            pg = pp.ParallelGauge()
            for _, pts, _ in iter_frames(f"{dataset}/{bag}", max_frames=max_frames):
                pg.update(pts)
            print(f"  {bag}: вызовов {calls}, расхождений {len(bad)}", flush=True)
    finally:
        pp._search_side = orig
    return calls, bad


def main():
    bad = random_checks()
    print("случайные массивы (10 000 проб):",
          ", ".join(f"{k} — расхождений {v}" for k, v in bad.items()))
    calls, diffs = real_checks()
    print(f"перебор ступенек на кадрах: {calls} вызовов, расхождений {len(diffs)}")
    for k, a, b in diffs[:3]:
        print(f"  вызов {k}:\n    numpy: {a}\n    numba: {b}")


if __name__ == "__main__":
    main()
