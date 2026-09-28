#!/usr/bin/env python3
"""Эксперимент 18: предсказывает ли свод, куда идёт путь по высоте.

Проверка как для оси по отложенным рельсам (§15): профиль по высоте
подгоняется ТОЛЬКО по своду, а сверяется с уровнем пола там, где пол ещё
виден (50–90 м), — пол в подгонку не входит. Сравнение с тем, как было:
прямая ближнего пола продолжена (поправка ноль).

    python exp_far_vprof.py [записи...]
"""

import numpy as np

from exp_far_cache import SIDE_S, SIDE_V, load, side_view
from rail_detection import far_detect as fd

RUNS = [("Dataset", "doubleT_platform"), ("Dataset", "roundT_doubleT"),
        ("Dataset", "roundT_pressureGate_roundT"), ("Dataset", "squareT_platform_squareT_switch"),
        ("Dataset", "doubleT_obstacle"),
        ("Synthetic_data", "box"), ("Synthetic_data", "human_smashed"),
        ("Synthetic_data", "human_smashed_diff_tunnels")]
CHECK = (50.0, 90.0)


def ceiling_only(H):
    """Вид сбоку без пола дальше VP_S0: пол в подгонку не попадает."""
    H2 = H.copy()
    vc = 0.5 * (SIDE_V[1:] + SIDE_V[:-1])
    far = np.searchsorted(SIDE_S, fd.VP_S0)
    H2[far:, vc <= 0.8] = 0
    return H2


def main():
    import sys
    only = sys.argv[1:]
    print(f"пол на {CHECK[0]:.0f}–{CHECK[1]:.0f} м против предсказания по своду: "
          "|ошибка| медиана / 90-й перц., м; «прямо» — поправка ноль")
    for tag, bag in RUNS:
        if only and bag not in only:
            continue
        cols, _ = load(f"output/exp18_cache/{tag}", bag)
        e_new, e_old = [], []
        for k in range(0, len(cols["idx"]), 2):
            if not cols["ok"][k]:
                continue
            H = side_view(cols, k)
            L = fd.side_levels(H, SIDE_S, SIDE_V)
            chk = (L[:, 0] >= CHECK[0]) & (L[:, 0] <= CHECK[1]) & np.isfinite(L[:, 1])
            if chk.sum() < 2:
                continue
            delta, info = fd.vertical_profile(ceiling_only(H), SIDE_S, SIDE_V, cols["limit"][k])
            e_new += list(np.abs(L[chk, 1] - delta(L[chk, 0])))
            e_old += list(np.abs(L[chk, 1]))
        if not e_new:
            print(f"  {bag:34s} пола на {CHECK} м не видно")
            continue
        print(f"  {bag:34s} по своду {np.median(e_new):.3f} / {np.percentile(e_new, 90):.3f}   "
              f"прямо {np.median(e_old):.3f} / {np.percentile(e_old, 90):.3f}   (полос {len(e_new)})")


if __name__ == "__main__":
    main()
