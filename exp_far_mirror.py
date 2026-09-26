#!/usr/bin/env python3
"""Эксперимент 18: зеркальная проверка находок (протокол §32).

Тот же вариант детектора на исходной записи и на её отражении (x -> -x,
/Volumes/T7/reversed) обязан дать те же находки кадр в кадр: габарит,
скопления, запас и профиль по высоте симметричны, так что любое расхождение —
от пути (или от решения на грани порога), и каждое надо объяснить.

    python exp_far_mirror.py                 # пять реальных записей разработки
    python exp_far_mirror.py --variant final --bags roundT_doubleT
"""

import argparse

import numpy as np

from exp20_split import guard
from exp_far_cache import SIDE_S, SIDE_V, load, side_view
from rail_detection import far_detect as fd

BAGS = ["doubleT_platform", "roundT_pressureGate_roundT", "squareT_platform_squareT_switch",
        "roundT_squareT_pressureGate_squareT", "doubleT_obstacle"]   # roundT_doubleT отложен (экспер. 20)
TOL = 1.0     # м: находки совпали, если дальности отличаются меньше


def detect(cache_dir, bag, p):
    cols, pts = load(cache_dir, bag)
    n = len(cols["idx"])
    return fd.run_sequence(p, n, lambda k: pts(k) if cols["ok"][k] else None,
                           cols["limit"], cols["ds"],
                           getside=lambda k: (side_view(cols, k), SIDE_S, SIDE_V),
                           alt_dx=cols.get("alt_dx"))


def same(a, b):
    return len(a) == len(b) and all(abs(x - y) < TOL for x, y in zip(sorted(a), sorted(b)))


def episodes(ks):
    out = []
    for k in ks:
        if out and k - out[-1][1] <= 2:
            out[-1][1] = k
        else:
            out.append([k, k])
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--variant", default="final")
    a.add_argument("--bags", nargs="+", default=BAGS)
    a.add_argument("--orig", default="output/exp18_cache/Dataset")
    a.add_argument("--mirror", default="output/exp18_cache/reversed")
    a.add_argument("--holdout", action="store_true",
                   help="отложенный замер: разрешить отложенные записи эксперимента 20")
    args = a.parse_args()
    guard(args.bags, args.holdout)
    p = fd.VARIANTS[args.variant]
    print(f"вариант {args.variant}: подтверждённые находки, исходная против отражённой")
    for bag in args.bags:
        do, dm = detect(args.orig, bag, p), detect(args.mirror, bag, p)
        n = min(len(do), len(dm))
        with_o = sum(1 for k in range(n) if do[k][1])
        with_m = sum(1 for k in range(n) if dm[k][1])
        diff = [k for k in range(n) if not same(do[k][1], dm[k][1])]
        ep = episodes(diff)
        print(f"  {bag:34s} кадров {n}: с находкой {with_o} / {with_m}, расходятся {len(diff)}"
              + (f" — эпизоды {', '.join(f'{x}–{y}' for x, y in ep)}" if ep else ""))
        for x, y in ep[:6]:
            k = x
            print(f"      кадр {k}: исходная {np.round(do[k][1], 1).tolist()}, "
                  f"отражённая {np.round(dm[k][1], 1).tolist()}")


if __name__ == "__main__":
    main()
