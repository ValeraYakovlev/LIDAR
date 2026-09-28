#!/usr/bin/env python3
"""Эксперимент 20: зеркальная проверка (протокол §32) по статистике живого прогона.

Тот же конвейер на записи и на её отражении (x -> -x) обязан дать отражённый
ответ: подтверждённые находки — кадр в кадр (дальность ± 1 м), ось габарита —
та же с обратным знаком. Расхождение само по себе не дефект, но указывает на
решение на грани порога, и каждый эпизод надо объяснить.

    python exp20_mirror.py --orig "output/Opus 5.5/double_track/v1/last_synth" \\
        --mirror "output/Opus 5.5/double_track/v1_mirror/last_synth" --variants final_b2_x final_b2
"""

import argparse
import json
from pathlib import Path

import numpy as np

from exp20_split import guard

TOL = 1.0
CX_D = np.arange(0.0, 151.0, 5.0)


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
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True)
    p.add_argument("--mirror", required=True)
    p.add_argument("--variants", nargs="+", required=True, help="первый — основной (лежит в корне)")
    p.add_argument("--bags", nargs="*", default=None)
    p.add_argument("--holdout", action="store_true")
    a = p.parse_args()
    bags = a.bags or sorted(f.stem for f in Path(a.orig).glob("*.json"))
    guard(bags, a.holdout)
    main_v = a.variants[0]
    for v in a.variants:
        sub = "" if v == main_v else v
        print(f"вариант {v}: подтверждённые находки и ось габарита, исходная против отражённой")
        for bag in bags:
            fo, fm = Path(a.orig) / sub / f"{bag}.json", Path(a.mirror) / sub / f"{bag}.json"
            if not (fo.exists() and fm.exists()):
                continue
            so, sm = json.load(open(fo))["stats"], json.load(open(fm))["stats"]
            n = min(len(so), len(sm))
            co = [s.get("conf", []) if s.get("ok") else [] for s in so[:n]]
            cm = [s.get("conf", []) if s.get("ok") else [] for s in sm[:n]]
            diff = [k for k in range(n) if not same(co[k], cm[k])]
            ep = episodes(diff)
            line = (f"  {bag:36s} кадров {n}: с находкой {sum(1 for c in co if c)} / "
                    f"{sum(1 for c in cm if c)}, расходятся {len(diff)}")
            if v == main_v:
                d50, d75, big = [], [], 0
                for k in range(n):
                    if not (so[k].get("ok") and sm[k].get("ok")) or "cx" not in so[k]:
                        continue
                    e = np.array(so[k]["cx"]) + np.array(sm[k]["cx"])
                    lim = min(so[k]["limit"], sm[k]["limit"])
                    d50.append(abs(e[10]) if lim >= 50 else np.nan)
                    d75.append(abs(e[15]) if lim >= 75 else np.nan)
                    big += bool(lim >= 50 and abs(e[10]) > 0.3)
                line += (f"; ось |исх + отр| на 50 / 75 м: медиана {np.nanmedian(d50):.3f} / "
                         f"{np.nanmedian(d75):.3f} м, кадров > 0.3 м на 50 м: {big}")
            if ep:
                line += " — эпизоды " + ", ".join(f"{x}–{y}" for x, y in ep[:8])
            print(line)
            for x, y in ep[:4]:
                print(f"      кадр {x}: исходная {np.round(co[x], 1).tolist()}, отражённая "
                      f"{np.round(cm[x], 1).tolist()}")


if __name__ == "__main__":
    main()
