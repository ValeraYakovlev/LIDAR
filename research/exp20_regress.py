#!/usr/bin/env python3
"""Эксперимент 20: регрессия по статистике живого прогона — мерило §34.

`make_far_gifs.py` пишет на каждый кадр находки (сырые, подтверждённые) и Δs.
Здесь они считаются тем же `exp_far_eval.score`, что и замер на кэше
эксперимента 18: New_synth (предметы 1–5 и, с --all-new-synth, 6–10), старая
синтетика, реальные записи разработки (ложные, человек и коробка на
doubleT_obstacle). Итоговая строка — в той же раскладке, что
`exp_far_eval.py --summary`, чтобы сравнивать с базой на кэше напрямую.

    python exp20_regress.py --root "output/Opus 5.5/double_track/v1" --variants final_b2_x final_b2
"""

import argparse
import json
from pathlib import Path

import numpy as np

import exp_far_eval as fe
from exp20_split import guard

GROUPS = [("new_synth", "cloud_with_fake_obj"),
          ("Synthetic_data", "box"), ("Synthetic_data", "human_smashed"),
          ("Synthetic_data", "human_smashed_diff_tunnels"),
          ("Dataset", "doubleT_obstacle"), ("Dataset", "doubleT_platform"),
          ("Dataset", "roundT_pressureGate_roundT"), ("Dataset", "squareT_platform_squareT_switch"),
          ("Dataset", "roundT_squareT_pressureGate_squareT")]


def stats_path(root, tag, bag, variant, main):
    d = Path(root) / tag
    return d / f"{bag}.json" if variant == main else d / variant / f"{bag}.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="папка прогона: <root>/<набор>/<запись>.json")
    p.add_argument("--variants", nargs="+", required=True, help="первый — основной (GIF)")
    p.add_argument("--tag-dir", default=None, help="подпапка набора, если не совпадает с тегом")
    p.add_argument("--json", default=None)
    p.add_argument("--false", action="store_true", help="перечислить кадры ложных подтверждённых")
    a = p.parse_args()
    guard([b for _, b in GROUPS])
    main_v = a.variants[0]
    allres = {}
    for tag, bag in GROUPS:
        for v in a.variants:
            sp = stats_path(a.root, tag, bag, v, main_v)
            if not sp.exists():
                continue
            st = json.load(open(sp))["stats"]
            det = [(s.get("raw", []), s.get("conf", [])) if s.get("ok") else ([], []) for s in st]
            cols = {"idx": np.arange(len(st)),
                    "ds": np.array([s.get("ds", np.nan) for s in st], float)}
            if tag == "new_synth" and "ds" not in st[1]:
                # у дополнительных вариантов Δs не пишется — он общий с основным
                ms = json.load(open(stats_path(a.root, tag, bag, main_v, main_v)))["stats"]
                cols["ds"] = np.array([s.get("ds", np.nan) for s in ms], float)
            sc = fe.score(tag, bag, cols, det)
            allres.setdefault(bag, {})[v] = sc
            if a.false:
                T, S, s_cut = fe.truth_for(tag, bag, cols)
                keep = (lambda sg: sg < s_cut) if s_cut >= 0 else (lambda sg: sg >= -s_cut)
                fl = []
                for k, (raw, conf) in enumerate(det):
                    tk = [t for t in T[k] if keep(S[k] + t[0])]
                    bad = [d for d in conf if keep(S[k] + d)
                           and not any(abs(d - t[0]) <= fe.tol(t[0]) for t in tk)]
                    if bad:
                        fl.append(f"{k}:{min(bad):.0f}")
                if fl:
                    print(f"  {bag} / {v}: ложные подтв. {', '.join(fl)}")
    fe.summary(allres, a.variants)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(allres, open(a.json, "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
