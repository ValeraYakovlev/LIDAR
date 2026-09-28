#!/usr/bin/env python3
"""Эксперимент 21: варианты детектора на кэшах ВСЕХ записей — предметы и ложные.

Кэш — `exp_far_cache.py` в текущей конфигурации трекера (`output/exp21_cache`).
Отложенных данных нет (все использованы в §31, §34, §36), поэтому мерило
идёт по всем записям сразу: низкие предметы (коробка на `doubleT_obstacle`,
New_synth 3 и 9, старая `box`) и ложные подтверждённые находки везде.

    python exp21_eval.py --variants final_b2_f low_rest low_rest_b0
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import exp_far_eval as fe
from exp_far_cache import load

CACHE = Path("output/exp21_cache")
LS = ["conv_r300_a30", "conv_r300_a45", "conv_r450_a15", "conv_r450_a30", "conv_r450_a45",
      "conv_r600_a15", "conv_r600_a30", "conv_r600_a45"]
REAL = ["doubleT_obstacle", "doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
        "roundT_squareT_pressureGate_squareT", "squareT_platform_squareT_switch"]
# (папка кэша, запись, тег правды для exp_far_eval.score)
RUNS = ([("Dataset", b, "Dataset") for b in REAL] + [("reversed", b, "Dataset") for b in REAL]
        + [("Synthetic_data", b, "Synthetic_data") for b in ("box", "human_smashed",
                                                             "human_smashed_diff_tunnels")]
        + [("new_synth", "cloud_with_fake_obj", "new_synth")]
        + [("last_synth", b, "Synthetic_data") for b in LS]
        + [("Last_synth_data", b, "Synthetic_data") for b in LS])


def _job(args):
    sub, bag, tag, v = args
    cols, pts = load(CACHE / sub, bag)
    det = fe.run_variant(v, cols, pts)
    out = {}
    if tag == "new_synth":
        for part, hold in (("dev", False), ("hold", True)):
            fe.HOLDOUT = hold
            out[part] = fe.score(tag, bag, cols, det)
        fe.HOLDOUT = False
    else:
        out["all"] = fe.score(tag, bag, cols, det)
    return sub, bag, v, out


def fmt(o, key="conf_max"):
    v = o.get(key)
    return "—" if v is None else f"{v:.0f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    jobs = [(s, b, t, v) for s, b, t in RUNS if (CACHE / s / f"{b}.npz").exists() for v in a.variants]
    res = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for sub, bag, v, out in ex.map(_job, jobs):
            res.setdefault(v, {})[f"{sub}/{bag}"] = out
    for v in a.variants:
        r = res[v]
        print(f"\n=== {v}")
        fl = {k: x["all"]["false_conf"] for k, x in r.items() if "all" in x}
        fl.update({f"{k}:{part}": x[part]["false_conf"] for k, x in r.items() if "all" not in x
                   for part in x})
        print("  ложные подтв. (кадров): " + ", ".join(f"{k.split('/')[-1] if 'last' not in k.lower() else k}={n}"
                                                     for k, n in fl.items() if n))
        print(f"  всего ложных, кроме roundT_doubleT: {sum(n for k, n in fl.items() if 'roundT_doubleT' not in k)}; "
              f"roundT_doubleT исх/зерк: {r.get('Dataset/roundT_doubleT', {}).get('all', {}).get('false_conf')} / "
              f"{r.get('reversed/roundT_doubleT', {}).get('all', {}).get('false_conf')}")
        for side in ("Dataset", "reversed"):
            o = r.get(f"{side}/doubleT_obstacle", {}).get("all", {}).get("objects", {})
            if o:
                print(f"  doubleT_obstacle ({side}): человек {o['person']['n_conf']}/72, "
                      f"коробка {o.get('box', {}).get('n_conf', 0)}/125 "
                      f"(непр. {fmt(o.get('box', {}), 'cont') if o.get('box', {}).get('cont') is None else round(o['box']['cont'], 2)})")
        ns = r.get("new_synth/cloud_with_fake_obj")
        if ns:
            objs = {**ns["dev"]["objects"], **ns["hold"]["objects"]}
            print("  New_synth, подтв. дальность (кадров): " + "  ".join(
                f"{k}:{fmt(objs[k])}({objs[k]['n_conf']})" for k in sorted(objs, key=int)))
        syn = [f"{b}:{fmt(r[f'Synthetic_data/{b}']['all']['objects'].get('obstacle', {}))}"
               f"({r[f'Synthetic_data/{b}']['all']['objects'].get('obstacle', {}).get('n_conf', 0)})"
               for b in ("box", "human_smashed", "human_smashed_diff_tunnels") if f"Synthetic_data/{b}" in r]
        print("  старая синтетика, подтв. дальность (кадров): " + "  ".join(syn))
        for sub in ("last_synth", "Last_synth_data"):
            w = [r[f"{sub}/{b}"]["all"]["objects"].get("obstacle", {}) for b in LS if f"{sub}/{b}" in r]
            if w:
                print(f"  выезд ({'исх' if sub == 'last_synth' else 'зерк'}): рабочий подтв. кадров "
                      + " ".join(str(o.get("n_conf", 0)) for o in w)
                      + "; первое подтв., м: " + " ".join(fmt(o) for o in w))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
