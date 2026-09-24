#!/usr/bin/env python3
"""Эксперимент 18: мерило обнаружения на кэше (`exp_far_cache.py`).

Одно мерило для эталона и для каждого варианта детектора:

- по каждому предмету (New_synth 1–5, три синтетические записи с разметкой,
  человек на doubleT_obstacle): дальность первой сырой и первой подтверждённой
  находки; непрерывность — доля кадров с подтверждённой находкой от первого
  подтверждения до 5 м (провалы и есть «вспышки»);
- для предмета, которого в габарите нет (New_synth 5), — кадры с находкой;
- ложные: кадры с находкой, не совпавшей ни с одним предметом.

На New_synth считается только участок разработки — длина пути меньше S_B
(`results/exp18/new_synth_objects.json`); всё дальше вырезается ДО подсчёта.

    python exp_far_eval.py                  # эталон и все варианты
    python exp_far_eval.py --variants base train
"""

import argparse
import json
from pathlib import Path

import numpy as np

from exp_far_cache import load
from exp_far_truth import track_length
from rail_detection import far_detect as fd

CACHE = Path("output/exp18_cache")
DEV = [("new_synth", "cloud_with_fake_obj"),
       ("Synthetic_data", "box"), ("Synthetic_data", "human_smashed"),
       ("Synthetic_data", "human_smashed_diff_tunnels"),
       ("Dataset", "doubleT_obstacle"),
       ("Dataset", "doubleT_platform"), ("Dataset", "roundT_doubleT"),
       ("Dataset", "roundT_pressureGate_roundT"), ("Dataset", "squareT_platform_squareT_switch")]
PERSON = (4, 75)        # doubleT_obstacle: человек на пути (§25)
NEAR_END = 5.0          # до какой дальности считается непрерывность


def tol(d):
    return max(2.0, 0.04 * d)


def truth_for(tag, bag, cols):
    """Покадровая правда: список [(дальность, в габарите?, имя)] на кадр и
    предел длины пути (для New_synth — S_B, иначе бесконечность)."""
    n = len(cols["idx"])
    T = [[] for _ in range(n)]
    S = track_length(cols)
    s_cut = np.inf
    if tag == "new_synth":
        info = json.load(open("results/exp18/new_synth_objects.json"))
        s_cut = info["S_B"]
        for o in info["objects"]:
            for k in range(n):
                d = o["s_obj"] - S[k]
                if -1.0 < d < 200.0:
                    T[k].append((d, o["positive"], f"{o['n']}"))
    elif tag == "Synthetic_data":
        tr = json.load(open(f"output/synthetic_truth/{bag}.json"))
        for r in tr["frames"]:
            if r["points"] > 0 and r["idx"] < n:
                T[r["idx"]].append((r["depth_min"], True, "obstacle"))
    elif bag == "doubleT_obstacle":
        for k in range(PERSON[0], PERSON[1] + 1):
            T[k].append((55.5, True, "person"))
    return T, S, s_cut


def score(tag, bag, cols, det, verbose=False):
    """det[k] = (raw: [dist...], conf: [dist...]) — находки варианта на кадре k."""
    T, S, s_cut = truth_for(tag, bag, cols)
    objs = {}
    false_raw = false_conf = 0
    for k, (raw, conf) in enumerate(det):
        # участок отложенной части вырезается до подсчёта
        raw = [d for d in raw if S[k] + d < s_cut]
        conf = [d for d in conf if S[k] + d < s_cut]
        tk = [t for t in T[k] if S[k] + t[0] < s_cut]
        for t in tk:
            o = objs.setdefault(t[2], {"pos": t[1], "raw": [], "conf": [], "seen": []})
            o["seen"].append((k, t[0]))
            if any(abs(d - t[0]) <= tol(t[0]) for d in raw):
                o["raw"].append((k, t[0]))
            if any(abs(d - t[0]) <= tol(t[0]) for d in conf):
                o["conf"].append((k, t[0]))
        unm = lambda ds: [d for d in ds if not any(abs(d - t[0]) <= tol(t[0]) for t in tk)]
        false_raw += bool(unm(raw))
        false_conf += bool(unm(conf))
    out = {"false_raw": false_raw, "false_conf": false_conf, "objects": {}}
    for name, o in sorted(objs.items()):
        r = {"pos": o["pos"], "n_raw": len(o["raw"]), "n_conf": len(o["conf"]),
             "raw_max": max((d for _, d in o["raw"]), default=None),
             "conf_max": max((d for _, d in o["conf"]), default=None)}
        if o["pos"] and r["conf_max"] is not None:
            span = [k for k, d in o["seen"] if NEAR_END <= d <= r["conf_max"]]
            got = {k for k, _ in o["conf"]}
            r["cont"] = float(np.mean([k in got for k in span])) if span else None
            r["gaps"] = int(sum(1 for a, b in zip(span, span[1:]) if a in got and b not in got))
        out["objects"][name] = r
    return out


def run_variant(name, cols, pts):
    """Покадровые находки варианта: (сырые дальности, подтверждённые дальности)."""
    n = len(cols["idx"])
    if name == "base_cached":
        # эталон как он записан в кэше при проходе трекера
        det = []
        for k in range(n):
            raw = [c["dist"] for c in cols["base_clusters"][k]]
            conf = [cols["base_conf"][k]] if np.isfinite(cols["base_conf"][k]) else []
            det.append((raw, conf))
        return det
    params = fd.VARIANTS[name]
    det = fd.run_sequence(params, n, lambda k: pts(k) if cols["ok"][k] else None,
                          cols["limit"], cols["ds"])
    return det


def fmt(v, f="{:.0f}"):
    return "—" if v is None else f.format(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=["base_cached"] + list(fd.VARIANTS))
    p.add_argument("--bags", nargs="+", default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    allres = {}
    for tag, bag in DEV:
        if a.bags and bag not in a.bags:
            continue
        cols, pts = load(CACHE / tag, bag)
        print(f"\n=== {bag} ({len(cols['idx'])} кадров)")
        for v in a.variants:
            det = run_variant(v, cols, pts)
            sc = score(tag, bag, cols, det)
            allres.setdefault(bag, {})[v] = sc
            objs = "  ".join(
                f"[{k}{'' if o['pos'] else ' (вне)'}: сыр {fmt(o['raw_max'])} / подтв "
                f"{fmt(o['conf_max'])} м, {o['n_conf']} кадр"
                + (f", непр {o['cont']:.2f}, провалов {o['gaps']}" if o.get("cont") is not None else "")
                + "]" for k, o in sc["objects"].items())
            print(f"  {v:14s} ложных: сырых {sc['false_raw']:4d}, подтв {sc['false_conf']:4d}   {objs}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(allres, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
