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

from exp_far_cache import SIDE_S, SIDE_V, load, side_view
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
# doubleT_obstacle: коробка на левом рельсе (u −0.98…−0.70, верх 0.16–0.30 м), 56.4 м —
# человек кладёт её на кадрах 30–45, лежит с кадра 50 до конца записи. Замечено
# заказчиком после отложенного замера эксперимента 18, проверено по точкам.
BOX = (50, 200, 56.4)
NEAR_END = 5.0          # до какой дальности считается непрерывность


def tol(d):
    # 6%: сумма Δs трекера на New_synth за разгон уходит на ~5% (предмет 1:
    # 98 м по находке на кадре 2 против 103 м по разметке вблизи)
    return max(2.0, 0.06 * d)


HOLDOUT = False   # --holdout: New_synth — только отложенная часть (длина пути >= S_B)


def truth_for(tag, bag, cols):
    """Покадровая правда: список [(дальность, в габарите?, имя)] на кадр и
    отбор по длине пути: для New_synth на разработке — меньше S_B, на
    отложенном замере — не меньше S_B; иначе — всё."""
    n = len(cols["idx"])
    T = [[] for _ in range(n)]
    S = track_length(cols)
    s_cut = np.inf
    if tag == "new_synth":
        info = json.load(open("results/exp18/new_synth_objects.json"))
        s_cut = info["S_B"]
        objs = info["objects"]
        if HOLDOUT:
            objs = json.load(open("results/exp18/new_synth_holdout_objects.json"))["objects"]
            s_cut = -info["S_B"]      # отрицательная граница: берётся всё, что дальше |s_cut|
        for o in objs:
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
        for k in range(BOX[0], min(BOX[1] + 1, n)):
            if k > PERSON[1]:
                T[k].append((BOX[2], True, "box"))
    return T, S, s_cut


def score(tag, bag, cols, det, verbose=False):
    """det[k] = (raw: [dist...], conf: [dist...]) — находки варианта на кадре k."""
    T, S, s_cut = truth_for(tag, bag, cols)
    keep = (lambda sg: sg < s_cut) if s_cut >= 0 else (lambda sg: sg >= -s_cut)
    objs = {}
    false_raw = false_conf = 0
    for k, (raw, conf) in enumerate(det):
        # участок другой части вырезается до подсчёта
        raw = [d for d in raw if keep(S[k] + d)]
        conf = [d for d in conf if keep(S[k] + d)]
        tk = [t for t in T[k] if keep(S[k] + t[0])]
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
    getside = None
    if "side" in cols:
        getside = lambda k: (side_view(cols, k), SIDE_S, SIDE_V)
    det = fd.run_sequence(params, n, lambda k: pts(k) if cols["ok"][k] else None,
                          cols["limit"], cols["ds"], getside=getside,
                          alt_dx=cols.get("alt_dx"))
    return det


def fmt(v, f="{:.0f}"):
    return "—" if v is None else f.format(v)


def _job(args):
    tag, bag, v, holdout = args
    global HOLDOUT
    HOLDOUT = holdout
    cols, pts = load(CACHE / tag, bag)
    return bag, v, score(tag, bag, cols, run_variant(v, cols, pts))


REAL_CLEAN = ["doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
              "squareT_platform_squareT_switch"]
SYN = ["box", "human_smashed", "human_smashed_diff_tunnels"]


def summary(allres, variants):
    """Одна строка на вариант: New_synth (предметы 1–4 — дальность подтверждения
    и непрерывность, 5 — кадры с находкой, ложные), синтетика (дальность
    подтверждения, ложные), реальные (ложные подтверждённые по прогонам, человек)."""
    print(f"\n{'вариант':16s} | New_synth: 1 / 2 / 3 / 4 подтв. м (непр.) | 5 кадр | ложн | "
          f"синт: box / hs / hsdt | ложн | реальн. ложн подтв: plat dbl gate sw obst | человек")
    for v in variants:
        ns = allres.get("cloud_with_fake_obj", {}).get(v)
        cell = ""
        if ns:
            o = ns["objects"]
            cell = " / ".join(f"{fmt(o[k]['conf_max'])}({fmt(o[k].get('cont'), '{:.2f}')})"
                              for k in "1234" if k in o)
            cell += f" | {o['5']['n_conf'] if '5' in o else '—':>4} | {ns['false_conf']:4d}"
        syn = [allres.get(b, {}).get(v) for b in SYN]
        sc = " / ".join(fmt(s["objects"]["obstacle"]["conf_max"]) if s else "—" for s in syn)
        sf = sum(s["false_conf"] for s in syn if s)
        real = [allres.get(b, {}).get(v) for b in REAL_CLEAN + ["doubleT_obstacle"]]
        rf = " ".join(f"{r['false_conf']:4d}" if r else "   —" for r in real)
        ob = allres.get("doubleT_obstacle", {}).get(v)
        person = f"{ob['objects']['person']['n_conf']}/72" if ob else "—"
        if ob and "box" in ob["objects"]:
            person += f", коробка {ob['objects']['box']['n_conf']}/125"
        print(f"{v:16s} | {cell} | {sc} | {sf:4d} | {rf} | {person}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=["base_cached"] + list(fd.VARIANTS))
    p.add_argument("--bags", nargs="+", default=None)
    p.add_argument("--json", default=None)
    p.add_argument("--summary", action="store_true", help="только сводная таблица")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--holdout", action="store_true",
                   help="отложенный замер: New_synth дальше S_B и roundT_squareT_pressureGate_squareT")
    a = p.parse_args()
    global HOLDOUT, DEV
    if a.holdout:
        HOLDOUT = True
        DEV = [("new_synth", "cloud_with_fake_obj"),
               ("Dataset", "roundT_squareT_pressureGate_squareT")]
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(tag, bag, v, HOLDOUT) for tag, bag in DEV if not a.bags or bag in a.bags
            for v in a.variants]
    allres = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for bag, v, sc in ex.map(_job, jobs):
            allres.setdefault(bag, {})[v] = sc
    if not a.summary:
        for tag, bag in DEV:
            if bag not in allres:
                continue
            print(f"\n=== {bag}")
            for v in a.variants:
                sc = allres[bag][v]
                objs = "  ".join(
                    f"[{k}{'' if o['pos'] else ' (вне)'}: сыр {fmt(o['raw_max'])} / подтв "
                    f"{fmt(o['conf_max'])} м, {o['n_conf']} кадр"
                    + (f", непр {o['cont']:.2f}, провалов {o['gaps']}" if o.get("cont") is not None else "")
                    + "]" for k, o in sc["objects"].items())
                print(f"  {v:14s} ложных: сырых {sc['false_raw']:4d}, подтв {sc['false_conf']:4d}   {objs}")
    summary(allres, a.variants)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(allres, f, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
