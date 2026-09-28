#!/usr/bin/env python3
"""Эксперимент 20: мерило выезда на двухпутный по покадровой статистике прогона.

Статистика — JSON, который пишет `make_far_gifs.py` рядом с GIF (на каждый кадр:
находки сырые и подтверждённые, центр габарита на глубинах `CX_D`, какой старт
выбран). Правда — `exp20_truth.py` (истинный путь по /tf, рабочий по разметке)
и пикеты сцены из README выгрузки (`results/exp20/scenes.json`).

Что считается по каждой записи:

- **путь:** ось габарита минус истинная ось пути вбок на 25 / 50 / 75 / 100 м —
  медиана и 90-й перцентиль модуля по кадрам; отдельно на участке выезда
  (поезд между «портал − 150 м» и «слияние + 30 м», то есть раструб в поле
  зрения) и доля кадров участка, где на 50 м ошибка больше 0.3 м;
- **рабочий:** дальность первой сырой и первой подтверждённой находки,
  непрерывность от подтверждения до 5 м — мерило §34 (`exp_far_eval.score`);
- **ложные:** кадры с подтверждённой (и сырой) находкой не у рабочего.

    python exp20_eval.py --stats "output/Opus 5.5/double_track/base"
    python exp20_eval.py --stats A B --names база итог        # сравнение
    python exp20_eval.py --stats DIR --mirror                  # зеркальные копии
"""

import argparse
import json
from pathlib import Path

import numpy as np

from exp20_split import DEV_LAST_SYNTH, HOLDOUT_LAST_SYNTH, guard
from exp_far_eval import score

CX_D = np.arange(0.0, 151.0, 5.0)
EVAL_D = (25.0, 50.0, 75.0, 100.0)
SCENES = json.load(open("results/exp20/scenes.json"))


def load_stats(stats_dir, bag):
    st = json.load(open(Path(stats_dir) / f"{bag}.json"))["stats"]
    return st


def path_errors(st, bag, mirror=False):
    """Ошибка оси габарита вбок на EVAL_D по кадрам: массив (кадров, len(EVAL_D)),
    nan — кадр без геометрии или правда туда не дотягивается."""
    z = np.load(f"output/exp20_truth/{bag}.npz")
    D, px = z["D"], z["path_x"] * (-1.0 if mirror else 1.0)
    n = len(st)
    err = np.full((n, len(EVAL_D)), np.nan)
    for k, s in enumerate(st):
        if not s.get("ok") or "cx" not in s or k >= len(px):
            continue
        cx = np.interp(EVAL_D, CX_D, s["cx"])
        tx = np.interp(EVAL_D, D, px[k], left=np.nan, right=np.nan)
        lim = s.get("limit", np.inf)
        e = cx - tx
        e[np.array(EVAL_D) > lim] = np.nan      # дальше предела пути габарит не ставится
        err[k] = e
    return err, z["s"]


def zone(bag, s_train):
    sc = SCENES[bag]
    return (s_train >= sc["portal"] - 150.0) & (s_train <= sc["merge"] + 30.0)


def evaluate(stats_dir, bag, mirror=False):
    st = load_stats(stats_dir, bag)
    err, s_train = path_errors(st, bag, mirror)
    zn = zone(bag, s_train[:len(st)])
    det = [(s.get("raw", []), s.get("conf", [])) if s.get("ok") else ([], []) for s in st]
    cols = {"idx": np.arange(len(st)), "ds": np.array([s.get("ds", np.nan) for s in st], float)}
    sc = score("Synthetic_data", bag, cols, det)
    a = np.abs(err)
    out = {"n": len(st), "false_conf": sc["false_conf"], "false_raw": sc["false_raw"],
           "worker": sc["objects"].get("obstacle", {}),
           "ds_meas": float(np.mean([s.get("ds_meas", False) for s in st if s.get("ok")])),
           "memory": float(np.mean([s.get("origin") == "память" for s in st if s.get("ok")])),
           "err_all": {}, "err_zone": {}}
    for j, d in enumerate(EVAL_D):
        for key, m in (("err_all", np.ones(len(st), bool)), ("err_zone", zn)):
            v = a[m, j]
            v = v[np.isfinite(v)]
            out[key][int(d)] = (float(np.median(v)), float(np.percentile(v, 90)), len(v)) if len(v) else None
    v50 = a[zn, 1]
    ok = np.isfinite(v50)
    out["zone_bad50"] = float(np.mean(v50[ok] > 0.3)) if ok.any() else None
    # ложные подтверждённые — в каких кадрах и на какой дальности (для разбора)
    tr = json.load(open(f"output/synthetic_truth/{bag}.json"))
    wk = {r["idx"]: r["depth_min"] for r in tr["frames"] if r["points"] > 0}
    fl = []
    for k, (raw, conf) in enumerate(det):
        bad = [d for d in conf if not (k in wk and abs(d - wk[k]) <= max(2.0, 0.06 * wk[k]))]
        if bad:
            fl.append((k, round(min(bad), 1)))
    out["false_frames"] = fl
    return out


def evaluate_cache(cache_dir, bag, variant):
    """Тот же замер рабочего и ложных, но находки — вариант детектора на кэше
    (`exp_far_cache.py` в новой конфигурации трекера): путь в кэше не хранится,
    поэтому без ошибки пути."""
    import exp_far_eval as fe
    from exp_far_cache import load
    cols, pts = load(cache_dir, bag)
    det = fe.run_variant(variant, cols, pts)
    sc = score("Synthetic_data", bag, cols, det)
    tr = json.load(open(f"output/synthetic_truth/{bag}.json"))
    wk = {r["idx"]: r["depth_min"] for r in tr["frames"] if r["points"] > 0}
    fl = [(k, round(min(b), 1)) for k, (raw, conf) in enumerate(det)
          for b in [[d for d in conf if not (k in wk and abs(d - wk[k]) <= max(2.0, 0.06 * wk[k]))]] if b]
    return {"n": len(det), "false_conf": sc["false_conf"], "false_raw": sc["false_raw"],
            "worker": sc["objects"].get("obstacle", {}), "false_frames": fl,
            "ds_meas": float(np.mean(cols["ds_meas"])), "memory": float("nan"),
            "err_all": {int(d): None for d in EVAL_D}, "err_zone": {int(d): None for d in EVAL_D},
            "zone_bad50": None}


def fmt(v, f="{:.0f}"):
    return "—" if v is None else f.format(v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", nargs="*", default=[], help="папки со статистикой make_far_gifs")
    p.add_argument("--cache", default=None, help="кэш exp_far_cache.py — варианты детектора за секунды")
    p.add_argument("--variants", nargs="*", default=[], help="варианты детектора для --cache")
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--bags", nargs="+", default=DEV_LAST_SYNTH)
    p.add_argument("--mirror", action="store_true", help="статистика по зеркальным копиям")
    p.add_argument("--holdout", action="store_true")
    p.add_argument("--json", default=None)
    p.add_argument("--false", action="store_true", help="перечислить кадры ложных")
    a = p.parse_args()
    if a.holdout:
        a.bags = HOLDOUT_LAST_SYNTH if a.bags == DEV_LAST_SYNTH else a.bags
    guard(a.bags, a.holdout)
    names = a.names or [Path(s).name for s in a.stats]
    res = {}
    for name, sd in zip(names, a.stats):
        for bag in a.bags:
            if (Path(sd) / f"{bag}.json").exists():
                res.setdefault(bag, {})[name] = evaluate(sd, bag, a.mirror)
    if a.cache:
        for v in a.variants:
            for bag in a.bags:
                if (Path(a.cache) / f"{bag}.npz").exists():
                    res.setdefault(bag, {})[v] = evaluate_cache(a.cache, bag, v)
        names = names + a.variants
    print(f"{'запись':14s} {'вариант':10s} | путь: |ошибка| медиана/90% на 50 и 75 м — весь прогон; "
          f"участок выезда | доля >0.3 м на 50 м | рабочий: сыр/подтв м, кадров, непр | ложн подтв/сыр | Δs изм | память")
    for bag in a.bags:
        for name in names:
            r = res.get(bag, {}).get(name)
            if r is None:
                continue
            ea, ez = r["err_all"], r["err_zone"]
            ea = {k: v for k, v in ea.items()}
            for e in (ea, ez):
                for d in (50, 75):
                    e.setdefault(d, None)
            cell = lambda e, d: "—" if e[d] is None else f"{e[d][0]:.2f}/{e[d][1]:.2f}"
            w = r["worker"]
            print(f"{bag:14s} {name:10s} | {cell(ea, 50)} {cell(ea, 75)} ; {cell(ez, 50)} {cell(ez, 75)} | "
                  f"{fmt(r['zone_bad50'], '{:.2f}')} | {fmt(w.get('raw_max'))}/{fmt(w.get('conf_max'))} м, "
                  f"{w.get('n_conf', 0)}, {fmt(w.get('cont'), '{:.2f}')} | {r['false_conf']}/{r['false_raw']} | "
                  f"{r['ds_meas']:.2f} | {r['memory']:.2f}")
            if a.false and r["false_frames"]:
                print(f"{'':26s}ложные подтв.: " + ", ".join(f"{k}:{d}" for k, d in r["false_frames"]))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.json, "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
