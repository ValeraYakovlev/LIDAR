#!/usr/bin/env python3
"""Эксперимент 14: оценка трёх взглядов на вид сверху — силуэт, пол, свод.

Эталона геометрии тоннеля нет, поэтому качество меряется тремя косвенными, но
НЕЗАВИСИМЫМИ от подходов способами:

  рельсы     — ни один из трёх подходов детектор рельсов не использует. Ось
               подхода сравнивается с центром колеи на 15-35 м после того, как
               постоянный сдвиг снят по ближним рельсам (до 15 м): на двухпутном
               участке ось тоннеля и ось пути законно не совпадают, а форма у
               них общая. Для сравнения то же считается для текущего метода
               (tunnel_frame + накопление) — это мерило «хорошо».
  дрожание   — поперечное положение оси на фиксированной глубине D меняется от
               кадра к кадру плавно (поезд проходит 1.5 м за кадр). Мера —
               вторая разность по времени |x_t − (x_{t−1} + x_{t+1})/2|: плавный
               поворот её не создаёт, а скачок подгонки создаёт.
  согласие   — два подхода смотрят на разные поверхности, и если на 60/90 м они
               независимо дают один и тот же увод оси вбок, это сильный довод,
               что оба правы. Сравнивается увод x(D) − x(15), без сдвига.

Отложенный прогон — `roundT_squareT_pressureGate_squareT` (самый трудный,
гермозатвор + стрелки + смена сечения). Он меряется только флагом --validate,
и только один раз, после того как решения по алгоритму приняты.

Запуск:
    python eval_views.py                       # 4 разработочных прогона
    python eval_views.py --validate            # отложенный прогон, один раз
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from rail_detection import bag_path, iter_frames, rail_samples
from rail_detection.tracker import TunnelTracker
from rail_detection.tunnel_frame import tracked_depth, tunnel_center_coeffs
from rail_detection.views import METHODS, curvature, fit_views, half_width, shape_x

DEV_RUNS = ["doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
            "squareT_platform_squareT_switch"]
VALIDATION_RUN = "roundT_squareT_pressureGate_squareT"

DEPTHS = np.array([15.0, 30.0, 45.0, 60.0, 90.0, 120.0])
RAIL_NEAR = 15.0
RAIL_FAR = (15.0, 35.0)
ALL = list(METHODS) + ["reference"]


def collect(dataset, bag, stride, max_frames=None, reference=True):
    """Один проход по записи: три подхода + текущий метод + рельсы на каждом кадре."""
    tracker = TunnelTracker() if reference else None
    priors = {}
    out = []
    t0 = time.time()
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=stride,
                                           max_frames=max_frames):
        _, res = fit_views(points, priors)
        rec = {"idx": idx}
        for m in METHODS:
            fit = res[m]["fit"]
            priors[m] = fit
            if fit is None:
                rec[m] = None
                continue
            rec[m] = {"x": shape_x(fit, DEPTHS), "reach": fit["reach"], "deg": fit["deg"],
                      "hw": half_width(fit), "k40": curvature(fit, 40.0),
                      "reach_side": fit["reach_side"]}
        if tracker is not None:
            ref = tracker.update(points, steps=stride)
            if ref is not None:
                reach = [r for r in (tracked_depth(ref, s) for s in ("left", "right")) if r]
                rec["reference"] = {"x": np.polyval(tunnel_center_coeffs(ref), DEPTHS),
                                    "reach": max(reach) if reach else None}
            else:
                rec["reference"] = None
        rd, rx, _, _ = rail_samples(points)
        rec["rails"] = (rd, rx)
        out.append(rec)
        print(f"\r  {bag}: кадр {idx}/{n_total}  ({time.time() - t0:.0f} с)", end="", flush=True)
    print()
    return out


def _x_at(r, d):
    """x оси на глубине d по записанным глубинам (линейно между ними)."""
    return float(np.interp(d, DEPTHS, r["x"]))


def rail_error(records, method, near=RAIL_NEAR, far=RAIL_FAR, min_near=3, min_far=2):
    """Ошибка ФОРМЫ оси на отложенных рельсах: сдвиг снят по ближним, мерится дальний.

    Рельсовые срезы, где колея ушла от оси подхода больше чем на 1.2 м после
    снятия сдвига, отброшены: это детектор рельсов схватил соседний путь на
    двухпутном участке (§25), а не ошибка оси — такая «ошибка» одинаково
    наказывала бы все подходы и прятала бы разницу между ними.
    """
    errs = []
    for rec in records:
        r = rec.get(method)
        rd, rx = rec["rails"]
        if r is None or len(rd) == 0:
            continue
        xm = np.array([_x_at(r, d) for d in rd])
        dn = rd <= near
        if dn.sum() < min_near:
            continue
        off = float(np.median(rx[dn] - xm[dn]))
        df = (rd > far[0]) & (rd <= far[1])
        e = np.abs(rx[df] - xm[df] - off)
        e = e[e < 1.2]
        if len(e) >= min_far:
            errs.append(float(np.median(e)))
    return np.array(errs)


def jitter(records, method, depth):
    """Вторая разность по времени оси на глубине depth — там, где она наблюдается
    во всех трёх соседних кадрах."""
    out = []
    for a, b, c in zip(records, records[1:], records[2:]):
        rs = [q.get(method) for q in (a, b, c)]
        if any(r is None or r["reach"] is None or r["reach"] < depth for r in rs):
            continue
        xa, xb, xc = (_x_at(r, depth) for r in rs)
        out.append(abs(xb - 0.5 * (xa + xc)))
    return np.array(out)


def coverage(records, method, depth):
    """Доля кадров, где ось подхода наблюдается до глубины depth."""
    ok = [rec.get(method) is not None and rec[method]["reach"] is not None
          and rec[method]["reach"] >= depth for rec in records]
    return float(np.mean(ok)) if ok else 0.0


def agreement(records, ma, mb, depth, base=15.0):
    """|увод_a(D) − увод_b(D)|, увод = x(D) − x(base): сравнивается форма, не сдвиг."""
    out = []
    for rec in records:
        a, b = rec.get(ma), rec.get(mb)
        if a is None or b is None or a["reach"] is None or b["reach"] is None:
            continue
        if min(a["reach"], b["reach"]) < depth:
            continue
        da = _x_at(a, depth) - _x_at(a, base)
        db = _x_at(b, depth) - _x_at(b, base)
        out.append(abs(da - db))
    return np.array(out)


def _q(a, p=50):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def summarize(runs):
    table = {}
    for bag, recs in runs.items():
        n = len(recs)
        row = {"frames": n}
        for m in ALL:
            if not any(m in r for r in recs):
                continue
            ok = [r.get(m) is not None for r in recs]
            reach = np.array([r[m]["reach"] for r in recs
                              if r.get(m) is not None and r[m]["reach"] is not None])
            re = rail_error(recs, m)
            row[m] = {
                "ok": float(np.mean(ok)),
                "reach_med": _q(reach), "reach_p10": _q(reach, 10),
                "rail_med": _q(re), "rail_p90": _q(re, 90), "rail_n": int(len(re)),
                **{f"cov{int(D)}": coverage(recs, m, D) for D in (60, 90)},
                **{f"jit{int(D)}_med": _q(jitter(recs, m, D)) for D in (30, 60, 90)},
                **{f"jit{int(D)}_p90": _q(jitter(recs, m, D), 90) for D in (30, 60, 90)},
            }
        pairs = [("silhouette", "floor"), ("silhouette", "ceiling"), ("floor", "ceiling"),
                 ("silhouette", "reference"), ("floor", "reference"), ("ceiling", "reference")]
        row["agree"] = {}
        for a, b in pairs:
            for D in ((40.0,) if b == "reference" else (60.0, 90.0)):
                ag = agreement(recs, a, b, D)
                row["agree"][f"{a}~{b}@{int(D)}"] = {"med": _q(ag), "p90": _q(ag, 90),
                                                    "n": int(len(ag))}
        table[bag] = row
    return table


def print_table(table):
    for bag, row in table.items():
        print(f"\n=== {bag} ({row['frames']} кадров) ===")
        print(f"{'подход':11s} {'есть':>5s} {'дальн.':>7s} {'p10':>5s} {'до60':>5s} {'до90':>5s} "
              f"{'рельсы':>7s} {'p90':>5s} {'n':>4s}  {'дрож30':>6s} {'дрож60':>6s} {'дрож90':>6s}"
              f" {'p90@60':>6s}")
        for m in ALL:
            if m not in row:
                continue
            s = row[m]
            print(f"{m:11s} {s['ok']:5.2f} {s['reach_med']:7.0f} {s['reach_p10']:5.0f} "
                  f"{s['cov60']:5.2f} {s['cov90']:5.2f} {s['rail_med']:7.3f} {s['rail_p90']:5.2f} "
                  f"{s['rail_n']:4d}  {s['jit30_med']:6.3f} {s['jit60_med']:6.3f} "
                  f"{s['jit90_med']:6.3f} {s['jit60_p90']:6.2f}")
        print("  согласие формы (|увод_a − увод_b|, м): медиана / p90 / кадров")
        for k, v in row["agree"].items():
            print(f"    {k:28s} {v['med']:6.2f} {v['p90']:6.2f} {v['n']:5d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=None)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--validate", action="store_true",
                   help=f"мерить ТОЛЬКО отложенный прогон {VALIDATION_RUN} (один раз!)")
    p.add_argument("--no-reference", action="store_true", help="не считать текущий метод")
    p.add_argument("--out", default="output/views_eval")
    a = p.parse_args()

    if a.validate:
        bags = [VALIDATION_RUN]
    else:
        bags = a.bags or DEV_RUNS
        if VALIDATION_RUN in bags:
            raise SystemExit(f"{VALIDATION_RUN} отложен для валидации: только через --validate")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for bag in bags:
        runs[bag] = collect(a.dataset, bag, a.stride, a.max_frames, not a.no_reference)
    table = summarize(runs)
    print_table(table)
    tag = "validation" if a.validate else "dev"
    with open(out / f"records_{tag}.pkl", "wb") as f:
        pickle.dump(runs, f)
    with open(out / f"summary_{tag}.json", "w") as f:
        json.dump(table, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nзаписано: {out}/summary_{tag}.json")


if __name__ == "__main__":
    main()
