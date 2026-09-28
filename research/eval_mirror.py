#!/usr/bin/env python3
"""Зеркальная проверка: метод на исходных записях против него же на отражённых.

Отражённая копия записи (`mirror_dataset.py`, x -> -x) — это «новый» прогон,
которого метод не видел: левые повороты стали правыми, соседний путь и
платформа — с другой стороны. Геометрически задача та же, поэтому правильный
метод обязан дать отражённый ответ: левая стена зеркального прогона — это
правая исходного со сменой знака, находки — в тех же кадрах на той же
дальности. Любое расхождение означает одно из двух:

  * метод несимметричен по построению (что-то подобрано под левые повороты);
  * решение стоит на грани порога, и миллиметровая разница — в Δs, в
    случайных выборках RANSAC, в порядке кромок — перекидывает его в другую
    сторону. Так на двухпутном прогоне у wall-parallel-v1 выбор «память /
    заново» на кадре 99 перевернулся, и зеркальный прогон получил
    подтверждённую ложную находку, которой у исходного нет (knowledge.md §32).

Сравниваются покадровые выходы `make_parallel_gifs.py` (<запись>.json) из двух
папок — сам метод здесь не запускается:

    python mirror_dataset.py                          # один раз: /Volumes/T7/reversed
    python make_parallel_gifs.py && python make_parallel_gifs.py --validate
    python make_parallel_gifs.py --dataset /Volumes/T7/reversed --out "output/Opus 5.5/reversed"
    python make_parallel_gifs.py --dataset /Volumes/T7/reversed --out "output/Opus 5.5/reversed" --validate
    python eval_mirror.py --orig "output/Opus 5.5" --mirror "output/Opus 5.5/reversed"
"""

import argparse
import json
from pathlib import Path

import numpy as np

BAGS = ["doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
        "roundT_squareT_pressureGate_squareT", "squareT_platform_squareT_switch",
        "doubleT_obstacle"]
PERSON_FRAMES = (4, 75)     # doubleT_obstacle: человек на путях (§25)
JUMP = 0.5                  # м: скачок стены у поезда между соседними кадрами
DIVERGE = 0.3               # м: зеркальный и исходный ответ разошлись
AGREE = 0.05                # м: ещё совпадают (медиана расхождения — миллиметры)


def run_summary(stats, bag):
    """Сводка одного прогона по покадровому выходу make_parallel_gifs.py."""
    ok = [s for s in stats if s["ok"]]
    raw = [s["idx"] for s in stats if s.get("raw") is not None]
    conf = [s["idx"] for s in stats if s.get("conf")]
    # Скачок стены у поезда, который не объясняется доехавшей ступенькой
    jumps = 0
    for a, b in zip(stats[:-1], stats[1:]):
        if not (a["ok"] and b["ok"]):
            continue
        for side, key in ((0, "wl"), (1, "wr")):
            arrived = any(sb <= 3.0 for sb, _ in a["steps"][side])
            if abs(b[key] - a[key]) > JUMP and not arrived:
                jumps += 1
    out = {"frames": len(stats), "geometry": len(ok), "raw": raw, "confirmed": conf,
           "wall_jumps": jumps,
           "reach_median": float(np.median([s["reach"] for s in ok])) if ok else None}
    if bag == "doubleT_obstacle":
        lo, hi = PERSON_FRAMES
        inside = [s for s in stats if lo <= s["idx"] <= hi]
        out["person"] = {"frames": len(inside),
                         "found": sum(1 for s in inside if s.get("raw") is not None),
                         "confirmed": sum(1 for s in inside if s.get("conf")),
                         "outside": sum(1 for i in raw if not lo <= i <= hi)}
    return out


def episodes(frames):
    """Подряд идущие номера кадров -> [(начало, конец)]."""
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [tuple(e) for e in out]


def compare(orig, mirr):
    """Зеркальная согласованность: левое зеркального = −правое исходного."""
    o = {s["idx"]: s for s in orig}
    m = {s["idx"]: s for s in mirr}
    common = sorted(k for k in o if k in m and o[k]["ok"] and m[k]["ok"])
    dl = np.array([abs(m[k]["wl"] + o[k]["wr"]) for k in common])
    dr = np.array([abs(m[k]["wr"] + o[k]["wl"]) for k in common])
    worst = np.maximum(dl, dr)
    w_of = dict(zip(common, worst))
    div = [k for k in common if w_of[k] > DIVERGE]
    eps = []
    for a, b in episodes(div):
        # Откуда расхождение началось: назад до последнего кадра, где ответы
        # ещё совпадали (AGREE). Решение могло перевернуться раньше, чем
        # расхождение переросло DIVERGE: на двухпутном у wall-parallel-v1
        # выбор перевернулся на кадре 99 (0.26 м), а 0.3 м превышено с 101.
        start = a
        while start - 1 in w_of and w_of[start - 1] > AGREE:
            start -= 1
        window = [k for k in range(start, a + 1) if k in w_of]
        origin_flip = [k for k in window if o[k]["origin"] != m[k]["origin"]]
        steps_flip = [k for k in window
                      if len(o[k]["steps"][0]) != len(m[k]["steps"][1])
                      or len(o[k]["steps"][1]) != len(m[k]["steps"][0])]
        eps.append({"frames": [a, b], "since": start,
                    "max_mismatch": float(max(w_of[k] for k in range(a, b + 1) if k in w_of)),
                    "start_choice_differs": origin_flip, "steps_differ": steps_flip})
    det = [k for k in sorted(o) if k in m
           and ((o[k].get("raw") is None) != (m[k].get("raw") is None)
                or bool(o[k].get("conf")) != bool(m[k].get("conf")))]
    return {"frames_compared": len(common),
            "wall_mismatch_median": [float(np.median(dl)), float(np.median(dr))],
            "wall_mismatch_p90": float(np.percentile(worst, 90)),
            "wall_mismatch_max": float(worst.max()),
            "divergent_frames": len(div), "episodes": eps,
            "detection_differs_frames": det}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig", default="output/Opus 5.5")
    p.add_argument("--mirror", default="output/Opus 5.5/reversed")
    p.add_argument("--bags", nargs="*", default=BAGS)
    p.add_argument("--out", default=None, help="куда записать сводку JSON")
    a = p.parse_args()
    report = {}
    for bag in a.bags:
        fo, fm = Path(a.orig) / f"{bag}.json", Path(a.mirror) / f"{bag}.json"
        if not (fo.exists() and fm.exists()):
            print(f"{bag}: нет {fo if not fo.exists() else fm}, пропущен")
            continue
        so, sm = json.load(open(fo))["stats"], json.load(open(fm))["stats"]
        ro, rm, c = run_summary(so, bag), run_summary(sm, bag), compare(so, sm)
        report[bag] = {"orig": ro, "mirror": rm, "symmetry": c}
        same = (not c["detection_differs_frames"]) and not c["episodes"]
        print(f"\n{bag}: {'СОВПАДАЕТ' if same else 'РАСХОДИТСЯ'}")
        for name, r in (("исходный ", ro), ("зеркальный", rm)):
            line = (f"  {name}: геометрия {r['geometry']}/{r['frames']}, находок сырых "
                    f"{len(r['raw'])}, подтв. {len(r['confirmed'])} {r['confirmed'][:8]}, "
                    f"скачков стен {r['wall_jumps']}, дальность {r['reach_median']:.0f} м")
            if "person" in r:
                q = r["person"]
                line += (f"; человек {q['found']}/{q['frames']}, подтв. {q['confirmed']}, "
                         f"вне интервала {q['outside']}")
            print(line)
        print(f"  стены: медиана расхождения {c['wall_mismatch_median'][0] * 100:.1f} / "
              f"{c['wall_mismatch_median'][1] * 100:.1f} см, p90 {c['wall_mismatch_p90'] * 100:.1f} см, "
              f"максимум {c['wall_mismatch_max']:.2f} м; кадров с расхождением > {DIVERGE} м: "
              f"{c['divergent_frames']} из {c['frames_compared']}")
        for e in c["episodes"]:
            why = []
            if e["start_choice_differs"]:
                why.append(f"выбор «память / заново» перевернулся в кадре "
                           f"{e['start_choice_differs'][0]}")
            if e["steps_differ"]:
                why.append(f"разное число ступенек с кадра {e['steps_differ'][0]}")
            since = f", расходится с кадра {e['since']}" if e["since"] < e["frames"][0] else ""
            print(f"    кадры {e['frames'][0]}–{e['frames'][1]}: до {e['max_mismatch']:.2f} м{since}"
                  + (f" ({'; '.join(why)})" if why else ""))
        if c["detection_differs_frames"]:
            print(f"  находки различаются в кадрах {c['detection_differs_frames']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
