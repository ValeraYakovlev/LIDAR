"""Разбор записи целиком, без ROS-графа: тот же конвейер, что в узле.

Нужен для замеров (каждый кадр по порядку, ответ не зависит от скорости машины)
и для сверки узла: узел с queue=all должен дать те же находки.

    metro_offline /bags/doubleT_obstacle --variant final --out /out/doubleT_obstacle.jsonl
"""

import argparse
import json
import sys
import time

import numpy as np

from rail_detection.loader import iter_frames

from .core import DEFAULT_VARIANT, Pipeline


def run(bag, variant, out_path=None, max_frames=None):
    pipe = Pipeline(variant)
    Pipeline.warm_up()
    fout = open(out_path, "w", encoding="utf-8") if out_path else None
    times, found = [], []
    t_all = time.perf_counter()
    for idx, points, n in iter_frames(bag, max_frames=max_frames):
        out = pipe.process(points)
        out.pop("geometry", None)
        out.update(frame=idx, variant=variant)
        times.append(out["proc_ms"])
        if out["detected"]:
            found.append((idx, out["distance"]))
        if fout:
            fout.write(json.dumps(out, ensure_ascii=False, default=float) + "\n")
        state = f"ПРЕПЯТСТВИЕ {out['distance']:.1f} м" if out["detected"] else \
            ("путь свободен" if out["ok"] else "путь не построен")
        print(f"кадр {idx}/{n}: {state} — {out['proc_ms']:.0f} мс", flush=True)
    if fout:
        fout.close()
    t = np.array(times)
    summary = {"bag": str(bag), "variant": variant, "frames": len(t),
               "frames_detected": len(found),
               "first_detection": found[0] if found else None,
               "max_distance": max((d for _, d in found), default=None),
               "proc_ms_median": float(np.median(t)) if len(t) else None,
               "proc_ms_p95": float(np.percentile(t, 95)) if len(t) else None,
               "wall_s": time.perf_counter() - t_all}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bag", help="папка записи (metadata.yaml + .db3)")
    ap.add_argument("--variant", default=DEFAULT_VARIANT)
    ap.add_argument("--out", help="JSON Lines по кадрам")
    ap.add_argument("--max-frames", type=int)
    a = ap.parse_args(argv)
    run(a.bag, a.variant, a.out, a.max_frames)


if __name__ == "__main__":
    sys.exit(main())
