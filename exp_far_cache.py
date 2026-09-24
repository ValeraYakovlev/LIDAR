#!/usr/bin/env python3
"""Эксперимент 18: кэш покадровых точек вокруг пути — в координатах габарита.

Трекер эталона (`ParallelGauge`, тег wall-parallel-v1) проходит КАЖДЫЙ кадр
записи, как на поезде. Путь и стены в эксперименте 18 не меняются, поэтому их
достаточно посчитать один раз, а варианты габарита, отбора скоплений и
подтверждения перебирать на кэше за секунды.

Что сохраняется на кадр:
- точки в широкой коробке вокруг пути в координатах габарита: s — длина вдоль
  пути, u — поперёк от середины между головками рельсов (с поворотом на крен),
  v — над плоскостью головок. Прорежены по сетке 5 см ровно так же, как перед
  DBSCAN эталона (`contrast_gauge._voxel`), так что результат эталона на кэше
  воспроизводится;
- Δs, которым трекер перенёс состояние (`tr["ds"]`), замер Δs, дальность пути;
- находки эталона на этом кадре (все скопления и подтверждённая дальность).

    python exp_far_cache.py --dataset /Volumes/T7/Synthetic_data --bags box
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from rail_detection import bag_path, iter_frames
from rail_detection.contrast_gauge import _voxel
from rail_detection.parallel_path import ParallelGauge

warnings.filterwarnings("ignore")

# Коробка кэша: шире габарита поезда с запасом на «рядом с габаритом» и выше
# свода квадратного тоннеля; снизу — чуть ниже головок рельсов.
U_MAX = 2.2
V_MIN, V_MAX = -0.15, 4.5
S_MIN, S_MAX = 1.0, 155.0


def run(dataset, bag, out_dir, max_frames=None):
    pg = ParallelGauge()
    rows, clusters = [], []
    ptr, S, U, V = [0], [], [], []
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=1,
                                           max_frames=max_frames):
        res = pg.update(points, steps=1)
        row = {"idx": idx, "ok": res is not None, "ds": np.nan, "ds_meas": False,
               "shift": np.nan, "reach": np.nan, "limit": np.nan,
               "base_raw": np.nan, "base_conf": np.nan}
        cl = []
        if res is not None:
            tr = res["track"]
            s, ug, vg = res["s"], res["ug"], res["vg"]
            m = (np.abs(ug) <= U_MAX) & (vg >= V_MIN) & (vg <= V_MAX) & (s > S_MIN) & (s <= S_MAX)
            vox = _voxel(np.column_stack([s[m], ug[m], vg[m]]))
            S.append(vox[:, 0].astype(np.float32))
            U.append(vox[:, 1].astype(np.float16))
            V.append(vox[:, 2].astype(np.float16))
            ptr.append(ptr[-1] + len(vox))
            cl = res["clusters"]
            row.update({"ds": tr["ds"], "ds_meas": bool(tr["ds_measured"]),
                        "shift": res["shift"] if res["shift"] is not None else np.nan,
                        "reach": res["reach"], "limit": res["limit"],
                        "base_raw": cl[0]["dist"] if cl else np.nan,
                        "base_conf": res["confirmed"] if res["confirmed"] else np.nan})
        else:
            ptr.append(ptr[-1])
        rows.append(row)
        clusters.append(cl)
        print(f"\r  {bag}: кадр {idx + 1}/{n_total}", end="", flush=True)
    print()
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(out_dir / f"{bag}.npz", ptr=np.array(ptr, np.int64),
                        s=np.concatenate(S) if S else np.zeros(0, np.float32),
                        u=np.concatenate(U) if U else np.zeros(0, np.float16),
                        v=np.concatenate(V) if V else np.zeros(0, np.float16),
                        **cols)
    with open(out_dir / f"{bag}.clusters.json", "w") as f:
        json.dump(clusters, f, default=float)
    print(f"  {bag}: {len(rows)} кадров, {ptr[-1]} вокселей")


def load(cache_dir, bag):
    """Кэш записи: словарь покадровых столбцов и функция точек кадра k -> (s, u, v)."""
    z = np.load(Path(cache_dir) / f"{bag}.npz")
    cols = {k: z[k] for k in z.files if k not in ("s", "u", "v", "ptr")}
    ptr, S = z["ptr"], z["s"]
    U, V = z["u"].astype(np.float32), z["v"].astype(np.float32)

    def pts(k):
        a, b = ptr[k], ptr[k + 1]
        return S[a:b], U[a:b], V[a:b]

    with open(Path(cache_dir) / f"{bag}.clusters.json") as f:
        cols["base_clusters"] = json.load(f)
    return cols, pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--bags", nargs="+", required=True)
    p.add_argument("--out", default="output/exp18_cache")
    p.add_argument("--max-frames", type=int, default=None)
    a = p.parse_args()
    if "roundT_squareT_pressureGate_squareT" in a.bags:
        raise SystemExit("roundT_squareT_pressureGate_squareT отложен (эксперимент 18): "
                         "только на финальном замере")
    tag = Path(a.dataset).name
    for bag in a.bags:
        run(a.dataset, bag, Path(a.out) / tag, a.max_frames)


if __name__ == "__main__":
    main()
