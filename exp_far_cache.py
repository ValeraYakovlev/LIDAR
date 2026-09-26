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
- находки эталона на этом кадре (все скопления и подтверждённая дальность);
- вид сбоку: гистограмма (s, v) полосы над путём во всю высоту (`side_view`);
- расхождение гипотез пути «память» / «заново» по дальности (`alt_dx`) и
  дальность каждой стены.

    python exp_far_cache.py --dataset /Volumes/T7/Synthetic_data --bags box
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from exp20_split import guard
from rail_detection import bag_path, iter_frames
from rail_detection.contrast_gauge import PATH_GRID, _voxel
from rail_detection.parallel_path import ParallelGauge, to_path_dict

warnings.filterwarnings("ignore")

# Коробка кэша: шире габарита поезда с запасом на «рядом с габаритом» и выше
# свода квадратного тоннеля; снизу — чуть ниже головок рельсов.
U_MAX = 2.2
V_MIN, V_MAX = -0.15, 4.5
S_MIN, S_MAX = 1.0, 155.0

# Вид сбоку (эксперимент 18): гистограмма точек полосы над путём |u| <= SIDE_U в
# плоскости (s, v) — пол, рельсы и свод во всю высоту, чтобы оценивать, идёт ли
# путь впереди вверх, вниз или прямо. Разреженно: только ненулевые клетки.
SIDE_U = 1.0
SIDE_S = np.arange(0.0, 156.0, 1.0)
SIDE_V = np.arange(-2.0, 6.55, 0.05)

# Расхождение двух гипотез пути («память» и «заново», §31) по дальности: где они
# расходятся, путь вдали не определён. Поперечное расстояние между путями на
# глубинах ALT_D (м); nan — второй гипотезы на кадре не было.
ALT_D = np.arange(10.0, 151.0, 10.0)


def run(dataset, bag, out_dir, max_frames=None):
    pg = ParallelGauge()
    rows, clusters = [], []
    ptr, S, U, V = [0], [], [], []
    side = []        # (кадр, бин s, бин v, число точек)
    for idx, points, n_total in iter_frames(bag_path(dataset, bag), stride=1,
                                           max_frames=max_frames):
        res = pg.update(points, steps=1)
        row = {"idx": idx, "ok": res is not None, "ds": np.nan, "ds_meas": False,
               "shift": np.nan, "reach": np.nan, "limit": np.nan,
               "base_raw": np.nan, "base_conf": np.nan,
               "alt_dx": np.full(len(ALT_D), np.nan), "reach_l": np.nan, "reach_r": np.nan}
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
            ms = np.abs(ug) <= SIDE_U
            H, _, _ = np.histogram2d(s[ms], vg[ms], bins=[SIDE_S, SIDE_V])
            i, j = np.nonzero(H)
            side.append(np.column_stack([np.full(len(i), idx), i, j, H[i, j]]).astype(np.int32))
            cl = res["clusters"]
            alt_dx = np.full(len(ALT_D), np.nan)
            if tr.get("alt") is not None:
                a = to_path_dict(tr["alt"]["curve"], PATH_GRID, "")
                alt_dx = np.interp(ALT_D, a["d"], a["x"]) - np.interp(ALT_D, res["path"]["d"],
                                                                      res["path"]["x"])
            row["alt_dx"] = alt_dx
            row.update({"reach_l": tr["reach_side"]["left"] if tr["reach_side"]["left"] is not None
                        else np.nan,
                        "reach_r": tr["reach_side"]["right"] if tr["reach_side"]["right"] is not None
                        else np.nan})
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
                        side=np.concatenate(side) if side else np.zeros((0, 4), np.int32),
                        **cols)
    with open(out_dir / f"{bag}.clusters.json", "w") as f:
        json.dump(clusters, f, default=float)
    print(f"  {bag}: {len(rows)} кадров, {ptr[-1]} вокселей")


def load(cache_dir, bag):
    """Кэш записи: словарь покадровых столбцов и функция точек кадра k -> (s, u, v)."""
    z = np.load(Path(cache_dir) / f"{bag}.npz")
    cols = {k: z[k] for k in z.files if k not in ("s", "u", "v", "ptr", "side")}
    ptr, S = z["ptr"], z["s"]
    U, V = z["u"].astype(np.float32), z["v"].astype(np.float32)

    def pts(k):
        a, b = ptr[k], ptr[k + 1]
        return S[a:b], U[a:b], V[a:b]

    with open(Path(cache_dir) / f"{bag}.clusters.json") as f:
        cols["base_clusters"] = json.load(f)
    if "side" in z.files:
        cols["side"] = z["side"]
    return cols, pts


def side_view(cols, k):
    """Вид сбоку кадра k: гистограмма (len(SIDE_S)-1, len(SIDE_V)-1) по (s, v)."""
    H = np.zeros((len(SIDE_S) - 1, len(SIDE_V) - 1), np.int32)
    sd = cols.get("side")
    if sd is None:
        return None
    a, b = np.searchsorted(sd[:, 0], [k, k + 1])
    H[sd[a:b, 1], sd[a:b, 2]] = sd[a:b, 3]
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--bags", nargs="+", required=True)
    p.add_argument("--out", default="output/exp18_cache")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--holdout", action="store_true",
                   help="отложенный замер: разрешить отложенные записи эксперимента 20")
    a = p.parse_args()
    guard(a.bags, a.holdout)
    tag = Path(a.dataset).name
    for bag in a.bags:
        run(a.dataset, bag, Path(a.out) / tag, a.max_frames)


if __name__ == "__main__":
    main()
