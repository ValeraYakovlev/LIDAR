#!/usr/bin/env python3
"""Эксперимент 20: правда по синтетическим записям Last_synth_data.

Алгоритм её не видит — здесь она читается отдельно, только для оценки:

- **истинный путь.** `/tf` (map → base_link, 100 Гц) — траектория поезда, то
  есть ось нашего пути. Будущая траектория, переведённая в систему лидара на
  кадре k (base_link → hesai_lidar из `/tf_static`), — это путь впереди, каким
  его должен найти трекер: x(d) на глубинах `D` (d = −y, как у трекера) и
  высота z(d). Разметка рельсов (`Track_Rails`) для этого не годится: дальше
  ~20 м рельсы в облако почти не попадают;
- **истинный Δs** — длина траектории между метками времени соседних кадров;
- **рабочий** (класс `Person`) — по `/lidar_points_labeled`, как в
  `synthetic_truth.py`: кадры, где он виден, глубина, x и z; плюс его длина по
  пути (дальность, которую должен назвать детектор).

    python exp20_truth.py                     # записи разработки
    python exp20_truth.py --bags conv_r300_a30
    python exp20_truth.py --holdout --bags conv_r450_a30     # только на финальном замере

Выход — `output/exp20_truth/<запись>.npz`, для рабочего — ещё
`output/synthetic_truth/<запись>.json` в формате `synthetic_truth.py` (его берут
`exp_far_eval.py` и `make_far_gifs.py`).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from exp20_split import DEV_LAST_SYNTH, guard

D = np.arange(0.0, 151.0, 1.0)     # глубины, на которых хранится истинный путь
OBJECT_CLASSES = ("Person",)


def _rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _tf(r):
    c = [c for c in r.connections if c.topic == "/tf_static"][0]
    for cc, _, raw in r.messages(connections=[c]):
        tr = r.deserialize(raw, cc.msgtype).transforms[0].transform
        t_bl = np.array([tr.translation.x, tr.translation.y, tr.translation.z])
        R_bl = _rot([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w])
    c = [c for c in r.connections if c.topic == "/tf"][0]
    T, P, Q = [], [], []
    for cc, t, raw in r.messages(connections=[c]):
        tr = r.deserialize(raw, cc.msgtype).transforms[0].transform
        T.append(t)
        P.append([tr.translation.x, tr.translation.y, tr.translation.z])
        Q.append([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w])
    return np.array(T), np.array(P), np.array(Q), t_bl, R_bl


def truth(bag_dir):
    bag_dir = Path(bag_dir)
    classes = json.load(open(bag_dir / "labels.json"))["categoryID_to_class"]
    ids = sorted(int(k) for k, v in classes.items() if v in OBJECT_CLASSES)
    with AnyReader([bag_dir]) as r:
        T, P, Q, t_bl, R_bl = _tf(r)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        c = [c for c in r.connections if c.topic == "/lidar_points"][0]
        stamps = np.array([t for _, t, _ in r.messages(connections=[c])])
        n = len(stamps)
        px = np.full((n, len(D)), np.nan)
        pz = np.full((n, len(D)), np.nan)
        s_frame = np.interp(stamps, T, arc)
        for k, t in enumerate(stamps):
            i = int(np.argmin(np.abs(T - t)))
            R_mb = _rot(Q[i])
            R_ml, O_ml = R_mb @ R_bl, P[i] + R_mb @ t_bl
            fut = (P[i:] - O_ml) @ R_ml            # точки траектории в системе лидара
            d = -fut[:, 1]
            ok = np.concatenate([[True], np.diff(d) > 0])   # глубина растёт вдоль пути
            ok &= np.maximum.accumulate(~ok) == 0
            d, fx, fz = d[ok], fut[ok, 0], fut[ok, 2]
            m = (D >= d[0]) & (D <= d[-1])
            px[k, m] = np.interp(D[m], d, fx)
            pz[k, m] = np.interp(D[m], d, fz)
        # рабочий — по разметке
        rows = []
        c = [c for c in r.connections if c.topic == "/lidar_points_labeled"][0]
        for k, (cc, t, raw) in enumerate(r.messages(connections=[c])):
            msg = r.deserialize(raw, cc.msgtype)
            dt = np.dtype({"names": [f.name for f in msg.fields],
                           "formats": ["<f4" if f.datatype == 7 else "<u2" for f in msg.fields],
                           "offsets": [f.offset for f in msg.fields], "itemsize": msg.point_step})
            p = np.frombuffer(msg.data, dtype=dt)
            m = np.isin(p["label"], ids)
            row = {"idx": k, "stamp": int(t), "points": int(m.sum())}
            if m.any():
                d, x, z = -p["y"][m], p["x"][m], p["z"][m]
                row.update({"depth_min": float(d.min()), "depth_med": float(np.median(d)),
                            "x_med": float(np.median(x)), "x_range": [float(x.min()), float(x.max())],
                            "z_range": [float(z.min()), float(z.max())]})
            rows.append(row)
    ds = np.concatenate([[np.nan], np.diff(s_frame)])
    return {"stamps": stamps, "D": D, "path_x": px, "path_z": pz, "s": s_frame, "ds": ds,
            "rows": rows, "classes": {i: classes[str(i)] for i in ids}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="output/last_synth")
    p.add_argument("--bags", nargs="+", default=DEV_LAST_SYNTH)
    p.add_argument("--out", default="output/exp20_truth")
    p.add_argument("--holdout", action="store_true")
    a = p.parse_args()
    guard(a.bags, a.holdout)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    Path("output/synthetic_truth").mkdir(parents=True, exist_ok=True)
    for bag in a.bags:
        t = truth(Path(a.dataset) / bag)
        np.savez_compressed(Path(a.out) / f"{bag}.npz", stamps=t["stamps"], D=t["D"],
                            path_x=t["path_x"], path_z=t["path_z"], s=t["s"], ds=t["ds"])
        with open(Path("output/synthetic_truth") / f"{bag}.json", "w") as f:
            json.dump({"bag": bag, "classes": t["classes"], "frames": t["rows"]}, f,
                      ensure_ascii=False, indent=0)
        seen = [r for r in t["rows"] if r["points"] > 0]
        print(f"{bag}: {len(t['stamps'])} кадров, путь {t['s'][-1]:.0f} м, Δs {np.nanmedian(t['ds']):.3f} м; "
              f"рабочий в {len(seen)} кадрах, {seen[0]['idx']}–{seen[-1]['idx']}, "
              f"{seen[0]['depth_min']:.0f} → {seen[-1]['depth_min']:.0f} м")


if __name__ == "__main__":
    main()
