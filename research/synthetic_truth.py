#!/usr/bin/env python3
"""Истинное положение препятствия в синтетических записях — по разметке.

Синтетические записи (/Volumes/T7/synthetic_data) несут рядом с облаком
/lidar_points топик /lidar_points_labeled: то же облако (только точки с
откликом) с классом каждой точки. Классы — в labels.json записи. Препятствие —
класс "Obstacle" или "SpiderMan" / "SpiderMan_Web" (человек и его «паутина»).

Алгоритм разметку не видит: загрузчик берёт только /lidar_points. Здесь она
читается ОТДЕЛЬНО, чтобы сверить находки с правдой: в каких кадрах препятствие
видно лидару, на какой глубине и где поперёк.

    python synthetic_truth.py                                  # все записи
    python synthetic_truth.py --bags box --out output/synthetic_truth
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

OBSTACLE_CLASSES = ("Obstacle", "SpiderMan", "SpiderMan_Web")
LABELED_TOPIC = "/lidar_points_labeled"


def truth(bag_dir):
    bag_dir = Path(bag_dir)
    classes = json.load(open(bag_dir / "labels.json"))["categoryID_to_class"]
    ids = sorted(int(k) for k, v in classes.items() if v in OBSTACLE_CLASSES)
    rows = []
    with AnyReader([bag_dir]) as r:
        conn = [c for c in r.connections if c.topic == LABELED_TOPIC][0]
        for k, (c, t, raw) in enumerate(r.messages(connections=[conn])):
            msg = r.deserialize(raw, c.msgtype)
            names = [f.name for f in msg.fields]
            dt = np.dtype({"names": names,
                           "formats": ["<f4" if f.datatype == 7 else "<u2" for f in msg.fields],
                           "offsets": [f.offset for f in msg.fields],
                           "itemsize": msg.point_step})
            p = np.frombuffer(msg.data, dtype=dt)
            m = np.isin(p["label"], ids)
            row = {"idx": k, "stamp": int(t), "points": int(m.sum())}
            if m.any():
                d, x, z = -p["y"][m], p["x"][m], p["z"][m]
                row.update({"depth_min": float(d.min()), "depth_med": float(np.median(d)),
                            "x_med": float(np.median(x)), "x_range": [float(x.min()), float(x.max())],
                            "z_range": [float(z.min()), float(z.max())]})
            rows.append(row)
            print(f"\r  {bag_dir.name}: кадр {k + 1}/{conn.msgcount}", end="", flush=True)
    print()
    return {"bag": bag_dir.name, "classes": {i: classes[str(i)] for i in ids}, "frames": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/synthetic_data")
    p.add_argument("--bags", nargs="*", default=["box", "human_smashed", "human_smashed_diff_tunnels"])
    p.add_argument("--out", default="output/synthetic_truth")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for bag in a.bags:
        t = truth(Path(a.dataset) / bag)
        with open(Path(a.out) / f"{bag}.json", "w") as f:
            json.dump(t, f, ensure_ascii=False, indent=0)
        seen = [r for r in t["frames"] if r["points"] > 0]
        if seen:
            print(f"{bag}: препятствие ({', '.join(t['classes'].values())}) видно в {len(seen)} "
                  f"кадрах из {len(t['frames'])}: кадры {seen[0]['idx']}–{seen[-1]['idx']}, "
                  f"глубина {seen[0]['depth_min']:.0f} → {seen[-1]['depth_min']:.0f} м")
        else:
            print(f"{bag}: препятствие в кадрах не видно")


if __name__ == "__main__":
    main()
