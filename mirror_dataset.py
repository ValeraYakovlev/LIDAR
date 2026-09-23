#!/usr/bin/env python3
"""Зеркальная копия датасета: отражение относительно оси Y (x -> -x).

Проверка на «новых» прогонах без новых записей: левый поворот становится
правым, соседний путь и платформа — с другой стороны, человек на путях — по
другую сторону оси. Алгоритм это видит как незнакомую геометрию, а подгонка
под шесть исходных файлов на отражённых копиях не работает.

Меняется ТОЛЬКО координата x каждой точки. y (глубина вперёд, ось Y сенсора
смотрит назад), z, интенсивность, кольцо и время точки, заголовок сообщения,
метка времени сообщения, метаданные — побайтно те же. Порядок точек в массиве не
меняется, поэтому азимутальные столбцы организованного облака (§29) идут теперь
в обратную сторону — наши методы порядок точек не используют.

Копия — в той же структуре, что исходный датасет (папка на запись, внутри
metadata.yaml и <запись>_0.db3), так что все скрипты работают с ней через
--dataset без правок.

    python mirror_dataset.py                                  # все 6 записей
    python mirror_dataset.py --bags roundT_doubleT --out /Volumes/T7/reversed
"""

import argparse
import shutil
import sqlite3
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from rail_detection.loader import POINT_DTYPE

ALL_BAGS = ["doubleT_platform", "roundT_doubleT", "roundT_pressureGate_roundT",
            "roundT_squareT_pressureGate_squareT", "squareT_platform_squareT_switch",
            "doubleT_obstacle"]


def data_layout(bag_dir):
    """Где в сериализованном PointCloud2 лежит массив точек: (смещение, длина,
    хвост после него). Берётся по первому сообщению через честную
    десериализацию; для остальных только сверяется."""
    with AnyReader([bag_dir]) as r:
        conn = r.connections[0]
        for c, _, raw in r.messages(connections=[conn]):
            msg = r.deserialize(raw, c.msgtype)
            assert msg.point_step == POINT_DTYPE.itemsize and not msg.is_bigendian
            names = [(f.name, f.offset) for f in msg.fields]
            assert names[0] == ("x", 0), names
            n = len(msg.data)
            off = len(raw) - 4 - n
            while off >= 4 and bytes(raw[off:off + 64]) != bytes(msg.data[:64]):
                off -= 1
            assert int.from_bytes(raw[off - 4:off], "little") == n
            return off, n, len(raw) - off - n
    raise RuntimeError(f"пустая запись {bag_dir}")


def mirror_blob(blob, off, n, tail):
    """Тот же PointCloud2, но с x -> -x. Проверяет, что раскладка сообщения та
    же, что у первого (иначе — ошибка, а не молчаливая порча данных)."""
    if len(blob) != off + n + tail or int.from_bytes(blob[off - 4:off], "little") != n:
        raise ValueError("раскладка сообщения отличается от первого")
    pts = np.frombuffer(blob, dtype=np.uint8, count=n, offset=off).copy().view(POINT_DTYPE)
    pts["x"] = -pts["x"]
    return blob[:off] + pts.view(np.uint8).tobytes() + blob[off + n:]


def mirror_bag(src_root, dst_root, bag):
    src, dst = Path(src_root) / bag, Path(dst_root) / bag
    db_name = f"{bag}_0.db3"
    dst.mkdir(parents=True, exist_ok=True)
    out_db = dst / db_name
    if out_db.exists():
        out_db.unlink()
    off, n, tail = data_layout(src)
    s = sqlite3.connect(f"file:{src / db_name}?mode=ro", uri=True)
    d = sqlite3.connect(out_db)
    tables = s.execute("select name, sql from sqlite_master where type='table'").fetchall()
    indexes = s.execute("select sql from sqlite_master where type='index' and sql is not null").fetchall()
    for _, sql in tables:
        d.execute(sql)
    for name, _ in tables:
        if name == "messages":
            continue
        rows = s.execute(f"select * from {name}").fetchall()
        if rows:
            d.executemany(f"insert into {name} values ({','.join('?' * len(rows[0]))})", rows)
    total = s.execute("select count(*) from messages").fetchone()[0]
    for k, (mid, tid, ts, blob) in enumerate(
            s.execute("select id, topic_id, timestamp, data from messages order by id")):
        d.execute("insert into messages values (?, ?, ?, ?)", (mid, tid, ts, mirror_blob(blob, off, n, tail)))
        print(f"\r  {bag}: {k + 1}/{total}", end="", flush=True)
    for (sql,) in indexes:
        d.execute(sql)
    d.commit()
    d.close()
    s.close()
    shutil.copy2(src / "metadata.yaml", dst / "metadata.yaml")
    print()
    return total


def verify(src_root, dst_root, bag, frames=(0, 1, -1)):
    """Сверка: число сообщений, метки времени, и на выбранных кадрах — x
    отражён, всё остальное побайтно то же."""
    src, dst = Path(src_root) / bag, Path(dst_root) / bag
    out = {}
    for root in (src, dst):
        with AnyReader([root]) as r:
            conn = r.connections[0]
            total = conn.msgcount
            idx = {i if i >= 0 else total + i for i in frames}
            stamps, pts = [], {}
            # по одному сообщению: целиком в память запись на 7 ГБ не влезет
            for k, (c, t, raw) in enumerate(r.messages(connections=[conn])):
                stamps.append(t)
                if k in idx:
                    pts[k] = np.frombuffer(r.deserialize(raw, c.msgtype).data,
                                           dtype=POINT_DTYPE).copy()
            out[root] = (len(stamps), stamps, pts)
    (n0, t0, p0), (n1, t1, p1) = out[src], out[dst]
    assert n0 == n1 and t0 == t1, "число сообщений или метки времени отличаются"
    for i in p0:
        a, b = p0[i], p1[i]
        assert np.array_equal(a["x"], -b["x"])
        for f in ("y", "z", "intensity", "ring", "timestamp"):
            assert np.array_equal(a[f], b[f]), f
    return n1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--out", default="/Volumes/T7/reversed")
    p.add_argument("--bags", nargs="*", default=ALL_BAGS)
    a = p.parse_args()
    for bag in a.bags:
        n = mirror_bag(a.dataset, a.out, bag)
        verify(a.dataset, a.out, bag)
        print(f"  {bag}: {n} сообщений, сверка пройдена (x отражён, остальное без изменений)")


if __name__ == "__main__":
    main()
