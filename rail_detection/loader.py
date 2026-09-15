"""Загрузка кадров лидара (sensor_msgs/PointCloud2) из ROS 2 bag (sqlite3)."""

from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

POINT_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4'),
    ('ring', '<u2'), ('timestamp', '<f8'),
])

# doubleT_obstacle исключён: там стоит препятствие (стоящий поезд), не типовая
# "чистая" сцена — см. knowledge.md §6, §11.
DEFAULT_BAGS = [
    "doubleT_platform",
    "roundT_doubleT",
    "roundT_pressureGate_roundT",
    "roundT_squareT_pressureGate_squareT",
    "squareT_platform_squareT_switch",
]


def bag_path(dataset_root, bag_name):
    return Path(dataset_root) / bag_name


def load_frame(bag_dir, frame_idx=None):
    """Загружает один кадр облака точек из bag.

    frame_idx=None -> берётся средний кадр записи.
    Возвращает (points: np.ndarray[POINT_DTYPE], n_frames: int).
    """
    bag_dir = Path(bag_dir)
    with AnyReader([bag_dir]) as reader:
        conns = reader.connections
        n = conns[0].msgcount
        target = n // 2 if frame_idx is None else frame_idx
        for i, (conn, ts, rawdata) in enumerate(reader.messages(connections=conns)):
            if i == target:
                msg = reader.deserialize(rawdata, conn.msgtype)
                points = np.frombuffer(msg.data, dtype=POINT_DTYPE)
                return points, n
    raise IndexError(f"кадр {target} не найден в {bag_dir} (всего кадров: {n})")


def iter_frames(bag_dir, stride=1, max_frames=None):
    """Эффективно проходит по кадрам bag ОДНИМ проходом (в отличие от load_frame,
    которую вызывать много раз для разных кадров дорого — она пересканирует bag
    с начала при каждом вызове). Отдаёт (frame_idx, points, n_total) по порядку.

    stride: брать каждый stride-й кадр (1 = все подряд).
    max_frames: остановиться после стольких ОТДАННЫХ кадров (не после стольких
        просмотренных) — None = пройти всю запись.
    """
    bag_dir = Path(bag_dir)
    yielded = 0
    with AnyReader([bag_dir]) as reader:
        conns = reader.connections
        n = conns[0].msgcount
        for i, (conn, ts, rawdata) in enumerate(reader.messages(connections=conns)):
            if i % stride != 0:
                continue
            msg = reader.deserialize(rawdata, conn.msgtype)
            points = np.frombuffer(msg.data, dtype=POINT_DTYPE)
            yield i, points, n
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                return
