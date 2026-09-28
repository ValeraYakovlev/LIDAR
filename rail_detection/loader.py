"""Загрузка кадров лидара (sensor_msgs/PointCloud2) из ROS 2 bag (sqlite3)."""

from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore, Stores

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


LIDAR_TOPIC = "/lidar_points"


def _lidar_connections(reader):
    """Соединения с облаком лидара.

    В исходных записях топик один. В синтетических (/Volumes/T7/synthetic_data)
    рядом лежат /lidar_points_labeled — то же облако с классом каждой точки, то
    есть готовый ответ, который алгоритм видеть не должен, — и /tf, /tf_static.
    Берётся /lidar_points; если его нет — первый топик PointCloud2.
    """
    conns = [c for c in reader.connections if c.topic == LIDAR_TOPIC]
    if not conns:
        conns = [c for c in reader.connections
                 if c.msgtype == "sensor_msgs/msg/PointCloud2"][:1]
    if not conns:
        raise RuntimeError("в записи нет облака точек (PointCloud2)")
    return conns


_DATATYPES = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def to_points(msg):
    """Облако сообщения как массив POINT_DTYPE.

    Исходные записи лежат ровно в этой раскладке (26 байт на точку) и читаются
    без копирования. Другие записи (New_synth_data: 16 байт, только x, y, z,
    intensity) собираются по описанию полей из самого сообщения; поля, которых
    в записи нет (ring, timestamp), заполняются нулями — конвейер пути и
    габарита их не использует.
    """
    names = [(f.name, f.offset, f.datatype) for f in msg.fields]
    if msg.point_step == POINT_DTYPE.itemsize and not msg.is_bigendian and \
            [(n, o) for n, o, _ in names] == [(n, POINT_DTYPE.fields[n][1]) for n in POINT_DTYPE.names]:
        return np.frombuffer(msg.data, dtype=POINT_DTYPE)
    have = [(n, o, d) for n, o, d in names if n in POINT_DTYPE.names]
    src_dt = np.dtype({"names": [n for n, _, _ in have],
                       "formats": [_DATATYPES[d] for _, _, d in have],
                       "offsets": [o for _, o, _ in have], "itemsize": msg.point_step})
    src = np.frombuffer(msg.data, dtype=src_dt)
    out = np.zeros(len(src), dtype=POINT_DTYPE)
    for n, _, _ in have:
        out[n] = src[n]
    return out


def bag_path(dataset_root, bag_name):
    return Path(dataset_root) / bag_name


def load_frame(bag_dir, frame_idx=None):
    """Загружает один кадр облака точек из bag.

    frame_idx=None -> берётся средний кадр записи.
    Возвращает (points: np.ndarray[POINT_DTYPE], n_frames: int).
    """
    bag_dir = Path(bag_dir)
    with AnyReader([bag_dir], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        conns = _lidar_connections(reader)
        n = conns[0].msgcount
        target = n // 2 if frame_idx is None else frame_idx
        for i, (conn, ts, rawdata) in enumerate(reader.messages(connections=conns)):
            if i == target:
                msg = reader.deserialize(rawdata, conn.msgtype)
                points = to_points(msg)
                return points, n
    raise IndexError(f"кадр {target} не найден в {bag_dir} (всего кадров: {n})")


def frame_count(bag_dir):
    """Число кадров в bag без чтения самих сообщений (берётся из индекса
    sqlite) — нужно, чтобы выбрать шаг прохода до начала чтения."""
    with AnyReader([Path(bag_dir)], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        return _lidar_connections(reader)[0].msgcount


def iter_selected_frames(bag_dir, frame_indices):
    """Отдаёт запрошенные кадры за ОДИН проход по bag, в порядке возрастания
    индекса. load_frame пересканирует запись с начала при каждом вызове, поэтому
    прогон по тестовой выборке (десятки кадров из одного многогигабайтного bag)
    через неё вырождается в десятки полных чтений файла.

    Отдаёт (frame_idx, points); чтение прекращается после последнего нужного кадра.
    """
    want = sorted({int(i) for i in frame_indices})
    if not want:
        return
    last = want[-1]
    want_set = set(want)
    bag_dir = Path(bag_dir)
    with AnyReader([bag_dir], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        conns = _lidar_connections(reader)
        for i, (conn, ts, rawdata) in enumerate(reader.messages(connections=conns)):
            if i in want_set:
                msg = reader.deserialize(rawdata, conn.msgtype)
                yield i, to_points(msg)
            if i >= last:
                return


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
    with AnyReader([bag_dir], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        conns = _lidar_connections(reader)
        n = conns[0].msgcount
        for i, (conn, ts, rawdata) in enumerate(reader.messages(connections=conns)):
            if i % stride != 0:
                continue
            msg = reader.deserialize(rawdata, conn.msgtype)
            points = to_points(msg)
            yield i, points, n
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                return
