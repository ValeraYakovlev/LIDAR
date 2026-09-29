"""ROS 2-узел обнаружения препятствия на пути поезда.

Подписка: облако лидара (sensor_msgs/PointCloud2). Топик по умолчанию ищется
сам: `/lidar_points`, иначе первый топик PointCloud2 (кроме размеченных
`*_labeled` — в синтетике это готовый ответ — и собственных `/obstacle/*`).

Публикация:
  /obstacle/detected       std_msgs/Bool       есть ли подтверждённое препятствие
  /obstacle/distance       std_msgs/Float32    до ближайшего, м вдоль пути (NaN — нет)
  /obstacle/status         std_msgs/String     JSON кадра: всё выше + задержка, пропуски,
                                               частота прихода, отставание потока
  /obstacle/markers        visualization_msgs/MarkerArray  габарит и препятствие для RViz2
  /obstacle/gauge_points   sensor_msgs/PointCloud2  точки, попавшие в габарит
  /obstacle/cloud_preview  sensor_msgs/PointCloud2  прореженное входное облако

Кадры обрабатываются в отдельном потоке. queue=all (по умолчанию) — все кадры
по порядку, ни один не пропускается (ответ как при разборе записи целиком); если
кадр обрабатывается дольше периода лидара, кадры ждут в очереди и задержка
растёт. queue=latest — берётся самый свежий кадр, пропущенные учитываются при
поиске Δs, задержка не растёт (режим для работы на поезде).
"""

import array
import collections
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Bool, ColorRGBA, Float32, String
from visualization_msgs.msg import Marker, MarkerArray

from rail_detection.loader import to_points

from .core import DEFAULT_VARIANT, Pipeline

PREFERRED_TOPIC = "/lidar_points"
# Известные имена топика лидара (реальные записи и синтетика): на них узел
# подписывается сразу, до появления издателя, — как на поезде, где детектор
# ждёт лидар. Иначе первые кадры приходят, пока узел ищет топик.
KNOWN_TOPICS = ("/sensing/lidar/hesai128/pointcloud", "/lidar_points")
CLOUD_TYPE = "sensor_msgs/msg/PointCloud2"

GREEN = ColorRGBA(r=0.1, g=0.9, b=0.3, a=0.9)
RED = ColorRGBA(r=1.0, g=0.15, b=0.1, a=0.95)
WHITE = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)


def pick_topic(topics):
    """Топик облака из списка (имя, [типы])."""
    clouds = [n for n, types in topics if CLOUD_TYPE in types
              and not n.endswith("_labeled") and not n.startswith("/obstacle/")]
    if PREFERRED_TOPIC in clouds:
        return PREFERRED_TOPIC
    return sorted(clouds)[0] if clouds else None


def xyz_cloud(header, xyz):
    """PointCloud2 из массива (N, 3) float32."""
    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, len(xyz)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate("xyz")]
    msg.is_bigendian, msg.point_step = False, 12
    msg.row_step, msg.is_dense = 12 * len(xyz), True
    buf = array.array("B")
    buf.frombytes(np.ascontiguousarray(xyz, np.float32).tobytes())
    msg.data = buf
    return msg


class ObstacleNode(Node):
    def __init__(self):
        super().__init__("obstacle_detector")
        self.declare_parameter("topic", "auto")
        self.declare_parameter("variant", DEFAULT_VARIANT)
        self.declare_parameter("queue", "all")             # all | latest
        self.declare_parameter("preview_stride", 10)       # 0 — не публиковать облако
        self.declare_parameter("log_file", "")             # JSON Lines по кадрам; папка — свой файл на запуск
        self.declare_parameter("reset_gap", 2.0)           # с: пауза в потоке — новая запись

        gp = lambda n: self.get_parameter(n).value
        self.variant = gp("variant")
        self.queue_mode = gp("queue")
        if self.queue_mode not in ("latest", "all"):
            raise ValueError("queue: latest или all")
        self.preview_stride = int(gp("preview_stride"))
        self.reset_gap = float(gp("reset_gap"))
        self.pipeline = Pipeline(self.variant)
        t0 = time.monotonic()
        Pipeline.warm_up()
        self.get_logger().info(f"конвейер прогрет за {time.monotonic() - t0:.1f} с")
        from rail_detection import jit
        from rail_detection import parallel as par
        self.get_logger().info(
            f"ускорения: потоков на кадр {par.workers()}, процессов-помощников "
            f"{par.N_HELPERS if par.use_process() else 0}, numba {'да' if jit.ENABLED else 'нет'}")
        path = gp("log_file")
        if path and os.path.isdir(path):      # папка (в образе — /out): свой файл на каждый запуск
            path = os.path.join(path, time.strftime("detections_%Y%m%d_%H%M%S.jsonl"))
        self.log = open(path, "a", encoding="utf-8") if path else None
        if self.log:
            self.get_logger().info(f"журнал кадров: {path}")

        self.pub_det = self.create_publisher(Bool, "/obstacle/detected", 10)
        self.pub_dist = self.create_publisher(Float32, "/obstacle/distance", 10)
        self.pub_status = self.create_publisher(String, "/obstacle/status", 10)
        self.pub_mark = self.create_publisher(MarkerArray, "/obstacle/markers", 10)
        self.pub_hits = self.create_publisher(PointCloud2, "/obstacle/gauge_points", 10)
        self.pub_prev = self.create_publisher(PointCloud2, "/obstacle/cloud_preview", 10)

        self.cv = threading.Condition()
        self.pending = collections.deque(maxlen=1 if self.queue_mode == "latest" else None)
        self.received = 0          # кадров пришло
        self.last_seq = None       # номер последнего обработанного
        self.last_rx = None        # время прихода последнего кадра
        self.rx_times = collections.deque(maxlen=21)   # приход последних кадров
        self.rx_origin = None      # (приход, метка лидара) первого кадра потока
        self.running = True
        self.subs = {}             # топик → подписка
        self.active_topic = None   # первый топик, с которого пришёл кадр
        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

        topic = gp("topic")
        self.find_timer = None
        if topic == "auto":
            for t in KNOWN_TOPICS:
                self._subscribe(t)
            self.get_logger().info("жду облако PointCloud2 (известные топики + поиск) …")
            self.find_timer = self.create_timer(0.5, self._find_topic)
        else:
            self._subscribe(topic)
        self.get_logger().info(f"вариант детектора: {self.variant}, очередь: {self.queue_mode}")

    # ------------------------------------------------------------ приём

    def _find_topic(self):
        """Облако под незнакомым именем: подписаться, когда появится издатель."""
        topic = pick_topic(self.get_topic_names_and_types())
        if topic and topic not in self.subs:
            self._subscribe(topic)

    def _subscribe(self, topic):
        # queue=all: очередь DDS на 10 с потока — пока поток приёма ждёт своей
        # очереди (обработка кадра держит интерпретатор), кадры не вытесняются
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST,
                         depth=100 if self.queue_mode == "all" else 10,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.subs[topic] = self.create_subscription(
            PointCloud2, topic, lambda m, t=topic: self._on_cloud(m, t), qos)

    def _choose(self, topic):
        """Первый топик, с которого пришёл кадр, — единственный: остальные подписки
        снимаются (в синтетике рядом идёт размеченное облако — его брать нельзя)."""
        self.active_topic = topic
        if self.find_timer is not None:
            self.find_timer.cancel()
        for t in [t for t in self.subs if t != topic]:
            self.destroy_subscription(self.subs.pop(t))
        self.get_logger().info(f"облако идёт из {topic}")

    def _on_cloud(self, msg, topic):
        if self.active_topic is None:
            self._choose(topic)
        elif topic != self.active_topic:
            return
        now = time.monotonic()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.cv:
            gap = self.last_rx is not None and now - self.last_rx > self.reset_gap
            if gap or self.rx_origin is None:
                self.rx_times.clear()
                self.rx_origin = (now, stamp)
            self.last_rx = now
            self.received += 1
            self.rx_times.append(now)
            # насколько кадр пришёл позже, чем по часам лидара от начала потока:
            # у живого лидара ~0, растёт — источник не успевает (диск, DDS)
            lag = (now - self.rx_origin[0]) - (stamp - self.rx_origin[1])
            self.pending.append((self.received, now, msg, gap, self._rx_stats(lag)))
            self.cv.notify()

    def _rx_stats(self, lag):
        t = self.rx_times
        dt = np.diff(t) if len(t) > 1 else np.zeros(0)
        return {"received": self.received,
                "rx_hz": float(len(dt) / (t[-1] - t[0])) if len(dt) else None,
                "rx_gap_ms": float(dt.max() * 1000.0) if len(dt) else None,
                "stream_lag_ms": float(lag * 1000.0)}

    # ------------------------------------------------------------ обработка

    def _loop(self):
        while self.running:
            with self.cv:
                while not self.pending and self.running:
                    self.cv.wait(0.2)
                if not self.running:
                    return
                seq, t_rx, msg, gap, rx = self.pending.popleft()
                backlog = len(self.pending)
            if gap:
                # запись проиграна заново или началась другая — путь и трекер с нуля
                self.pipeline.reset()
                self.last_seq = None
                self.get_logger().info("перерыв в потоке — состояние сброшено")
            steps = 1 if self.last_seq is None else max(1, seq - self.last_seq)
            skipped = steps - 1
            wait_ms = (time.monotonic() - t_rx) * 1000.0
            try:
                points = to_points(msg)
                out = self.pipeline.process(points, steps=steps)
            except Exception as e:  # кадр с ошибкой не должен ронять узел
                self.get_logger().error(f"кадр {seq}: {e!r}")
                continue
            self.last_seq = seq
            out.update(rx)
            out.update(frame=seq, skipped=skipped, backlog=backlog, wait_ms=wait_ms,
                       latency_ms=(time.monotonic() - t_rx) * 1000.0,
                       variant=self.variant, frame_id=msg.header.frame_id,
                       stamp=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            self._publish(msg, points, out)

    def _publish(self, msg, points, out):
        # сначала ответ (он нужен поезду), потом картинка для RViz2
        geom = out.pop("geometry", None)
        dist = out["distance"]
        self.pub_det.publish(Bool(data=out["detected"]))
        self.pub_dist.publish(Float32(data=float(dist) if dist is not None else math.nan))
        self.pub_status.publish(String(data=json.dumps(out, ensure_ascii=False, default=float)))
        t0 = time.monotonic()
        self.pub_mark.publish(self._markers(msg.header, out, geom))
        if geom is not None:
            self.pub_hits.publish(xyz_cloud(msg.header, geom["gauge_hits"]))
        if self.preview_stride > 0:
            sub = points[::self.preview_stride]
            self.pub_prev.publish(xyz_cloud(msg.header, np.column_stack(
                [sub["x"], sub["y"], sub["z"]])))
        out["viz_ms"] = (time.monotonic() - t0) * 1000.0
        if self.log:
            self.log.write(json.dumps(out, ensure_ascii=False, default=float) + "\n")
            self.log.flush()

        if not out["ok"]:
            state = "путь не построен"
        elif out["detected"]:
            state = f"ПРЕПЯТСТВИЕ {dist:.1f} м"
        else:
            state = f"путь свободен (видно до {out['limit']:.0f} м)"
        extra = f", пропущено {out['skipped']}" if out["skipped"] else ""
        extra += f", в очереди {out['backlog']}" if out["backlog"] else ""
        self.get_logger().info(f"кадр {out['frame']}: {state} — "
                               f"{out['proc_ms']:.0f} мс{extra}")

    def _markers(self, header, out, geom):
        arr = MarkerArray()
        arr.markers.append(Marker(header=header, action=Marker.DELETEALL))
        colour = RED if out["detected"] else GREEN
        life = Duration(sec=1)

        # крупная надпись впереди лидара слева от пути — ответ кадра (не над
        # путём: там она закрывает препятствие); латиницей: шрифт сцены RViz2
        # (Ogre) кириллицу не рисует
        if not out["ok"]:
            head, hc = "NO TRACK", WHITE
        elif out["detected"]:
            head, hc = f"OBSTACLE {out['distance']:.1f} m", RED
        else:
            head, hc = f"CLEAR to {out['limit']:.0f} m", GREEN
        hud = Marker(header=header, ns="status", id=0, type=Marker.TEXT_VIEW_FACING,
                     action=Marker.ADD, lifetime=life, color=hc, text=head)
        hud.pose.position.x, hud.pose.position.y, hud.pose.position.z = -6.0, -30.0, 6.0
        hud.pose.orientation.w = 1.0
        hud.scale.z = 2.0
        arr.markers.append(hud)
        if geom is None:
            return arr

        lines = Marker(header=header, ns="gauge", id=0, type=Marker.LINE_LIST,
                       action=Marker.ADD, lifetime=life, color=colour)
        lines.scale.x = 0.06
        lines.pose.orientation.w = 1.0
        c = geom["corridor"]
        edges = [c["left_bottom"], c["right_bottom"], c["left_top"], c["right_top"]]
        for e in edges:                                   # вдоль пути
            for a, b in zip(e[:-1], e[1:]):
                lines.points += [Point(x=float(a[0]), y=float(a[1]), z=float(a[2])),
                                 Point(x=float(b[0]), y=float(b[1]), z=float(b[2]))]
        n = len(edges[0])
        for i in range(0, n, 5):                          # сечения каждые 10 м
            ring = [c["left_bottom"][i], c["right_bottom"][i], c["right_top"][i],
                    c["left_top"][i], c["left_bottom"][i]]
            for a, b in zip(ring[:-1], ring[1:]):
                lines.points += [Point(x=float(a[0]), y=float(a[1]), z=float(a[2])),
                                 Point(x=float(b[0]), y=float(b[1]), z=float(b[2]))]
        arr.markers.append(lines)

        for k, b in enumerate(geom["boxes"]):
            box = Marker(header=header, ns="obstacle", id=k, type=Marker.CUBE,
                         action=Marker.ADD, lifetime=life, color=RED)
            box.color.a = 0.6
            cx, cy, cz = (float(t) for t in b["centre"])
            box.pose.position.x, box.pose.position.y, box.pose.position.z = cx, cy, cz
            box.pose.orientation.w = 1.0
            # размеры скопления: вдоль пути (−y лидара), поперёк (x), по высоте (z)
            box.scale.x, box.scale.y, box.scale.z = (float(b["size"][1]), float(b["size"][0]),
                                                     float(b["size"][2]))
            arr.markers.append(box)
            txt = Marker(header=header, ns="obstacle_text", id=k, type=Marker.TEXT_VIEW_FACING,
                         action=Marker.ADD, lifetime=life, color=WHITE,
                         text=f"{b['dist']:.1f} m")
            txt.pose.position.x, txt.pose.position.y = cx, cy
            txt.pose.position.z = cz + float(b["size"][2]) / 2 + 1.0
            txt.pose.orientation.w = 1.0
            txt.scale.z = 1.8
            arr.markers.append(txt)
        return arr

    def destroy_node(self):
        self.running = False
        with self.cv:
            self.cv.notify_all()
        self.worker.join(timeout=2.0)
        if self.log:
            self.log.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
