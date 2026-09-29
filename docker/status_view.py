#!/usr/bin/env python3
"""Ответ узла по кадрам — строка на каждый /obstacle/status, для окна рядом с RViz2.

То же, что пишет узел в свой журнал, но читается из ROS-графа: окно работает в
образе показа и не зависит от того, где запущен узел.
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

RED, GREEN, GREY, RESET = "\033[1;31m", "\033[1;32m", "\033[0;37m", "\033[0m"


class StatusView(Node):
    def __init__(self):
        super().__init__("status_view")
        self.create_subscription(String, "/obstacle/status", self.on_status, 10)
        print(f"{GREY}ответ узла obstacle_detector по кадрам (топик /obstacle/status){RESET}",
              flush=True)

    def on_status(self, msg):
        s = json.loads(msg.data)
        if not s.get("ok"):
            state = f"{GREY}путь не построен{RESET}"
        elif s.get("detected"):
            state = f"{RED}ПРЕПЯТСТВИЕ {s['distance']:6.1f} м{RESET}"
        else:
            state = f"{GREEN}путь свободен{RESET}, видно до {s['limit']:3.0f} м"
        extra = f", пропущено {s['skipped']}" if s.get("skipped") else ""
        extra += f", в очереди {s['backlog']}" if s.get("backlog") else ""
        print(f"кадр {s['frame']:5d}   {state}   — обработка {s['proc_ms']:4.0f} мс, "
              f"задержка {s['latency_ms']:4.0f} мс{extra}", flush=True)


def main():
    rclpy.init()
    node = StatusView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
