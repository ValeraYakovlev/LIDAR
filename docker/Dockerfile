# Обнаружение препятствия на пути поезда метро — ROS 2 Humble (Ubuntu 22.04).
#
#   docker build -f docker/Dockerfile -t metro-obstacle .                 # узел
#   docker build -f docker/Dockerfile --target viz -t metro-obstacle-viz .  # RViz2 в браузере
#
# Все зависимости ставятся здесь; ручной установки после сборки не нужно.
# Узел (runtime) — ПОСЛЕДНЯЯ стадия: docker build без --target собирает её.

# ---------------------------------------------------------------- показ (--target viz)
# RViz2 в виртуальном экране, доступ из браузера через noVNC. Порт 6080
# слушается только на 127.0.0.1: снаружи — через SSH-туннель.
FROM osrf/ros:humble-desktop-jammy AS viz

ENV DEBIAN_FRONTEND=noninteractive
RUN mkdir /tmp/apt-off && mv /etc/apt/sources.list.d/* /tmp/apt-off/ \
    && apt-get update && apt-get install -y --no-install-recommends \
        tigervnc-standalone-server novnc websockify openbox fonts-dejavu \
        xterm ffmpeg \
    && mv /tmp/apt-off/* /etc/apt/sources.list.d/ && rm -rf /tmp/apt-off /var/lib/apt/lists/*

COPY ros2/metro_obstacle/rviz/metro.rviz /opt/viz/metro.rviz
COPY docker/status_view.py docker/make_video.sh /opt/viz/
COPY docker/viz.sh /viz.sh
ENV VIZ_GEOMETRY=1600x900 VIZ_PORT=6080 VIZ_BIND=127.0.0.1 FIXED_FRAME=lidar_livox
CMD ["/viz.sh"]


# ---------------------------------------------------------------- узел (по умолчанию)
FROM ros:humble-ros-base-jammy AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Из репозитория ROS ничего не ставится: rosbag2 с sqlite3, launch_ros,
# visualization_msgs уже есть в базовом образе. Его список пакетов на время apt
# отключается — packages.ros.org бывает недоступен (из Yandex Cloud — таймаут
# на индексе), а архив Ubuntu доступен всегда.
RUN mkdir /tmp/apt-off && mv /etc/apt/sources.list.d/* /tmp/apt-off/ \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-pytest \
    && mv /tmp/apt-off/* /etc/apt/sources.list.d/ && rm -rf /tmp/apt-off /var/lib/apt/lists/*

# Версии закреплены: на них сделаны все замеры (requirements.txt). numpy из pip
# встаёт поверх системного 1.21 — узлам ROS на Python он не мешает (rclpy его не
# использует), C++-инструменты (ros2 bag play) numpy не касаются.
COPY requirements.txt /opt/metro/requirements.txt
RUN pip3 install --no-cache-dir -r /opt/metro/requirements.txt

COPY rail_detection /opt/metro/rail_detection
COPY ros2/metro_obstacle /opt/ws/src/metro_obstacle
RUN . /opt/ros/humble/setup.sh \
    && cd /opt/ws && colcon build --event-handlers console_direct- \
    && rm -rf build log

# numba (экспер. 23) компилирует при первом вызове и кладёт код в кэш рядом с
# модулем — компиляция здесь, при сборке, чтобы готовый код был в образе.
RUN cd /opt/metro && PYTHONPATH=/opt/metro python3 -c "\
import numpy as np; from rail_detection import parallel_path as pp; \
s = np.linspace(1.0, 100.0, 40); \
pp._search_side(s, np.full(40, 1.8), np.ones(40), 1.8, pp.TUKEY * pp.SIGMA_EDGE)" \
    && ls /opt/metro/rail_detection/__pycache__ | grep -c "jit.*nb"

# Ускорение §35 — без изменения ответа (проверено exp_speed.py check).
ENV PYTHONPATH=/opt/metro \
    RAIL_WORKERS=4 \
    RAIL_PROCESS=1 \
    RAIL_JIT=1 \
    MPLBACKEND=Agg

# Журнал кадров — в /out (смонтировать папку хоста: -v "$PWD/out":/out):
# на каждый запуск узла свой файл detections_<дата>_<время>.jsonl.
RUN mkdir -p /out
COPY docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "metro_obstacle", "detector.launch.py", "log_file:=/out"]
