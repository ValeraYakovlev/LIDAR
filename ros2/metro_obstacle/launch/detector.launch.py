"""Узел обнаружения; rviz:=true — ещё и RViz2 с готовой раскладкой."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARGS = {
    "topic": ("auto", "топик PointCloud2; auto — найти самому"),
    "variant": ("low_rest_b0", "вариант детектора (rail_detection.far_detect.VARIANTS)"),
    "queue": ("all", "all — все кадры по порядку, latest — только свежий кадр"),
    "preview_stride": ("10", "прореживание облака для показа; 0 — не публиковать"),
    "log_file": ("", "JSON Lines по кадрам"),
    "rviz": ("false", "запустить RViz2"),
}


def generate_launch_description():
    cfg = LaunchConfiguration
    rviz_cfg = os.path.join(get_package_share_directory("metro_obstacle"), "rviz", "metro.rviz")
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v, description=d) for k, (v, d) in ARGS.items()]
        + [
            Node(package="metro_obstacle", executable="obstacle_detector", output="screen",
                 emulate_tty=True,
                 parameters=[{"topic": cfg("topic"), "variant": cfg("variant"),
                              "queue": cfg("queue"),
                              "preview_stride": cfg("preview_stride"),
                              "log_file": cfg("log_file")}]),
            Node(package="rviz2", executable="rviz2", arguments=["-d", rviz_cfg],
                 condition=IfCondition(cfg("rviz"))),
        ])
