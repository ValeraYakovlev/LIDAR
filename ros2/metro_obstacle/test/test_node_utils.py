import numpy as np

from metro_obstacle.node import pick_topic, xyz_cloud
from std_msgs.msg import Header

CLOUD = ["sensor_msgs/msg/PointCloud2"]


def test_pick_topic_prefers_lidar_points_and_skips_labels():
    topics = [("/lidar_points_labeled", CLOUD), ("/lidar_points", CLOUD), ("/tf", ["tf2_msgs/msg/TFMessage"])]
    assert pick_topic(topics) == "/lidar_points"


def test_pick_topic_real_bag_and_own_outputs():
    topics = [("/obstacle/gauge_points", CLOUD), ("/sensing/lidar/hesai128/pointcloud", CLOUD)]
    assert pick_topic(topics) == "/sensing/lidar/hesai128/pointcloud"
    assert pick_topic([("/obstacle/cloud_preview", CLOUD)]) is None


def test_xyz_cloud_roundtrip():
    xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
    msg = xyz_cloud(Header(frame_id="lidar"), xyz)
    back = np.frombuffer(msg.data, np.float32).reshape(-1, 3)
    assert msg.width == 4 and np.array_equal(back, xyz)
