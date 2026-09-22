import sys
import os
sys.path.append('D:\\')
sys.path.append('D:\\LIDAR')
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rail_detection import POINT_DTYPE
from pipeline_procrustes_clearance import DynamicClearancePipeline
import numpy as np

dataset_dir = Path(r'D:\Датасет')
typestore = get_typestore(Stores.LATEST)

# Collect per-frame lidar projection data
u_lidar_list = []  # lateral offset of lidar from box center (in Frenet u)
v_lidar_list = []  # vertical position of lidar in Frenet v
v_min_list = []     # bottom of clearance box in Frenet v (rail_top_v + 0.10)
elastic_floor_0_list = []  # elastic_floor polynomial evaluated at d=0
elastic_axis_0_list = []   # elastic_axis polynomial evaluated at d=0
rail_top_v_list = []

pipeline = DynamicClearancePipeline()

for bag_dir in dataset_dir.iterdir():
    if not bag_dir.is_dir(): continue
    db3_files = list(bag_dir.glob('*.db3'))
    if not db3_files: continue
    
    reader = AnyReader(db3_files, default_typestore=typestore)
    with reader:
        count = 0
        for conn, ts, rawdata in reader.messages():
            if conn.msgtype != 'sensor_msgs/msg/PointCloud2': continue
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            pts = np.frombuffer(msg.data, dtype=POINT_DTYPE)
            _, _, payload = pipeline.process_pointcloud(pts)
            
            if payload['status'] == 'success':
                frame = payload['frame_geometry']
                rail_top_v = payload['clearance_boxes']['rail_top_v']
                
                ea = frame['elastic_axis']
                ef = frame['elastic_floor']
                
                # At d=0 (the lidar position, since d = -y and lidar is at y=0):
                axis_at_0 = ea(0)      # track center X in sensor coords at d=0
                floor_at_0 = ef(0)     # floor Z in sensor coords at d=0
                
                # Lidar is at sensor (x=0, z=0).
                # In Frenet frame at d=0:
                #   u_lidar = (0 - axis_at_0) * cos(heading(0))
                #   v_lidar = 0 - floor_at_0
                heading_0 = np.arctan(ea.deriv()(0))
                u_lid = (0.0 - axis_at_0) * np.cos(heading_0)
                v_lid = 0.0 - floor_at_0
                
                v_min = rail_top_v + 0.10  # bottom of clearance box in v-space
                
                u_lidar_list.append(u_lid)
                v_lidar_list.append(v_lid)
                v_min_list.append(v_min)
                elastic_floor_0_list.append(floor_at_0)
                elastic_axis_0_list.append(axis_at_0)
                rail_top_v_list.append(rail_top_v)
            
            count += 1
            if count >= 3: break

print(f'Samples collected: {len(u_lidar_list)}')
print(f'--- Lidar projection on rear face of clearance box (Frenet frame) ---')
print(f'  Mean u_lidar (lateral offset from box center): {np.mean(u_lidar_list):.4f} m')
print(f'  Mean v_lidar (lidar height in Frenet v):       {np.mean(v_lidar_list):.4f} m')
print(f'  Mean v_min (box bottom in Frenet v):           {np.mean(v_min_list):.4f} m')
print(f'  Mean rail_top_v:                               {np.mean(rail_top_v_list):.4f} m')
print(f'  Mean elastic_floor(0):                         {np.mean(elastic_floor_0_list):.4f} m')
print(f'  Mean elastic_axis(0):                          {np.mean(elastic_axis_0_list):.4f} m')
print()
print(f'--- Lidar position RELATIVE to box rear face ---')
u_rel = np.mean(u_lidar_list)
v_rel = np.mean(v_lidar_list) - np.mean(v_min_list)
print(f'  Lateral offset from box center: {u_rel:.4f} m')
print(f'  Height above box bottom:        {v_rel:.4f} m')
print(f'  (Box width={2.7}, height={3.5})')
print(f'  Fraction across width:          {(u_rel + 2.7/2) / 2.7:.3f}')
print(f'  Fraction up height:             {v_rel / 3.5:.3f}')
