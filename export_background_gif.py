import sys
import argparse
import re
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg') # Безопасный рендер без окон
import matplotlib.pyplot as plt
import open3d as o3d

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rail_detection import POINT_DTYPE
from rail_detection.tunnel_frame import fit_tunnel_geometry, to_track_coords, eval_fit, slope_from_fit

def track_to_sensor(d, u, v, frame):
    xc = eval_fit(frame["axis_fit"], d)
    heading = np.arctan(slope_from_fit(frame["axis_fit"], d))
    z = v + np.polyval(frame["floor_coeffs"], d)
    x = xc + u / np.cos(heading)
    y = -d
    return x, y, z

def process_frame(points):
    geom = fit_tunnel_geometry(points)
    if geom is None:
        return None, None
        
    frame = geom["frame"]
    x = points['x'].astype(float)
    y = points['y'].astype(float)
    z = points['z'].astype(float)
        
    d, u, v = to_track_coords(x, y, z, frame)
    valid = (d >= 0) & (d <= 40)
    pts_valid = points[valid]
    d_val = d[valid]
    u_val = u[valid]
    v_val = v[valid]
    
    boxes = [{"start": start_d, "end": start_d + 1.0} for start_d in np.arange(0, 40, 1.0)]
    ref_mask = (d_val >= 0) & (d_val <= 3.0)
    if np.sum(ref_mask) < 100:
        return None, None
        
    pts_ref_2d = np.column_stack((u_val[ref_mask], v_val[ref_mask], np.zeros(np.sum(ref_mask))))
    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points = o3d.utility.Vector3dVector(pts_ref_2d)
    pcd_ref = pcd_ref.voxel_down_sample(voxel_size=0.05)
    
    u_aligned = np.copy(u_val)
    v_aligned = np.copy(v_val)
    ideal_tunnel_points = []
    
    ref_2d_hom = np.column_stack((np.asarray(pcd_ref.points)[:, :2], np.zeros(len(pcd_ref.points)), np.ones(len(pcd_ref.points))))
    
    for b in boxes:
        b_mask = (d_val >= b["start"]) & (d_val <= b["end"])
        if np.sum(b_mask) < 10:
            b["T_inv"] = np.eye(4)
        elif b["end"] <= 3.0:
            b["T_inv"] = np.eye(4)
        else:
            pts_seg_raw = np.column_stack((u_val[b_mask], v_val[b_mask], np.zeros(np.sum(b_mask))))
            pcd_seg = o3d.geometry.PointCloud()
            pcd_seg.points = o3d.utility.Vector3dVector(pts_seg_raw)
            pcd_seg = pcd_seg.voxel_down_sample(voxel_size=0.05)
            
            reg = o3d.pipelines.registration.registration_icp(
                pcd_seg, pcd_ref, 0.5, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            T = reg.transformation
            b["T_inv"] = np.linalg.inv(T)
            
            pts_hom = np.column_stack((pts_seg_raw, np.ones(len(pts_seg_raw))))
            pts_aligned = (T @ pts_hom.T).T
            u_aligned[b_mask] = pts_aligned[:, 0]
            v_aligned[b_mask] = pts_aligned[:, 1]
            
        seg_2d = (b["T_inv"] @ ref_2d_hom.T).T
        for d_step in [b["start"], b["start"] + 0.5]:
            xc_v = eval_fit(frame["axis_fit"], d_step)
            heading_v = np.arctan(slope_from_fit(frame["axis_fit"], d_step))
            z_base = np.polyval(frame["floor_coeffs"], d_step)
            
            x_v = xc_v + seg_2d[:, 0] / np.cos(heading_v)
            y_v = np.full(len(seg_2d), -d_step)
            z_v = seg_2d[:, 1] + z_base
            
            ideal_tunnel_points.extend(np.column_stack((x_v, y_v, z_v)))
                
    pts_query = np.column_stack((u_aligned, v_aligned, np.zeros(len(u_aligned))))
    pcd_query = o3d.geometry.PointCloud()
    pcd_query.points = o3d.utility.Vector3dVector(pts_query)
    
    distances = np.asarray(pcd_query.compute_point_cloud_distance(pcd_ref))
    is_anomaly = distances > 0.15
    
    colors = np.full((len(pts_valid), 3), [0.5, 0.5, 0.5])
    colors[is_anomaly] = [1.0, 0.0, 0.0]
    
    actual_xyz = np.column_stack((pts_valid['x'], pts_valid['y'], pts_valid['z']))
    pcd_actual = o3d.geometry.PointCloud()
    pcd_actual.points = o3d.utility.Vector3dVector(actual_xyz)
    pcd_actual.colors = o3d.utility.Vector3dVector(colors)
    
    pcd_ideal = o3d.geometry.PointCloud()
    if len(ideal_tunnel_points) > 0:
        pcd_ideal.points = o3d.utility.Vector3dVector(ideal_tunnel_points)
        pcd_ideal.paint_uniform_color([0.0, 1.0, 0.0])
        
    return pcd_actual, pcd_ideal

def export_gif(bag_folder="doubleT_obstacle"):
    # Извлекаем путь к датасету
    try:
        dataset_path = re.search(r"default=r'(D:\\[^']+)'", open('d:/LIDAR/viz_clearance_video.py', encoding='utf-8').read()).group(1)
    except:
        dataset_path = r"D:\Дата"
        
    bag_dir = Path(dataset_path) / bag_folder
    
    if not bag_dir.exists():
        print(f"Путь не найден: {bag_dir}")
        return
        
    db3_files = list(bag_dir.glob('*.db3'))
    if not db3_files:
        print(f"Нет файлов .db3 в {bag_dir}")
        return
    
    out_dir = Path(r"D:\LIDAR\output")
    out_dir.mkdir(exist_ok=True)
    output_filename = out_dir / f"{bag_folder}_background.gif"
    
    print(f"Читаем ROS bag директорию: {bag_dir}")
    typestore = get_typestore(Stores.LATEST)
    
    frames = []
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 10), facecolor='black')
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.90, wspace=0.1)
    
    frame_idx = 0
    # Открываем напрямую .db3 файлы, чтобы избежать ошибок с metadata.yaml
    reader = AnyReader(db3_files, default_typestore=typestore)
    with reader:
        for i, (conn, ts, rawdata) in enumerate(reader.messages()):
            if conn.msgtype != 'sensor_msgs/msg/PointCloud2':
                continue
                
            if hasattr(typestore, 'deserialize_cdr'):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            else:
                msg = reader.deserialize(rawdata, conn.msgtype) if hasattr(reader, 'deserialize') else typestore.deserialize_cdr(rawdata, conn.msgtype)
                
            pts = np.frombuffer(msg.data, dtype=POINT_DTYPE)
            
            pcd_a, pcd_i = process_frame(pts)
            if pcd_a is None or pcd_i is None:
                continue
                
            frame_idx += 1
            pts_a = np.asarray(pcd_a.points)
            colors_a = np.asarray(pcd_a.colors)
            pts_i = np.asarray(pcd_i.points)
            
            for ax in axes:
                ax.clear()
                ax.set_facecolor('black')
                ax.set_xlim(-5, 5)
                # Переворачиваем ось Y: 2 внизу, -40 вверху (начало координат внизу экрана)
                ax.set_ylim(2, -40)
                ax.set_aspect('equal')
                ax.axis('off')
            
            # Subsample for faster rendering
            decimate = 10
            
            # Разделяем серые (норма) и красные (аномалии)
            is_red = colors_a[:, 0] > 0.8
            is_gray = ~is_red
            
            # Panel 1: Ideal Tunnel
            axes[0].scatter(pts_i[::decimate, 0], pts_i[::decimate, 1], c='lime', s=0.5, alpha=0.3)
            axes[0].set_title("Ideal Reference (ICP)", color='lime', fontsize=14, pad=10)
            
            # Panel 2: Matched Walls
            axes[1].scatter(pts_a[is_gray][::decimate, 0], pts_a[is_gray][::decimate, 1], c='gray', s=0.5, alpha=0.5)
            axes[1].set_title("Matched Walls", color='gray', fontsize=14, pad=10)
            
            # Panel 3: Anomalies / Obstacles
            axes[2].scatter(pts_a[is_red][::decimate, 0], pts_a[is_red][::decimate, 1], c='red', s=3.0, alpha=1.0)
            axes[2].set_title("Obstacles (>15cm)", color='red', fontsize=14, pad=10)
            
            # Добавляем номер кадра как общий заголовок над всеми графиками
            fig.suptitle(f"Frame: {frame_idx:03d}", color='white', fontsize=16, y=0.98)
            
            fig.canvas.draw()
            # COPY THE ARRAY so it's not a view of a mutable buffer that gets overwritten!
            img = np.array(fig.canvas.buffer_rgba(), copy=True)
            frames.append(Image.fromarray(img))
            
            print(f"Кадр {frame_idx:03d} добавлен в GIF...", end='\r')
                
    print(f"\nСохранение GIF файла '{output_filename}' ({len(frames)} кадров)...")
    
    if frames:
        frames[0].save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=100,
            loop=0
        )
        print("Готово!")
    plt.close(fig)

if __name__ == "__main__":
    export_gif("roundT_squareT_pressureGate_squareT")
