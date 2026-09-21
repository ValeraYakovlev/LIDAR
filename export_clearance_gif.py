import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from rail_detection import POINT_DTYPE
from pipeline_procrustes_clearance import DynamicClearancePipeline
from viz_clearance_video import create_box_sequence_geometry

def export_gif(dataset_name):
    # 1. Настройка путей
    base_dir = Path(r"D:\Датасет")
    # Если нужно брать из viz_clearance_video.py:
    import re
    try:
        dataset_path = re.search(r"default=r'(D:\\[^']+)'", open('d:/LIDAR/viz_clearance_video.py', encoding='utf-8').read()).group(1)
        base_dir = Path(dataset_path)
    except:
        pass
        
    dataset_dir = base_dir / dataset_name
    
    db3_files = list(dataset_dir.glob("*.db3"))
    if not db3_files:
        print(f"Ошибка: Не найдены файлы .db3 в директории {dataset_dir}")
        return
    
    out_dir = Path(r"D:\LIDAR\output")
    out_dir.mkdir(exist_ok=True)
    output_filename = out_dir / f"{dataset_name}_clearance.gif"
    
    print(f"Читаем ROS bag директорию: {dataset_dir}")
    
    # 2. Инициализация пайплайна
    pipeline = DynamicClearancePipeline()
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    # 3. Настройка Open3D визуализатора
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=1200, window_name="Render GIF (Top-Down)", visible=False)
    
    # Геометрия
    pcd_outlier = o3d.geometry.PointCloud()
    pcd_obstacle = o3d.geometry.PointCloud()
    ls_boxes = o3d.geometry.LineSet()
    
    is_first_frame = True
    frames = []
    
    # Открываем директорию целиком, чтобы прочитать все .db3 куски
    reader = AnyReader(db3_files, default_typestore=typestore)
    frame_idx = 0
    with reader:
        messages_iter = reader.messages() if hasattr(reader, 'messages') else []
        for i, (conn, ts, rawdata) in enumerate(messages_iter):
            if conn.msgtype != 'sensor_msgs/msg/PointCloud2':
                continue
                
            if hasattr(typestore, 'deserialize_cdr'):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
            else:
                msg = reader.deserialize(rawdata, conn.msgtype) if hasattr(reader, 'deserialize') else typestore.deserialize_cdr(rawdata, conn.msgtype)
                
            pts = np.frombuffer(msg.data, dtype=POINT_DTYPE)
            
            # Прогоняем через алгоритм
            obstacle_points, min_dist, payload = pipeline.process_pointcloud(pts)
            if payload["status"] != "success":
                continue
                
            frame_idx += 1
            frame_geom = payload["frame_geometry"]
            box_params = payload["clearance_boxes"]
            
            xyz_all = np.vstack((pts['x'], pts['y'], pts['z'])).T
            if len(obstacle_points) > 0:
                xyz_obs = np.vstack((obstacle_points['x'], obstacle_points['y'], obstacle_points['z'])).T
            else:
                xyz_obs = np.empty((0, 3))
                
            xyz_all_decimated = xyz_all[::10]
            
            pcd_outlier.points = o3d.utility.Vector3dVector(xyz_all_decimated)
            pcd_outlier.paint_uniform_color([0.5, 0.5, 0.5])
            
            pcd_obstacle.points = o3d.utility.Vector3dVector(xyz_obs)
            colors = payload.get("obstacle_colors")
            if colors is not None and len(colors) > 0:
                pcd_obstacle.colors = o3d.utility.Vector3dVector(colors)
            if "clearance_boxes" in payload:
                box_params = payload["clearance_boxes"]
                pts, lns, cls = create_box_sequence_geometry(
                    box_params["boundaries"], 
                    box_params["width"], 
                    box_params["height"], 
                    payload["frame_geometry"], 
                    box_params.get("rail_top_v", 0.0)
                )
                if len(pts) > 0:
                    ls_boxes.points = o3d.utility.Vector3dVector(pts)
                    ls_boxes.lines = o3d.utility.Vector2iVector(lns)
                    ls_boxes.colors = o3d.utility.Vector3dVector(cls)
                else:
                    ls_boxes.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
                    ls_boxes.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
                    ls_boxes.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))
            
            if is_first_frame:
                vis.add_geometry(pcd_outlier)
                vis.add_geometry(pcd_obstacle)
                vis.add_geometry(ls_boxes)
                
                # ЖЕСТКАЯ ФИКСАЦИЯ КАМЕРЫ (Крупно, строго сверху)
                view_ctrl = vis.get_view_control()
                # Смотрим из +Z вниз на объект
                view_ctrl.set_front([0.0, 0.0, 1.0])
                # Вектор UP: чтобы вперед (-y) было направлено вверх на экране
                view_ctrl.set_up([0.0, -1.0, 0.0])
                # Смещаем центр обзора ближе к началу (чтобы начало координат было внизу экрана)
                view_ctrl.set_lookat([0.0, -45.0, 0.0])
                # Зум значительно приближен, чтобы не "висеть высоко в воздухе"
                view_ctrl.set_zoom(0.22)
                
                is_first_frame = False
            else:
                vis.update_geometry(pcd_outlier)
                vis.update_geometry(pcd_obstacle)
                vis.update_geometry(ls_boxes)
            
            vis.poll_events()
            vis.update_renderer()
            
            # Захват кадра (без альфа канала, float -> uint8)
            img_float = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            # COPY THE ARRAY to prevent repeating last frame bug in Open3D offscreen buffers
            img_uint8 = np.array(img_float * 255.0, copy=True).astype(np.uint8)
            frames.append(Image.fromarray(img_uint8))
            
            print(f"Кадр {frame_idx:03d} добавлен в GIF...", end='\r')
            
    vis.destroy_window()
    print(f"\nСохранение GIF файла '{output_filename}' ({len(frames)} кадров)...")
    
    if frames:
        # Сохраняем в GIF со скоростью ~10 FPS (100 ms)
        frames[0].save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=100,
            loop=0
        )
        print("Готово!")

if __name__ == "__main__":
    export_gif("roundT_squareT_pressureGate_squareT")
