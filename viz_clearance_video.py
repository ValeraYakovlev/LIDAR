import numpy as np
import open3d as o3d
import cv2
import os
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rail_detection import POINT_DTYPE
from pipeline_procrustes_clearance import DynamicClearancePipeline
from rail_detection.tunnel_frame import eval_fit, slope_from_fit

def track_to_sensor(d, u, v, frame):
    """Переводит координаты из (глубина, смещение вбок, высота над полом) в (x, y, z) сенсора."""
    if "elastic_axis" in frame and frame["elastic_axis"] is not None:
        xc = frame["elastic_axis"](d)
        heading = np.arctan(frame["elastic_deriv"](d))
        z = v + frame["elastic_floor"](d)
    else:
        xc = eval_fit(frame["axis_fit"], d)
        heading = np.arctan(slope_from_fit(frame["axis_fit"], d))
        z = v + np.polyval(frame["floor_coeffs"], d)
        
    x = xc + u / np.cos(heading)
    y = -d
    return x, y, z

def create_box_sequence_linesets(boundaries, width, height, frame, rail_top_v):
    """
    Рисует последовательность коробок. Использует ICP трансформации, если они есть.
    """
    linesets = []

    for b in boundaries:
        d_s = b["start"]
        d_e = b["end"]
        
        v_min = rail_top_v + 0.10
        v_max = v_min + height
        
        u_min = -width / 2.0
        u_max = width / 2.0
        
        # 4 угла в 2D (координаты U, V, Z=0, W=1)
        corners_2d = np.array([
            [u_min, v_min, 0, 1],
            [u_max, v_min, 0, 1],
            [u_max, v_max, 0, 1],
            [u_min, v_max, 0, 1]
        ])
        
        if "T_inv" in b:
            # Применяем обратное преобразование ICP
            T_inv = b["T_inv"]
            c_transformed = (T_inv @ corners_2d.T).T
            u_c = c_transformed[:, 0]
            v_c = c_transformed[:, 1]
        else:
            u_c = np.array([u_min, u_max, u_max, u_min])
            v_c = np.array([v_min, v_min, v_max, v_max])
            
        corners = []
        # Передняя грань (d_s)
        for i in range(4):
            corners.append(track_to_sensor(d_s, u_c[i], v_c[i], frame))
        # Задняя грань (d_e)
        for i in range(4):
            corners.append(track_to_sensor(d_e, u_c[i], v_c[i], frame))
        
        # Линии (ребра параллелепипеда) - строго без диагоналей!
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0], # Периметр задней грани
            [4, 5], [5, 6], [6, 7], [7, 4], # Периметр передней грани
            [0, 4], [1, 5], [2, 6], [3, 7]  # Продольные ребра
        ]
        
        # Если смещение ICP больше 0.1, можем подсветить коробку
        colors = [[0, 1, 0] for _ in range(len(lines))]
        
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(corners)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(colors)
        linesets.append(ls)
        
    return linesets

import argparse

def main(bag_folder_name="doubleT_obstacle"):
    """
    Запуск визуализации габарита.
    :param bag_folder_name: Название папки с сумкой в D:\Датасет (например, 'doubleT_obstacle')
    """
    # Можно передать через командную строку, либо напрямую в вызов функции
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', default=bag_folder_name, help='Имя папки с bag-файлом (например, doubleT_platform)')
    parser.add_argument('--dataset', default=r'D:\Датасет', help='Путь к папке со всеми датасетами')
    args = parser.parse_args()
    
    bag_dir = Path(args.dataset) / args.bag
    if not bag_dir.exists():
        print(f"Папка {bag_dir} не найдена!")
        return
        
    # Находим .db3 файл напрямую
    db3_files = list(bag_dir.glob('*.db3'))
    if not db3_files:
        print(f"Не найден .db3 файл в {bag_dir}")
        return
    db3_path = db3_files[0]
        
    print(f"Используем файл: {db3_path}")
    
    # Параметры из ТЗ: 3.1м ширина, 3.7м высота, 1м длина. Зазор 0.25м.
    pipeline = DynamicClearancePipeline()
    typestore = get_typestore(Stores.LATEST)
    
    # Настраиваем визуализатор
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Clearance Trench Interactive", width=1280, height=720, visible=True)
    
    pcd_inlier = o3d.geometry.PointCloud()
    pcd_outlier = o3d.geometry.PointCloud()
    geometries_added = False
    box_linesets = []
    
    from rosbags.highlevel import AnyReader
    
    print("Начинаем бесконечный цикл воспроизведения. Закройте окно для выхода.")
    import cv2
    import time
    
    cv2.namedWindow("Timeline", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Timeline", 800, 50)
    def on_trackbar(val):
        pass
    cv2.createTrackbar("Frame", "Timeline", 0, 1, on_trackbar)
    
    frame_cache = []
    bag_finished = False
    
    print("Проигрывание началось. Используйте ползунок в окне Timeline для перемотки.")
    try:
        proc_times = []
        reader = AnyReader([db3_path], default_typestore=typestore)
        with reader:
            messages_iter = iter(reader.messages()) if hasattr(reader, 'messages') else iter([])
            
            while True:
                pos = cv2.getTrackbarPos("Frame", "Timeline")
                
                # Если ползунок в конце и датасет не закончился - обрабатываем новый кадр
                if (pos >= len(frame_cache) - 1 or len(frame_cache) == 0) and not bag_finished:
                    try:
                        conn, ts, rawdata = next(messages_iter)
                        while conn.msgtype != 'sensor_msgs/msg/PointCloud2':
                            conn, ts, rawdata = next(messages_iter)
                            
                        if hasattr(typestore, 'deserialize_cdr'):
                            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                        else:
                            msg = reader.deserialize(rawdata, conn.msgtype) if hasattr(reader, 'deserialize') else typestore.deserialize_cdr(rawdata, conn.msgtype)
                            
                        pts = np.frombuffer(msg.data, dtype=POINT_DTYPE)
                        
                        t0 = time.perf_counter()
                        obstacle_points, min_dist, payload = pipeline.process_pointcloud(pts)
                        t1 = time.perf_counter()
                        proc_times.append((t1 - t0) * 1000.0)
                        
                        if payload["status"] == "success":
                            xyz_all = np.vstack((pts['x'], pts['y'], pts['z'])).T
                            xyz_all_decimated = xyz_all[::10]
                            
                            if len(obstacle_points) > 0:
                                xyz_obs = np.vstack((obstacle_points['x'], obstacle_points['y'], obstacle_points['z'])).T
                            else:
                                xyz_obs = np.empty((0, 3))
                                
                            frame_cache.append({
                                "xyz_all_decimated": xyz_all_decimated,
                                "xyz_obs": xyz_obs,
                                "payload": payload,
                                "min_dist": min_dist,
                                "frame_idx": len(frame_cache)
                            })
                            
                            max_val = max(1, len(frame_cache) - 1)
                            cv2.setTrackbarMax("Frame", "Timeline", max_val)
                            cv2.setTrackbarPos("Frame", "Timeline", max_val)
                            pos = max_val
                            
                    except StopIteration:
                        bag_finished = True
                        print("\nКонец файла. Теперь можно свободно листать ползунок туда-сюда.")
                
                # Рендерим кадр под ползунком
                if len(frame_cache) > 0:
                    pos = min(pos, len(frame_cache) - 1)
                    cached = frame_cache[pos]
                    
                    pcd_outlier.points = o3d.utility.Vector3dVector(cached["xyz_all_decimated"])
                    pcd_outlier.paint_uniform_color([0.5, 0.5, 0.5])
                    
                    if len(cached["xyz_obs"]) > 0:
                        pcd_inlier.points = o3d.utility.Vector3dVector(cached["xyz_obs"])
                        colors = cached["payload"].get("obstacle_colors")
                        if colors is not None:
                            pcd_inlier.colors = o3d.utility.Vector3dVector(colors)
                        else:
                            pcd_inlier.paint_uniform_color([1.0, 0.0, 0.0])
                    else:
                        pcd_inlier.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
                        if hasattr(pcd_inlier, 'colors'):
                            pcd_inlier.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))
                        
                    for ls in box_linesets:
                        vis.remove_geometry(ls, reset_bounding_box=False)
                    box_linesets.clear()
                    
                    box_params = cached["payload"]["clearance_boxes"]
                    new_linesets = create_box_sequence_linesets(
                        box_params.get("boundaries", []), 
                        box_params["width"], 
                        box_params["height"], 
                        cached["payload"]["frame_geometry"], 
                        box_params.get("rail_top_v", 0.0)
                    )
                    
                    for ls in new_linesets:
                        box_linesets.append(ls)
                        vis.add_geometry(ls, reset_bounding_box=False)
                        
                    if not geometries_added:
                        vis.add_geometry(pcd_outlier)
                        vis.add_geometry(pcd_inlier)
                        ctr = vis.get_view_control()
                        ctr.set_lookat([0, -20, 0])
                        ctr.set_up([0, 0, 1])
                        ctr.set_front([0.0, 1.0, 0.3])
                        ctr.set_zoom(0.3)
                        geometries_added = True
                    else:
                        vis.update_geometry(pcd_outlier)
                        vis.update_geometry(pcd_inlier)
                        
                    med_time = np.median(proc_times) if proc_times else 0.0
                    time_str = f"| Pipeline Median: {med_time:.1f} ms"
                    
                    if cached["min_dist"] != np.inf:
                        print(f"Frame {cached['frame_idx']:03d}: Препятствие на {cached['min_dist']:5.2f} м {time_str}       ", end='\r')
                    else:
                        print(f"Frame {cached['frame_idx']:03d}: Путь свободен           {time_str}       ", end='\r')
                        
                if not vis.poll_events():
                    break
                vis.update_renderer()
                
                key = cv2.waitKey(10) & 0xFF
                if key == 27: # ESC
                    break
                    
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("\nПросмотр завершен.")

if __name__ == "__main__":
    # Вы можете поменять имя папки здесь, если запускаете из VS Code
    # Например: main("doubleT_platform")
    main("roundT_squareT_pressureGate_squareT")
