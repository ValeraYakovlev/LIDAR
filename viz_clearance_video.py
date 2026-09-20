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

def create_box_sequence_geometry(boundaries, width, height, frame, rail_top_v):
    """
    Рисует последовательность коробок единым мешем (points, lines, colors).
    Промежутки между коробками отрисовываются как соединительные деформируемые линии (сцепки).
    """
    all_points = []
    all_lines = []
    all_colors = []
    
    offset = 0
    has_prev_box = False
    
    for b in boundaries:
        if b.get("is_link", False):
            continue
            
        d_s = b["start"]
        d_e = b["end"]
        
        v_min = rail_top_v + 0.10
        v_max = v_min + height
        
        u_min = -width / 2.0
        u_max = width / 2.0
        
        corners_2d = np.array([
            [u_min, v_min, 0, 1],
            [u_max, v_min, 0, 1],
            [u_max, v_max, 0, 1],
            [u_min, v_max, 0, 1]
        ])
        
        if "T_inv" in b:
            c_transformed = (b["T_inv"] @ corners_2d.T).T
            u_c, v_c = c_transformed[:, 0], c_transformed[:, 1]
        else:
            u_c = np.array([u_min, u_max, u_max, u_min])
            v_c = np.array([v_min, v_min, v_max, v_max])
            
        corners = []
        for i in range(4): corners.append(track_to_sensor(d_s, u_c[i], v_c[i], frame))
        for i in range(4): corners.append(track_to_sensor(d_e, u_c[i], v_c[i], frame))
        
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0], # front
            [4, 5], [5, 6], [6, 7], [7, 4], # rear
            [0, 4], [1, 5], [2, 6], [3, 7]  # longitudinal
        ]
        
        for p in corners: all_points.append(p)
        for l in lines: all_lines.append([l[0] + offset, l[1] + offset])
        for _ in lines: all_colors.append([0, 1, 0])
        
        # Если есть предыдущая коробка, рисуем деформируемую сцепку (4 линии)
        if has_prev_box:
            for i in range(4):
                all_lines.append([offset - 4 + i, offset + i])
                all_colors.append([0, 1, 0])
                
        has_prev_box = True
        offset += 8
        
    return np.array(all_points, dtype=np.float64), np.array(all_lines, dtype=np.int32), np.array(all_colors, dtype=np.float64)

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
    ls_boxes = o3d.geometry.LineSet()
    
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
                        
                    box_params = cached["payload"]["clearance_boxes"]
                    pts, lns, cls = create_box_sequence_geometry(
                        box_params.get("boundaries", []), 
                        box_params["width"], 
                        box_params["height"], 
                        cached["payload"]["frame_geometry"], 
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
                        
                    if not geometries_added:
                        vis.add_geometry(pcd_outlier)
                        vis.add_geometry(pcd_inlier)
                        vis.add_geometry(ls_boxes)
                        ctr = vis.get_view_control()
                        ctr.set_lookat([0, -20, 0])
                        ctr.set_up([0, 0, 1])
                        ctr.set_front([0.0, 1.0, 0.3])
                        ctr.set_zoom(0.3)
                        geometries_added = True
                    else:
                        vis.update_geometry(pcd_outlier)
                        vis.update_geometry(pcd_inlier)
                        vis.update_geometry(ls_boxes)
                        
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
