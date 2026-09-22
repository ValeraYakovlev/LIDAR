import numpy as np
import time
import open3d as o3d
import cv2
import argparse
from pathlib import Path
import sys

# Подключаем локальные модули для POINT_DTYPE
sys.path.append(r'D:\LIDAR')
try:
    from rail_detection import POINT_DTYPE
except ImportError:
    # Fallback, если не удалось импортировать
    POINT_DTYPE = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('intensity', np.float32)
    ])

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.highlevel import AnyReader

class RigidKinematicClearance:
    def __init__(self, width: float, height: float, length: float, d_start: float):
        """
        Инициализация жесткого кинематического габарита.
        :param width: Ширина вагона (W) (м)
        :param height: Высота вагона (H) (м)
        :param length: Длина жесткой базы (L) (м)
        :param d_start: Смещение от лидара до задней грани первого звена (м)
        """
        self.W = width
        self.H = height
        self.L = length
        self.d_start = d_start
        
        self.rotation_matrix = np.eye(3)
        self.box_origin = np.zeros(3)

    def build_local_frame(self, lidar_origin: np.ndarray, normal: np.ndarray, tangent: np.ndarray):
        """
        Векторизованное построение локального базиса O(1).
        """
        Z = tangent / np.linalg.norm(tangent)
        X = np.cross(Z, normal)
        X = X / np.linalg.norm(X)
        Y = np.cross(X, Z)
        Y = Y / np.linalg.norm(Y)
        
        self.rotation_matrix = np.column_stack((X, Y, Z))
        self.box_origin = lidar_origin + self.d_start * Z
        
        return self.rotation_matrix, self.box_origin

    def filter_points_in_rigid_box(self, point_cloud: np.ndarray):
        """
        Запускает луч (габарит поезда). Луч летит прямо, пока не врежется в плотную стену
        (много точек в срезе). При ударе отражается, и из точки удара запускается новый луч.
        На выходе получаем последовательность длинных жестких боксов (сегментов).
        """
        pts_local = (point_cloud - self.box_origin) @ self.rotation_matrix
        
        # Оставляем только точки в пределах высоты вагона
        y_mask = (pts_local[:, 1] >= 0.0) & (pts_local[:, 1] <= self.H)
        pts_2d = pts_local[:, [0, 2]]
        pts_valid = pts_2d[y_mask]
        
        pos = np.array([0.0, 0.0])
        dir = np.array([0.0, 1.0])
        
        path_segments = [] # (start_pos, dir, length)
        
        step_size = 0.5
        
        for iteration in range(10): # Максимум 10 отскоков
            current_t = 0.0
            bounce_found = False
            
            while current_t < 200.0:
                perp = np.array([-dir[1], dir[0]])
                
                vecs = pts_valid - pos
                local_y = vecs @ dir
                local_x = vecs @ perp
                
                # Проверяем срез на текущем шаге
                mask_hit = (local_y > current_t) & (local_y <= current_t + step_size) & (np.abs(local_x) <= self.W / 2.0)
                hit_pts = pts_valid[mask_hit]
                
                # "когда очень много точек распределены примерно равномерно"
                if len(hit_pts) > 20:
                    avg_x = np.mean(local_x[mask_hit])
                    side = -1 if avg_x < 0 else 1
                    
                    cov = np.cov((hit_pts - np.mean(hit_pts, axis=0)).T)
                    if not np.any(np.isnan(cov)) and not np.any(np.isinf(cov)):
                        vals, vecs_eig = np.linalg.eigh(cov)
                        wall_dir = vecs_eig[:, 1]
                        if wall_dir @ dir < 0:
                            wall_dir = -wall_dir
                        wall_normal = np.array([-wall_dir[1], wall_dir[0]])
                        if wall_normal @ perp * side > 0:
                            wall_normal = -wall_normal
                            
                        wall_normal = wall_normal / (np.linalg.norm(wall_normal) + 1e-9)
                        
                        # Отражение направления
                        new_dir = dir - 2 * np.dot(dir, wall_normal) * wall_normal
                        new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-9)
                        
                        # Сохраняем завершенный длинный жесткий бокс
                        path_segments.append((pos.copy(), dir.copy(), current_t))
                        
                        # Новая позиция для следующего бокса
                        pos = pos + dir * current_t
                        dir = new_dir
                        bounce_found = True
                        break
                    
                current_t += step_size
                
            if not bounce_found:
                # Улетели в бесконечность или конец облака
                path_segments.append((pos.copy(), dir.copy(), current_t))
                break
                
        # --- Построение геометрии ---
        mask = np.zeros(len(point_cloud), dtype=bool)
        ls_points = []
        ls_lines = []
        point_idx_offset = 0
        
        for p0, d, length in path_segments:
            if length <= 0: continue
            
            perp = np.array([-d[1], d[0]])
            
            # Маскируем точки, попавшие именно в этот жесткий бокс
            vecs = pts_2d - p0
            local_y = vecs @ d
            local_x = vecs @ perp
            seg_mask = (local_y >= 0.0) & (local_y <= length) & (np.abs(local_x) <= self.W / 2.0) & y_mask
            mask |= seg_mask
            
            # Построение 3D-проволоки для ОДНОГО длинного жесткого бокса
            w2 = self.W / 2.0
            h = self.H
            l = length
            
            p0_3d = np.array([p0[0], 0.0, p0[1]])
            d_3d = np.array([d[0], 0.0, d[1]])
            perp_3d = np.array([perp[0], 0.0, perp[1]])
            up_3d = np.array([0.0, h, 0.0])
            
            c0 = p0_3d - perp_3d * w2
            c1 = p0_3d + perp_3d * w2
            c2 = c1 + up_3d
            c3 = c0 + up_3d
            c4 = c0 + d_3d * l
            c5 = c1 + d_3d * l
            c6 = c5 + up_3d
            c7 = c4 + up_3d
            
            box_pts = np.vstack((c0, c1, c2, c3, c4, c5, c6, c7))
            box_global = box_pts @ self.rotation_matrix.T + self.box_origin
            ls_points.extend(box_global)
            
            box_lines = np.array([
                [0,1], [1,2], [2,3], [3,0], 
                [4,5], [5,6], [6,7], [7,4], 
                [0,4], [1,5], [2,6], [3,7]
            ]) + point_idx_offset
            ls_lines.extend(box_lines)
            point_idx_offset += 8
            
        return mask, np.array(ls_points), np.array(ls_lines)

def run_zero_allocation_vis(bag_folder_name="doubleT_obstacle", dataset_path=r"D:\Датасет"):
    """
    Демонстрация с чтением из ROS bag (D:\Датасет) с UI в точности как в viz_clearance_video.py.
    """
    bag_dir = Path(dataset_path) / bag_folder_name
    if not bag_dir.exists():
        print(f"Папка {bag_dir} не найдена! Проверьте путь.")
        return
        
    db3_files = list(bag_dir.glob('*.db3'))
    if not db3_files:
        print(f"Не найден .db3 файл в {bag_dir}")
        return
        
    print(f"Читаем ROS bag директорию: {bag_dir}")
    typestore = get_typestore(Stores.LATEST)
    
    # =====================================================================
    # Параметры поезда (НЕ запретной зоны!): ширина 2.1м, высота 3.0м
    # Смещение начала габарита вперед (d_start = 1.5м)
    # =====================================================================
    clearance = RigidKinematicClearance(width=2.1, height=3.0, length=15.0, d_start=1.5)
    
    # Жёсткий базис (лидар выровнен с путём: Z вверх, движение вперёд по -Y)
    lidar_orig = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])   # Z вверх
    tangent = np.array([0.0, -1.0, 0.0])  # Движение вперёд по -Y
    clearance.build_local_frame(lidar_orig, normal, tangent)
    
    # =====================================================================
    # Выравнивание габарита по проекции лидара из pipeline_procrustes_clearance.py
    # Рассчитано по всему датасету (calc_means.py):
    #   x = -0.045 (центр бокса левее лидара)
    #   z = -1.109 (дно бокса ниже лидара)
    # Добавляем это смещение к уже вычисленному стартовому положению (d_start)
    # =====================================================================
    clearance.box_origin = clearance.box_origin + np.array([-0.045, 0.0, -1.109])
    
    print("\nЗапуск zero-allocation визуализации Open3D (Timeline UI)...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Clearance Trench Interactive", width=1280, height=720, visible=True)
    
    pcd_inlier = o3d.geometry.PointCloud()
    pcd_outlier = o3d.geometry.PointCloud()
    ls_boxes = o3d.geometry.LineSet()
    
    # Добавляем красную сферу для отображения точки проекции лидара (начало отсчета габарита)
    lidar_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    lidar_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    lidar_sphere.translate(clearance.box_origin)
    
    geometries_added = False
    
    cv2.namedWindow("Timeline", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Timeline", 800, 50)
    def on_trackbar(val): pass
    cv2.createTrackbar("Frame", "Timeline", 0, 1, on_trackbar)
    
    frame_cache = []
    bag_finished = False
    
    print("Проигрывание началось. Используйте ползунок в окне Timeline для перемотки.")
    
    try:
        reader = AnyReader(db3_files, default_typestore=typestore)
        with reader:
            messages_iter = iter(reader.messages()) if hasattr(reader, 'messages') else iter([])
            
            while True:
                pos = cv2.getTrackbarPos("Frame", "Timeline")
                
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
                        xyz_all = np.vstack((pts['x'], pts['y'], pts['z'])).T
                        
                        current_pc = xyz_all[::5] 
                        
                        t0 = time.perf_counter()
                        mask, box_pts, box_lines = clearance.filter_points_in_rigid_box(current_pc)
                        t1 = time.perf_counter()
                        
                        xyz_obs = current_pc[mask]
                        xyz_out = current_pc[~mask]
                        
                        frame_cache.append({
                            "xyz_out": xyz_out.copy(),
                            "xyz_obs": xyz_obs.copy(),
                            "box_pts": box_pts.copy(),
                            "box_lines": box_lines.copy(),
                            "time_ms": (t1 - t0) * 1000.0,
                            "frame_idx": len(frame_cache)
                        })
                        
                        max_val = max(1, len(frame_cache) - 1)
                        cv2.setTrackbarMax("Frame", "Timeline", max_val)
                        cv2.setTrackbarPos("Frame", "Timeline", max_val)
                        pos = max_val
                        
                    except StopIteration:
                        bag_finished = True
                        print("\nКонец файла. Теперь можно свободно листать ползунок туда-сюда.")
                
                if len(frame_cache) > 0:
                    pos = min(pos, len(frame_cache) - 1)
                    cached = frame_cache[pos]
                    
                    if len(cached["xyz_out"]) > 0:
                        pcd_outlier.points = o3d.utility.Vector3dVector(cached["xyz_out"])
                        pcd_outlier.paint_uniform_color([0.5, 0.5, 0.5])
                    else:
                        pcd_outlier.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
                        
                    if len(cached["xyz_obs"]) > 0:
                        pcd_inlier.points = o3d.utility.Vector3dVector(cached["xyz_obs"])
                        pcd_inlier.paint_uniform_color([1.0, 0.0, 0.0])
                    else:
                        pcd_inlier.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
                        
                    ls_boxes.points = o3d.utility.Vector3dVector(cached["box_pts"])
                    ls_boxes.lines = o3d.utility.Vector2iVector(cached["box_lines"])
                    colors = [[0, 1, 0] for _ in range(len(cached["box_lines"]))]
                    ls_boxes.colors = o3d.utility.Vector3dVector(colors)
                        
                    if not geometries_added:
                        vis.add_geometry(pcd_outlier)
                        vis.add_geometry(pcd_inlier)
                        vis.add_geometry(ls_boxes)
                        vis.add_geometry(lidar_sphere)
                        
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
                        
                    print(f"Frame {cached['frame_idx']:03d} | Время фильтрации (в габарите): {cached['time_ms']:.3f} ms       ", end='\r')
                    
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

if __name__ == '__main__':
    run_zero_allocation_vis("roundT_pressureGate_roundT")
