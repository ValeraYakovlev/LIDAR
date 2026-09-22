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
        
        :param lidar_origin: (3,) координаты источника лидара (обычно 0, 0, 0)
        :param normal: (3,) вектор нормали путевой решетки (от RANSAC)
        :param tangent: (3,) вектор направления (тангенс пути)
        :return: rotation_matrix, box_origin
        """
        Z = tangent / np.linalg.norm(tangent)
        X = np.cross(Z, normal)
        X = X / np.linalg.norm(X)
        Y = np.cross(X, Z)
        Y = Y / np.linalg.norm(Y)
        
        self.rotation_matrix = np.column_stack((X, Y, Z))
        self.box_origin = lidar_origin + self.d_start * Z
        
        return self.rotation_matrix, self.box_origin

    def filter_points_in_rigid_box(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        За O(N) определяет точки, попавшие внутрь жесткого вагона.
        """
        pts_local = (point_cloud - self.box_origin) @ self.rotation_matrix
        
        mask = (
            (pts_local[:, 0] >= -self.W / 2.0) & (pts_local[:, 0] <= self.W / 2.0) &
            (pts_local[:, 1] >= 0.0) & (pts_local[:, 1] <= self.H) &
            (pts_local[:, 2] >= 0.0) & (pts_local[:, 2] <= self.L)
        )
        return mask

    def get_box_wireframe(self) -> o3d.geometry.LineSet:
        """
        Генерирует LineSet для отрисовки 3D-габарита вагона в Open3D.
        """
        w2 = self.W / 2.0
        h = self.H
        l = self.L
        
        corners_local = np.array([
            [-w2, 0, 0], [w2, 0, 0], [w2, h, 0], [-w2, h, 0], 
            [-w2, 0, l], [w2, 0, l], [w2, h, l], [-w2, h, l]  
        ])
        
        corners_global = corners_local @ self.rotation_matrix.T + self.box_origin
        
        lines = [
            [0,1], [1,2], [2,3], [3,0], 
            [4,5], [5,6], [6,7], [7,4], 
            [0,4], [1,5], [2,6], [3,7]  
        ]
        
        colors = [[0, 1, 0] for _ in range(len(lines))] # Зеленый цвет габарита
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_global)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set

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
    # =====================================================================
    clearance = RigidKinematicClearance(width=2.1, height=3.0, length=100.0, d_start=0.0)
    
    # Жёсткий базис (лидар выровнен с путём: Z вверх, движение вперёд по -Y)
    lidar_orig = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])   # Z вверх
    tangent = np.array([0.0, -1.0, 0.0])  # Движение вперёд по -Y
    clearance.build_local_frame(lidar_orig, normal, tangent)
    
    # =====================================================================
    # Выравнивание габарита по проекции лидара из pipeline_procrustes_clearance.py
    # Рассчитано по всему датасету (calc_means.py):
    #   - Лидар проецируется на заднюю грань на 0.045м ПРАВЕЕ центра бокса
    #   - Лидар проецируется на 1.109м ВЫШЕ нижней грани бокса
    # Следовательно, центр нижней грани бокса находится:
    #   x = -0.045 (центр бокса левее лидара)
    #   y =  0.0   (задняя грань совпадает с плоскостью лидара)
    #   z = -1.109  (дно бокса ниже лидара)
    # =====================================================================
    clearance.box_origin = np.array([-0.045, 0.0, -1.109])
    
    ls_boxes = clearance.get_box_wireframe()
    
    print("\nЗапуск zero-allocation визуализации Open3D (Timeline UI)...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Clearance Trench Interactive", width=1280, height=720, visible=True)
    
    pcd_inlier = o3d.geometry.PointCloud()
    pcd_outlier = o3d.geometry.PointCloud()
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
                
                # --- ЧТЕНИЕ КАДРА ИЗ BAG-ФАЙЛА ---
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
                        
                        # Децимируем облако в 5 раз для плавного рендера 
                        current_pc = xyz_all[::5] 
                        
                        t0 = time.perf_counter()
                        mask = clearance.filter_points_in_rigid_box(current_pc)
                        t1 = time.perf_counter()
                        
                        xyz_obs = current_pc[mask]
                        xyz_out = current_pc[~mask]
                        
                        frame_cache.append({
                            "xyz_out": xyz_out.copy(),
                            "xyz_obs": xyz_obs.copy(),
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
                
                # --- ОБНОВЛЕНИЕ РЕНДЕРА ИЗ КЕША ПО ПОЛЗУНКУ ---
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
    # Введите нужную папку из D:\Датасет здесь
    run_zero_allocation_vis("roundT_pressureGate_roundT")
