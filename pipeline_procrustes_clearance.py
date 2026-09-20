import numpy as np
from rail_detection.tunnel_frame import fit_track_frame, to_track_coords

class DynamicClearancePipeline:
    def __init__(self, box_width=2.7, box_height=3.5, box_length=1.0, box_gap=0.45, points_per_box=1000):
        """
        Инициализация параметров габарита.
        :param box_width: Ширина габарита (м).
        :param box_height: Высота габарита (м).
        :param box_length: Минимальная длина единичного сегмента габарита (м).
        :param box_gap: Зазор между сегментами габарита (м).
        :param points_per_box: Целевое количество точек в одном звене для динамического удлинения.
        """
        self.box_width = box_width
        self.box_height = box_height
        self.box_length = box_length
        self.box_gap = box_gap
        self.points_per_box = points_per_box
        
        # Переменные для временного сглаживания (EMA) и оптимизации
        self.alpha = 0.3  # Коэффициент сглаживания
        self.prev_axis_coeffs = None
        self.prev_floor_coeffs = None
        self.prev_rail_top = None
        self.prev_dx = 0.0
        self.prev_dz = 0.0
        
    def process_pointcloud(self, points):
        """
        Обработка облака точек для нахождения препятствий в динамическом габарите.
        :param points: структурированный массив numpy (с полями x, y, z, ...).
        :return: (obstacle_points, min_distance, ros_payload)
                 obstacle_points - массив точек, попавших в габарит и классифицированных как препятствие
                 min_distance - расстояние до ближайшего препятствия (м)
                 ros_payload - задел под ROS (dict с результатами)
        """
        from rail_detection.tunnel_frame import fit_tunnel_geometry, tunnel_center_coeffs, to_track_coords
        from rail_detection.curvature import ransac_poly_fit
        
        # 1. Используем глобальную робастную геометрию тоннеля (с опорой на стены!)
        # Она гораздо стабильнее, чем голые рельсы на горизонте.
        geom = fit_tunnel_geometry(points)
        if geom is None:
            return np.array([]), np.inf, {"status": "no_rails_detected"}
        
        frame = geom["frame"]
        
        # Латеральная ось (x) - берем ИДЕАЛЬНУЮ робастную дугу (которую не "колбасит"),
        # рассчитанную модулем rail_detection по рельсам вблизи и стенам вдали.
        coeffs_x = tunnel_center_coeffs(geom)
        elastic_axis = np.poly1d(coeffs_x)
        elastic_deriv = elastic_axis.deriv()
        
        # Вертикальная ось (z) - используем RANSAC, чтобы отбросить выбросы детектора,
        # из-за которых профиль пола дрыгался. Строим параболу (спуск/подъем).
        rail_records = frame.get("rail_records", [])
        if len(rail_records) >= 3:
            d_vals = np.array([(r["depth_lo"] + r["depth_hi"]) / 2.0 for r in rail_records])
            z_vals = np.array([r["shoulder_z"] for r in rail_records])
            
            # Убираем дубликаты
            d_vals, unique_idx = np.unique(d_vals, return_index=True)
            z_vals = z_vals[unique_idx]
            
            if len(d_vals) >= 4: # Для RANSAC параболы нужно хотя бы 4 точки
                # ransac_poly_fit(depths, xs, degree, threshold)
                fit_z = ransac_poly_fit(d_vals, z_vals, 2, 0.15)
                if fit_z is not None:
                    elastic_floor = np.poly1d(fit_z["coeffs"])
                else:
                    elastic_floor = np.poly1d(frame["floor_coeffs"])
            else:
                elastic_floor = np.poly1d(frame["floor_coeffs"])
        else:
            elastic_floor = np.poly1d(frame["floor_coeffs"])
            
        frame["elastic_axis"] = elastic_axis
        frame["elastic_deriv"] = elastic_deriv
        frame["elastic_floor"] = elastic_floor
        
        # --- 1.1 Временное сглаживание базовой геометрии (Temporal Smoothing) ---
        curr_axis = frame["elastic_axis"].coeffs
        curr_floor = frame["elastic_floor"].coeffs
        
        if self.prev_axis_coeffs is not None:
            # Используем np.polyadd для корректного сложения массивов разной длины (на случай изменения степени полинома)
            smooth_axis = np.polyadd(self.alpha * curr_axis, (1.0 - self.alpha) * self.prev_axis_coeffs)
            smooth_floor = np.polyadd(self.alpha * curr_floor, (1.0 - self.alpha) * self.prev_floor_coeffs)
        else:
            smooth_axis = curr_axis
            smooth_floor = curr_floor
            
        self.prev_axis_coeffs = smooth_axis.copy()
        self.prev_floor_coeffs = smooth_floor.copy()
        
        frame["elastic_axis"] = np.poly1d(smooth_axis)
        frame["elastic_deriv"] = frame["elastic_axis"].deriv()
        frame["elastic_floor"] = np.poly1d(smooth_floor)
        # ------------------------------------------------------------------------

        # 2. Выделяем координаты
        x = points['x'].astype(float)
        y = points['y'].astype(float)
        z = points['z'].astype(float)
        
        d = -y
        
        if "elastic_axis" in frame and frame["elastic_axis"] is not None:
            xc = frame["elastic_axis"](d)
            heading = np.arctan(frame["elastic_deriv"](d))
            u = (x - xc) * np.cos(heading)
            v = z - frame["elastic_floor"](d)
        else:
            d, u, v = to_track_coords(x, y, z, frame)
            
        # 3. Вычисление вертикальной привязки к рельсам со сглаживанием
        gauge = frame.get("gauge", 1.52)
        rail_mask = (np.abs(np.abs(u) - gauge / 2.0) <= 0.15) & (v >= -0.1) & (v <= 0.3) & (d > 0)
        
        rail_d_start = 0.0
        if np.any(rail_mask):
            curr_rail_top = np.percentile(v[rail_mask], 99)
            rail_d_start = np.min(d[rail_mask])
            if self.prev_rail_top is not None:
                rail_top_v = self.alpha * curr_rail_top + (1.0 - self.alpha) * self.prev_rail_top
            else:
                rail_top_v = curr_rail_top
            self.prev_rail_top = rail_top_v
        else:
            rail_top_v = self.prev_rail_top if self.prev_rail_top is not None else 0.0
            
        # 4. Отбор передних точек (d > 0)
        valid_d = d[d > 0]
        if len(valid_d) == 0:
            return np.array([]), np.inf, {"status": "no_forward_points"}
            
        # Начинаем строить габарит там, где фактически начинаются рельсы (или от 0, если не видно)
        d_start = rail_d_start if rail_d_start > 0 else np.min(valid_d)
        
        # 5. Динамическая логика звеньев (зависимость от плотности точек)
        boxes = []
        current_d = d_start
        N_target = getattr(self, 'points_per_box', 1000)  # Регулируется здесь (по умолчанию 1000)
        
        while current_d < np.max(valid_d):
            pts_ahead = valid_d[valid_d >= current_d]
            if len(pts_ahead) == 0:
                break
                
            pts_ahead_sorted = np.sort(pts_ahead)
            if len(pts_ahead_sorted) >= N_target:
                d_end_target = pts_ahead_sorted[N_target - 1]
            else:
                d_end_target = pts_ahead_sorted[-1]
                
            d_end = max(d_end_target, current_d + self.box_length)
            pts_count = np.sum((valid_d >= current_d) & (valid_d <= d_end))
            
            b = {
                "start": float(current_d), 
                "end": float(d_end), 
                "length": float(d_end - current_d), 
                "pts_count": int(pts_count)
            }
            boxes.append(b)
            current_d = d_end + self.box_gap
            if len(boxes) >= 150: 
                break
        
        # Отладка в консоль: показываем длину каждого звена и количество точек в нем
        debug_str = " | ".join([f"L={b['length']:.1f}m ({b['pts_count']} pts)" for b in boxes])
        print(f"[DEBUG SEGMENTS] {len(boxes)} звеньев: {debug_str}")
        
        mask_longitudinal = np.zeros_like(d, dtype=bool)
        d_mid_full = np.copy(d)
        weight_full = np.ones(len(d), dtype=float)
        
        for b in boxes:
            mask = (d >= b["start"]) & (d <= b["end"])
            mask_longitudinal |= mask
            d_mid_full[mask] = (b["start"] + b["end"]) / 2.0
            if b["pts_count"] > 0:
                weight_full[mask] = 1.0 / b["pts_count"]
            
        # --- 6. Выравнивание ICP и Вычитание эталонного профиля ---
        import open3d as o3d
        
        valid_idx = np.where(mask_longitudinal)[0]
        d_val = d[valid_idx]
        u_val = u[valid_idx]
        v_val = v[valid_idx]
        
        u_aligned = np.copy(u_val)
        v_aligned = np.copy(v_val)
        
        # Определяем эталон (первые метры тоннеля, например первые 3 сегмента)
        if len(boxes) >= 3:
            ref_end_d = boxes[2]["end"]
        elif len(boxes) > 0:
            ref_end_d = boxes[-1]["end"]
        else:
            ref_end_d = 0.0
            
        ref_mask = (d_val >= 0.0) & (d_val <= ref_end_d)
        
        pcd_ref = None
        if np.sum(ref_mask) > 100:
            pts_ref = np.column_stack((u_val[ref_mask], v_val[ref_mask], np.zeros(np.sum(ref_mask))))
            pcd_ref = o3d.geometry.PointCloud()
            pcd_ref.points = o3d.utility.Vector3dVector(pts_ref)
            pcd_ref = pcd_ref.voxel_down_sample(voxel_size=0.05)
        
        # Регистрация каждого звена
        from scipy.spatial.transform import Rotation as R
        from scipy.ndimage import gaussian_filter1d
        
        transforms = []
        for b in boxes:
            b_mask = (d_val >= b["start"]) & (d_val <= b["end"])
            
            if np.sum(b_mask) < 10 or pcd_ref is None:
                b["T_inv"] = np.eye(4)
                transforms.append(np.eye(4))
                continue
                
            if b["end"] <= ref_end_d:
                b["T_inv"] = np.eye(4)
                transforms.append(np.eye(4))
                continue
                
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
            T_inv = np.linalg.inv(T)
            b["T_inv_raw"] = T_inv
            transforms.append(T_inv)
            
        # Сглаживание трансформаций
        if len(transforms) > 0:
            translations = np.array([T[:3, 3] for T in transforms])
            rotations = [R.from_matrix(T[:3, :3]).as_euler('xyz', degrees=False) for T in transforms]
            rotations = np.array(rotations)
            
            # Применяем фильтр Гаусса (sigma=1.0)
            translations_smooth = gaussian_filter1d(translations, sigma=1.0, axis=0)
            rotations_smooth = gaussian_filter1d(rotations, sigma=1.0, axis=0)
            
            for i, b in enumerate(boxes):
                T_smooth = np.eye(4)
                T_smooth[:3, :3] = R.from_euler('xyz', rotations_smooth[i]).as_matrix()
                T_smooth[:3, 3] = translations_smooth[i]
                b["T_inv"] = T_smooth

        # Добавляем звенья-сцепки (links) для зоны препятствий в промежутках
        all_segments = []
        for i, b in enumerate(boxes):
            all_segments.append(b)
            if i < len(boxes) - 1:
                # Сцепка между текущим и следующим звеном
                link = {
                    "start": b["end"],
                    "end": boxes[i+1]["start"],
                    "is_link": True,
                }
                # Интерполируем трансформацию для сцепки как среднее
                T_link = np.eye(4)
                rot1 = R.from_matrix(b["T_inv"][:3, :3])
                rot2 = R.from_matrix(boxes[i+1]["T_inv"][:3, :3])
                T_link[:3, :3] = R.from_euler('xyz', (rot1.as_euler('xyz') + rot2.as_euler('xyz')) / 2).as_matrix()
                T_link[:3, 3] = (b["T_inv"][:3, 3] + boxes[i+1]["T_inv"][:3, 3]) / 2.0
                link["T_inv"] = T_link
                all_segments.append(link)

        # Расширяем mask_longitudinal, чтобы включить точки в сцепках
        for seg in all_segments:
            mask = (d >= seg["start"]) & (d <= seg["end"])
            mask_longitudinal |= mask

        # Обновляем valid_idx и массивы
        valid_idx = np.where(mask_longitudinal)[0]
        d_val = d[valid_idx]
        u_val = u[valid_idx]
        v_val = v[valid_idx]
        
        u_aligned = np.copy(u_val)
        v_aligned = np.copy(v_val)

        # Выравниваем точки всего облака по сглаженным сегментам и сцепкам
        for seg in all_segments:
            s_mask = (d_val >= seg["start"]) & (d_val <= seg["end"])
            if np.sum(s_mask) == 0: continue
            
            T_inv = seg["T_inv"]
            pts_seg_raw = np.column_stack((u_val[s_mask], v_val[s_mask], np.zeros(np.sum(s_mask))))
            pts_hom = np.column_stack((pts_seg_raw, np.ones(len(pts_seg_raw))))
            
            # Применяем прямую трансформацию (T_forward), так как T_inv хранит обратную
            T_forward = np.linalg.inv(T_inv)
            pts_aligned = (T_forward @ pts_hom.T).T
            
            u_aligned[s_mask] = pts_aligned[:, 0]
            v_aligned[s_mask] = pts_aligned[:, 1]


        # --- 7. Вычитание фона и поиск препятствий внутри габарита ---
        v_min_base = rail_top_v + 0.10
        v_max_base = v_min_base + self.box_height
        u_min_base = -self.box_width / 2.0
        u_max_base =  self.box_width / 2.0
        
        mask_lateral = (u_aligned >= u_min_base) & (u_aligned <= u_max_base)
        mask_vertical = (v_aligned >= v_min_base) & (v_aligned <= v_max_base)
        mask_inside = mask_lateral & mask_vertical
        
        mask_bg_sub = np.zeros_like(mask_inside, dtype=bool)
        
        if pcd_ref is not None and np.any(mask_inside):
            inside_idx = np.where(mask_inside)[0]
            pts_query = np.column_stack((u_aligned[inside_idx], v_aligned[inside_idx], np.zeros(len(inside_idx))))
            
            pcd_query = o3d.geometry.PointCloud()
            pcd_query.points = o3d.utility.Vector3dVector(pts_query)
            
            # Векторизованный поиск расстояний до эталона
            distances = np.asarray(pcd_query.compute_point_cloud_distance(pcd_ref))
            
            # Отклонение более 15 см означает, что это инородный объект (препятствие)
            is_obstacle = distances > 0.15
            mask_bg_sub[inside_idx[is_obstacle]] = True
            
        obstacle_points = points[valid_idx][mask_bg_sub]
        obstacle_d = d_val[mask_bg_sub]
        
        # Вычисляем градиент вероятности (цвета) на выровненных координатах
        obstacle_colors = None
        if len(obstacle_points) > 0:
            obs_u = u_aligned[mask_bg_sub]
            obs_v = v_aligned[mask_bg_sub]
            
            d_left = obs_u - u_min_base
            d_right = u_max_base - obs_u
            d_top = v_max_base - obs_v
            
            # Вероятность максимальна в центре с отступом 50см (кроме низа)
            danger = np.minimum.reduce([d_left / 0.5, d_right / 0.5, d_top / 0.5])
            danger = np.clip(danger, 0.0, 1.0)
            
            # Цвет: края (0.0) -> Зеленый, середина (0.5) -> Желтый, центр (1.0) -> Красный
            R = np.clip(2.0 * danger, 0.0, 1.0)
            G = np.clip(2.0 * (1.0 - danger), 0.0, 1.0)
            B = np.zeros_like(danger)
            obstacle_colors = np.vstack([R, G, B]).T
        
        # Отчет
        min_distance = np.min(obstacle_d) if len(obstacle_d) > 0 else np.inf
        
        ros_payload = {
            "status": "success",
            "obstacle_detected": len(obstacle_points) > 0,
            "min_distance_m": min_distance,
            "obstacle_points_count": len(obstacle_points),
            "frame_geometry": frame,
            "obstacle_colors": obstacle_colors,
            "clearance_boxes": {
                "boundaries": all_segments,
                "width": float(self.box_width),
                "height": float(self.box_height),
                "gap": float(self.box_gap),
                "rail_top_v": float(rail_top_v)
            }
        }
        
        return obstacle_points, min_distance, ros_payload
