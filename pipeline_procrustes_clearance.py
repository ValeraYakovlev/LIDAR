import numpy as np
from rail_detection.tunnel_frame import fit_track_frame, to_track_coords

class DynamicClearancePipeline:
    def __init__(self, box_width=2.7, box_height=3.5, box_length=1.0, box_gap=0.45, points_per_box=1000):
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
        
        self.frame_count = 0
        self.fixed_boxes = None
        
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
            
        self.frame_count += 1
        
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
        if self.fixed_boxes is not None:
            boxes = [b.copy() for b in self.fixed_boxes]
        else:
            boxes = []
            current_d = d_start
            N_target = getattr(self, 'points_per_box', 1000)
            
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
            
            if self.frame_count >= 10:
                self.fixed_boxes = [b.copy() for b in boxes]
        
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
                
        # 5.5 Строим эталон тоннеля ДЛЯ ТЕКУЩЕГО КАДРА по первым 5 звеньям
        ref_kdtree_2d = None
        if len(boxes) > 0:
            limit_idx = min(5, len(boxes))
            ref_end_d = boxes[limit_idx - 1]["end"]
            
            ref_mask = (d > 0.0) & (d <= ref_end_d)
            if np.any(ref_mask):
                pts_2d = np.column_stack((u[ref_mask], v[ref_mask]))
                from scipy.spatial import cKDTree
                ref_kdtree_2d = cKDTree(pts_2d)
            
        # --- 6. Эвристика оптимизации изгиба (Procrustes Flex) ---
        valid_idx = np.where(mask_longitudinal)[0]
        d_val = d[valid_idx]
        u_val = u[valid_idx]
        v_val = v[valid_idx]
        d_mid_val = d_mid_full[valid_idx]
        weight_val = weight_full[valid_idx]
        
        v_min_base = rail_top_v + 0.10
        v_max_base = v_min_base + self.box_height
        u_min_base = -self.box_width / 2.0
        u_max_base =  self.box_width / 2.0
        
        cand_mask = (u_val >= u_min_base - 1.5) & (u_val <= u_max_base + 1.5) & \
                    (v_val >= v_min_base - 0.5) & (v_val <= v_max_base + 0.5)
        d_cand = d_val[cand_mask]
        u_cand = u_val[cand_mask]
        v_cand = v_val[cand_mask]
        d_mid_cand = d_mid_val[cand_mask]
        weight_cand = weight_val[cand_mask]
        
        # Оптимизация оси должна избегать стен тоннеля, а не препятствий внутри!
        # Фильтруем точки: оставляем только те, что принадлежат "фону" тоннеля
        if ref_kdtree_2d is not None:
            pts_2d = np.column_stack((u_cand, v_cand))
            distances, _ = ref_kdtree_2d.query(pts_2d)
            is_tunnel = distances < 0.15
            
            d_cand = d_cand[is_tunnel]
            u_cand = u_cand[is_tunnel]
            v_cand = v_cand[is_tunnel]
            d_mid_cand = d_mid_cand[is_tunnel]
            weight_cand = weight_cand[is_tunnel]

        
        # Динамические веса для штрафа изгиба
        far_pts_count = np.sum(d_cand > 20.0)
        density_factor = np.clip(far_pts_count / 1000.0, 0.05, 1.0)
        
        inertia_weight = 200.0 * density_factor
        center_weight = 500.0 * density_factor
        
        # --- Coarse Grid Search + L-BFGS-B ---
        best_cost = np.inf
        best_dx_far = 0.0
        best_dz_far = 0.0
        
        def cost_func(params):
            dx_f, dz_f = params
            # Вычисляем сдвиг, привязанный к центрам звеньев (d_mid_cand вместо d_cand)
            # Это дает идеальное соответствие прямым граням визуальных прямоугольников!
            dx_arr = dx_f * (d_mid_cand / 40.0)**2
            dz_arr = dz_f * (d_mid_cand / 40.0)**2
            
            u_min_pad = u_min_base - 0.10
            u_max_pad = u_max_base + 0.10
            v_min_pad = v_min_base - 0.10
            v_max_pad = v_max_base + 0.10
            
            dist_u = np.minimum(u_cand - (u_min_pad + dx_arr), (u_max_pad + dx_arr) - u_cand)
            dist_v = np.minimum(v_cand - (v_min_pad + dz_arr), (v_max_pad + dz_arr) - v_cand)
            
            inside_mask = (dist_u > 0) & (dist_v > 0)
            if np.any(inside_mask):
                penetration = np.minimum(dist_u[inside_mask], dist_v[inside_mask])
                w = weight_cand[inside_mask]
                
                # Штраф высчитывается отдельно (нормируется) по каждому звену!
                # Теперь звено со 100 точками дает такой же штраф, как звено с 10 000 точек.
                cost_points = np.sum(penetration * w) * 100000.0 + np.sum(w) * 10000.0
            else:
                cost_points = 0.0
                
            return cost_points + \
                   (dx_f**2 + dz_f**2) * center_weight + \
                   ((dx_f - self.prev_dx)**2 + (dz_f - self.prev_dz)**2) * inertia_weight

        for dx_far in np.arange(-1.2, 1.21, 0.3):
            for dz_far in np.arange(-0.5, 0.51, 0.2):
                cost = cost_func([dx_far, dz_far])
                if cost < best_cost:
                    best_cost = cost
                    best_dx_far = dx_far
                    best_dz_far = dz_far
                    
        from scipy.optimize import minimize
        res = minimize(cost_func, [best_dx_far, best_dz_far], method='L-BFGS-B', bounds=[(-1.2, 1.2), (-0.5, 0.5)])
        best_dx_far, best_dz_far = res.x
        
        final_dx_far = self.alpha * best_dx_far + (1.0 - self.alpha) * self.prev_dx
        final_dz_far = self.alpha * best_dz_far + (1.0 - self.alpha) * self.prev_dz
        self.prev_dx = final_dx_far
        self.prev_dz = final_dz_far
        
        # Проверяем реальные точки препятствий (также с квантованием сдвига d_mid_full)
        u_shifted = u - final_dx_far * (d_mid_full / 40.0)**2
        v_shifted = v - final_dz_far * (d_mid_full / 40.0)**2
        
        mask_lateral = (u_shifted >= u_min_base) & (u_shifted <= u_max_base)
        mask_vertical = (v_shifted >= v_min_base) & (v_shifted <= v_max_base)
        # ---------------------------------------------------------
        
        # Финальная маска
        mask_inside_boxes = mask_lateral & mask_vertical & mask_longitudinal
        obstacle_points = points[mask_inside_boxes]
        
        obs_u = u_shifted[mask_inside_boxes]
        obs_v = v_shifted[mask_inside_boxes]
        obstacle_d = d[mask_inside_boxes]
        
        # 7. Вычитание фона тоннеля (Background Subtraction) для препятствий
        if ref_kdtree_2d is not None and len(obstacle_points) > 0:
            obs_pts_2d = np.column_stack((u[mask_inside_boxes], v[mask_inside_boxes]))
            dist_to_wall, _ = ref_kdtree_2d.query(obs_pts_2d)
            # Если точка дальше 15 см от известной стены - это реальное препятствие (а не стена, попавшая в габарит)
            true_obstacle_mask = dist_to_wall > 0.15
            
            obstacle_points = obstacle_points[true_obstacle_mask]
            obstacle_d = obstacle_d[true_obstacle_mask]
            obs_u = obs_u[true_obstacle_mask]
            obs_v = obs_v[true_obstacle_mask]
            
        obstacle_colors = None
        if len(obstacle_points) > 0:
            abs_u = np.abs(obs_u)
            
            core_u_max = 1.05
            core_v_max = v_min_base + 3.0
            
            mid_u_max = 1.2
            mid_v_max = v_min_base + 3.25
            
            colors = np.zeros((len(abs_u), 3))
            colors[:, 1] = 1.0 # Базово все зеленые
            
            mask_yellow = (abs_u <= mid_u_max) & (obs_v <= mid_v_max)
            colors[mask_yellow] = [1.0, 1.0, 0.0]
            
            mask_red = (abs_u <= core_u_max) & (obs_v <= core_v_max)
            colors[mask_red] = [1.0, 0.0, 0.0]
            
            obstacle_colors = colors
        
        # Обновляем полиномы эластичных осей для визуализации
        new_axis = frame["elastic_axis"].coeffs.copy()
        if len(new_axis) < 3:
            new_axis = np.pad(new_axis, (3 - len(new_axis), 0), 'constant')
        new_axis[-3] += final_dx_far / (40.0**2)
        frame["elastic_axis"] = np.poly1d(new_axis)
        
        new_floor = frame["elastic_floor"].coeffs.copy()
        if len(new_floor) < 3:
            new_floor = np.pad(new_floor, (3 - len(new_floor), 0), 'constant')
        new_floor[-3] += final_dz_far / (40.0**2)
        frame["elastic_floor"] = np.poly1d(new_floor)
        
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
                "boundaries": boxes,
                "width": float(self.box_width),
                "height": float(self.box_height),
                "gap": float(self.box_gap),
                "rail_top_v": float(rail_top_v)
            },
            "frame_idx": self.frame_count
        }
        
        return obstacle_points, min_distance, ros_payload
