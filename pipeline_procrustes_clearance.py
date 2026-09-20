import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from rail_detection.tunnel_frame import fit_tunnel_geometry, tunnel_center_coeffs, to_track_coords
from rail_detection.curvature import ransac_poly_fit

class DynamicClearancePipeline:
    def __init__(self, box_width=2.7, box_height=3.5, box_length=1.0, box_gap=0.45, points_per_box=1000):
        self.box_width = box_width
        self.box_height = box_height
        self.box_length = box_length
        self.box_gap = box_gap
        self.points_per_box = points_per_box
        
        self.alpha = 0.3
        self.prev_axis_coeffs = None
        self.prev_floor_coeffs = None
        self.prev_rail_top = None
        
        # Stateful caching for high FPS
        self.frame_count = 0
        self.fixed_boxes = [] # Static boxes after 10 frames
        self.last_T_inv = {} # T_inv from previous frame
        
        self.ref_kdtree_2d = None
        self.pcd_ref_3d = None
        self.ref_accumulated_pts = []

    def process_pointcloud(self, points):
        self.frame_count += 1
        
        geom = fit_tunnel_geometry(points)
        if geom is None:
            return np.array([]), np.inf, {"status": "no_rails_detected"}
        
        frame = geom["frame"]
        
        coeffs_x = tunnel_center_coeffs(geom)
        elastic_axis = np.poly1d(coeffs_x)
        elastic_deriv = elastic_axis.deriv()
        
        rail_records = frame.get("rail_records", [])
        if len(rail_records) >= 3:
            d_vals = np.array([(r["depth_lo"] + r["depth_hi"]) / 2.0 for r in rail_records])
            z_vals = np.array([r["shoulder_z"] for r in rail_records])
            
            d_vals, unique_idx = np.unique(d_vals, return_index=True)
            z_vals = z_vals[unique_idx]
            
            if len(d_vals) >= 4:
                fit_z = ransac_poly_fit(d_vals, z_vals, 2, 0.15)
                elastic_floor = np.poly1d(fit_z["coeffs"]) if fit_z is not None else np.poly1d(frame["floor_coeffs"])
            else:
                elastic_floor = np.poly1d(frame["floor_coeffs"])
        else:
            elastic_floor = np.poly1d(frame["floor_coeffs"])
            
        curr_axis = elastic_axis.coeffs
        curr_floor = elastic_floor.coeffs
        
        if self.prev_axis_coeffs is not None:
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
        
        x = points['x'].astype(float)
        y = points['y'].astype(float)
        z = points['z'].astype(float)
        d = -y
        
        xc = frame["elastic_axis"](d)
        heading = np.arctan(frame["elastic_deriv"](d))
        u = (x - xc) * np.cos(heading)
        v = z - frame["elastic_floor"](d)
        
        gauge = frame.get("gauge", 1.52)
        rail_mask = (np.abs(np.abs(u) - gauge / 2.0) <= 0.15) & (v >= -0.1) & (v <= 0.3) & (d > 0)
        
        rail_d_start = 0.0
        if np.any(rail_mask):
            curr_rail_top = np.percentile(v[rail_mask], 99)
            rail_d_start = np.min(d[rail_mask])
            rail_top_v = self.alpha * curr_rail_top + (1.0 - self.alpha) * self.prev_rail_top if self.prev_rail_top is not None else curr_rail_top
            self.prev_rail_top = rail_top_v
        else:
            rail_top_v = self.prev_rail_top if self.prev_rail_top is not None else 0.0
            
        # 4. Filter forward points
        valid_mask = d > 0
        valid_d = d[valid_mask]
        
        if len(valid_d) == 0:
            return np.array([]), np.inf, {"status": "no_forward_points"}
            
        d_start = rail_d_start if rail_d_start > 0 else np.min(valid_d)
        
        # 5. Fast Box Generation (Stateful + searchsorted)
        valid_d_sorted = np.sort(valid_d)
        max_d = valid_d_sorted[-1]
        
        boxes = []
        current_d = d_start
        N_target = self.points_per_box
        
        # Use cached boxes if available
        if self.frame_count > 10 and self.fixed_boxes:
            for b in self.fixed_boxes:
                if b["end"] > max_d:
                    break
                boxes.append(b.copy())
            if boxes:
                current_d = boxes[-1]["end"] + self.box_gap
                
        # Generate the rest without loop sorting
        if current_d < max_d:
            idx = np.searchsorted(valid_d_sorted, current_d)
            while idx < len(valid_d_sorted):
                target_idx = min(idx + N_target - 1, len(valid_d_sorted) - 1)
                d_end_target = valid_d_sorted[target_idx]
                d_end = max(d_end_target, current_d + self.box_length)
                
                next_idx = np.searchsorted(valid_d_sorted, d_end, side='right')
                pts_count = next_idx - idx
                
                boxes.append({
                    "start": float(current_d), 
                    "end": float(d_end), 
                    "length": float(d_end - current_d), 
                    "pts_count": int(pts_count),
                    "is_link": False
                })
                current_d = d_end + self.box_gap
                idx = np.searchsorted(valid_d_sorted, current_d)
                if len(boxes) >= 150: 
                    break

        if self.frame_count == 10:
            # Cache boxes up to 30 meters
            self.fixed_boxes = [b for b in boxes if b["end"] < 30.0]
            
        # 6. Optimize ICP (Keyframes + Warm Start + Interpolation)
        # Build Reference 2D KDTree (Accumulate over first 5 frames)
        if self.ref_kdtree_2d is None:
            ref_mask = (d >= 0.0) & (d <= 3.0)
            if np.sum(ref_mask) > 100:
                ref_2d = np.column_stack((u[ref_mask], v[ref_mask]))
                self.ref_accumulated_pts.append(ref_2d)
                
                if self.frame_count >= 5 or len(self.ref_accumulated_pts) >= 5:
                    all_ref = np.vstack(self.ref_accumulated_pts)
                    self.ref_kdtree_2d = cKDTree(all_ref)
                    
                    pts_ref_3d = np.column_stack((all_ref[:, 0], all_ref[:, 1], np.zeros(len(all_ref))))
                    self.pcd_ref_3d = o3d.geometry.PointCloud()
                    self.pcd_ref_3d.points = o3d.utility.Vector3dVector(pts_ref_3d)
                    self.pcd_ref_3d = self.pcd_ref_3d.voxel_down_sample(voxel_size=0.05)

        kf_step = 4
        kf_indices = list(range(0, len(boxes), kf_step))
        if len(boxes) - 1 not in kf_indices and len(boxes) > 0:
            kf_indices.append(len(boxes) - 1)
            
        kf_transforms = {}
        for idx in kf_indices:
            b = boxes[idx]
            b_mask = (d >= b["start"]) & (d <= b["end"])
            
            init_T = self.last_T_inv.get(idx, np.eye(4))
            
            if np.sum(b_mask) < 10 or self.pcd_ref_3d is None:
                kf_transforms[idx] = init_T
                continue
                
            pts_seg = np.column_stack((u[b_mask], v[b_mask], np.zeros(np.sum(b_mask))))
            pcd_seg = o3d.geometry.PointCloud()
            pcd_seg.points = o3d.utility.Vector3dVector(pts_seg)
            pcd_seg = pcd_seg.voxel_down_sample(voxel_size=0.05)
            
            reg = o3d.pipelines.registration.registration_icp(
                pcd_seg, self.pcd_ref_3d, 0.5, np.linalg.inv(init_T),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5)
            )
            T_inv = np.linalg.inv(reg.transformation)
            kf_transforms[idx] = T_inv
            self.last_T_inv[idx] = T_inv

        # Interpolate across non-keyframes
        for i in range(len(boxes)):
            if i in kf_transforms:
                boxes[i]["T_inv"] = kf_transforms[i]
            else:
                l_idx = max([k for k in kf_indices if k < i])
                r_idx = min([k for k in kf_indices if k > i])
                alpha = (i - l_idx) / (r_idx - l_idx)
                
                t_l = kf_transforms[l_idx][:3, 3]
                t_r = kf_transforms[r_idx][:3, 3]
                t_interp = t_l * (1 - alpha) + t_r * alpha
                
                try:
                    from scipy.spatial.transform import Slerp
                    key_rots = R.from_matrix([kf_transforms[l_idx][:3, :3], kf_transforms[r_idx][:3, :3]])
                    slerp = Slerp([0, 1], key_rots)
                    R_interp = slerp([alpha])[0].as_matrix()
                except:
                    R_interp = R.from_matrix(kf_transforms[l_idx][:3, :3]).as_matrix()
                
                T_inv = np.eye(4)
                T_inv[:3, :3] = R_interp
                T_inv[:3, 3] = t_interp
                boxes[i]["T_inv"] = T_inv
                self.last_T_inv[i] = T_inv
                
        # Gaussian smoothing
        if len(boxes) > 0:
            translations = np.array([b["T_inv"][:3, 3] for b in boxes])
            rotations = np.array([R.from_matrix(b["T_inv"][:3, :3]).as_euler('xyz', degrees=False) for b in boxes])
            
            translations_smooth = gaussian_filter1d(translations, sigma=1.0, axis=0)
            rotations_smooth = gaussian_filter1d(rotations, sigma=1.0, axis=0)
            
            for i, b in enumerate(boxes):
                T_smooth = np.eye(4)
                T_smooth[:3, :3] = R.from_euler('xyz', rotations_smooth[i]).as_matrix()
                T_smooth[:3, 3] = translations_smooth[i]
                b["T_inv"] = T_smooth

            # Физическое ограничение сцепки: длина граней строго <= box_gap
            u_min_c, u_max_c = -self.box_width / 2.0, self.box_width / 2.0
            v_min_c, v_max_c = rail_top_v + 0.10, rail_top_v + 0.10 + self.box_height
            corners_2d = np.array([
                [u_min_c, v_min_c, 0, 1], [u_max_c, v_min_c, 0, 1],
                [u_max_c, v_max_c, 0, 1], [u_min_c, v_max_c, 0, 1]
            ])
            for i in range(1, len(boxes)):
                b_prev = boxes[i-1]
                b_curr = boxes[i]
                
                c_prev = (b_prev["T_inv"] @ corners_2d.T).T
                c_curr = (b_curr["T_inv"] @ corners_2d.T).T
                
                # Максимальное боковое смещение в 2D между углами соседних вагонов
                max_lat_sq = np.max((c_curr[:, 0] - c_prev[:, 0])**2 + (c_curr[:, 1] - c_prev[:, 1])**2)
                
                # Фильтр аномалий (предотвращает "сплющивание" в двумерную грань на концах)
                # Если смещение больше длины сцепки - это физически невозможно (ошибка ICP на разреженных данных)
                if max_lat_sq >= self.box_gap**2:
                    b_curr["T_inv"] = b_prev["T_inv"].copy()
                    required_gap = self.box_gap
                else:
                    required_gap = np.sqrt(self.box_gap**2 - max_lat_sq)
                    
                b_curr["start"] = b_prev["end"] + required_gap
                b_curr["end"] = b_curr["start"] + b_curr["length"]

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

        # Vectorized alignment
        u_aligned = np.copy(u)
        v_aligned = np.copy(v)
        
        # Batch points transformation
        for b in all_segments:
            s_mask = (d >= b["start"]) & (d <= b["end"])
            if np.sum(s_mask) == 0: continue
            
            T_forward = np.linalg.inv(b["T_inv"])
            pts_seg_raw = np.column_stack((u[s_mask], v[s_mask], np.zeros(np.sum(s_mask))))
            pts_hom = np.column_stack((pts_seg_raw, np.ones(len(pts_seg_raw))))
            pts_aligned = (T_forward @ pts_hom.T).T
            
            u_aligned[s_mask] = pts_aligned[:, 0]
            v_aligned[s_mask] = pts_aligned[:, 1]
            
        # 7. Fast 2D Background Subtraction
        v_min_base = rail_top_v + 0.10
        v_max_base = v_min_base + self.box_height
        u_min_base = -self.box_width / 2.0
        u_max_base =  self.box_width / 2.0
        
        mask_inside = (u_aligned >= u_min_base) & (u_aligned <= u_max_base) & (v_aligned >= v_min_base) & (v_aligned <= v_max_base)
        mask_inside &= (d > d_start)
        
        mask_bg_sub = np.zeros_like(mask_inside, dtype=bool)
        
        if self.ref_kdtree_2d is not None and np.any(mask_inside):
            inside_idx = np.where(mask_inside)[0]
            pts_query_2d = np.column_stack((u_aligned[inside_idx], v_aligned[inside_idx]))
            
            distances, _ = self.ref_kdtree_2d.query(pts_query_2d, workers=-1)
            is_obstacle = distances > 0.15
            mask_bg_sub[inside_idx[is_obstacle]] = True
            
        obstacle_points = points[mask_bg_sub]
        obstacle_d = d[mask_bg_sub]
        
        obstacle_colors = None
        if len(obstacle_points) > 0:
            obs_u = u_aligned[mask_bg_sub]
            obs_v = v_aligned[mask_bg_sub]
            abs_u = np.abs(obs_u)
            
            # Размеры зон (половина ширины и полная высота от рельса)
            # Внешняя невидимая зона (Green): до 2.7m ширина, до 3.5m высота
            # Промежуточная зона (Yellow): до 2.4m ширина, до 3.25m высота
            # Зона физического центра/поезда (Red): до 2.1m ширина, до 3.0m высота
            
            core_u_max = 2.1 / 2.0
            core_v_max = rail_top_v + 0.10 + 3.0
            
            mid_u_max = 2.4 / 2.0
            mid_v_max = rail_top_v + 0.10 + 3.25
            
            # По умолчанию все точки зеленые (попали в невидимую внешнюю зону)
            colors = np.zeros((len(abs_u), 3))
            colors[:, 1] = 1.0 # (0, 1, 0)
            
            # Точки, которые уже глубоко в границах (желтые)
            mask_yellow = (abs_u <= mid_u_max) & (obs_v <= mid_v_max)
            colors[mask_yellow] = [1.0, 1.0, 0.0]
            
            # Точки прямо в центре / внутри корпуса поезда (красные)
            mask_red = (abs_u <= core_u_max) & (obs_v <= core_v_max)
            colors[mask_red] = [1.0, 0.0, 0.0]
            
            obstacle_colors = colors
        
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
