import os
import numpy as np
from scipy.spatial import cKDTree
from xgboost import XGBClassifier
from collections import deque

from rail_detection.parallel_path import ParallelGauge
from rail_detection.far_detect import VARIANTS
from rail_detection.loader import iter_frames
from rail_detection import DEFAULT_BAGS, bag_path

class ObstaclePipeline:
    def __init__(self, model_path="xgb_obstacle.json"):
        self.model = XGBClassifier()
        if os.path.exists(model_path):
            self.model.load_model(model_path)
            self.has_model = True
        else:
            print(f"ВНИМАНИЕ: Модель {model_path} не найдена. Предсказания будут случайными.")
            self.has_model = False
            
        p = VARIANTS["final_b2"]
        self.u_min = -p["half"]
        self.u_max = p["half"]
        self.v_min = p["bottom"]
        self.v_max = p["top"]
        
        self.gauge = ParallelGauge()
        self.cumulative_ds = 0.0
        
        self.slice_history = deque()
        self.frame_idx = 0
        
    def process_frame(self, pts):
        res = self.gauge.update(pts)
        self.frame_idx += 1
        
        # Удаляем срезы старше 5 кадров
        while self.slice_history and self.slice_history[0]["frame_idx"] <= self.frame_idx - 5:
            self.slice_history.popleft()
            
        if res is None:
            return self._check_history()
            
        ds = res.get("ds", 0.0)
        if ds is not None and np.isfinite(ds):
            self.cumulative_ds += ds
            
        s = res["s"]
        u = res["u"]
        v = res["v"]
        
        # 1. Вычитание стен по эталону (первые 3 метра текущего кадра)
        ref_mask = (s >= 0.0) & (s <= 3.0)
        if np.sum(ref_mask) > 100:
            pts_2d_ref = np.column_stack((u[ref_mask], v[ref_mask]))
            ref_kdtree = cKDTree(pts_2d_ref)
            pts_all_2d = np.column_stack((u, v))
            dist_to_wall, _ = ref_kdtree.query(pts_all_2d)
            wall_mask = dist_to_wall > 0.15
        else:
            wall_mask = np.ones_like(s, dtype=bool)
            
        # 2. Выделение точек внутри габарита и без стен
        mask = (u >= self.u_min) & (u <= self.u_max) & (v >= self.v_min) & (v <= self.v_max) & (s > 0) & wall_mask
        s_val = s[mask]
        u_val = u[mask]
        v_val = v[mask]
        
        if len(s_val) == 0:
            return self._check_history()
            
        # 3. Габарит засек препятствие: разделяем на кластеры и отступаем всю длину каждого
        sort_idx = np.argsort(s_val)
        s_val = s_val[sort_idx]
        u_val = u_val[sort_idx]
        v_val = v_val[sort_idx]
        
        gaps = np.diff(s_val) > 1.0
        split_indices = np.where(gaps)[0] + 1
        
        current_slices = []
        vectors = []
        
        for c_s, c_u, c_v in zip(np.split(s_val, split_indices), np.split(u_val, split_indices), np.split(v_val, split_indices)):
            if len(c_s) == 0:
                continue
                
            c_s_min = c_s[0]
            c_s_max = c_s[-1]
            
            bins_s = np.arange(c_s_min, c_s_max + 0.03, 0.03)
            if len(bins_s) < 2:
                continue
                
            bin_indices = np.digitize(c_s, bins_s) - 1
            
            for i in range(len(bins_s) - 1):
                in_bin = (bin_indices == i)
                if not np.any(in_bin):
                    vectors.append(np.zeros(100))
                else:
                    u_bin = c_u[in_bin]
                    v_bin = c_v[in_bin]
                    H, _, _ = np.histogram2d(u_bin, v_bin, bins=[10, 10], 
                                             range=[[self.u_min, self.u_max], [self.v_min, self.v_max]])
                    vectors.append(H.flatten() / 100.0)
                current_slices.append(bins_s[i])
                
        if not vectors:
            return self._check_history()
            
        # 4. Прогоняем векторы через модель XGBoost
        if self.has_model:
            preds = self.model.predict(np.array(vectors))
        else:
            preds = np.zeros(len(vectors))
            
        # 5. Сохраняем результаты предсказаний в историю
        for s_dist, pred in zip(current_slices, preds):
            self.slice_history.append({
                "global_s": self.cumulative_ds + s_dist,
                "is_obst": bool(pred),
                "frame_idx": self.frame_idx
            })
            
        return self._check_history()
        
    def _check_history(self):
        if not self.slice_history:
            return None
            
        # Кластеризация истории по глобальному расстоянию (gap < 1.0 м)
        history = list(self.slice_history)
        history.sort(key=lambda x: x["global_s"])
        
        clusters = []
        current_cluster = [history[0]]
        
        for item in history[1:]:
            if item["global_s"] - current_cluster[-1]["global_s"] < 1.0:
                current_cluster.append(item)
            else:
                clusters.append(current_cluster)
                current_cluster = [item]
        clusters.append(current_cluster)
        
        detected_distances = []
        for cluster in clusters:
            # Требуем, чтобы препятствие подтверждалось на протяжении 5 кадров
            frame_idxs = set(item["frame_idx"] for item in cluster)
            
            total_slices = len(cluster)
            ones = sum(1 for item in cluster if item["is_obst"])
            
            # Процент единиц от всех срезов объекта за 5 кадров
            ratio = ones / max(1, total_slices)
            
            # Если процент > 0.95 и препятствие было видно хотя бы в 3-5 кадрах
            if ratio > 0.95 and len(frame_idxs) >= 3: 
                # Нашли реальное препятствие! Вычисляем расстояние от текущего положения поезда
                dist_to_train = min(item["global_s"] for item in cluster) - self.cumulative_ds
                if dist_to_train > 0:
                    detected_distances.append(dist_to_train)
                    
        if detected_distances:
            return min(detected_distances)
        return None

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/Volumes/T7/Dataset")
    p.add_argument("--bags", nargs="*", default=None)
    p.add_argument("--model_path", default="xgb_obstacle.json")
    a = p.parse_args()
    
    # Для запуска из корня репозитория путь к модели: ml/xgb_obstacle.json
    pipeline = ObstaclePipeline(a.model_path)
    
    for bag in a.bags or DEFAULT_BAGS:
        print(f"\n--- Запуск пайплайна на записи {bag} ---")
        try:
            for idx, points, n_total in iter_frames(bag_path(a.dataset, bag), stride=1):
                dist = pipeline.process_frame(points)
                if dist is not None:
                    print(f"Кадр {idx}/{n_total}: Найдено ПРЕПЯТСТВИЕ на расстоянии {dist:.2f} м!")
        except Exception as e:
            print(f"Ошибка при обработке {bag}: {e}")
