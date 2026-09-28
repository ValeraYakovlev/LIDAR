import os
import glob
import json
import numpy as np
import sys

# Добавляем корневую папку LIDAR в sys.path, чтобы импортировать пайплайн
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rail_detection.parallel_path import ParallelGauge
from rail_detection.far_detect import VARIANTS
from rosbags.typesys import Stores, get_typestore

DATASET_ROOT = "D:/synth_data"
DEBUG_MODE = sys.gettrace() is not None
OUT_DIR = os.path.join(DATASET_ROOT, "processed_slices_debug" if DEBUG_MODE else "processed_slices")

PF = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}

def to_numpy(msg):
    dt = np.dtype({"names": [f.name for f in msg.fields], "formats": [PF[f.datatype] for f in msg.fields],
                   "offsets": [f.offset for f in msg.fields], "itemsize": msg.point_step})
    return np.frombuffer(msg.data, dtype=dt, count=msg.width * msg.height)

def process_bag(bag_dir):
    bag_name = os.path.basename(bag_dir)
    out_file = os.path.join(OUT_DIR, f"{bag_name}.npz")
    if os.path.exists(out_file):
        print(f"[{bag_name}] Уже обработано, пропускаем.")
        return
        
    print(f"[{bag_name}] Начало обработки...")
    mcap_files = glob.glob(os.path.join(bag_dir, "mcap", "*.mcap"))
    db3_files = glob.glob(os.path.join(bag_dir, "db3", "*.db3")) + glob.glob(os.path.join(bag_dir, "*.db3"))
    if not mcap_files and not db3_files:
        print(f"[{os.path.basename(bag_dir)}] Не найден ни mcap, ни db3 файл")
        return
    
    labels_file = os.path.join(bag_dir, "yaml", "labels.json")
    if not os.path.exists(labels_file):
        labels_file = os.path.join(bag_dir, "labels.json")
    
    if not os.path.exists(labels_file):
        print(f"[{os.path.basename(bag_dir)}] Не найден labels.json")
        return
        
    with open(labels_file, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)["categoryID_to_class"]
    id2name = {int(k): v for k, v in labels_dict.items()}
    
    # Классы препятствий
    obstacle_names = {"Chair", "SpiderMan", "SpiderMan_Web"}
    obstacle_ids = set([k for k, v in id2name.items() if v in obstacle_names])
    
    gauge = ParallelGauge()
    # Используем вариант габарита из main
    p = VARIANTS["final_b2"]
    u_max = p["half"]
    u_min = -p["half"]
    v_min = p["bottom"]
    v_max = p["top"]
    
    ts = get_typestore(Stores.ROS2_HUMBLE)
    
    all_vectors = []
    all_labels = []

    def iter_pts():
        has_mcap_lib = False
        if mcap_files:
            try:
                from mcap.reader import make_reader
                has_mcap_lib = True
            except ImportError:
                pass
                
        if mcap_files and has_mcap_lib:
            with open(mcap_files[0], "rb") as f:
                for schema, channel, message in make_reader(f).iter_messages(topics=["/lidar_points_labeled"]):
                    yield to_numpy(ts.deserialize_cdr(message.data, schema.name))
        elif db3_files:
            import sqlite3
            con = sqlite3.connect("file:" + db3_files[0].replace("\\", "/") + "?mode=ro&immutable=1", uri=True)
            topics = {name: (tid, typ) for tid, name, typ in con.execute("SELECT id, name, type FROM topics")}
            tid, typ = topics["/lidar_points_labeled"]
            for _, data in con.execute("SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,)):
                yield to_numpy(ts.deserialize_cdr(bytes(data), typ))
        else:
            raise RuntimeError("mcap library not installed and no db3 file found")

    count = 0
    for pts in iter_pts():
            
            # Удаляем дубликаты
            _, idx = np.unique(pts[["x", "y", "z"]], return_index=True)
            pts = pts[idx]
            
            # Сохраняем метки ДО фильтрации внутри ParallelGauge
            x = pts['x'].astype(float)
            y = pts['y'].astype(float)
            z = pts['z'].astype(float)
            keep = (np.abs(x) + np.abs(y) + np.abs(z)) > 0.1
            labels_kept = pts["label"][keep]
            
            res = gauge.update(pts, steps=1)
            if res is None:
                count += 1
                continue
                
            s = res["s"]
            u = res["u"]
            v = res["v"]
            
            # Выделяем точки, попавшие внутрь статической зоны габарита
            from scipy.spatial import cKDTree
            ref_mask = (s >= 0.0) & (s <= 3.0)
            if np.sum(ref_mask) > 100:
                pts_2d_ref = np.column_stack((u[ref_mask], v[ref_mask]))
                ref_kdtree = cKDTree(pts_2d_ref)
                pts_all_2d = np.column_stack((u, v))
                dist_to_wall, _ = ref_kdtree.query(pts_all_2d)
                wall_mask = dist_to_wall > 0.15
            else:
                wall_mask = np.ones_like(s, dtype=bool)
            mask = (u >= u_min) & (u <= u_max) & (v >= v_min) & (v <= v_max) & (s > 0) & wall_mask
            s_val = s[mask]
            u_val = u[mask]
            v_val = v[mask]
            label_val = labels_kept[mask]
            
            if len(s_val) == 0:
                count += 1
                continue
                
            # Разделяем точки внутри габарита на отдельные препятствия (кластеры с разрывом > 1.0 м)
            sort_idx = np.argsort(s_val)
            s_val = s_val[sort_idx]
            u_val = u_val[sort_idx]
            v_val = v_val[sort_idx]
            label_val = label_val[sort_idx]
            
            gaps = np.diff(s_val) > 1.0
            split_indices = np.where(gaps)[0] + 1
            
            for c_s, c_u, c_v, c_lbl in zip(np.split(s_val, split_indices), np.split(u_val, split_indices), np.split(v_val, split_indices), np.split(label_val, split_indices)):
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
                        # Пропускаем пустые срезы - берем только те, где остались какие-то точки (шум/стены или препятствия)
                        continue
                    
                    u_bin = c_u[in_bin]
                    v_bin = c_v[in_bin]
                    lbl_bin = c_lbl[in_bin]
                    
                    H, _, _ = np.histogram2d(u_bin, v_bin, bins=[10, 10], range=[[u_min, u_max], [v_min, v_max]])
                    vector = H.flatten() / 100.0
                    
                    # Проверяем, есть ли точки реального препятствия
                    bin_label = 1 if np.any(np.isin(lbl_bin, list(obstacle_ids))) else 0
                    
                    all_vectors.append(vector)
                    all_labels.append(bin_label)
                    
            count += 1
            if count % 100 == 0:
                print(f"[{os.path.basename(bag_dir)}] Обработано {count} кадров...")
                
    if all_vectors:
        bag_name = os.path.basename(bag_dir)
        os.makedirs(OUT_DIR, exist_ok=True)
        out_file = os.path.join(OUT_DIR, f"{bag_name}.npz")
        np.savez_compressed(out_file, vectors=np.array(all_vectors), labels=np.array(all_labels))
        print(f"[{bag_name}] Сохранено {len(all_vectors)} срезов в {out_file}")

if __name__ == "__main__":
    bag_dirs = [os.path.join(DATASET_ROOT, d) for d in os.listdir(DATASET_ROOT) 
                if os.path.isdir(os.path.join(DATASET_ROOT, d)) and ("_obst" in d or "path" in d)]
    
    print(f"Найдено {len(bag_dirs)} папок для обработки.")
    for bd in bag_dirs:
        process_bag(bd)
