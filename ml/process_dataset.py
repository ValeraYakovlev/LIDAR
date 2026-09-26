import os
import glob
import json
import numpy as np
import sys

# Добавляем корневую папку LIDAR в sys.path, чтобы импортировать пайплайн
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_procrustes_clearance import DynamicClearancePipeline
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
    print(f"[{os.path.basename(bag_dir)}] Начало обработки...")
    mcap_files = glob.glob(os.path.join(bag_dir, "mcap", "*.mcap"))
    db3_files = glob.glob(os.path.join(bag_dir, "db3", "*.db3"))
    if not mcap_files and not db3_files:
        print(f"[{os.path.basename(bag_dir)}] Не найден ни mcap, ни db3 файл")
        return
    
    labels_file = os.path.join(bag_dir, "yaml", "labels.json")
    if not os.path.exists(labels_file):
        print(f"[{os.path.basename(bag_dir)}] Не найден labels.json")
        return
        
    with open(labels_file, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)["categoryID_to_class"]
    id2name = {int(k): v for k, v in labels_dict.items()}
    
    # Классы, которые считаются препятствиями (согласно README)
    obstacle_names = {"Chair", "SpiderMan", "SpiderMan_Web"}
    obstacle_ids = set([k for k, v in id2name.items() if v in obstacle_names])
    
    pipeline = DynamicClearancePipeline()
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
            
            # Удаляем дубликаты, как рекомендовано в README (раздел 8)
            _, idx = np.unique(pts[["x", "y", "z"]], return_index=True)
            pts = pts[idx]
            
            # Пропускаем кадр через пайплайн для получения геометрии габарита
            try:
                # Пайплайн подавляет принты? В нем есть print("[DEBUG SEGMENTS] ...")
                # Чтобы не засорять вывод, можно перенаправить stdout, но оставим как есть.
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    _, _, ros_payload = pipeline.process_pointcloud(pts)
            except Exception as e:
                print(f"Ошибка в пайплайне на кадре {count}: {e}")
                continue
                
            if ros_payload["status"] != "success":
                continue
                
            frame = ros_payload["frame_geometry"]
            rail_top_v = ros_payload["clearance_boxes"]["rail_top_v"]
            box_width = ros_payload["clearance_boxes"]["width"]
            box_height = ros_payload["clearance_boxes"]["height"]
            
            # Извлекаем координаты d, u, v для всех точек
            d = -pts['y'].astype(float)
            xc = frame["elastic_axis"](d)
            heading = np.arctan(frame["elastic_axis"].deriv()(d))
            u = (pts['x'].astype(float) - xc) * np.cos(heading)
            v = pts['z'].astype(float) - frame["elastic_floor"](d)
            
            v_min = rail_top_v + 0.10
            v_max = v_min + box_height
            u_min = -box_width / 2.0
            u_max = box_width / 2.0
            
            # Выделяем точки, попавшие внутрь габарита (впереди лидара, d > 0)
            mask = (u >= u_min) & (u <= u_max) & (v >= v_min) & (v <= v_max) & (d > 0)
            d_val = d[mask]
            u_val = u[mask]
            v_val = v[mask]
            label_val = pts["label"][mask]
            
            if len(d_val) == 0:
                count += 1
                continue
                
            # Нарезаем на срезы по 3 см вдоль оси d (от 0 до максимума)
            d_max_val = np.max(d_val)
            bins_d = np.arange(0.0, d_max_val + 0.03, 0.03)
            
            if len(bins_d) < 2:
                count += 1
                continue
                
            bin_indices = np.digitize(d_val, bins_d) - 1
            
            # Идем по каждому срезу
            for i in range(len(bins_d) - 1):
                in_bin = (bin_indices == i)
                if not np.any(in_bin):
                    continue  # Пустой срез не сохраняем
                    
                u_bin = u_val[in_bin]
                v_bin = v_val[in_bin]
                lbl_bin = label_val[in_bin]
                
                # Векторизуем в сетку 10x10 квадратиков
                H, _, _ = np.histogram2d(u_bin, v_bin, bins=[10, 10], range=[[u_min, u_max], [v_min, v_max]])
                vector = H.flatten()
                
                # Нормализация на количество квадратиков (100)
                vector = vector / 100.0
                
                # Определяем метку препятствия
                has_obst = any(lbl in obstacle_ids for lbl in lbl_bin)
                
                all_vectors.append(vector)
                all_labels.append(1 if has_obst else 0)
            
            count += 1
            if count % 100 == 0:
                print(f"[{os.path.basename(bag_dir)}] Обработано {count} кадров...")
                
    if all_vectors:
        bag_name = os.path.basename(bag_dir)
        os.makedirs(OUT_DIR, exist_ok=True)
        out_file = os.path.join(OUT_DIR, f"{bag_name}.npz")
        np.savez_compressed(out_file, vectors=np.array(all_vectors), labels=np.array(all_labels))
        print(f"[{bag_name}] Сохранено {len(all_vectors)} непустых срезов в {out_file}")

if __name__ == "__main__":
    # Находим все папки с датасетами
    bag_dirs = [os.path.join(DATASET_ROOT, d) for d in os.listdir(DATASET_ROOT) 
                if os.path.isdir(os.path.join(DATASET_ROOT, d)) and ("_obst" in d or "path" in d)]
    
    print(f"Найдено {len(bag_dirs)} папок для обработки.")
    for bd in bag_dirs:
        process_bag(bd)
