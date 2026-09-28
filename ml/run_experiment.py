"""
Эксперимент: сравнение ML-моделей с эталонным алгоритмом FarDetector на реальных данных.
Запускается из VS Code. Прогресс выводится в консоль, результат — в ml/experiment_results.txt.
"""
import sys
import os
import time
import json

sys.path.insert(0, r"d:\LIDAR")

from rail_detection.parallel_path import ParallelGauge
from rail_detection.far_detect import FarDetector, VARIANTS
from rail_detection.loader import iter_frames
from rail_detection import bag_path

# Импорт ML-пайплайна
sys.path.insert(0, r"d:\LIDAR\ml")
from ros_pipeline import ObstaclePipeline

# ─────────────── Настройки ───────────────
DATASET_DIR = "D:/Датасет"
BAGS = [
    "doubleT_obstacle",
    "doubleT_platform",
    "roundT_doubleT",
    "roundT_pressureGate_roundT",
    "roundT_squareT_pressureGate_squareT",
    "squareT_platform_squareT_switch",
]

MODELS = {
    "xgb_obstacles": r"d:\LIDAR\ml\xgb_obstacle_obstacles.json",
    "xgb_icp":       r"d:\LIDAR\ml\xgb_obstacle_icp.json",
    "xgb_tunnel":    r"d:\LIDAR\ml\xgb_obstacle_tunnel.json",
}

OUTPUT_FILE = r"d:\LIDAR\ml\experiment_results.txt"


# ─────────── Прогон эталона (FarDetector из main) ───────────
def run_far_detector(bag_name):
    """Прогоняет стандартный алгоритм FarDetector (main) по записи."""
    gauge = ParallelGauge()
    det = FarDetector(VARIANTS["final_b2"])

    results = []
    times = []

    for idx, pts, n_total in iter_frames(bag_path(DATASET_DIR, bag_name), stride=1):
        t0 = time.perf_counter()
        res = gauge.update(pts)
        if res is None:
            times.append(time.perf_counter() - t0)
            continue
        out = det.update(res)
        dt = time.perf_counter() - t0
        times.append(dt)

        confirmed = out.get("confirmed", []) if isinstance(out, dict) else []
        if confirmed:
            for dist in confirmed:
                results.append({
                    "frame": idx,
                    "n_total": n_total,
                    "distance": round(float(dist), 2),
                    "time_ms": round(dt * 1000, 1),
                })

    return results, times


# ─────────── Прогон ML-модели (ObstaclePipeline) ───────────
def run_ml_pipeline(bag_name, model_path):
    """Прогоняет ML-пайплайн (ObstaclePipeline) по записи."""
    pipeline = ObstaclePipeline(model_path)

    results = []
    times = []

    for idx, pts, n_total in iter_frames(bag_path(DATASET_DIR, bag_name), stride=1):
        t0 = time.perf_counter()
        dist = pipeline.process_frame(pts)
        dt = time.perf_counter() - t0
        times.append(dt)

        if dist is not None:
            results.append({
                "frame": idx,
                "n_total": n_total,
                "distance": round(float(dist), 2),
                "time_ms": round(dt * 1000, 1),
            })

    return results, times


# ─────────── ML без модели (только геометрия) ───────────
def run_ml_no_model(bag_name):
    """Прогоняет ML-пайплайн без обученной модели (все предсказания = 0)."""
    pipeline = ObstaclePipeline("__nonexistent_model__.json")

    results = []
    times = []

    for idx, pts, n_total in iter_frames(bag_path(DATASET_DIR, bag_name), stride=1):
        t0 = time.perf_counter()
        dist = pipeline.process_frame(pts)
        dt = time.perf_counter() - t0
        times.append(dt)

        if dist is not None:
            results.append({
                "frame": idx,
                "n_total": n_total,
                "distance": round(float(dist), 2),
                "time_ms": round(dt * 1000, 1),
            })

    return results, times


# ─────────── Форматирование ───────────
def summarize(label, bag, results, times):
    import numpy as np
    n_det = len(results)
    t_arr = np.array(times) * 1000  # мс
    med_ms = np.median(t_arr)
    p95_ms = np.percentile(t_arr, 95)
    n_frames = len(times)

    lines = []
    lines.append(f"  [{label}] {bag}: {n_det} обнаружений за {n_frames} кадров")
    lines.append(f"    Время: медиана {med_ms:.0f} мс, 95-й перцентиль {p95_ms:.0f} мс")
    if results:
        dists = [r["distance"] for r in results]
        frames = [r["frame"] for r in results]
        lines.append(f"    Первое обнаружение: кадр {min(frames)}, расстояние {max(dists):.1f} м")
        lines.append(f"    Ближайшее расстояние: {min(dists):.1f} м")
        lines.append(f"    Кадры с детекцией: {sorted(set(frames))[:10]}{'...' if len(set(frames)) > 10 else ''}")
    return lines


# ─────────── Главный цикл ───────────
if __name__ == "__main__":
    import numpy as np

    all_output = []
    all_output.append("=" * 70)
    all_output.append("ЭКСПЕРИМЕНТ: сравнение ML-моделей с FarDetector (main)")
    all_output.append(f"Датасет: {DATASET_DIR}")
    all_output.append(f"Записи: {BAGS}")
    all_output.append(f"Модели: {list(MODELS.keys())}")
    all_output.append("=" * 70)

    # Названия методов для прогона
    methods = [("FarDetector (main)", None)]  # эталон
    methods.append(("ML без модели", "__no_model__"))
    for name, path in MODELS.items():
        methods.append((f"ML: {name}", path))

    for bag in BAGS:
        all_output.append("")
        all_output.append("-" * 60)
        all_output.append(f"Запись: {bag}")
        all_output.append("-" * 60)
        print(f"\n{'=' * 60}")
        print(f"  Запись: {bag}")
        print(f"{'=' * 60}")

        for method_name, method_arg in methods:
            print(f"  > {method_name}...", end=" ", flush=True)
            t_start = time.perf_counter()

            if method_arg is None:
                # FarDetector
                results, times = run_far_detector(bag)
            elif method_arg == "__no_model__":
                results, times = run_ml_no_model(bag)
            else:
                results, times = run_ml_pipeline(bag, method_arg)

            elapsed = time.perf_counter() - t_start
            print(f"gotovo ({elapsed:.1f} s, {len(results)} detections)")

            lines = summarize(method_name, bag, results, times)
            for ln in lines:
                print(f"    {ln}")
                all_output.append(ln)

    # Сохраняем
    all_output.append("")
    all_output.append("=" * 70)
    all_output.append("EXPERIMENT DONE")
    all_output.append("=" * 70)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_output))

    print(f"\n{'=' * 60}")
    print(f"  Rezultaty sokhraneny v {OUTPUT_FILE}")
    print(f"{'=' * 60}")
