#!/usr/bin/env python3
"""Создание реалистичной GIF-анимации с помощью перспективной проекции."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

sys.path.append('d:/LIDAR')
from rail_detection.loader import load_frame, iter_frames, bag_path
from rail_detection import DEFAULT_BAGS

ALL_BAGS = list(DEFAULT_BAGS) + ["doubleT_obstacle"]

def build_perspective_gif(dataset, bag, out_dir, stride, target_frames, fps):
    print(f"\n=== Перспектива: {bag} (шаг {stride}) ===")
    bag_dir = bag_path(dataset, bag)
    
    from rail_detection.loader import frame_count
    n = frame_count(bag_dir)
    if stride is None:
        stride = max(1, round(n / target_frames))
        
    images = []
    
    # Настройки виртуальной камеры
    W, H = 1200, 800
    fov_h = np.radians(90) # Угол обзора 90 градусов
    f = W / (2 * np.tan(fov_h / 2))
    cx, cy = W / 2, H / 2
    
    for idx, points, n_total in iter_frames(bag_dir, stride=stride, max_frames=target_frames):
        # Координаты (в локальной системе камеры, где Z смотрит вперед)
        Z_c = -points['y']
        X_c = points['x']
        Y_c = -points['z']
        intensity = points['intensity']
        
        # Отсекаем все, что сзади
        front = Z_c > 0.5
        X_c, Y_c, Z_c, intensity = X_c[front], Y_c[front], Z_c[front], intensity[front]
        dist = np.sqrt(X_c**2 + Y_c**2 + Z_c**2)
        
        # Проекция на пиксели
        u = (f * X_c / Z_c + cx).astype(np.int32)
        v = (f * Y_c / Z_c + cy).astype(np.int32)
        
        # Отсекаем то, что за кадром
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, intensity, dist = u[valid], v[valid], intensity[valid], dist[valid]
        
        # Сортировка по расстоянию (сначала дальние, чтобы ближние рисовались поверх них)
        order = np.argsort(dist)[::-1]
        u, v, intensity, dist = u[order], v[order], intensity[order], dist[order]
        
        # Отрисовка на матрицу изображения
        img_intensity = np.zeros((H, W))
        img_dist_inv = np.zeros((H, W))  # Обратное расстояние для Z-буфера
        
        img_intensity[v, u] = intensity
        img_dist_inv[v, u] = 1.0 / dist
        
        # Расширение точек (чтобы картинка была плотной, а не из редких пикселей)
        img_intensity_dense = maximum_filter(img_intensity, size=3)
        img_dist_inv_dense = maximum_filter(img_dist_inv, size=3)
        
        img_dist_dense = np.full((H, W), np.nan)
        mask = img_dist_inv_dense > 0
        img_dist_dense[mask] = 1.0 / img_dist_inv_dense[mask]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=100)
        
        # Панель 1: Интенсивность (как Ч/Б фото)
        vmax_int = np.percentile(intensity, 99) if intensity.size > 0 else 255.0
        im1 = ax1.imshow(img_intensity_dense, cmap='gray', vmin=0, vmax=vmax_int)
        ax1.set_title(f"{bag} - Кадр {idx}/{n_total} | Интенсивность отражения (как видео)")
        ax1.axis('off')
        
        # Панель 2: Расстояние
        cmap = plt.cm.viridis.copy()
        cmap.set_bad('black')
        im2 = ax2.imshow(img_dist_dense, cmap=cmap, vmin=0, vmax=60)
        ax2.set_title("Оптическая проекция | Расстояние до лидара")
        ax2.axis('off')
        fig.colorbar(im2, ax=ax2, label='Расстояние (м)', fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=96))
        
        plt.close(fig)
        print(f"\r  отрисовано кадров: {len(images)}", end="", flush=True)
        
    print()
    if not images:
        return
        
    path = out_dir / f"perspective_{bag}.gif"
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(f"  Сохранено: {path} ({path.stat().st_size / 1e6:.1f} МБ)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="D:/Датасет")
    p.add_argument("--bags", nargs="*", default=ALL_BAGS)
    p.add_argument("--out", default="d:/LIDAR/output")
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--target-frames", type=int, default=50)
    p.add_argument("--fps", type=float, default=10.0)
    a = p.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bag in a.bags:
        build_perspective_gif(a.dataset, bag, out_dir, a.stride, a.target_frames, a.fps)

if __name__ == "__main__":
    main()

