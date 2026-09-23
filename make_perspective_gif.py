#!/usr/bin/env python3
"""
Создание реалистичных GIF-анимаций (Интенсивность, Расстояние, Градиент).
Скрипт разносит 3 типа визуализаций по разным папкам и гифкам.
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

sys.path.append('d:/LIDAR')
from rail_detection.loader import load_frame, iter_frames, bag_path
from rail_detection import DEFAULT_BAGS

ALL_BAGS = list(DEFAULT_BAGS) + ["doubleT_obstacle"]

def process_bag(bag, dataset="D:/Датасет", out_dir="d:/LIDAR/output", stride=1, target_frames=None, fps=10.0):
    """
    Функция обработки одного датасета. Удобно вызывать напрямую из интерпретатора.
    
    Пример:
        >>> from make_perspective_gif import process_bag
        >>> process_bag("roundT_pressureGate_roundT")
    """
    print(f"\n=== Обработка: {bag} ===")
    bag_dir = bag_path(dataset, bag)
    
    out_dir = Path(out_dir)
    dir_int = out_dir / 'intensity'
    dir_dist = out_dir / 'distance'
    dir_grad = out_dir / 'gradient'
    
    dir_int.mkdir(parents=True, exist_ok=True)
    dir_dist.mkdir(parents=True, exist_ok=True)
    dir_grad.mkdir(parents=True, exist_ok=True)
    
    images_int, images_dist, images_grad = [], [], []
    
    # Настройки виртуальной камеры (FOV 80 градусов как в test_camera.py)
    W, H_proj = 1200, 800
    crop_top = 200 # Обрезаем верхнюю черную полосу
    H = H_proj - crop_top
    
    fov_h = np.radians(80) 
    f = W / (2 * np.tan(fov_h / 2))
    cx, cy = W / 2, H_proj / 2
    
    for idx, points, n_total in iter_frames(bag_dir, stride=stride, max_frames=target_frames):
        Z_c = -points['y']
        X_c = points['x']
        Y_c = -points['z']
        intensity = points['intensity']
        
        front = Z_c > 0.5
        X_c, Y_c, Z_c, intensity = X_c[front], Y_c[front], Z_c[front], intensity[front]
        dist = np.sqrt(X_c**2 + Y_c**2 + Z_c**2)
        
        u = (f * X_c / Z_c + cx).astype(np.int32)
        v = (f * Y_c / Z_c + cy).astype(np.int32)
        
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H_proj)
        u, v, intensity, dist = u[valid], v[valid], intensity[valid], dist[valid]
        
        # Сортировка по удаленности (Z-буфер)
        order = np.argsort(dist)[::-1]
        u, v, intensity, dist = u[order], v[order], intensity[order], dist[order]
        
        img_int_proj = np.zeros((H_proj, W))
        img_dist_inv_proj = np.zeros((H_proj, W))
        
        img_int_proj[v, u] = intensity
        img_dist_inv_proj[v, u] = 1.0 / dist
        
        # Обрезаем верхнюю черную полосу (камера смотрит чуть ниже на рельсы)
        img_int = img_int_proj[crop_top:, :]
        img_dist_inv = img_dist_inv_proj[crop_top:, :]
        
        # 1. Интенсивность (стиль test_camera)
        img_int_dense = ndimage.maximum_filter(img_int, size=3)
        
        # 2. Расстояние (сглаживаем черные полосы по вертикали)
        img_dist_inv_dense = ndimage.maximum_filter(img_dist_inv, footprint=np.ones((5, 3)))
        img_dist_dense = np.full((H, W), np.nan)
        mask = img_dist_inv_dense > 0
        img_dist_dense[mask] = 1.0 / img_dist_inv_dense[mask]
        
        # 3. Градиент (Стиль v2, но БЕЗ каши)
        img_dist_inv_solid = ndimage.maximum_filter(img_dist_inv, footprint=np.ones((7, 3)))
        img_dist_solid = np.full((H, W), 60.0)
        solid_mask = img_dist_inv_solid > 0
        img_dist_solid[solid_mask] = 1.0 / img_dist_inv_solid[solid_mask]
        
        grad_x = ndimage.sobel(img_dist_solid, axis=1) / 8.0
        grad_y = ndimage.sobel(img_dist_solid, axis=0) / 8.0
        grad_map = np.hypot(grad_x, grad_y)
        
        eroded_mask = ndimage.binary_erosion(solid_mask, iterations=3)
        grad_map[~eroded_mask] = np.nan
        
        def render_panel(data, title, cmap, vmin, vmax, cbar_label):
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{bag} - Кадр {idx}/{n_total} | {title}")
            ax.axis('off')
            
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="2%", pad=0.1)
            fig.colorbar(im, cax=cax, label=cbar_label)
            
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            plt.close(fig)
            
            return Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=256)
        
        images_int.append(render_panel(img_int_dense, "Интенсивность", 'gray', 0, 150, "Интенсивность"))
        
        cmap_dist = plt.cm.viridis.copy()
        cmap_dist.set_bad('black')
        images_dist.append(render_panel(img_dist_dense, "Расстояние", cmap_dist, 0, 60, "Расстояние (м)"))
        
        cmap_grad = plt.cm.inferno.copy()
        cmap_grad.set_bad('black')
        images_grad.append(render_panel(grad_map, "Градиент", cmap_grad, 0, 2.0, "Градиент (м)"))
        
        print(f"\r  обработано кадров: {len(images_int)}/{n_total}", end="", flush=True)
        
    print()
    
    if not images_int:
        return
        
    kwargs = dict(save_all=True, duration=int(1000 / fps), loop=0, optimize=True)
    images_int[0].save(dir_int / f"{bag}.gif", append_images=images_int[1:], **kwargs)
    images_dist[0].save(dir_dist / f"{bag}.gif", append_images=images_dist[1:], **kwargs)
    images_grad[0].save(dir_grad / f"{bag}.gif", append_images=images_grad[1:], **kwargs)
    print(f"  Успешно сохранены 3 GIF файла в {out_dir}/[intensity, distance, gradient]/")

def process_all(dataset="D:/Датасет", out_dir="d:/LIDAR/output", stride=1, target_frames=None, fps=10.0):
    """
    Удобная функция для запуска обработки всех датасетов из интерпретатора.
    
    Пример:
        >>> from make_perspective_gif import process_all
        >>> process_all()
    """
    for bag in ALL_BAGS:
        process_bag(bag, dataset, out_dir, stride, target_frames, fps)

def main():
    p = argparse.ArgumentParser(description="Генерация отдельных GIF-файлов.")
    p.add_argument("--dataset", default="D:/Датасет")
    p.add_argument("--bags", nargs="*", default=[], help="Список датасетов для обработки (через пробел)")
    p.add_argument("--all", action="store_true", help="Запустить для всех датасетов")
    p.add_argument("--out", default="d:/LIDAR/output")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--target-frames", type=int, default=None)
    p.add_argument("--fps", type=float, default=10.0)
    a = p.parse_args()

    bags_to_process = ALL_BAGS if a.all else a.bags
    
    if not bags_to_process:
        print("Укажите датасеты через --bags <имя> или используйте флаг --all для запуска всех.")
        return

    for bag in bags_to_process:
        process_bag(bag, a.dataset, a.out, a.stride, a.target_frames, a.fps)

if __name__ == "__main__":
    # main()
    process_bag("roundT_pressureGate_roundT")
    process_bag("doubleT_obstacle")
