#!/usr/bin/env python3
"""Создание GIF-анимаций градиентов (расстояний) между соседними точками лидара."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append('d:/LIDAR')
from rail_detection.loader import load_frame, iter_frames, bag_path
from rail_detection import DEFAULT_BAGS

ALL_BAGS = list(DEFAULT_BAGS) + ["doubleT_obstacle"]

def build_gradient_gif(dataset, bag, out_dir, stride, target_frames, fps):
    print(f"\n=== Градиенты: {bag} (шаг {stride}) ===")
    bag_dir = bag_path(dataset, bag)
    
    # Сначала считаем общее количество кадров
    from rail_detection.loader import frame_count
    n = frame_count(bag_dir)
    if stride is None:
        stride = max(1, round(n / target_frames))
        
    images = []
    for idx, points, n_total in iter_frames(bag_dir, stride=stride, max_frames=target_frames):
        # Определяем размерность сетки динамически
        N = len(np.unique(points['ring']))
        M = len(points) // N
        
        # Извлекаем XYZ
        xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)
        grid = xyz.reshape(M, N, 3)
        
        # Вычисляем расстояния между соседними точками
        dist_h = np.linalg.norm(grid[:-1, :, :] - grid[1:, :, :], axis=-1) # (M-1, N)
        dist_v = np.linalg.norm(grid[:, :-1, :] - grid[:, 1:, :], axis=-1) # (M, N-1)
        
        # Плотная карта градиентов (без пустых исходных точек)
        grad_map = np.sqrt(dist_h[:, :-1]**2 + dist_v[:-1, :]**2).T # (N-1, M-1)
        
        # Расстояние от каждой точки до лидара
        dist_origin = np.linalg.norm(grid, axis=-1).T # (N, M)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=100)
        
        # Панель 1: Градиенты
        vmax_grad = np.percentile(grad_map, 99) if grad_map.size > 0 else 5.0
        im1 = ax1.imshow(grad_map, cmap='inferno', vmin=0, vmax=vmax_grad, aspect='auto')
        ax1.set_title(f"{bag} - Кадр {idx}/{n_total} | Градиент (расстояние между точками)")
        ax1.axis('off')
        fig.colorbar(im1, ax=ax1, label='Градиент (м)')
        
        # Панель 2: Проекция исходного облака (расстояние до лидара)
        vmax_orig = np.percentile(dist_origin, 99) if dist_origin.size > 0 else 60.0
        im2 = ax2.imshow(dist_origin, cmap='viridis', vmin=0, vmax=vmax_orig, aspect='auto')
        ax2.set_title("Проекция облака | Расстояние до лидара")
        ax2.axis('off')
        fig.colorbar(im2, ax=ax2, label='Расстояние (м)')
        
        plt.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        images.append(Image.fromarray(buf).convert("P", palette=Image.ADAPTIVE, colors=96))
        
        plt.close(fig)
        print(f"\r  отрисовано кадров: {len(images)}", end="", flush=True)
        
    print()
    if not images:
        return
        
    path = out_dir / f"gradient_{bag}.gif"
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
        build_gradient_gif(a.dataset, bag, out_dir, a.stride, a.target_frames, a.fps)

if __name__ == "__main__":
    main()

