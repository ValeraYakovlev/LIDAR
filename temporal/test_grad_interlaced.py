import sys, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append('d:/LIDAR')
from rail_detection.loader import load_frame

dataset_dir = Path('D:/Датасет')
points, n = load_frame(dataset_dir / 'doubleT_platform')

xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)
grid = xyz.reshape(2400, 128, 3)

dist_h = np.linalg.norm(grid[:-1, :] - grid[1:, :], axis=-1) # shape (2399, 128)
dist_v = np.linalg.norm(grid[:, :-1] - grid[:, 1:], axis=-1) # shape (2400, 127)

N, M = 128, 2400
map_2d = np.full((N*2-1, M*2-1), np.nan)
map_2d[::2, 1::2] = dist_h.T
map_2d[1::2, ::2] = dist_v.T

fig, ax = plt.subplots(figsize=(24, 6), dpi=100)
cmap = plt.cm.inferno.copy()
cmap.set_bad('black')
im = ax.imshow(map_2d, cmap=cmap, vmin=0, vmax=2.0)
fig.colorbar(im, ax=ax)
plt.savefig('d:/LIDAR/test_grad_interlaced.png', bbox_inches='tight')

