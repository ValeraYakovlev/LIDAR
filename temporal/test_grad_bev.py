import sys, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append('d:/LIDAR')
from rail_detection.loader import load_frame

dataset_dir = Path('D:/Датасет')
points, n = load_frame(dataset_dir / 'doubleT_platform')

xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)
grid = xyz.reshape(2400, 128, 3)

p1_h = grid[:-1, :, :]
p2_h = grid[1:, :, :]
dist_h = np.linalg.norm(p1_h - p2_h, axis=-1)
mid_h = (p1_h + p2_h) / 2

p1_v = grid[:, :-1, :]
p2_v = grid[:, 1:, :]
dist_v = np.linalg.norm(p1_v - p2_v, axis=-1)
mid_v = (p1_v + p2_v) / 2

mid_x = np.concatenate([mid_h[:, :, 0].flatten(), mid_v[:, :, 0].flatten()])
mid_y = np.concatenate([mid_h[:, :, 1].flatten(), mid_v[:, :, 1].flatten()])
dist = np.concatenate([dist_h.flatten(), dist_v.flatten()])

mid_depth = -mid_y
valid = (mid_depth > 0) & (mid_depth < 60) & (np.abs(mid_x) < 6.0)

fig, ax = plt.subplots(figsize=(6, 10), dpi=150)
sc = ax.scatter(mid_x[valid], mid_depth[valid], c=dist[valid], cmap='inferno', s=0.8, vmin=0, vmax=0.5)
ax.set_xlim(-6, 6)
ax.set_ylim(0, 60)
ax.set_facecolor('black')
fig.colorbar(sc, ax=ax)
plt.savefig('d:/LIDAR/test_grad_bev.png', bbox_inches='tight')

