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

fig, axs = plt.subplots(2, 1, figsize=(15, 10))
im1 = axs[0].imshow(dist_h.T, cmap='inferno', aspect='auto', vmin=0, vmax=2.0)
axs[0].set_title('Horizontal gradients (azimuth)')
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(dist_v.T, cmap='inferno', aspect='auto', vmin=0, vmax=2.0)
axs[1].set_title('Vertical gradients (ring)')
fig.colorbar(im2, ax=axs[1])
plt.savefig('d:/LIDAR/test_grad_image.png')

fig2, ax2 = plt.subplots(figsize=(10, 10))
ax2.scatter(mid_h[:, :, 0].flatten(), mid_h[:, :, 1].flatten(), c=dist_h.flatten(), cmap='inferno', s=0.1, vmin=0, vmax=2.0)
plt.savefig('d:/LIDAR/test_grad_scatter.png')
