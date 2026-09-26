import sys, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import scipy.ndimage as ndimage
sys.path.append('d:/LIDAR')
from rail_detection.loader import load_frame

dataset_dir = Path('D:/Датасет')
points, _ = load_frame(dataset_dir / 'roundT_pressureGate_roundT')

N = len(np.unique(points['ring']))
M = len(points) // N
xyz = np.stack([points['x'], points['y'], points['z']], axis=-1)
grid = xyz.reshape(M, N, 3)

dist_h = np.linalg.norm(grid[:-1, :, :] - grid[1:, :, :], axis=-1)
dist_v = np.linalg.norm(grid[:, :-1, :] - grid[:, 1:, :], axis=-1)
dist_h = np.pad(dist_h, ((0, 1), (0, 0)), mode='edge')
dist_v = np.pad(dist_v, ((0, 0), (0, 1)), mode='edge')
grad_flat = np.sqrt(dist_h**2 + dist_v**2).flatten()

Z_c = -points['y']
X_c = points['x']
Y_c = -points['z']
intensity = points['intensity']

front = Z_c > 0.5
X_c, Y_c, Z_c = X_c[front], Y_c[front], Z_c[front]
intensity = intensity[front]
grad_flat = grad_flat[front]
dist = np.sqrt(X_c**2 + Y_c**2 + Z_c**2)

W, H = 1200, 800
fov_h = np.radians(90) 
f = W / (2 * np.tan(fov_h / 2))
cx, cy = W / 2, H / 2

u = (f * X_c / Z_c + cx).astype(np.int32)
v = (f * Y_c / Z_c + cy).astype(np.int32)

valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
u, v, intensity, dist, grad_flat = u[valid], v[valid], intensity[valid], dist[valid], grad_flat[valid]

order = np.argsort(dist)[::-1]
u, v, intensity, dist, grad_flat = u[order], v[order], intensity[order], dist[order], grad_flat[order]

img_intensity = np.zeros((H, W))
img_dist_inv = np.zeros((H, W))
img_grad = np.zeros((H, W))

img_intensity[v, u] = intensity
img_dist_inv[v, u] = 1.0 / dist
img_grad[v, u] = grad_flat

img_intensity_dense = ndimage.maximum_filter(img_intensity, size=3)
img_dist_inv_dense = ndimage.maximum_filter(img_dist_inv, size=3)
img_grad_dense = ndimage.maximum_filter(img_grad, size=3)

img_dist_dense = np.full((H, W), np.nan)
mask = img_dist_inv_dense > 0
img_dist_dense[mask] = 1.0 / img_dist_inv_dense[mask]
img_grad_dense[~mask] = np.nan

fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=100)

im1 = axes[0].imshow(img_intensity_dense, cmap='gray', vmin=0, vmax=150)
axes[0].set_title("Intensity")
axes[0].axis('off')
div1 = make_axes_locatable(axes[0])
cax1 = div1.append_axes("right", size="2%", pad=0.1)
fig.colorbar(im1, cax=cax1, label='Intensity')

cmap = plt.cm.viridis.copy()
cmap.set_bad('black')
im2 = axes[1].imshow(img_dist_dense, cmap=cmap, vmin=0, vmax=60)
axes[1].set_title("Distance")
axes[1].axis('off')
div2 = make_axes_locatable(axes[1])
cax2 = div2.append_axes("right", size="2%", pad=0.1)
fig.colorbar(im2, cax=cax2, label='Distance (m)')

cmap_grad = plt.cm.inferno.copy()
cmap_grad.set_bad('black')
im3 = axes[2].imshow(img_grad_dense, cmap=cmap_grad, vmin=0, vmax=5.0)
axes[2].set_title("Gradient")
axes[2].axis('off')
div3 = make_axes_locatable(axes[2])
cax3 = div3.append_axes("right", size="2%", pad=0.1)
fig.colorbar(im3, cax=cax3, label='Gradient (m)')

plt.tight_layout()
plt.savefig('d:/LIDAR/test_camera_v3.png')

