import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from pathlib import Path
import sys
sys.path.append('d:/LIDAR')
from rail_detection.loader import load_frame
from algorithm import compute_distance_minus_gradient

points, _ = load_frame(Path('D:/Датасет/roundT_pressureGate_roundT'))
xyz, metric = compute_distance_minus_gradient(points)

W, H_proj = 1200, 800
crop_top = 200
H = H_proj - crop_top
fov_h = np.radians(80) 
f = W / (2 * np.tan(fov_h / 2))
cx, cy = W / 2, H_proj / 2

X_c, Z_c, Y_c = xyz[:, 0], -xyz[:, 1], -xyz[:, 2]
front = Z_c > 0.5
X_c, Y_c, Z_c, metric = X_c[front], Y_c[front], Z_c[front], metric[front]
dist = np.sqrt(X_c**2 + Y_c**2 + Z_c**2)

u = (f * X_c / Z_c + cx).astype(np.int32)
v = (f * Y_c / Z_c + cy).astype(np.int32)

valid = (u >= 0) & (u < W) & (v >= 0) & (v < H_proj)
u, v, metric, dist = u[valid], v[valid], metric[valid], dist[valid]

order = np.argsort(dist)[::-1]
u, v, metric = u[order], v[order], metric[order]

img_metric_proj = np.full((H_proj, W), np.nan)
img_metric_proj[v, u] = metric
img_metric = img_metric_proj[crop_top:, :]

# Create validity mask (where LiDAR points exist roughly)
mask_points = np.zeros((H, W), dtype=bool)
# need to recompute valid v after crop_top
valid_crop = (v >= crop_top)
u_crop = u[valid_crop]
v_crop = v[valid_crop] - crop_top
mask_points[v_crop, u_crop] = True
valid_mask = ndimage.maximum_filter(mask_points, footprint=np.ones((7, 5)))

# Vertical interpolation
for x in range(W):
    col = img_metric[:, x]
    val = ~np.isnan(col)
    if val.sum() > 1:
        idx = np.arange(H)
        img_metric[:, x] = np.interp(idx, idx[val], col[val])

img_metric[~valid_mask] = np.nan

plt.figure(figsize=(12, 6), dpi=100)
plt.imshow(img_metric, cmap='coolwarm', vmin=-1.0, vmax=1.0)
plt.colorbar()
plt.tight_layout()
plt.savefig('test_interp.png')

