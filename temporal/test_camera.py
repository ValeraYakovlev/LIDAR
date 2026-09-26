import sys, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append('d:/LIDAR')
from rail_detection.loader import load_frame

dataset_dir = Path('D:/Датасет')
points, _ = load_frame(dataset_dir / 'roundT_pressureGate_roundT')

x = points['x']
y = points['y']
z = points['z']
intensity = points['intensity']

Z_c = -y
X_c = x
Y_c = -z

front = Z_c > 1.0
X_c = X_c[front]
Y_c = Y_c[front]
Z_c = Z_c[front]
intensity = intensity[front]
dist = np.sqrt(X_c**2 + Y_c**2 + Z_c**2)

W, H = 1200, 800
fov_h = np.radians(80) 
f = W / (2 * np.tan(fov_h / 2))
cx, cy = W / 2, H / 2

u = (f * X_c / Z_c + cx).astype(np.int32)
v = (f * Y_c / Z_c + cy).astype(np.int32)

valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
u = u[valid]
v = v[valid]
intensity = intensity[valid]
dist = dist[valid]

img_intensity = np.zeros((H, W))
img_dist = np.full((H, W), np.nan)

order = np.argsort(dist)[::-1]
u = u[order]
v = v[order]
intensity = intensity[order]
dist = dist[order]

img_intensity[v, u] = intensity
img_dist[v, u] = dist

from scipy.ndimage import maximum_filter
img_intensity_dense = maximum_filter(img_intensity, size=3)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.imshow(img_intensity_dense, cmap='gray', vmin=np.percentile(intensity, 5), vmax=np.percentile(intensity, 99))
ax1.set_title("Perspective Projection - Intensity")
ax1.axis('off')

cmap = plt.cm.viridis.copy()
cmap.set_bad('black')
im2 = ax2.imshow(img_dist, cmap=cmap, vmin=0, vmax=60)
ax2.set_title("Perspective Projection - Distance")
ax2.axis('off')

plt.tight_layout()
plt.savefig('d:/LIDAR/test_camera.png')

