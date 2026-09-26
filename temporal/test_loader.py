import sys
import numpy as np
import os
from pathlib import Path

sys.path.append('d:/LIDAR')
from rail_detection.loader import load_frame

# search for dataset dir safely
dataset_dir = None
for name in os.listdir('D:/'):
    if 'Датасет' in name or name.encode('utf-8', 'ignore') == b'\xd0\x94\xd0\xb0\xd1\x82\xd0\xb0\xd1\x81\xd0\xb5\xd1\x82': 
        dataset_dir = Path('D:/') / name
        break
if dataset_dir is None:
    dataset_dir = Path('D:/Датасет')

print("dataset_path exists:", dataset_dir.exists())

try:
    bag_path = dataset_dir / 'doubleT_platform'
    points, n = load_frame(bag_path)
    print("Loaded points:", points.shape)
    print(points.dtype)
except Exception as e:
    import traceback
    traceback.print_exc()

