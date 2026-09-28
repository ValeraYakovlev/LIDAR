
import os
import numpy as np
from rail_detection.loader import iter_frames
from rail_detection.parallel_path import ParallelGauge
from rail_detection.far_detect import FarDetector, VARIANTS
import sys

dataset = sys.argv[1]
bag = sys.argv[2]
gauge = ParallelGauge()
detector = FarDetector(VARIANTS['final_b2'])

for idx, points, n_total in iter_frames(os.path.join(dataset, bag), stride=1):
    res = gauge.update(points)
    if res is None: continue
    dets = detector.update(res)
    # dets is list of ([distances], [TrackWatch confidences])
    for d, conf in dets:
        if conf:
            print(f"Кадр {idx}/{n_total}: Чистый метод нашел препятствие на {min([c['dist'] for c in conf]):.2f} м!")
