import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from cython_dbscan.dbscan_core import dbscan

coords = np.array([
    [52.0, 4.0],
    [52.0, 4.001],
    [48.0, 2.0],
    [48.0002, 2.0001]
], dtype=np.float64)

labels = dbscan(coords, eps=0.5, min_samples=2)
print(labels)
