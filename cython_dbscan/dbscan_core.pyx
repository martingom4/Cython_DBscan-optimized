# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# distutils: language=c++
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
"""
DBSCAN optimizado con Cython.
"""

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport sqrt, sin, cos, atan2
from libcpp.vector cimport vector

# Constante precomputada
cdef double DEG_TO_RAD = 3.141592653589793 / 180.0

@cython.cdivision(True)
cdef inline double haversine(double lat1, double lon1, double lat2, double lon2) nogil:
    cdef double R = 6371.0
    cdef double dlat = (lat2 - lat1) * DEG_TO_RAD
    cdef double dlon = (lon2 - lon1) * DEG_TO_RAD
    cdef double a = sin(dlat / 2.0) ** 2 + cos(lat1 * DEG_TO_RAD) * cos(lat2 * DEG_TO_RAD) * sin(dlon / 2.0) ** 2
    cdef double c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

@cython.boundscheck(False)
@cython.wraparound(False)
def dbscan(double[:, ::1] coords, double eps, int min_samples):
    """
    Algoritmo DBSCAN aplicado sobre coordenadas geográficas.
    Usa vectores C++ y cálculos sin envoltorios de Python para máxima velocidad.
    """
    cdef Py_ssize_t n = coords.shape[0]
    cdef int[:] labels = np.full(n, -1, dtype=np.int32)
    cdef int cluster_id = 0
    cdef Py_ssize_t i, j, point
    cdef vector[int] neighbors, seeds, new_neighbors
    cdef double lat_i, lon_i, lat_j, lon_j, d

    for i in range(n):
        if labels[i] != -1:
            continue

        lat_i = coords[i, 0]
        lon_i = coords[i, 1]
        neighbors.clear()

        for j in range(n):
            lat_j = coords[j, 0]
            lon_j = coords[j, 1]
            d = haversine(lat_i, lon_i, lat_j, lon_j)
            if d <= eps:
                neighbors.push_back(j)

        if neighbors.size() < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seeds = neighbors[:]

        while seeds.size() > 0:
            point = seeds.back()
            seeds.pop_back()

            if labels[point] == -1:
                labels[point] = cluster_id
            elif labels[point] != -1:
                continue

            labels[point] = cluster_id
            new_neighbors.clear()
            lat_i = coords[point, 0]
            lon_i = coords[point, 1]

            for j in range(n):
                lat_j = coords[j, 0]
                lon_j = coords[j, 1]
                d = haversine(lat_i, lon_i, lat_j, lon_j)
                if d <= eps:
                    new_neighbors.push_back(j)

            if new_neighbors.size() >= min_samples:
                seeds.insert(seeds.end(), new_neighbors.begin(), new_neighbors.end())

        cluster_id += 1

    return np.asarray(labels)
