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
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from cython.parallel cimport prange
from libcpp.map cimport map as cpp_map
from libc.stdint cimport int64_t
from libc.math cimport floor

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


# --- New spatial hashing and neighbor precomputation ---
@cython.cdivision(True)
cdef inline int64_t compute_cell_id(double lat, double lon,
                                    double cell_size_lat,
                                    double cell_size_lon) nogil:
    cdef double fx = lon / cell_size_lon
    cdef double fy = lat / cell_size_lat
    cdef int64_t x = <int64_t>fx
    cdef int64_t y = <int64_t>fy
    cdef int64_t mask = <int64_t>0xFFFFFFFF
    return (x << 32) | (y & mask)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef vector[vector[int]] precompute_neighbors(double[:, ::1] coords,
                                       int64_t[::1] cell_ids,
                                       double eps,
                                       double cell_size_lat,
                                       double cell_size_lon,
                                       cpp_map[int64_t, vector[int]] cell_to_points) nogil:
    cdef Py_ssize_t n = coords.shape[0]
    cdef vector[vector[int]] neighbor_list
    neighbor_list.resize(n)
    cdef vector[vector[int]] neighbors_by_index
    neighbors_by_index.resize(n)
    cdef Py_ssize_t i, k, pt
    cdef int64_t cid, neigh_cid, x, y
    cdef int64_t mask = <int64_t>0xFFFFFFFF
    cdef size_t pts_size
    cdef int dxs[9]
    cdef int dys[9]
    cdef int m, dx, dy
    cdef vector[int]* neighbors_local
    # neighbor offset arrays
    dxs[0] = -1; dxs[1] = -1; dxs[2] = -1; dxs[3] =  0; dxs[4] =  0; dxs[5] =  0; dxs[6] =  1; dxs[7] =  1; dxs[8] =  1
    dys[0] = -1; dys[1] =  0; dys[2] =  1; dys[3] = -1; dys[4] =  0; dys[5] =  1; dys[6] = -1; dys[7] =  0; dys[8] =  1
    for i in prange(n, nogil=True, schedule='dynamic'):
        neighbors_local = &neighbors_by_index[i]
        neighbors_local[0].clear()
        neighbors_local[0].reserve(64)
        cid = cell_ids[i]
        x = cid >> 32
        y = cid & mask

        for m in range(9):
            dx = dxs[m]; dy = dys[m]
            neigh_cid = ((x + dx) << 32) | ((y + dy) & mask)
            if cell_to_points.find(neigh_cid) != cell_to_points.end():
                neighbors_local[0].insert(neighbors_local[0].end(),
                                       cell_to_points[neigh_cid].begin(),
                                       cell_to_points[neigh_cid].end())

        for k in range(neighbors_local[0].size()):
            pt = neighbors_local[0][k]
            if haversine(coords[i, 0], coords[i, 1],
                         coords[pt, 0], coords[pt, 1]) <= eps:
                neighbor_list[i].push_back(pt)
    return neighbor_list

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
    cdef vector[int] seeds, new_neighbors, neighbors
    cdef int point

    # prepare cell IDs and spatial hash
    cdef np.ndarray[np.int64_t, ndim=1] cell_ids_py = np.empty(n, dtype=np.int64)
    cdef int64_t[::1] cell_ids = cell_ids_py
    cdef cpp_map[int64_t, vector[int]] cell_to_points
    cdef double cell_size_lat = eps / 110.574
    cdef double cell_size_lon = eps / 111.320

    for i in range(n):
        cell_ids[i] = compute_cell_id(coords[i, 0], coords[i, 1],
                                      cell_size_lat, cell_size_lon)
        cell_to_points[cell_ids[i]].push_back(i)

    # precompute neighbor lists in parallel
    cdef vector[vector[int]] neighbor_list = precompute_neighbors(coords,
                                                           cell_ids,
                                                           eps,
                                                           cell_size_lat,
                                                           cell_size_lon,
                                                           cell_to_points)

    # DBSCAN cluster expansion using precomputed neighborhood
    for i in range(n):
        neighbors = neighbor_list[i]
        if neighbors.size() < min_samples:
            labels[i] = -1
            continue
        labels[i] = cluster_id
        seeds = neighbors
        while seeds.size() > 0:
            point = seeds.back()
            seeds.pop_back()
            if labels[point] == -1:
                labels[point] = cluster_id
            elif labels[point] != -1:
                continue
            new_neighbors = neighbor_list[point]
            if new_neighbors.size() >= min_samples:
                seeds.insert(seeds.end(),
                             new_neighbors.begin(),
                             new_neighbors.end())
            labels[point] = cluster_id
        cluster_id += 1

    return np.asarray(labels)
