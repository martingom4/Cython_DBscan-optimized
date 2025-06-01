from cython_dbscan.dbscan_core import dbscan
import numpy as np
import pandas as pd
import time

def segment_dataframe(df: pd.DataFrame, eps: float = 0.5, min_samples: int = 10,
                      min_segment: int = 10, max_segment: int = 10000) -> (pd.DataFrame, list):
    """
    Aplica el algoritmo DBSCAN agrupado por la columna 'res_8'.

    Parámetros:
        df (pd.DataFrame): DataFrame que debe tener las columnas 'res_8', 'latitude' y 'longitude'.
        eps (float): Radio de vecindad para DBSCAN.
        min_samples (int): Número mínimo de puntos para formar un cluster.
        min_segment (int): Mínimo número de registros para procesar un segmento.
        max_segment (int): Máximo número de registros para procesar un segmento.

    Retorna:
        (pd.DataFrame, list): DataFrame con la columna 'cluster' asignada y una lista de errores ocurridos.
    """
    # Validar columnas requeridas
    if not {"res_8", "latitude", "longitude"}.issubset(df.columns):
        raise ValueError("El DataFrame debe contener las columnas 'res_8', 'latitude' y 'longitude'")

    # Copia segura del DataFrame original
    df = df.copy()

    start_time = time.time()

    all_labels = -np.ones(len(df), dtype=np.int32)
    cluster_counter = 0
    errores = []

    for res8_id, group in df.groupby("res_8"):
        try:
            n = len(group)
            if n < min_segment or n > max_segment:
                continue  # evita segmentos extremos

            coords_seg = group[["latitude", "longitude"]].to_numpy(dtype=np.float64)
            coords_seg = np.ascontiguousarray(coords_seg)  # muy importante

            labels_seg = dbscan(coords_seg, eps=eps, min_samples=min_samples)

            labels_seg = np.where(labels_seg != -1, labels_seg + cluster_counter, -1)
            all_labels[group.index] = labels_seg

            max_label = int(labels_seg.max())
            if max_label > -1:
                cluster_counter += max_label + 1
        except Exception as e:
            errores.append((res8_id, str(e)))

    df["cluster"] = all_labels
    print(f"Segmentación completada en {time.time() - start_time:.2f} segundos")
    return df, errores
