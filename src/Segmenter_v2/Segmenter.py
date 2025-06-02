# idea pricipal 2.0
"""
poder crear json con clave valor, de res_8
es decir hacer una segmentacion de hexagonos de la data de res_8 con k.ring(origin, 1 ) que genera 6 vecinos cercanos, y ir clasificando cada hexagono con un valor de 0 a 5, donde 0 es el centro y 1-5 son los vecinos cercanos pero todo como 0 el central y 1-5 los vecinos cercanos, unicamente con k.ring(origin,1)
"""

"""
origin = '881f222c31fffff' # un hexagono de h3 existente

test = h3.grid_disk(origin, 1)

print(f"Hexágonos vecinos de {origin} a distancia 1: {test}")
"""
import h3
import json
import dask.dataframe as dd
import pandas as pd

def get_res8_parent(hexagon):
    """
    Obtiene el hexágono padre de resolución 8 para un hexágono dado.
    """
    father = h3.cell_to_parent(hexagon, 7)
    return father

def get_res7_children(hexagon):
    """
    Obtiene los hexágonos hijos de resolución 7 para un hexágono dado.
    """
    children = list(h3.cell_to_children(hexagon))
    return children


def build_matrix_from_res8(data):
    res8_to_res15 = data.groupby("res_8")["H3_int_index_15"].apply(set).to_dict()

    latlon_map = data.groupby("H3_int_index_15").agg({
        "latitude": "first",
        "longitude": "first"
    }).to_dict("index")

    rows = []
    visited_res7 = set()
    segment_id = 0

    for res_8 in res8_to_res15.keys():
        res_7 = get_res8_parent(res_8)
        if res_7 in visited_res7:
            continue

        visited_res7.add(res_7)
        children_res8 = get_res7_children(res_7)

        for child in children_res8:
            if child in res8_to_res15:
                for res15 in res8_to_res15[child]:
                    if res15 in latlon_map:
                        rows.append({
                            "res_7": res_7,
                            "res_8": child,
                            "segment_id": segment_id,
                            "res_15_id": res15,
                            "lat": latlon_map[res15]["latitude"],
                            "lon": latlon_map[res15]["longitude"]
                        })
            else:
                # Añadir res_8 vacío con el segment_id correspondiente
                rows.append({
                    "res_7": res_7,
                    "res_8": child,
                    "segment_id": segment_id,
                    "res_15_id": None,
                    "lat": None,
                    "lon": None
                })

            segment_id += 1

    return dd.from_pandas(pd.DataFrame(rows), npartitions=4)


