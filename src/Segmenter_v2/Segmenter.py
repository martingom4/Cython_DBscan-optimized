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

def get_res8_neighbors(hexagon, k=1):
    return list(h3.grid_disk(hexagon, k))

def build_segment_matrix_index(data):
    segment_index = {}
    segmenter_count = 0

    # Mapea cada res_8 a sus res_15
    res8_to_res15 = data.groupby("res_8")["H3_int_index_15"].apply(set).to_dict()
    origin_hexes = list(res8_to_res15.keys())

    unique_res15 = data["H3_int_index_15"].unique()
    res15_to_latlon = {res15: {"id": int(res15), "lat": h3.cell_to_latlng(h3.int_to_str(res15))[0], "lon": h3.cell_to_latlng(h3.int_to_str(res15))[1]} for res15 in unique_res15}

    for hexagon in origin_hexes:
        neighbors = get_res8_neighbors(hexagon, k=1)
        neighbors_list = []
        for neighbor in neighbors:
            neighbors_list.append({
                "res_8": neighbor,
                "res_15_points": [res15_to_latlon[res15] for res15 in res8_to_res15.get(neighbor, [])]
            })

        segment_index[hexagon] = {
            "segment_id": segmenter_count,
            "neighbors": neighbors_list
        }
        segmenter_count += 1

    return segment_index

def save_segment_index_to_json(segment_index):
    with open("segment_index.json", "w") as f:
        json.dump(segment_index, f, indent=2)
