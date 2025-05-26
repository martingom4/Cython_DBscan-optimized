import pandas as pd
from collections import defaultdict

def segment_by_h3(df: pd.DataFrame, res_column="res_8") -> dict:
    """Segmentación granular: una celda H3 = un segmento."""
    segments = defaultdict(pd.DataFrame)
    grouped = df.groupby(res_column)
    for h3_cell, group in grouped:
        segments[h3_cell] = group.reset_index(drop=True)
    return segments

def segment_by_prefix(df: pd.DataFrame, res_column="res_8", prefix_len=5) -> dict:
    """Segmentación por prefijo: agrupa por primeros N caracteres del H3."""
    segments = defaultdict(pd.DataFrame)
    df["h3_prefix"] = df[res_column].astype(str).str[:prefix_len]
    grouped = df.groupby("h3_prefix")
    for prefix, group in grouped:
        segments[prefix] = group.reset_index(drop=True)
    return segments
