from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# -------------------------
# Config
# -------------------------
HistogramNorm = Literal["prob", "l1"]


@dataclass(frozen=True)
class POIInitEmbeddingConfig:
    split_tag: str = "train"
    hour_col: str = "Hour"
    poi_col: str = "PoiId"
    cat_col: str = "PoiCategoryId"
    lat_col: str = "Latitude"
    lon_col: str = "Longitude"
    time_col: str = "LocalTime"
    laplace_alpha: float = 1.0
    hist_norm: HistogramNorm = "prob"


@dataclass(frozen=True)
class TrajGraphConfig:
    split_col: str = "SplitTag"
    split_value: str = "train"
    user_col: str = "UserId"
    traj_col: str = "TrajectoryId"
    poi_col: str = "PoiId"
    time_col: str = "LocalTime"
    drop_self_loops: bool = False
    min_weight: int = 1
    log1p_weight: bool = True


# -------------------------
# I/O
# -------------------------
def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file. Raise an error if the file does not exist or is unreadable."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def get_project_root(explicit: Optional[str] = None) -> Path:
    """
    Resolve project root (STELLAR).
    Assumes this file is located at: STELLAR/Make_Embedding/make_poi_init_embedding.py
    """
    if explicit is not None:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[1]


# -------------------------
# POI base embedding
# -------------------------
def _mode(series: pd.Series):
    """Return the most frequent value (mode). If empty, return None."""
    cnt = Counter(series.dropna())
    if not cnt:
        return None
    return cnt.most_common(1)[0][0]


def _ensure_hour_column(df: pd.DataFrame, *, hour_col: str, time_col: str) -> pd.DataFrame:
    """
    Ensure `hour_col` exists and is valid (0~23).
    If missing or contains NaN, derive it from `time_col`.
    """
    out = df.copy()

    need_hour = hour_col not in out.columns or out[hour_col].isna().any()
    if need_hour:
        if time_col not in out.columns:
            raise ValueError(f"Either '{hour_col}' or '{time_col}' column must exist.")
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        out = out.dropna(subset=[time_col])
        out[hour_col] = out[time_col].dt.hour

    out[hour_col] = out[hour_col].astype(int).clip(0, 23)
    return out


def build_poi_base_embedding(
    df: pd.DataFrame,
    *,
    cfg: POIInitEmbeddingConfig = POIInitEmbeddingConfig(),
) -> pd.DataFrame:
    """
    Build POI initial features using:
      - (normalized) latitude/longitude
      - POI category (mode)
      - log1p check-in count (normalized)
      - 24-dim hourly check-in histogram (Laplace-smoothed; normalized)

    Returns a DataFrame with columns:
      PoiId, Latitude, Longitude, PoiCategoryId, hour_00..hour_23, LogCheckinCount
    """
    # Use only the specified split (default: train)
    df = df.loc[df["SplitTag"] == cfg.split_tag].copy()

    # Ensure Hour column
    df = _ensure_hour_column(df, hour_col=cfg.hour_col, time_col=cfg.time_col)

    # Aggregate static POI attributes
    agg_lat = df.groupby(cfg.poi_col)[cfg.lat_col].mean()
    agg_lon = df.groupby(cfg.poi_col)[cfg.lon_col].mean()
    agg_cat = df.groupby(cfg.poi_col)[cfg.cat_col].agg(_mode)
    agg_cnt = df.groupby(cfg.poi_col).size().rename("CheckinCount")

    # Hourly histogram features
    def _hour_hist(g: pd.DataFrame) -> pd.Series:
        hist = np.bincount(g[cfg.hour_col].values, minlength=24).astype(np.float32)

        if cfg.laplace_alpha > 0:
            hist += cfg.laplace_alpha

        if cfg.hist_norm == "prob":
            hist = hist / hist.sum()
        elif cfg.hist_norm == "l1":
            s = np.abs(hist).sum()
            hist = hist / s if s > 0 else hist
        else:
            raise ValueError(f"Unsupported hist_norm: {cfg.hist_norm}")

        return pd.Series({f"hour_{h:02d}": float(hist[h]) for h in range(24)})

    hour_feats = df.groupby(cfg.poi_col).apply(_hour_hist)

    # Combine all features (keep original naming)
    feats = pd.concat([agg_lat, agg_lon, agg_cat, agg_cnt, hour_feats], axis=1).reset_index()
    feats.rename(
        columns={
            cfg.poi_col: "PoiId",
            cfg.lat_col: "Latitude",
            cfg.lon_col: "Longitude",
            cfg.cat_col: "PoiCategoryId",
        },
        inplace=True,
    )

    # Long-tail adjustment for count
    feats["LogCheckinCount"] = np.log1p(feats["CheckinCount"])
    feats.drop(columns=["CheckinCount"], inplace=True)

    # Normalize numeric columns (same as original)
    scaler = MinMaxScaler()
    cols_to_scale = ["LogCheckinCount", "Latitude", "Longitude"]
    feats[cols_to_scale] = scaler.fit_transform(feats[cols_to_scale])

    return feats


# -------------------------
# POI trajectory graph
# -------------------------
def build_poi_traj_graph(
    df: pd.DataFrame,
    *,
    cfg: TrajGraphConfig = TrajGraphConfig(),
) -> pd.DataFrame:
    """
    Build a directed POI transition graph from trajectories:
      - Edge (p_i -> p_{i+1}) within the same trajectory, sorted by time.
      - Weight = frequency of transitions in the chosen split.
      - Optional: drop self-loops, minimum weight filtering, log1p transform.
    """
    use_cols = [cfg.user_col, cfg.poi_col, cfg.traj_col, cfg.time_col, cfg.split_col]
    gdf = df.loc[df[cfg.split_col] == cfg.split_value, use_cols].copy()

    gdf[cfg.time_col] = pd.to_datetime(gdf[cfg.time_col], errors="coerce")
    gdf = gdf.dropna(subset=[cfg.time_col]).sort_values([cfg.traj_col, cfg.time_col, cfg.poi_col])

    gdf["next_poi"] = gdf.groupby(cfg.traj_col)[cfg.poi_col].shift(-1)
    edges = gdf.dropna(subset=["next_poi"])[[cfg.poi_col, "next_poi"]].copy()
    edges.columns = ["src", "dst"]

    if cfg.drop_self_loops:
        edges = edges[edges["src"] != edges["dst"]]

    edges_df = (
        edges.groupby(["src", "dst"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
        .sort_values(["src", "dst"])
        .reset_index(drop=True)
    )

    if cfg.min_weight > 1:
        edges_df = edges_df[edges_df["weight"] >= cfg.min_weight].reset_index(drop=True)

    if cfg.log1p_weight:
        edges_df["weight"] = edges_df["weight"].apply(np.log1p)

    return edges_df


# -------------------------
# Script entry
# -------------------------
def run_for_dataset(dataset_name: str, *, project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end generation for one dataset:
      - POI base features -> data/{dataset}/graph/{dataset}_poi_feat.csv
      - POI trajectory graph -> data/{dataset}/graph/{dataset}_traj_graph.csv
    """
    data_root = project_root / "data"
    raw_path = data_root / dataset_name / "raw" / f"{dataset_name.upper()}.csv"
    graph_dir = data_root / dataset_name / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv(raw_path)

    poi_feats = build_poi_base_embedding(df, cfg=POIInitEmbeddingConfig(split_tag="train"))
    traj_graph = build_poi_traj_graph(df, cfg=TrajGraphConfig(split_value="train"))

    print(
    f"[{dataset_name.upper()}] POI init embedding: {poi_feats.shape} "
    f"(#POIs={poi_feats['PoiId'].nunique()}, dim={poi_feats.shape[1] - 1})"
    )
    print(
        f"[{dataset_name.upper()}] POI traj graph: {traj_graph.shape} "
        f"(#edges={len(traj_graph)}, avg_w={traj_graph['weight'].mean():.4f})"
    )

    poi_feats.to_csv(graph_dir / f"{dataset_name}_poi_feat.csv", index=False)
    traj_graph.to_csv(graph_dir / f"{dataset_name}_traj_graph.csv", index=False)

    return poi_feats, traj_graph


def main():
    project_root = get_project_root()

    # NYC / TKY
    run_for_dataset("nyc", project_root=project_root)
    run_for_dataset("tky", project_root=project_root)


if __name__ == "__main__":
    main()
