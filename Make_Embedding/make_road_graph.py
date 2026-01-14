from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import geopandas as gpd
import geohash2
import osmnx as ox
from shapely.geometry import Polygon


# -------------------------
# I/O
# -------------------------
def read_data(dataset_name: str, file_name: str, data_root: Path) -> pd.DataFrame:
    """
    Read a preprocessed CSV file from {data_root}/{dataset_name}/raw/{file_name}.
    """
    data_path = data_root / dataset_name / "raw" / file_name
    return pd.read_csv(data_path)


# -------------------------
# Geohash utilities
# -------------------------
def geohash_to_polygon(gh: str) -> Polygon:
    """Convert a geohash string to its bounding box polygon."""
    lat, lon, lat_err, lon_err = geohash2.decode_exactly(gh)
    lat_min, lat_max = lat - lat_err, lat + lat_err
    lon_min, lon_max = lon - lon_err, lon + lon_err
    return Polygon(
        [
            (lon_min, lat_min),
            (lon_min, lat_max),
            (lon_max, lat_max),
            (lon_max, lat_min),
            (lon_min, lat_min),
        ]
    )


def point_to_geohash(point, precision: int = 6) -> str:
    """Convert a shapely Point to geohash."""
    return geohash2.encode(point.y, point.x, precision=precision)


def extract_first_if_list(val):
    """
    Extract the first element if the value is a list or a list-like string.
    Keeps original behavior for compatibility.
    """
    if isinstance(val, list):
        return val[0]
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
        except Exception:
            pass
    return val


# -------------------------
# Main builder
# -------------------------
def make_road_graph(
    dataset_name: str,
    *,
    project_root: Optional[Path] = None,
    geohash_precision: int = 6,
    plot_graph: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a geohash-level road graph from OSM data and generate:
      - node features aggregated within each geohash
      - edge list between geohashes with weights (#crossing road segments)

    Output files (kept identical to original):
      - {project_root}/data/{dataset}/graph/{dataset}_geohash_node_features.csv
      - {project_root}/data/{dataset}/graph/{dataset}_geohash_edge_features.csv
      - {project_root}/Make_Embedding/{dataset}_boundary.geojson (same as before: CWD-dependent originally)
        -> For backward compatibility, this function saves boundary geojson into the current working directory
           unless you change `boundary_out_dir` later in the pipeline.

    Notes:
      - Prints are kept as-is to allow easy output comparison with the original script.
    """
    if project_root is None:
        # STELLAR/Make_Embedding/make_road_graph.py -> project_root is STELLAR
        project_root = Path(__file__).resolve().parents[1]

    data_root = project_root / "data"
    graph_out_dir = data_root / dataset_name / "graph"
    graph_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load preprocessed data
    data = read_data(dataset_name, f"{dataset_name.upper()}.csv", data_root=data_root)

    geometry = gpd.points_from_xy(data["Longitude"], data["Latitude"])
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs="EPSG:4326")

    # 2) City boundary
    boundary = gdf.unary_union.envelope
    boundary_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[boundary], crs="EPSG:4326")

    # 3) OSM road network
    graph = ox.graph_from_polygon(boundary, network_type="drive")
    if plot_graph:
        ox.plot_graph(graph)

    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)

    road = gdf_edges[
        [
            "osmid",
            "highway",
            "lanes",
            "name",
            "oneway",
            "bridge",
            "access",
            "tunnel",
            "junction",
            "length",
            "geometry",
        ]
    ].copy()
    road = road.reset_index()
    node = gdf_nodes.reset_index()

    # Merge to get start/end node geometries
    main_road = road.merge(
        node[["osmid", "geometry"]],
        left_on="u",
        right_on="osmid",
        suffixes=("", "_start"),
    )
    main_road = main_road.merge(
        node[["osmid", "geometry"]],
        left_on="v",
        right_on="osmid",
        suffixes=("", "_end"),
    )

    main_road = main_road.loc[
        :, ["geometry_start", "geometry_end", "highway", "oneway", "bridge", "tunnel", "length"]
    ].copy()
    print(f"{dataset_name} Number of road segments: {main_road.shape[0]}")

    # 4) Geohash IDs in dataset
    data["geohash"] = data.apply(
        lambda row: geohash2.encode(row["Latitude"], row["Longitude"], precision=geohash_precision),
        axis=1,
    )

    geohash_polygons = pd.DataFrame(data["geohash"].unique(), columns=["geohash"])
    geohash_polygons["geometry"] = geohash_polygons["geohash"].apply(geohash_to_polygon)
    geohash_polygons = gpd.GeoDataFrame(geohash_polygons, geometry="geometry", crs="EPSG:4326")

    geohash_ids = geohash_polygons["geohash"].tolist()
    print(f"{dataset_name} Number of unique geohashes: {len(geohash_ids)}")

    # 5) Map road segment endpoints to geohashes
    main_road["geohash_start"] = main_road["geometry_start"].apply(
        lambda p: point_to_geohash(p, precision=geohash_precision)
    )
    main_road["geohash_end"] = main_road["geometry_end"].apply(
        lambda p: point_to_geohash(p, precision=geohash_precision)
    )

    # 6) Preprocess categorical columns (kept identical)
    main_road["highway"] = main_road["highway"].apply(extract_first_if_list)
    main_road["bridge"] = main_road["bridge"].apply(extract_first_if_list)
    main_road["tunnel"] = main_road["tunnel"].apply(extract_first_if_list)
    main_road.loc[(main_road["highway"] == "crossing"), "highway"] = "unclassified"
    main_road.loc[(main_road["bridge"] == "movable") | (main_road["bridge"] == "partial"), "bridge"] = "yes"
    main_road.loc[main_road["tunnel"] == "passage", "tunnel"] = "yes"
    main_road["bridge"] = main_road["bridge"].fillna("no")
    main_road["tunnel"] = main_road["tunnel"].fillna("no")

    cols_to_encode = ["highway", "oneway", "bridge", "tunnel"]
    main_road = pd.get_dummies(main_road, columns=cols_to_encode, prefix=cols_to_encode)
    print(f"{dataset_name} Road graph shape after one-hot encoding: {main_road.shape}")

    # 7) Node features (within geohash)
    internal_edges = main_road[main_road["geohash_start"] == main_road["geohash_end"]]

    cols_to_sum = [
        "highway_busway",
        "highway_living_street",
        "highway_motorway",
        "highway_motorway_link",
        "highway_primary",
        "highway_primary_link",
        "highway_residential",
        "highway_secondary",
        "highway_secondary_link",
        "highway_tertiary",
        "highway_tertiary_link",
        "highway_trunk",
        "highway_trunk_link",
        "highway_unclassified",
        "oneway_False",
        "oneway_True",
        "bridge_no",
        "bridge_yes",
        "tunnel_building_passage",
        "tunnel_no",
        "tunnel_yes",
    ]

    node_features = internal_edges.groupby("geohash_start")[cols_to_sum].sum().reset_index()
    node_features = node_features.rename(columns={"geohash_start": "geohash"})
    print(f"{dataset_name} Node features shape: {node_features.shape}")

    # 8) Edge list (between geohashes)
    external_edges = main_road[main_road["geohash_start"] != main_road["geohash_end"]].copy()
    external_edges["geo_pair"] = external_edges.apply(
        lambda row: tuple(sorted([row["geohash_start"], row["geohash_end"]])), axis=1
    )

    edge_list_df = external_edges.groupby("geo_pair").size().reset_index(name="weight")
    edge_list_df[["geohash_start", "geohash_end"]] = pd.DataFrame(
        edge_list_df["geo_pair"].tolist(), index=edge_list_df.index
    )
    edge_list = edge_list_df[["geohash_start", "geohash_end", "weight"]]
    print(f"{dataset_name} Edge list shape: {edge_list.shape}")

    filtered_edges = edge_list[
        edge_list["geohash_start"].isin(geohash_ids) & edge_list["geohash_end"].isin(geohash_ids)
    ].copy()

    # 9) Save (same filenames as original)
    # NOTE: original saved boundary geojson to CWD; we keep that behavior for easy comparison.
    #boundary_gdf.to_file(f"{dataset_name}_boundary.geojson", driver="GeoJSON")

    final_nodes = node_features[node_features["geohash"].isin(geohash_ids)].copy()
    final_edges = filtered_edges.loc[:, ["geohash_start", "geohash_end", "weight"]].copy()

    final_nodes.to_csv(graph_out_dir / f"{dataset_name}_geohash_node_features.csv", index=False)
    final_edges.to_csv(graph_out_dir / f"{dataset_name}_geohash_edge_features.csv", index=False)

    return final_nodes, final_edges


if __name__ == "__main__":
    make_road_graph("nyc")
    make_road_graph("tky")
