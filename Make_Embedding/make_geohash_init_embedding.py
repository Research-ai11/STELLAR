from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import geopandas as gpd
import geohash2
from shapely.geometry import Polygon


# -------------------------
# I/O
# -------------------------
def read_data(dataset_name: str, file_name: str, data_root: Path) -> pd.DataFrame:
    """Read CSV from {data_root}/{dataset_name}/raw/{file_name}."""
    data_path = data_root / dataset_name / "raw" / file_name
    return pd.read_csv(data_path)


# -------------------------
# Time utilities
# -------------------------
def assign_time_period(hour: int) -> str:
    """Assign a coarse time period label from an hour (0-23)."""
    if 5 <= hour < 10:
        return "morning"
    elif 10 <= hour < 16:
        return "afternoon"
    elif 16 <= hour < 21:
        return "evening"
    else:
        return "night"


def assign_timezone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'hour' and 'time_period' columns based on 'LocalTime'.
    This function returns a copy to avoid side-effects on the input.
    """
    out = df.copy()
    out["hour"] = pd.to_datetime(out["LocalTime"]).dt.hour
    out["time_period"] = out["hour"].apply(assign_time_period)
    return out


# -------------------------
# Geohash utilities
# -------------------------
def geohash_to_polygon(gh: str) -> Polygon:
    """Convert a geohash to its bounding box polygon."""
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


def df_to_geohash_gdf(
    df: pd.DataFrame,
    *,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    precision: int = 6,
) -> gpd.GeoDataFrame:
    """
    Map POI coordinates to geohash cells and build a GeoDataFrame that contains
    per-geohash category distribution ratios, a major category, and polygon geometry.

    Note:
      - The original behavior (category distribution computed from the given df subset)
        is preserved to keep outputs identical.

    Returns:
      GeoDataFrame with columns:
        - geohash
        - [category ratio columns...]
        - count
        - major_category
        - geometry (geohash polygon)
    """
    tmp = df.copy()

    # 1) Geohash encoding
    tmp["geohash"] = tmp.apply(
        lambda row: geohash2.encode(row[lat_col], row[lon_col], precision=precision), axis=1
    )

    # 2) Category distribution per geohash
    all_categories = sorted(tmp["UpperCategory"].dropna().unique())
    cat_count = tmp.groupby(["geohash", "UpperCategory"]).size().unstack(fill_value=0)
    cat_count.columns.name = None

    cat_ratio = cat_count.div(cat_count.sum(axis=1), axis=0)

    # Ensure all categories exist as columns (kept as original)
    for cat in all_categories:
        if cat not in cat_ratio.columns:
            cat_ratio[cat] = 0.0
    cat_ratio = cat_ratio[all_categories]

    # 3) Add count, major_category, geometry
    cat_ratio["count"] = cat_count.sum(axis=1)
    cat_ratio["major_category"] = cat_ratio[all_categories].idxmax(axis=1)
    cat_ratio["geometry"] = cat_ratio.index.map(geohash_to_polygon)

    # 4) Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(cat_ratio.reset_index(), geometry="geometry", crs="EPSG:4326")
    return gdf


# -------------------------
# Feature builders
# -------------------------
def get_time_gdf(df: pd.DataFrame, *, precision: int = 6) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create geohash GeoDataFrames for each time period."""
    df = assign_timezone(df)

    df_morning = df.loc[df["time_period"] == "morning"].copy()
    df_afternoon = df.loc[df["time_period"] == "afternoon"].copy()
    df_evening = df.loc[df["time_period"] == "evening"].copy()
    df_night = df.loc[df["time_period"] == "night"].copy()

    gdf_morning = df_to_geohash_gdf(df_morning, precision=precision)
    gdf_afternoon = df_to_geohash_gdf(df_afternoon, precision=precision)
    gdf_evening = df_to_geohash_gdf(df_evening, precision=precision)
    gdf_night = df_to_geohash_gdf(df_night, precision=precision)

    return gdf_morning, gdf_afternoon, gdf_evening, gdf_night


def get_holiday_gdf(df: pd.DataFrame, *, precision: int = 6) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create geohash GeoDataFrames for weekday vs holiday subsets."""
    df = assign_timezone(df)

    df_weekday = df.loc[df["Holiday"] == False].copy()
    df_holiday = df.loc[df["Holiday"] == True].copy()

    gdf_weekday = df_to_geohash_gdf(df_weekday, precision=precision)
    gdf_holiday = df_to_geohash_gdf(df_holiday, precision=precision)

    return gdf_weekday, gdf_holiday


def get_major_category_embedding(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split out major category columns into a separate dataframe and drop them from df.

    Returns:
      (df_without_major_category_cols, major_category_df)
    """
    category_cols = ["geohash"] + [col for col in df.columns if "major_category" in col]
    out = df.copy()
    out[category_cols] = out[category_cols].astype(str)

    main_cat_df = out[category_cols].copy()
    out = out.drop(columns=category_cols[1:])
    return out, main_cat_df


# -------------------------
# Main entry
# -------------------------
def make_geohash_timebase_node(
    dataset_name: str,
    *,
    project_root: Optional[Path] = None,
    geohash_precision: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the initial geohash embedding (time/weekday/holiday category distributions)
    and merge with road-based accessibility node features.

    Output files (same as original):
      - data/{dataset}/graph/{dataset}_geohash_basic_embedding.csv
      - data/{dataset}/graph/{dataset}_main_category_embedding.csv
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    data_root = project_root / "data"
    graph_dir = data_root / dataset_name / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Train data
    train_data = read_data(dataset_name, f"{dataset_name.upper()}_train.csv", data_root=data_root)

    # Geohash accessibility features (from road graph)
    acc_path = graph_dir / f"{dataset_name}_geohash_node_features.csv"
    acc_df = pd.read_csv(acc_path)

    print(f"{dataset_name} Train data shape: {train_data.shape}")

    # Time-period and weekday/holiday GeoDataFrames
    gdf_morning, gdf_afternoon, gdf_evening, gdf_night = get_time_gdf(train_data, precision=geohash_precision)
    gdf_weekday, gdf_holiday = get_holiday_gdf(train_data, precision=geohash_precision)
    print(f"{dataset_name} Time-based GeoDataFrames created.")
    print(f"morning: {gdf_morning.shape}, afternoon: {gdf_afternoon.shape}, evening: {gdf_evening.shape}, night: {gdf_night.shape}")
    print(f"weekday: {gdf_weekday.shape}, holiday: {gdf_holiday.shape}")

    # Weekday & Holiday embedding
    wh_df = pd.merge(
        gdf_weekday.iloc[:, :-1],
        gdf_holiday.iloc[:, :-1],
        on="geohash",
        how="outer",
        suffixes=("_weekday", "_holiday"),
    )
    print("Weekday and Holiday embedding shape:", wh_df.shape)

    # Time-period embedding
    print(gdf_morning.shape, gdf_afternoon.shape, gdf_evening.shape, gdf_night.shape)
    ma_df = pd.merge(
        gdf_morning.iloc[:, :-1],
        gdf_afternoon.iloc[:, :-1],
        on="geohash",
        how="outer",
        suffixes=("_morning", "_afternoon"),
    )
    en_df = pd.merge(
        gdf_evening.iloc[:, :-1],
        gdf_night.iloc[:, :-1],
        on="geohash",
        how="outer",
        suffixes=("_evening", "_night"),
    )
    time_df = pd.merge(ma_df, en_df, on="geohash", how="outer")
    print("Time-based embedding shape:", time_df.shape)

    # Final time-based embedding composition
    time_fun_df = pd.merge(wh_df, time_df, on="geohash", how="outer")
    time_fun_df.fillna(0, inplace=True)
    print(f"{dataset_name} Final time-based geohash embedding shape:", time_fun_df.shape)
    print(f"{dataset_name} Geohash Accessibility shape:", acc_df.shape)

    # Zero preprocessing for major_category columns
    category_cols = [col for col in time_fun_df.columns if "major_category" in col]
    for col in category_cols:
        time_fun_df[col] = time_fun_df[col].astype(str)
        time_fun_df[col] = time_fun_df[col].replace({"0": "Unknown", "0.0": "Unknown", "nan": "Unknown"})

    time_fun_df, main_cat_df = get_major_category_embedding(time_fun_df)
    print(f"{dataset_name} Major category dataframe shape:", main_cat_df.shape)

    # Merge with accessibility features
    final_geohash_embedding = pd.merge(time_fun_df, acc_df, on="geohash", how="outer")
    final_geohash_embedding.fillna(0, inplace=True)
    print(f"{dataset_name} Final geohash embedding shape:", final_geohash_embedding.shape)

    # Save outputs (same paths as original)
    final_geohash_embedding.to_csv(graph_dir / f"{dataset_name}_geohash_basic_embedding.csv", index=False)
    main_cat_df.to_csv(graph_dir / f"{dataset_name}_main_category_embedding.csv", index=False)

    return final_geohash_embedding, main_cat_df


if __name__ == "__main__":
    make_geohash_timebase_node("nyc")
    make_geohash_timebase_node("tky")
