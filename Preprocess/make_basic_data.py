from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from file_reader import FileReader
from preprocess_fn import (
    add_holidays_nyc,
    add_holidays_tky,
    make_uppercat,
    remove_trajectories_with_anomalies,
)

DatasetName = Literal["nyc", "tky"]

# 논문용 최종 코드

# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class PreprocessConfig:
    poi_min_freq: int = 10
    user_min_freq: int = 10


# -------------------------
# Logging
# -------------------------
def setup_logger() -> logging.Logger:
    logger = logging.getLogger("preprocess.make_base_data")

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        logger.setLevel(logging.INFO)
        logger.propagate = False  # root logger로 전달 차단

    return logger


# -------------------------
# Helpers
# -------------------------
def _raw_filename(dataset: DatasetName) -> str:
    return "dataset_TSMC2014_NYC.txt" if dataset == "nyc" else "dataset_TSMC2014_TKY.txt"


def _add_calendar_and_category_features(df: pd.DataFrame, dataset: DatasetName) -> pd.DataFrame:
    """Add holiday indicators and upper-level category features."""
    if dataset == "nyc":
        df = add_holidays_nyc(df)
        df = make_uppercat(df, opt="NYC")
    elif dataset == "tky":
        df = add_holidays_tky(df)
        df = make_uppercat(df, opt="TKY")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return df


def _log_basic_stats(logger: logging.Logger, df: pd.DataFrame, title: str) -> None:
    """Log core dataset statistics for reproducibility."""
    logger.info("%s shape: %s", title, df.shape)
    # Defensive: columns must exist
    required_cols = {"UserId", "PoiId", "TrajectoryId"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"{title} missing required columns: {missing}")

    logger.info(
        "%s | #users=%d, #pois=%d, #trajectories=%d",
        title,
        df["UserId"].nunique(),
        df["PoiId"].nunique(),
        df["TrajectoryId"].nunique(),
    )


def _split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/val/test subsets."""
    required = {"SplitTag"}
    if not required.issubset(df.columns):
        raise KeyError(f"Missing required column(s): {required - set(df.columns)}")

    train = df[df["SplitTag"] == "train"]
    val = df[df["SplitTag"] == "validation"]
    test = df[df["SplitTag"] == "test"]
    return train, val, test


def _save_splits(
    df: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
    dataset: DatasetName,
) -> None:
    """Save processed full dataframe and split files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = dataset.upper()
    df.to_csv(out_dir / f"{prefix}.csv", index=False)
    train.to_csv(out_dir / f"{prefix}_train.csv", index=False)
    val.to_csv(out_dir / f"{prefix}_val.csv", index=False)
    test.to_csv(out_dir / f"{prefix}_test.csv", index=False)


# -------------------------
# Main pipeline
# -------------------------
def preprocess_dataset(dataset: DatasetName, cfg: PreprocessConfig, project_root: Path) -> None:
    """
    End-to-end preprocessing pipeline.

    Steps (kept identical to the original logic):
      1) Load raw dataset.
      2) Filter by min POI/user frequency.
      3) Split train/val/test.
      4) Generate trajectory IDs.
      5) Remove anomalous trajectories.
      6) Filter again + split again + re-generate trajectory IDs. (Required by design.)
      7) Add holidays and upper-category features.
      8) Save processed outputs (full + split CSVs).

    Notes:
      - split_train_test() and generate_traj_id() are intentionally called twice.
    """
    logger = setup_logger()

    # Load
    df = FileReader.read_dataset(file_name=_raw_filename(dataset), dataset_name=dataset)

    # First round: filter/split/traj-id (required)
    df = FileReader.do_filter(df, poi_min_freq=cfg.poi_min_freq, user_min_freq=cfg.user_min_freq)
    df = FileReader.split_train_test(df)
    df = FileReader.generate_traj_id(df)

    _log_basic_stats(logger, df, title=f"{dataset.upper()} (before anomaly removal)")

    # Remove anomalies + second round (required)
    clean_df = remove_trajectories_with_anomalies(df)
    clean_df = FileReader.do_filter(clean_df, poi_min_freq=cfg.poi_min_freq, user_min_freq=cfg.user_min_freq)
    clean_df = FileReader.split_train_test(clean_df)
    clean_df = FileReader.generate_traj_id(clean_df)

    # Add features
    clean_df = _add_calendar_and_category_features(clean_df, dataset)

    _log_basic_stats(logger, clean_df, title=f"{dataset.upper()} (after anomaly removal)")

    # Split & log
    train, val, test = _split_df(clean_df)

    logger.info(
        "%s split shapes | train=%s, val=%s, test=%s",
        dataset.upper(),
        train.shape,
        val.shape,
        test.shape,
    )
    logger.info(
        "%s split trajectories | train=%d, val=%d, test=%d",
        dataset.upper(),
        train["TrajectoryId"].nunique(),
        val["TrajectoryId"].nunique(),
        test["TrajectoryId"].nunique(),
    )

    # Save
    out_dir = project_root / "data" / dataset / "raw"
    _save_splits(clean_df, train, val, test, out_dir=out_dir, dataset=dataset)

    logger.info("Saved processed files to: %s", out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess TSMC2014 NYC/TKY datasets.")
    parser.add_argument("--dataset", type=str, choices=["nyc", "tky", "all"], default="all")
    parser.add_argument("--poi_min_freq", type=int, default=10)
    parser.add_argument("--user_min_freq", type=int, default=10)
    parser.add_argument(
        "--project_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),  # STELLAR root (preprocess/..)
        help="Path to the project root directory (STELLAR).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(poi_min_freq=args.poi_min_freq, user_min_freq=args.user_min_freq)
    project_root = Path(args.project_root).resolve()

    if args.dataset == "all":
        preprocess_dataset("nyc", cfg, project_root)
        preprocess_dataset("tky", cfg, project_root)
    else:
        preprocess_dataset(args.dataset, cfg, project_root)


if __name__ == "__main__":
    main()
