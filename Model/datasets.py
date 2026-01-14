# Model/datasets.py
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set, Tuple

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

# TrajectoryId format: "{user}_{traj_n}"
TRAJ_PATTERN = re.compile(r"^(.+?)_(\d+)$")


def parse_traj_id(traj_id: str) -> Tuple[int, int]:
    """
    Parse a trajectory identifier into (user_id, trajectory_index).

    Expected format:
        "{user}_{traj_n}" (e.g., "123_7")

    Returns:
        user_id (int), traj_n (int)

    Raises:
        ValueError: if the format does not match.
    """
    m = TRAJ_PATTERN.match(str(traj_id))
    if not m:
        raise ValueError(f"Invalid TrajectoryId format: {traj_id}")

    # NOTE: Kept consistent with the original behavior (user is cast to int).
    user_id = int(m.group(1))
    traj_n = int(m.group(2))
    return user_id, traj_n


def build_prev_traj_mapping(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a mapping from each trajectory to its immediate previous trajectory.

    For TrajectoryId = "{user}_{traj_n}", for traj_n >= 2, map to "{user}_{traj_n-1}"
    only if the previous trajectory actually exists in the provided dataframe.

    Important:
        - This mapping is built from the *full* dataframe (not split-filtered),
          so that previous trajectories may come from other splits.
    """
    traj_ids = df["TrajectoryId"].astype(str).unique().tolist()
    exist = set(traj_ids)

    mapping: Dict[str, str] = {}
    for tid in traj_ids:
        user, n = parse_traj_id(tid)
        if n >= 2:
            prev_tid = f"{user}_{n - 1}"
            if prev_tid in exist:
                mapping[tid] = prev_tid
    return mapping


class TrajectoryDataset(Dataset):
    """
    Convert raw trajectory check-ins into training-ready sequences.

    Each sample returns:
        - traj_id: str
        - user_idx: int (indexed via user_id2idx, padding not used here)
        - input_seq:  List[(poi_idx, (weekday, time_frac, holiday))]
        - label_seq:  List[(poi_idx, (weekday, time_frac, holiday))]
        - prev_seq:   List[(poi_idx, weekday, time_frac, holiday)] from the previous trajectory
                      (or a single padded element if previous trajectory is missing)

    Notes:
        - Only POIs present in `poi_id2idx` are retained.
        - Short trajectories are filtered by `short_traj_thres` (minimum transitions).
        - Users not in `user_set` are dropped (typically users not in training set).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split_tag: str,
        user_set: Set[int],
        user_id2idx: Dict[int, int],
        poi_id2idx: Dict[int, int],
        short_traj_thres: int,
        sort_by: str = "traj_n",
        time_col: str = "LocalTime",
        prev_pad_val: int = 0,
    ):
        self.split_tag = split_tag

        self.traj_seqs: List[str] = []
        self.user_idxs: List[int] = []
        self.input_seqs: List[List[Tuple[int, Tuple[int, float, bool]]]] = []
        self.label_seqs: List[List[Tuple[int, Tuple[int, float, bool]]]] = []

        # Previous-trajectory sequences (list of tuples):
        # [(poi_idx, weekday, time_frac, holiday), ...]
        self.prev_poi_seqs: List[List[Tuple[int, int, float, bool]]] = []
        self.prev_pad_val = prev_pad_val

        # ---------------------------------------------------------------------
        # 0) Filter by split
        # ---------------------------------------------------------------------
        use_df = df[df["SplitTag"] == split_tag].copy()
        use_df["TrajectoryId"] = use_df["TrajectoryId"].astype(str)

        # ---------------------------------------------------------------------
        # 1) Previous trajectory mapping (built on full df)
        # ---------------------------------------------------------------------
        prev_mapping = build_prev_traj_mapping(df)

        # ---------------------------------------------------------------------
        # 2) Grouping caches to avoid repeated slicing
        # ---------------------------------------------------------------------
        grouped_all = df.groupby("TrajectoryId", sort=False)       # full data
        grouped_use = use_df.groupby("TrajectoryId", sort=False)   # split-filtered

        # ---------------------------------------------------------------------
        # 3) Build ordering keys for sorting trajectories per user
        # ---------------------------------------------------------------------
        order_key_traj_n: Dict[str, Tuple[int, int]] = {}
        order_key_time: Dict[str, Tuple[int, pd.Timestamp]] = {}

        if sort_by == "traj_n":
            for tid in grouped_all.groups.keys():
                u, n = parse_traj_id(tid)
                order_key_traj_n[tid] = (u, n)

        elif sort_by == "time":
            # Use a local view for datetime conversion to avoid mutating the caller's df
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df_time = df.copy()
                df_time[time_col] = pd.to_datetime(df_time[time_col], errors="coerce")
                grouped_all_time = df_time.groupby("TrajectoryId", sort=False)
            else:
                grouped_all_time = grouped_all

            for tid, tdf in grouped_all_time:
                u, _ = parse_traj_id(tid)
                tmin = tdf[time_col].min()
                order_key_time[tid] = (u, tmin)

        else:
            raise ValueError("sort_by must be 'traj_n' or 'time'")

        # ---------------------------------------------------------------------
        # 4) Collect trajectory ids per user (only for the selected split)
        # ---------------------------------------------------------------------
        user2tids: DefaultDict[int, List[str]] = defaultdict(list)
        for tid in grouped_use.groups.keys():
            u, _ = parse_traj_id(tid)
            user2tids[u].append(tid)

        # Sort trajectories within each user
        for u, tids in user2tids.items():
            if sort_by == "traj_n":
                tids.sort(key=lambda t: order_key_traj_n[t][1])  # ascending traj_n
            else:
                tids.sort(key=lambda t: order_key_time[t][1])    # ascending start time
            user2tids[u] = tids

        dropped_user = 0
        dropped_short = 0

        # ---------------------------------------------------------------------
        # 5) Build sequences
        # ---------------------------------------------------------------------
        for user, tids in tqdm(user2tids.items(), desc=f"Building {split_tag} dataset"):
            # Drop users not in user_set (e.g., not in train users)
            if user not in user_set:
                dropped_user += len(tids)
                continue

            # Map raw user id to contiguous user index
            user_idx = user_id2idx[user]

            for tid in tids:
                traj_df = grouped_use.get_group(tid)

                # Keep only POIs that exist in the training POI vocabulary
                tmp = [
                    (poi_id2idx[p], w, float(t), bool(hol))
                    for p, w, t, hol in zip(
                        traj_df["PoiId"],
                        traj_df["Weekday"],
                        traj_df["TimeFraction"],
                        traj_df["Holiday"],
                    )
                    if p in poi_id2idx
                ]

                # Require minimum number of transitions: (len(pois) - 1) >= short_traj_thres
                if (len(tmp) - 1) < short_traj_thres:
                    dropped_short += 1
                    continue

                poi_idxs = [x[0] for x in tmp]
                weekdays = [x[1] for x in tmp]
                times = [x[2] for x in tmp]
                holidays = [x[3] for x in tmp]

                # Build input/label sequences (next-step prediction)
                input_seq: List[Tuple[int, Tuple[int, float, bool]]] = []
                label_seq: List[Tuple[int, Tuple[int, float, bool]]] = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], (weekdays[i], times[i], holidays[i])))
                    label_seq.append((poi_idxs[i + 1], (weekdays[i + 1], times[i + 1], holidays[i + 1])))

                # Build previous-trajectory sequence
                prev_seq: List[Tuple[int, int, float, bool]] = []
                if tid in prev_mapping:
                    prev_tid = prev_mapping[tid]
                    prev_df = grouped_all.get_group(prev_tid)

                    prev_seq = [
                        (poi_id2idx[p], w, float(t), bool(hol))
                        for p, w, t, hol in zip(
                            prev_df["PoiId"],
                            prev_df["Weekday"],
                            prev_df["TimeFraction"],
                            prev_df["Holiday"],
                        )
                        if p in poi_id2idx
                    ]

                # If no previous trajectory exists, use a single padded element
                if len(prev_seq) == 0:
                    prev_seq = [(self.prev_pad_val, 0, 0.0, False)]

                # Store
                self.traj_seqs.append(tid)
                self.user_idxs.append(user_idx)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.prev_poi_seqs.append(prev_seq)

        logger.info(f"[{split_tag}] dropped_user={dropped_user}, dropped_short={dropped_short}")

    def __len__(self) -> int:
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs) == len(self.prev_poi_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index: int):
        return (
            self.traj_seqs[index],
            self.user_idxs[index],
            self.input_seqs[index],
            self.label_seqs[index],
            self.prev_poi_seqs[index],
        )
