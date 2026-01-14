# Make_Embedding/make_user_neighbors.py
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
import geohash2


def read_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def build_space_table(geohash_embedding_path: str) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Load geohash space embeddings as a lookup table (with padding row at index 0).

    Returns:
        space_table: [num_geohashes+1, dim] (0th row is padding)
        geohash2idx: mapping geohash -> index (starts from 1)
    """
    df = pd.read_csv(geohash_embedding_path)
    df = df.rename(columns={df.columns[0]: "geohash"}).sort_values("geohash").reset_index(drop=True)

    geohash2idx = {gh: i + 1 for i, gh in enumerate(df["geohash"].tolist())}  # 0 is padding
    table = torch.tensor(df[[c for c in df.columns if c != "geohash"]].values, dtype=torch.float)
    space_table = torch.cat([torch.zeros(1, table.shape[1]), table], dim=0)  # padding row
    return space_table, geohash2idx


def build_mappings(train_df: pd.DataFrame, geohash2idx: Dict[str, int]):
    """
    Build id->index mappings used in neighbor construction.

    NOTE: Behavior is kept identical to your original implementation.
    """
    poi_ids = sorted(train_df["PoiId"].unique().tolist())
    cat_ids = sorted(train_df["PoiCategoryId"].unique().tolist())
    user_ids = sorted(train_df["UserId"].unique().tolist())

    user_set = set(user_ids)

    poi_id2idx = {pid: i + 1 for i, pid in enumerate(poi_ids)}
    cat_id2idx = {cid: i + 1 for i, cid in enumerate(cat_ids)}
    user_id2idx = {uid: i + 1 for i, uid in enumerate(user_ids)}

    poi_info = (
        train_df.loc[:, ["PoiId", "PoiCategoryId", "Latitude", "Longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    poi_info["geohash"] = poi_info.apply(
        lambda r: geohash2.encode(r["Latitude"], r["Longitude"], precision=6), axis=1
    )
    poi_info["geohash_idx"] = poi_info["geohash"].map(geohash2idx).fillna(0).astype(int)

    poi_idx2cat_idx = {
        poi_id2idx[pid]: cat_id2idx[cid]
        for pid, cid in zip(poi_info["PoiId"], poi_info["PoiCategoryId"])
        if pid in poi_id2idx and cid in cat_id2idx
    }
    poi_idx2geo_idx = {poi_id2idx[pid]: gid for pid, gid in zip(poi_info["PoiId"], poi_info["geohash_idx"])}

    poi_idx2latlon = {
        poi_id2idx[row["PoiId"]]: (row["Latitude"], row["Longitude"])
        for _, row in poi_info.iterrows()
        if row["PoiId"] in poi_id2idx
    }

    sizes = dict(num_pois=len(poi_ids), num_users=len(user_ids), num_cats=len(cat_ids))
    mappings = dict(
        user_id2idx=user_id2idx,
        poi_id2idx=poi_id2idx,
        cat_id2idx=cat_id2idx,
        geohash2idx=geohash2idx,
        poi_idx2cat_idx=poi_idx2cat_idx,
        poi_idx2geo_idx=poi_idx2geo_idx,
        poi_idx2latlon=poi_idx2latlon,
    )
    return sizes, mappings, user_set


def build_user_checkin_vectors(
    train_df: pd.DataFrame,
    user_id2idx: dict,
    poi_id2idx: dict,
    num_users: int,
    num_pois: int,
    user_col: str = "UserId",
    poi_col: str = "PoiId",
    device: str = "cpu",
    eps: float = 1e-8,
):
    df = train_df[[user_col, poi_col]].copy()
    df["u"] = df[user_col].map(user_id2idx)
    df["p"] = df[poi_col].map(poi_id2idx)
    df = df.dropna(subset=["u", "p"])

    u = torch.tensor(df["u"].astype(int).to_numpy(), dtype=torch.long, device=device)
    p = torch.tensor(df["p"].astype(int).to_numpy(), dtype=torch.long, device=device)

    ones = torch.ones_like(u, dtype=torch.float32, device=device)
    counts = torch.zeros((num_users, num_pois), dtype=torch.float32, device=device)

    if u.numel() > 0:
        counts.index_put_((u, p), ones, accumulate=True)

    user_activity_tensor = counts.sum(dim=1)

    row_sums = counts.sum(dim=1, keepdim=True) + eps
    probs = counts / row_sums
    return probs, user_activity_tensor


def build_user_timeslot_probs(
    train_df: pd.DataFrame,
    user_id2idx: dict,
    num_users: int,
    time_col: str = "TimeSlot96",
    num_bins: int = 96,
    user_col: str = "UserId",
    device: str = "cpu",
    eps: float = 1e-8,
):
    df = train_df[[user_col, time_col]].copy()
    df["u"] = df[user_col].map(user_id2idx)
    df["t"] = df[time_col].astype(int)
    df = df.dropna(subset=["u", "t"])

    u = torch.tensor(df["u"].astype(int).to_numpy(), dtype=torch.long, device=device)
    t_numpy = df["t"].astype(int).to_numpy()
    t_clipped = np.clip(t_numpy, 0, num_bins - 1)
    t = torch.tensor(t_clipped, dtype=torch.long, device=device)

    ones = torch.ones_like(u, dtype=torch.float32, device=device)
    counts = torch.zeros((num_users, num_bins), dtype=torch.float32, device=device)

    if u.numel() > 0:
        counts.index_put_((u, t), ones, accumulate=True)

    row_sums = counts.sum(dim=1, keepdim=True) + eps
    probs = counts / row_sums
    return probs


def build_user_geohash_probs(
    train_df: pd.DataFrame,
    user_id2idx: dict,
    poi_id2idx: dict,
    poi_idx2geo_idx,
    num_users: int,
    num_geohashes: int,
    user_col: str = "UserId",
    poi_col: str = "PoiId",
    device: str = "cpu",
    eps: float = 1e-8,
):
    df = train_df[[user_col, poi_col]].copy()
    df["g"] = df[poi_col].map(poi_id2idx).map(poi_idx2geo_idx)
    df["u"] = df[user_col].map(user_id2idx)
    df = df.dropna(subset=["u", "g"])

    u = torch.tensor(df["u"].astype(int).to_numpy(), dtype=torch.long, device=device)
    g = torch.tensor(df["g"].astype(int).to_numpy(), dtype=torch.long, device=device)

    ones = torch.ones_like(u, dtype=torch.float32, device=device)
    counts = torch.zeros((num_users, num_geohashes), dtype=torch.float32, device=device)

    if u.numel() > 0:
        counts.index_put_((u, g), ones, accumulate=True)

    row_sums = counts.sum(dim=1, keepdim=True) + eps
    probs = counts / row_sums
    return probs


def build_user_category_probs(
    train_df: pd.DataFrame,
    user_id2idx: dict,
    cat_id2idx: dict,
    num_users: int,
    num_cats: int,
    user_col: str = "UserId",
    cat_col: str = "PoiCategoryId",
    device: str = "cpu",
    eps: float = 1e-8,
):
    df = train_df[[user_col, cat_col]].copy()
    df["u"] = df[user_col].map(user_id2idx)
    df["c"] = df[cat_col].map(cat_id2idx)
    df = df.dropna(subset=["u", "c"])

    u = torch.tensor(df["u"].astype(int).to_numpy(), dtype=torch.long, device=device)
    c = torch.tensor(df["c"].astype(int).to_numpy(), dtype=torch.long, device=device)

    ones = torch.ones_like(u, dtype=torch.float32, device=device)
    counts = torch.zeros((num_users, num_cats), dtype=torch.float32, device=device)

    if u.numel() > 0:
        counts.index_put_((u, c), ones, accumulate=True)

    row_sums = counts.sum(dim=1, keepdim=True) + eps
    probs = counts / row_sums
    return probs


def quick_diag(name: str, X: torch.Tensor):
    print(f"[{name}] shape={tuple(X.shape)} dtype={X.dtype}")
    row_sums = X.sum(dim=1)
    print("  row_sums: min=", row_sums.min().item(), " max=", row_sums.max().item())
    nz_rows = (row_sums > 0).sum().item()
    print("  nonzero_rows:", nz_rows, "/", X.size(0))


def _row_l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = torch.linalg.norm(x, dim=1, keepdim=True)
    return x / (n + eps)


@torch.no_grad()
def similarity_topk_stats(X: torch.Tensor, K0: int = 64, exclude_self: bool = True):
    U = X.size(0)
    Z = _row_l2norm(X)
    S_row = Z @ Z.T

    if exclude_self:
        S_row.fill_diagonal_(-float("inf"))

    K0 = min(K0, U - 1)
    vals, idxs = torch.topk(S_row, k=K0, dim=1)

    flat = vals.flatten()
    flat = flat[torch.isfinite(flat)]

    def q(p):
        return torch.quantile(flat, torch.tensor(p, device=flat.device))

    global_stats = {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "min": flat.min().item(),
        "p25": q(0.25).item(),
        "p50": q(0.50).item(),
        "p75": q(0.75).item(),
        "p80": q(0.80).item(),
        "p90": q(0.90).item(),
        "p95": q(0.95).item(),
        "p99": q(0.99).item(),
        "max": flat.max().item(),
    }

    per_user_p80 = torch.quantile(vals, 0.80, dim=1)
    per_user_p90 = torch.quantile(vals, 0.90, dim=1)
    per_user_stats = {
        "p80_mean": per_user_p80.mean().item(),
        "p80_std": per_user_p80.std(unbiased=False).item(),
        "p90_mean": per_user_p90.mean().item(),
        "p90_std": per_user_p90.std(unbiased=False).item(),
    }

    return {"global": global_stats, "per_user": per_user_stats, "topk_vals": vals, "topk_idxs": idxs}


def print_view_similarity_reports(poi_probs, time_probs, geo_probs, cat_probs, K0=64, name_prefix="NYC"):
    views = {"POI": poi_probs, "TIME": time_probs, "GEO": geo_probs, "CAT": cat_probs}
    for vn, X in views.items():
        rep = similarity_topk_stats(X, K0=K0)
        g, pu = rep["global"], rep["per_user"]
        print(f"[{name_prefix}] View={vn} | TopK={K0}")
        print(
            f"  global: mean={g['mean']:.3f}, p50={g['p50']:.3f}, p75={g['p75']:.3f}, "
            f"p80={g['p80']:.3f}, p90={g['p90']:.3f}, p95={g['p95']:.3f}, p99={g['p99']:.3f}, max={g['max']:.3f}"
        )
        print(
            f"  per-user: p80_mean={pu['p80_mean']:.3f} (±{pu['p80_std']:.3f}), "
            f"p90_mean={pu['p90_mean']:.3f} (±{pu['p90_std']:.3f})"
        )


def analyze_hybrid_similarity_stats(
    poi_probs,
    time_probs,
    geo_probs,
    cat_probs,
    user_activity,
    topk_candidates=64,
    w_poi=0.5,
    w_time=0.2,
    w_geo=0.2,
    w_cat=0.1,
    shrink_m=100.0,
):
    """
    Compute hybrid user-user similarity and return global summary stats
    over Top-K candidate similarities (used to choose a tau threshold).

    Notes:
      - This function is intended to be run once to inspect similarity scale.
      - Behavior is kept identical to the original implementation.
    """
    U = poi_probs.size(0)

    # (1) Cosine similarity per view: S_view = normalize(X) @ normalize(X)^T
    A = _row_l2norm(poi_probs);  S_poi  = A @ A.T
    B = _row_l2norm(time_probs); S_time = B @ B.T
    C = _row_l2norm(geo_probs);  S_geo  = C @ C.T
    D = _row_l2norm(cat_probs);  S_cat  = D @ D.T

    # (2) Hybrid similarity: weighted sum across views
    S = w_poi * S_poi + w_time * S_time + w_geo * S_geo + w_cat * S_cat

    # Exclude self-similarity (diagonal) from candidate selection
    S.fill_diagonal_(float("-inf"))

    # (3) Activity-based shrinkage for cold users:
    #     sigma(u) = sqrt(n_u / (n_u + m)), then S <- sigma(u) * S * sigma(v)
    if shrink_m is not None and shrink_m > 0:
        n = user_activity.to(S.dtype).clamp_min(0)
        sigma = torch.sqrt(n / (n + shrink_m))
        S = (sigma.view(-1, 1) * S) * sigma.view(1, -1)

        # Ensure padding user (index 0) stays neutral
        S[0, :] = 0.0
        S[:, 0] = 0.0

    # (4) Top-K candidate similarities per user (first-stage candidate pool)
    vals, _ = torch.topk(S, k=min(topk_candidates, U - 1), dim=1)

    # (5) Aggregate stats over all candidate similarities
    flat = vals.flatten()
    flat = flat[torch.isfinite(flat) & (flat > 0)]  # remove -inf and non-positive

    if flat.numel() == 0:
        print("[WARN] No valid similarity values found. Returning empty stats.")
        return {}

    def q(p):
        return torch.quantile(flat, torch.tensor(p, device=flat.device))

    global_stats = {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "min": flat.min().item(),
        "p25": q(0.25).item(),
        "p50": q(0.50).item(),  # median
        "p75": q(0.75).item(),
        "p90": q(0.90).item(),
        "max": flat.max().item(),
    }
    return global_stats


def build_user_neighbors_hybrid(
    poi_probs,
    time_probs,
    geo_probs,
    cat_probs,
    user_activity,
    topk_candidates=64,
    topk_min=6,
    topk_max=12,
    w_poi=0.5,
    w_time=0.2,
    w_geo=0.2,
    w_cat=0.1,
    tau=None,
    percentile=80,
    shrink_m=100.0,
):
    """
    Build a fixed-size neighbor list per user using a hybrid similarity:

      S(u,v) = w_poi * cos(poi_u, poi_v)
             + w_time * cos(time_u, time_v)
             + w_geo  * cos(geo_u,  geo_v)
             + w_cat  * cos(cat_u,  cat_v)

    Selection procedure (kept identical to original):
      1) Compute S and take Top-K0 candidates per user.
      2) Filter by absolute threshold tau (optional).
      3) Filter by per-user percentile among remaining candidates (optional).
      4) Enforce K in [topk_min, topk_max] via backfilling from Top-K0 if needed.
      5) Normalize weights to sum to 1 for each user.
    """
    U = poi_probs.size(0)

    # (1) Cosine similarity per view
    A = _row_l2norm(poi_probs);  S_poi  = A @ A.T
    B = _row_l2norm(time_probs); S_time = B @ B.T
    C = _row_l2norm(geo_probs);  S_geo  = C @ C.T
    D = _row_l2norm(cat_probs);  S_cat  = D @ D.T

    # (2) Hybrid similarity and self-exclusion
    S = w_poi * S_poi + w_time * S_time + w_geo * S_geo + w_cat * S_cat
    S.fill_diagonal_(float("-inf"))

    # (3) Activity-based shrinkage (cold-start smoothing)
    if shrink_m is not None and shrink_m > 0:
        n = user_activity.to(S.dtype).clamp_min(0)
        sigma = torch.sqrt(n / (n + shrink_m))
        S = (sigma.view(-1, 1) * S) * sigma.view(1, -1)

        # Keep padding user 0 neutral
        S[0, :] = 0.0
        S[:, 0] = 0.0

    # (4) First-stage candidate pool
    vals, idxs = torch.topk(S, k=min(topk_candidates, U - 1), dim=1)

    # (5) Candidate filtering mask
    keep = torch.ones_like(vals, dtype=torch.bool)

    # Absolute threshold filter
    if tau is not None:
        keep &= vals >= tau

    # Per-user percentile filter (computed after tau filtering)
    if percentile is not None:
        thr = torch.quantile(
            vals.masked_fill(~keep, float("-inf")).clamp_min(-1),
            q=percentile / 100.0,
            dim=1,
            keepdim=True,
        )
        keep &= vals >= thr

    # (6) Enforce neighbor count range and compute weights
    backfilled_user_indices: List[int] = []
    neigh_idx, neigh_w = [], []

    for u in range(U):
        v_u = vals[u][keep[u]]
        i_u = idxs[u][keep[u]]

        # Sort by similarity (descending)
        if v_u.numel() > 0:
            order = torch.argsort(v_u, descending=True)
            v_u, i_u = v_u[order], i_u[order]

        # Backfill to reach topk_min (using the original Top-K0 pool)
        if v_u.numel() < topk_min:
            backfilled_user_indices.append(u)
            extra_needed = topk_min - v_u.numel()

            base_i, base_v = idxs[u], vals[u]
            taken = set(i_u.tolist())

            extras = [
                (base_v[j].item(), base_i[j].item())
                for j in range(base_i.numel())
                if base_i[j].item() not in taken
            ]
            extras.sort(reverse=True)

            for _ in range(min(extra_needed, len(extras))):
                vv, ii = extras.pop(0)
                v_u = torch.cat([v_u, torch.tensor([vv], device=S.device)])
                i_u = torch.cat([i_u, torch.tensor([ii], device=S.device)])

        # Truncate to topk_max
        if v_u.numel() > topk_max:
            v_u, i_u = v_u[:topk_max], i_u[:topk_max]

        # Clamp negatives and normalize to probability weights
        v_u = torch.clamp(v_u, min=0)
        w_u = v_u / (v_u.sum() + 1e-8)

        neigh_idx.append(i_u)
        neigh_w.append(w_u)

    # (7) Pack variable-length neighbor lists into fixed [U, K] tensors
    K = topk_max
    pad_idx = torch.zeros((U, K), dtype=torch.long, device=S.device)
    pad_w = torch.zeros((U, K), dtype=torch.float32, device=S.device)

    for u in range(U):
        k = neigh_idx[u].numel()
        pad_idx[u, :k] = neigh_idx[u]
        pad_w[u, :k] = neigh_w[u]

    return pad_idx, pad_w, backfilled_user_indices


def check_neighbors(pad_idx: torch.Tensor):
    """
    Print descriptive statistics of neighbor counts per user.
    (User index 0 is reserved for padding and excluded.)
    """
    valid_pad_idx = pad_idx[1:]  # exclude padding user 0
    neighbor_counts = (valid_pad_idx != 0).sum(dim=1).float()

    if neighbor_counts.numel() > 0:
        print("[INFO] Neighbor count statistics per user (excluding padding user 0)")
        print(f"  - mean:   {neighbor_counts.mean().item():.2f}")
        print(f"  - std:    {neighbor_counts.std().item():.2f}")
        print(f"  - min:    {neighbor_counts.min().item():.0f}")
        print(f"  - p25:    {torch.quantile(neighbor_counts, 0.25).item():.0f}")
        print(f"  - p50:    {neighbor_counts.median().item():.0f}")
        print(f"  - p75:    {torch.quantile(neighbor_counts, 0.75).item():.0f}")
        print(f"  - max:    {neighbor_counts.max().item():.0f}")
    else:
        print("[WARN] No valid users found to compute neighbor statistics.")
