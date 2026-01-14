# Model/utils.py
from __future__ import annotations

import glob
import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, Iterator, Iterable, List, Tuple, Union

import geohash2
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from collections import Counter, defaultdict
from scipy.sparse.linalg import eigsh
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PAD_LABEL = -100


# -----------------------------------------------------------------------------
# Time feature helpers
# -----------------------------------------------------------------------------
def frac_to_sincos(frac: float) -> List[float]:
    """
    Convert a fractional value in [0,1] (e.g., normalized time-of-day) into
    a 2D sin/cos representation.
    """
    s = math.sin(2.0 * math.pi * frac)
    c = math.cos(2.0 * math.pi * frac)
    return [s, c]


def _hour_to_sin_cos(h: int) -> Tuple[float, float]:
    """
    Convert an integer hour (0..23) to sin/cos features.
    """
    theta = 2.0 * math.pi * (float(h) / 24.0)
    return float(np.sin(theta)), float(np.cos(theta))


def _time_tuple_to_feat(t: Tuple[int, float, bool]) -> Tuple[float, float, float]:
    """
    Convert a time tuple into a normalized 3D feature:
      t = (weekday:int, time_frac:float, holiday:bool)
    Returns:
      (weekday_frac, time_frac, holiday_flag)
    """
    weekday, tf, hol = t
    weekday_frac = float(weekday) / 7.0
    time_frac = float(tf)  # assumed already normalized to [0,1]
    holiday_flag = 1.0 if bool(hol) else 0.0
    return weekday_frac, time_frac, holiday_flag


# -----------------------------------------------------------------------------
# Embedding / mapping utilities
# -----------------------------------------------------------------------------
def build_space_table(geohash_embedding_path: str) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Load geohash space embeddings as a lookup table (padding row at index 0).

    Args:
        geohash_embedding_path: CSV path. First column must be geohash id.

    Returns:
        space_table: Tensor of shape [num_geohashes + 1, dim], where row 0 is padding.
        geohash2idx: dict mapping geohash string -> index (starts from 1).
    """
    df = pd.read_csv(geohash_embedding_path)
    df = df.rename(columns={df.columns[0]: "geohash"}).sort_values("geohash").reset_index(drop=True)

    geohash2idx = {gh: i + 1 for i, gh in enumerate(df["geohash"].tolist())}  # 0 is padding
    table = torch.tensor(df[[c for c in df.columns if c != "geohash"]].values, dtype=torch.float32)
    space_table = torch.cat([torch.zeros(1, table.shape[1]), table], dim=0)  # prepend padding row
    return space_table, geohash2idx


def build_mappings(train_df: pd.DataFrame, geohash2idx: Dict[str, int]):
    """
    Build ID-to-index mappings (with padding index 0) from the training split.

    NOTE:
        Behavior is intentionally kept consistent with the original implementation.

    Returns:
        sizes: dict with counts (num_pois, num_users, num_cats) excluding padding.
        mappings: dict containing id2idx and derived lookup tables.
        user_set: set of raw user IDs (for short-term preference dict usage).
    """
    poi_ids = sorted(train_df["PoiId"].unique().tolist())
    cat_ids = sorted(train_df["PoiCategoryId"].unique().tolist())
    user_ids = sorted(train_df["UserId"].unique().tolist())

    user_set = set(user_ids)

    poi_id2idx = {pid: i + 1 for i, pid in enumerate(poi_ids)}
    cat_id2idx = {cid: i + 1 for i, cid in enumerate(cat_ids)}
    user_id2idx = {uid: i + 1 for i, uid in enumerate(user_ids)}

    # Attach geohash to each POI (precision fixed to 6 to match original behavior)
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
    poi_idx2geo_idx = {
        poi_id2idx[pid]: gid
        for pid, gid in zip(poi_info["PoiId"], poi_info["geohash_idx"])
        if pid in poi_id2idx
    }
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


def mapping_to_index_tensor(
    mapping: Union[Dict[int, int], List[int], np.ndarray, torch.Tensor],
    length: int,
    device: torch.device,
) -> torch.LongTensor:
    """
    Convert an index mapping into a dense LongTensor of shape [length].

    Args:
        mapping: dict[int->int] or list/ndarray/tensor.
        length: total length including padding (e.g., num_pois + 1).
        device: target device.

    Returns:
        LongTensor of shape [length] on the given device.
    """
    if isinstance(mapping, dict):
        t = torch.zeros(length, dtype=torch.long, device=device)
        for k, v in mapping.items():
            kk = int(k)
            if 0 <= kk < length:
                t[kk] = int(v)
        return t

    if isinstance(mapping, torch.Tensor):
        return mapping.to(device=device, dtype=torch.long)

    return torch.as_tensor(mapping, device=device, dtype=torch.long)


# -----------------------------------------------------------------------------
# PyG graph utility
# -----------------------------------------------------------------------------
def df_to_pyg_data(
    df: pd.DataFrame,
    poi_id2idx: Dict[int, int],
    src_col: str = "src",
    dst_col: str = "dst",
    weight_col: str = "weight",
    device: torch.device | None = None,
) -> Data:
    """
    Convert an edge DataFrame into a PyG Data object.

    - Node indices follow poi_id2idx (padding index 0 is not expected inside edges).
    - Duplicate edges are merged via `coalesce` using mean weight (same as original code).

    Args:
        df: edge list dataframe with columns [src, dst, (weight)].
        poi_id2idx: mapping from raw POI id -> contiguous index.
        src_col, dst_col, weight_col: column names.
        device: optional torch device.

    Returns:
        PyG Data with edge_index, edge_weight, num_nodes.
    """
    s = df[src_col].map(poi_id2idx)
    t = df[dst_col].map(poi_id2idx)
    w = df[weight_col] if weight_col in df.columns else pd.Series([1.0] * len(df))

    mask = (~s.isna()) & (~t.isna())
    s = s[mask].astype(int).values
    t = t[mask].astype(int).values
    w = w[mask].astype(float).values

    edge_index = torch.tensor([s, t], dtype=torch.long)
    edge_weight = torch.tensor(w, dtype=torch.float32)

    # Merge duplicates
    edge_index, edge_weight = coalesce(
        edge_index,
        edge_weight,
        num_nodes=len(poi_id2idx),
        reduce="mean",
    )

    data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=len(poi_id2idx))
    if device is not None:
        data = data.to(device)
    return data


# -----------------------------------------------------------------------------
# Experiment utilities
# -----------------------------------------------------------------------------
def fit_delimiter(string: str = "", length: int = 80, delimiter: str = "=") -> str:
    """
    Create a centered delimiter string:
      e.g., "==== title ===="
    """
    result_len = length - len(string)
    half_len = math.floor(result_len / 2)
    return delimiter * half_len + string + delimiter * half_len


def init_torch_seeds(seed: int = 0) -> None:
    """
    Initialize random seeds for reproducibility.
    If seed==0, use deterministic mode (slower but reproducible).
    """
    torch.manual_seed(seed)
    if seed == 0:
        cudnn.benchmark, cudnn.deterministic = False, True
    else:
        cudnn.benchmark, cudnn.deterministic = True, False


def zipdir(path: str, ziph, include_format: Tuple[str, ...]) -> None:
    """
    Add files under `path` into an opened zip handle.
    Only include files whose extension is in `include_format`.
    """
    for root, _dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, ".."))
                ziph.write(filename, arcname)


def increment_path(path: str, exist_ok: bool = True, sep: str = "") -> str:
    """
    Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1, ...

    Args:
        path: base path
        exist_ok: if True, return `path` even if it exists
        sep: separator between stem and number

    Returns:
        incremented path string.
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)

    dirs = glob.glob(f"{path}{sep}*")
    matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
    idxs = [int(m.groups()[0]) for m in matches if m]
    n = max(idxs) + 1 if idxs else 2
    return f"{path}{sep}{n}"


# -----------------------------------------------------------------------------
# Feature normalization / Laplacian
# -----------------------------------------------------------------------------
def get_normalized_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features along axis=0: (X - mean) / std.

    Args:
        X: shape (num_nodes, num_features)

    Returns:
        X_norm, means, stds
    """
    means = np.mean(X, axis=0)
    X_centered = X - means.reshape((1, -1))
    stds = np.std(X_centered, axis=0)
    X_norm = X_centered / stds.reshape((1, -1))
    return X_norm, means, stds


def calculate_laplacian_matrix(adj_mat: np.ndarray, mat_type: str) -> np.ndarray:
    """
    Compute a Laplacian-related matrix used in graph neural networks.

    Supported `mat_type`:
      - 'com_lap_mat': combinatorial Laplacian (D - A)
      - 'wid_rw_normd_lap_mat': rescaled random-walk normalized Laplacian for ChebConv
      - 'hat_rw_normd_lap_mat': renormalized adjacency for GCNConv

    NOTE:
        Logic is kept consistent with the original code.
    """
    n_vertex = adj_mat.shape[0]
    deg_mat = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == "com_lap_mat":
        return deg_mat - adj_mat

    if mat_type == "wid_rw_normd_lap_mat":
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which="LM", return_eigenvectors=False)[0]
        return 2 * rw_normd_lap_mat / lambda_max_rw - id_mat

    if mat_type == "hat_rw_normd_lap_mat":
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        return np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)

    raise ValueError(f"Unknown mat_type: {mat_type}")


# -----------------------------------------------------------------------------
# Loss / metrics
# -----------------------------------------------------------------------------
def masked_mse_loss(input: torch.Tensor, target: torch.Tensor, mask_value: float = -1.0) -> torch.Tensor:
    """
    MSE computed only on entries where target != mask_value.
    """
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    return out.mean()


def top_k_acc(y_true_seq: Iterable[int], y_pred_seq: Iterable[np.ndarray], k: int) -> float:
    """
    Top-K hit rate for a list of predictions (each prediction is a score vector).
    """
    hit = 0
    y_true_seq = list(y_true_seq)
    y_pred_seq = list(y_pred_seq)

    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        top_k_rec = y_pred.argsort()[-k:][::-1]
        if np.any(top_k_rec == y_true):
            hit += 1
    return hit / max(len(y_true_seq), 1)


def mAP_metric(y_true_seq: Iterable[int], y_pred_seq: Iterable[np.ndarray], k: int) -> float:
    """
    mAP@K for Next-POI setting (one positive item per query).
    Uses reciprocal rank within top-K list, averaged over queries.
    """
    rlt = 0.0
    y_true_seq = list(y_true_seq)
    y_pred_seq = list(y_pred_seq)

    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1.0 / (r_idx[0] + 1)
    return rlt / max(len(y_true_seq), 1)


def MRR_metric(y_true_seq: Iterable[int], y_pred_seq: Iterable[np.ndarray]) -> float:
    """
    Mean Reciprocal Rank (MRR) over full ranking.
    """
    rlt = 0.0
    y_true_seq = list(y_true_seq)
    y_pred_seq = list(y_pred_seq)

    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1.0 / (r_idx + 1)
    return rlt / max(len(y_true_seq), 1)


def top_k_acc_last_timestep(y_true_seq: List[int], y_pred_seq: List[np.ndarray], k: int) -> int:
    """
    Hit@K computed only on the last timestep.
    """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    return 1 if np.any(top_k_rec == y_true) else 0


def mAP_metric_last_timestep(y_true_seq: List[int], y_pred_seq: List[np.ndarray], k: int) -> float:
    """
    mAP@K computed only on the last timestep (single positive item).
    """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    return 1.0 / (r_idx[0] + 1) if len(r_idx) != 0 else 0.0


def MRR_metric_last_timestep(y_true_seq: List[int], y_pred_seq: List[np.ndarray]) -> float:
    """
    MRR computed only on the last timestep.
    """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1.0 / (r_idx + 1)


def array_round(x: List[float], k: int = 4) -> List[float]:
    """
    Round list of floats to k decimals (for logging).
    """
    return list(np.around(np.array(x), k))


# -----------------------------------------------------------------------------
# Optimizer utils
# -----------------------------------------------------------------------------
def build_adamw_param_groups(modules: Iterable[torch.nn.Module], weight_decay: float):
    """
    Create AdamW parameter groups:
      - apply weight_decay to normal params
      - do NOT apply weight_decay to bias / LayerNorm-like params

    NOTE:
        Grouping rule is kept consistent with the original implementation.
    """
    decay, no_decay = [], []
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            if (
                name.endswith(".bias")
                or "LayerNorm" in name
                or "layer_norm" in name
                or name.endswith(".ln")
                or ".ln." in name
            ):
                no_decay.append(param)
            else:
                decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# -----------------------------------------------------------------------------
# Re-ranking utilities (hybrid)
# -----------------------------------------------------------------------------
def cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity computed after L2-normalization.
    """
    a = F.normalize(a, dim=dim, eps=eps)
    b = F.normalize(b, dim=dim, eps=eps)
    return (a * b).sum(dim=dim)


@torch.no_grad()
def rerank_topk_last_step(
    last_logits: torch.Tensor,        # (V,)
    last_poi_idx: int,                # previous POI index
    K: int,
    T_prior: torch.Tensor,            # (V, V) transition prior
    poi_idx2cat_idx: torch.Tensor,    # (V,)
    poi_idx2geo_idx: torch.Tensor,    # (V,)
    space_emb: torch.nn.Embedding,    # geohash embedding (padding_idx=0)
    betas: Tuple[float, float, float, float] = (1.0, 0.4, 0.15, 0.25),
    eps: float = 1e-12,
):
    """
    Re-rank the model top-K candidates using:
      - model logits
      - transition prior log-prob
      - category match
      - spatial cosine similarity (geohash space embedding)

    Returns:
        best_idx: int
        cands: Tensor[K]
        final_scores: Tensor[K]
    """
    V = last_logits.size(0)
    scores, cands = torch.topk(last_logits, k=min(K, V), dim=0)

    prior = torch.log(T_prior[last_poi_idx, cands].clamp_min(eps))
    cat_last = poi_idx2cat_idx[last_poi_idx]
    cat_cands = poi_idx2cat_idx[cands]
    cat_match = (cat_cands == cat_last).float()

    geo_last = poi_idx2geo_idx[last_poi_idx].clamp_min(0)
    geo_cands = poi_idx2geo_idx[cands].clamp_min(0)

    emb_last = space_emb(geo_last)                     # (D,)
    emb_cand = space_emb(geo_cands)                    # (K,D)
    geo_cos = cosine_sim(emb_cand, emb_last.unsqueeze(0), dim=1)

    b_logit, b_prior, b_cat, b_geo = betas
    final = (b_logit * scores) + (b_prior * prior) + (b_cat * cat_match) + (b_geo * geo_cos)

    best_rel = torch.argmax(final)
    best_idx = int(cands[best_rel].item())
    return best_idx, cands, final


# -----------------------------------------------------------------------------
# Last-step metrics (no shifting)
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_last_step_metrics(
    logits_poi: torch.Tensor,  # (B, L, V)
    y_poi: torch.Tensor,       # (B, L)
    lengths: Iterable[int],    # true lengths per sample
    topk: Tuple[int, ...] = (1, 5, 10, 20),
    pad_label: int = PAD_LABEL,
) -> Dict[str, float]:
    """
    Compute metrics at the last valid timestep (L_i - 1) for each sample:
      - Hit@K
      - MRR
      - mAP@20 (single positive item)

    Alignment rule:
      logits[:, t] corresponds to y[:, t] (NO external shifting).
    """
    B, L, V = logits_poi.shape
    kmax = min(max(topk), V)

    hit = {k: 0.0 for k in topk}
    mrr = 0.0
    map20 = 0.0
    valid = 0

    for i, L_i in enumerate(lengths):
        L_i = int(L_i)
        if L_i <= 0:
            continue

        last_idx = L_i - 1
        last_logits = logits_poi[i, last_idx]
        last_label = int(y_poi[i, last_idx].item())

        if last_label == pad_label or last_label < 0 or last_label >= V:
            continue

        topk_inds = torch.topk(last_logits, k=kmax).indices

        for k in topk:
            kk = min(k, V)
            if last_label in topk_inds[:kk].tolist():
                hit[k] += 1.0

        target_logit = last_logits[last_label]
        rank = int((last_logits > target_logit).sum().item()) + 1
        mrr += 1.0 / rank

        if rank <= 20:
            map20 += 1.0 / rank

        valid += 1

    denom = max(valid, 1)
    return {
        **{f"top{k}": hit[k] / denom for k in topk},
        "mrr": mrr / denom,
        "map20": map20 / denom,
        "valid": float(valid),
    }


# -----------------------------------------------------------------------------
# Debug logging helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def _log_batch_predictions_new(
    batch: Dict[str, torch.Tensor],
    logits_poi: torch.Tensor,       # (B, L, V)
    y_poi: torch.Tensor,            # (B, L)
    attention_mask: torch.Tensor,   # (B, L) 1=valid, 0=pad
    phase: str = "train",
    sample_idx: int = 0,
    topk: int = 5,
    preview_steps: int = 3,
    pad_label: int = PAD_LABEL,
) -> None:
    """
    Log a single example from a collated batch (dict-based batch).

    Rules:
      - logits[:, t] aligns with y[:, t] (no shifting).
      - last valid step is computed from attention_mask and pad_label.
      - does not assume any specific prefixing scheme.
    """
    try:
        B = batch["user_idxs"].size(0)
        if B == 0:
            return

        sample_idx = min(sample_idx, B - 1)

        traj_id = batch["traj_ids"][sample_idx]
        user_idx_int = int(batch["user_idxs"][sample_idx].item())
        input_pois_tensor = batch["x_poi_idxs"][sample_idx]      # (L,)
        input_times_tensor = batch["x_time_feats"][sample_idx]   # (L, 3)
        y_time_frac_tensor = batch.get("y_time_frac", None)

        amask = attention_mask[sample_idx].bool()
        yrow = y_poi[sample_idx]
        valid_mask = amask & (yrow != pad_label)
        valid_idxs = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if valid_idxs.numel() == 0:
            return

        t_last = int(valid_idxs.max().item())
        L_eff = int(amask.sum().item())

        input_pois = input_pois_tensor[:L_eff].tolist()
        label_pois = yrow[:L_eff].tolist()

        last_logits = logits_poi[sample_idx, t_last].detach().float()
        k_last = int(max(1, min(topk, last_logits.size(0))))
        topk_vals, topk_inds = torch.topk(last_logits, k=k_last)

        last_gold_raw = yrow[t_last].item()
        last_gold = int(last_gold_raw) if last_gold_raw != pad_label else -1

        t_start = max(0, t_last - (max(1, preview_steps) - 1))
        t_end = t_last

        preview_lines = []
        for t in range(t_start, t_end + 1):
            poi_id = int(input_pois_tensor[t].item())
            time_tuple = input_times_tensor[t]
            wd = float(time_tuple[2].item())
            tf = float(time_tuple[1].item())

            gold_raw = yrow[t].item()
            gold_t = int(gold_raw) if gold_raw != pad_label else -1

            tf_next_str = "N/A"
            if y_time_frac_tensor is not None:
                tf_next = float(y_time_frac_tensor[sample_idx, t].item())
                tf_next_str = f"{tf_next:.4f}" if tf_next != -1.0 else "PAD"

            lt = logits_poi[sample_idx, t].detach().float()
            k = int(max(1, min(topk, lt.size(0))))
            tk_inds = torch.topk(lt, k=k).indices.tolist()

            preview_lines.append(
                f"t={t:02d} "
                f"[in_poi={poi_id}, time=(wd={wd:.2f}, tf={tf:.4f})] -> "
                f"[label_poi={gold_t}, next_tf={tf_next_str}] "
                f"top{topk}={tk_inds}"
            )

        logger.info(
            f"[{phase}] traj_id={traj_id}, user_idx={user_idx_int} "
            f"L(effective)={L_eff} (predict@t={t_last} -> label@t={t_last})"
        )
        logger.info(f"[{phase}] input_pois[:10]={input_pois[:10]}")
        logger.info(f"[{phase}] label_pois[:10]={label_pois[:10]}")
        logger.info(
            f"[{phase}] last_step gold={last_gold} -> top{topk}={topk_inds.tolist()} "
            f"(scores≈{[f'{v:.3f}' for v in topk_vals.tolist()]})"
        )
        if preview_lines:
            logger.info(f"[{phase}] preview (last up to {preview_steps} steps):")
            for line in preview_lines:
                logger.info("  " + line)
            logger.info("-" * 80)

    except Exception as e:
        logger.warning(
            f"[{phase}] prediction logging failed: {e} "
            f"(batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'N/A'})"
        )


# -----------------------------------------------------------------------------
# Trajectory collators
# -----------------------------------------------------------------------------
class TrajectoryCollator:
    """
    Collate function for trajectory samples.

    Input (list of samples):
        [(traj_id, user_idx, input_seq, label_seq, prev_poi_seq), ...]

    Output (dict of padded tensors):
        - x_* : input-side padded tensors
        - y_* : label-side padded tensors
        - attention_mask : 1 for valid steps, 0 for padding
    """

    def __init__(self, poi_id2cat_idx: Dict[int, int], poi_id2geo_idx: Dict[int, int], pad_label: int = PAD_LABEL):
        self.poi_id2cat_idx = poi_id2cat_idx
        self.poi_id2geo_idx = poi_id2geo_idx
        self.pad_label = pad_label
        self.poi_default_cat = 0
        self.poi_default_geo = 0

    def __call__(self, batch: list):
        user_idxs_list: List[int] = []
        traj_ids_list: List[str] = []
        lengths: List[int] = []

        input_poi_list, input_cat_list, input_geo_list = [], [], []
        input_time_list = []   # (L_i, 3)
        prev_poi_list = []

        label_poi_list, label_cat_list = [], []
        label_time_sincos_list, label_time_frac_list = [], []

        for (traj_id, user_idx, input_seq, label_seq, prev_poi_seq) in batch:
            traj_ids_list.append(traj_id)
            user_idxs_list.append(user_idx)
            lengths.append(len(input_seq))

            # ---- Input sequence ----
            in_pois, in_cats, in_geos, in_times = [], [], [], []
            for poi_id, time_tuple in input_seq:
                in_pois.append(poi_id)
                in_cats.append(self.poi_id2cat_idx.get(poi_id, self.poi_default_cat))
                in_geos.append(self.poi_id2geo_idx.get(poi_id, self.poi_default_geo))
                in_times.append(time_tuple)

            input_poi_list.append(torch.tensor(in_pois, dtype=torch.long))
            input_cat_list.append(torch.tensor(in_cats, dtype=torch.long))
            input_geo_list.append(torch.tensor(in_geos, dtype=torch.long))
            input_time_list.append(torch.tensor(in_times, dtype=torch.float32))
            prev_poi_list.append(torch.tensor(prev_poi_seq, dtype=torch.long))

            # ---- Label sequence ----
            lbl_pois, lbl_cats, lbl_times_sincos, lbl_times_frac = [], [], [], []
            for poi_id, time_tuple in label_seq:
                lbl_pois.append(poi_id)
                lbl_cats.append(self.poi_id2cat_idx.get(poi_id, self.poi_default_cat))
                tf_next = float(time_tuple[1])
                lbl_times_sincos.append(frac_to_sincos(tf_next))
                lbl_times_frac.append(tf_next)

            label_poi_list.append(torch.tensor(lbl_pois, dtype=torch.long))
            label_cat_list.append(torch.tensor(lbl_cats, dtype=torch.long))
            label_time_sincos_list.append(torch.tensor(lbl_times_sincos, dtype=torch.float32))
            label_time_frac_list.append(torch.tensor(lbl_times_frac, dtype=torch.float32))

        # ---- Padding ----
        x_poi_idxs = pad_sequence(input_poi_list, batch_first=True, padding_value=0)
        x_cat_idxs = pad_sequence(input_cat_list, batch_first=True, padding_value=0)
        x_geo_idxs = pad_sequence(input_geo_list, batch_first=True, padding_value=0)
        x_time_feats = pad_sequence(input_time_list, batch_first=True, padding_value=0.0)
        x_prev_poi_idxs = pad_sequence(prev_poi_list, batch_first=True, padding_value=0)

        y_poi = pad_sequence(label_poi_list, batch_first=True, padding_value=self.pad_label)
        y_cat = pad_sequence(label_cat_list, batch_first=True, padding_value=self.pad_label)
        y_time = pad_sequence(label_time_sincos_list, batch_first=True, padding_value=0.0)
        y_time_frac = pad_sequence(label_time_frac_list, batch_first=True, padding_value=-1.0)

        B, L_max = x_poi_idxs.shape
        attention_mask = torch.zeros((B, L_max), dtype=torch.long)
        for i, L_i in enumerate(lengths):
            attention_mask[i, :L_i] = 1

        return {
            "traj_ids": traj_ids_list,
            "user_idxs": torch.tensor(user_idxs_list, dtype=torch.long),
            "x_poi_idxs": x_poi_idxs,
            "x_cat_idxs": x_cat_idxs,
            "x_geo_idxs": x_geo_idxs,
            "x_time_feats": x_time_feats,     # (B, L_max, 3)
            "prev_ids": x_prev_poi_idxs,      # (B, L_max)
            "y_poi": y_poi,                   # (B, L_max)
            "y_cat": y_cat,                   # (B, L_max)
            "y_time": y_time,                 # (B, L_max, 2)
            "y_time_frac": y_time_frac,       # (B, L_max)
            "attention_mask": attention_mask, # (B, L_max)
        }


class TrajectoryCollator2:
    """
    Variant collator that additionally returns target time features (3D) for conditioning.
    """

    def __init__(self, poi_id2cat_idx: Dict[int, int], poi_id2geo_idx: Dict[int, int], pad_label: int = PAD_LABEL):
        self.poi_id2cat_idx = poi_id2cat_idx
        self.poi_id2geo_idx = poi_id2geo_idx
        self.pad_label = pad_label
        self.poi_default_cat = 0
        self.poi_default_geo = 0

    def __call__(self, batch: list):
        user_idxs_list, traj_ids_list, lengths = [], [], []

        input_poi_list, input_cat_list, input_geo_list = [], [], []
        input_time_list = []   # (L_i, 3)
        prev_poi_list = []

        label_poi_list, label_cat_list = [], []
        label_time_sincos_list, label_time_frac_list = [], []
        label_time_3d_list = []  # target-time feats (3D)

        for (traj_id, user_idx, input_seq, label_seq, prev_poi_seq) in batch:
            traj_ids_list.append(traj_id)
            user_idxs_list.append(user_idx)
            lengths.append(len(input_seq))

            # ---- Input sequence ----
            in_pois, in_cats, in_geos, in_times = [], [], [], []
            for poi_id, time_tuple in input_seq:
                in_pois.append(poi_id)
                in_cats.append(self.poi_id2cat_idx.get(poi_id, self.poi_default_cat))
                in_geos.append(self.poi_id2geo_idx.get(poi_id, self.poi_default_geo))
                in_times.append(_time_tuple_to_feat(time_tuple))

            input_poi_list.append(torch.tensor(in_pois, dtype=torch.long))
            input_cat_list.append(torch.tensor(in_cats, dtype=torch.long))
            input_geo_list.append(torch.tensor(in_geos, dtype=torch.long))
            input_time_list.append(torch.tensor(in_times, dtype=torch.float32))
            prev_poi_list.append(torch.tensor(prev_poi_seq, dtype=torch.long))

            # ---- Label sequence ----
            lbl_pois, lbl_cats = [], []
            lbl_times_sincos, lbl_times_frac = [], []
            lbl_times_3d = []

            for poi_id, time_tuple in label_seq:
                lbl_pois.append(poi_id)
                lbl_cats.append(self.poi_id2cat_idx.get(poi_id, self.poi_default_cat))

                tf_next = float(time_tuple[1])
                lbl_times_sincos.append(frac_to_sincos(tf_next))
                lbl_times_frac.append(tf_next)
                lbl_times_3d.append(_time_tuple_to_feat(time_tuple))

            label_poi_list.append(torch.tensor(lbl_pois, dtype=torch.long))
            label_cat_list.append(torch.tensor(lbl_cats, dtype=torch.long))
            label_time_sincos_list.append(torch.tensor(lbl_times_sincos, dtype=torch.float32))
            label_time_frac_list.append(torch.tensor(lbl_times_frac, dtype=torch.float32))
            label_time_3d_list.append(torch.tensor(lbl_times_3d, dtype=torch.float32))

        # ---- Padding ----
        x_poi_idxs = pad_sequence(input_poi_list, batch_first=True, padding_value=0)
        x_cat_idxs = pad_sequence(input_cat_list, batch_first=True, padding_value=0)
        x_geo_idxs = pad_sequence(input_geo_list, batch_first=True, padding_value=0)
        x_time_feats = pad_sequence(input_time_list, batch_first=True, padding_value=0.0)
        x_prev_poi_idxs = pad_sequence(prev_poi_list, batch_first=True, padding_value=0)

        y_poi = pad_sequence(label_poi_list, batch_first=True, padding_value=self.pad_label)
        y_cat = pad_sequence(label_cat_list, batch_first=True, padding_value=self.pad_label)
        y_time = pad_sequence(label_time_sincos_list, batch_first=True, padding_value=0.0)
        y_time_frac = pad_sequence(label_time_frac_list, batch_first=True, padding_value=-1.0)

        tgt_time_feats = pad_sequence(label_time_3d_list, batch_first=True, padding_value=0.0)

        B, L_max = x_poi_idxs.shape
        attention_mask = torch.zeros((B, L_max), dtype=torch.long)
        for i, L_i in enumerate(lengths):
            attention_mask[i, :L_i] = 1

        return {
            "traj_ids": traj_ids_list,
            "user_idxs": torch.tensor(user_idxs_list, dtype=torch.long),
            "x_poi_idxs": x_poi_idxs,
            "x_cat_idxs": x_cat_idxs,
            "x_geo_idxs": x_geo_idxs,
            "x_time_feats": x_time_feats,      # input-time feats (B, L, 3)
            "prev_ids": x_prev_poi_idxs,
            "y_poi": y_poi,
            "y_cat": y_cat,
            "y_time": y_time,                  # sin/cos (B, L, 2) for auxiliary loss
            "y_time_frac": y_time_frac,
            "attention_mask": attention_mask,
            "tgt_time_feats": tgt_time_feats,  # target-time feats (B, L, 3) for conditioning
        }


# -----------------------------------------------------------------------------
# Parameter counting utility
# -----------------------------------------------------------------------------
def count_params_from_modules(modules: Iterable[torch.nn.Module]):
    """
    Count parameters across modules, removing duplicates by parameter object id.

    Returns:
        total_trainable, total_all, per_module(list of dict), mem_info(dict)

    Notes:
        Memory estimation is parameter-tensor-only (optimizer states and gradients excluded).
    """
    seen_all = set()
    seen_train = set()
    per_module = []

    def mod_counts_unique(mod):
        s_all, s_tr = set(), set()
        t_all = t_tr = 0
        for p in mod.parameters(recurse=True):
            pid = id(p)
            if pid not in s_all:
                s_all.add(pid)
                t_all += p.numel()
            if p.requires_grad and pid not in s_tr:
                s_tr.add(pid)
                t_tr += p.numel()
        return t_tr, t_all, s_tr, s_all

    for i, mod in enumerate(modules):
        tr, al, s_tr, s_al = mod_counts_unique(mod)
        per_module.append({"name": f"{i}:{mod.__class__.__name__}", "trainable": tr, "total": al})
        seen_all |= s_al
        seen_train |= s_tr

    id2numel = {}
    id2numel_train = {}
    for mod in modules:
        for p in mod.parameters(recurse=True):
            pid = id(p)
            if pid in seen_all and pid not in id2numel:
                id2numel[pid] = p.numel()
            if p.requires_grad and pid in seen_train and pid not in id2numel_train:
                id2numel_train[pid] = p.numel()

    total_all = sum(id2numel.values())
    total_train = sum(id2numel_train.values())

    bytes_fp32_all = total_all * 4
    bytes_fp16_all = total_all * 2
    bytes_fp32_train = total_train * 4
    bytes_fp16_train = total_train * 2

    mem_info = {
        "all_params": total_all,
        "trainable_params": total_train,
        "params_only_fp32_MB": bytes_fp32_all / (1024**2),
        "params_only_fp16_MB": bytes_fp16_all / (1024**2),
        "trainable_params_fp32_MB": bytes_fp32_train / (1024**2),
        "trainable_params_fp16_MB": bytes_fp16_train / (1024**2),
        # Reference: AdamW optimizer states typically require ~2x parameter memory (m and v).
    }
    return total_train, total_all, per_module, mem_info

def _iter_pairs_from_dataset(dataset) -> Iterator[Tuple[int, int, int]]:
    """
    Iterate over (user_idx, prev_poi, next_poi) triples from a trajectory dataset.

    Notes:
        - Assumes each dataset item is:
            (traj_id, user_idx, input_seq, label_seq, prev_poi_seq)
        - input_seq[t][0] is the observed POI at timestep t (prev POI)
        - label_seq[t][0] is the target POI at timestep t (next POI)
    """
    for _traj_id, user_idx, input_seq, label_seq, _prev_poi_seq in dataset:
        # iterate over prediction positions
        for t in range(len(label_seq)):
            prev_poi = int(input_seq[t][0])
            next_poi = int(label_seq[t][0])
            yield int(user_idx), prev_poi, next_poi


def _build_freq_tables(train_dataset):
    """
    Build frequency tables for simple popularity / Markov baselines.

    Returns:
        global_next: Counter(next_poi)
        user_next: dict[user] -> Counter(next_poi)
        user_prev_next: dict[(user, prev_poi)] -> Counter(next_poi)
    """
    global_next = Counter()
    user_next = defaultdict(Counter)
    user_prev_next = defaultdict(Counter)

    for u, prev, nxt in _iter_pairs_from_dataset(train_dataset):
        global_next[nxt] += 1
        user_next[u][nxt] += 1
        user_prev_next[(u, prev)][nxt] += 1

    return global_next, user_next, user_prev_next


def _top1_from_counter(counter: Counter):
    """Return the most frequent key in a Counter, or None if empty."""
    return counter.most_common(1)[0][0] if counter else None