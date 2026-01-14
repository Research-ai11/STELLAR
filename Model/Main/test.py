"""
Test script for STELLAR next-POI prediction.

This script:
  1) Loads the full dataset CSV and selects SplitTag == 'test'
  2) Loads a trained checkpoint (mappings + model weights)
  3) Rebuilds all model components exactly as in training
  4) Computes POI embeddings (feature + category + GCN)
  5) Runs evaluation on test trajectories (last-timestep metrics)
  6) Saves a CSV summary of test metrics

Recommended for paper-code release:
- Keep path resolution and reproducibility utilities in a single place.
- Make diagnostics optional (coverage / frequency baselines).
"""

from __future__ import annotations

import logging
import os
import random
import pathlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from param_parser import parameter_parser
from STELLAR.Model.datasets import TrajectoryDataset
from STELLAR.Model.utils import (
    TrajectoryCollator2,
    increment_path,
    zipdir,
    compute_last_step_metrics,
    build_space_table,
    df_to_pyg_data,
    _iter_pairs_from_dataset,
    _build_freq_tables,
    _top1_from_counter,
)

from model import (
    UserEmbedding,
    POIEncoderGCN,
    CatEmbedding,
    TimeEmbedding,
    CheckInFusion,
    PFA,
    NextPOIWithPFA,
    get_batch_inputs_embeds,
    build_transition_prior_dense,
)

# ---------------------------------------------------------------------
# Constants / deterministic settings
# ---------------------------------------------------------------------
PAD_LABEL = -100
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set seeds for Python/NumPy/PyTorch and enable deterministic behavior."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Make DataLoader workers deterministic."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------
# Run directory / logging
# ---------------------------------------------------------------------
def setup_logger(save_dir: Path) -> None:
    """Configure file + console logging."""
    save_dir.mkdir(parents=True, exist_ok=True)

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=str(save_dir / "log_testing.txt"),
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger("").addHandler(console)

    logging.getLogger("matplotlib.font_manager").disabled = True


def snapshot_code(save_dir: Path) -> None:
    """Zip code snapshot for reproducibility."""
    zip_path = save_dir / "code.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipdir(pathlib.Path().absolute(), zipf, include_format=[".py"])


def prepare_test_run(args) -> Any:
    """Create output directory, set seed, configure logging, and snapshot code."""
    set_seed(args.seed)

    save_dir = Path(increment_path(Path(args.project) / args.data / (f"test_{args.name}"), exist_ok=args.exist_ok))
    args.save_dir = str(save_dir)

    setup_logger(save_dir)
    snapshot_code(save_dir)

    logging.info("Test configuration:")
    logging.info(args)
    return args


# ---------------------------------------------------------------------
# Path resolution (paper-friendly)
# ---------------------------------------------------------------------
@dataclass
class DataPaths:
    data_path: str
    poi_traj_graph: str
    geohash_embedding: str
    poi_feat: str
    user_neighbors: str


def resolve_data_paths(dataset_key: str) -> DataPaths:
    if dataset_key == "nyc":
        return DataPaths(
            data_path="../../data/nyc/raw/NYC.csv",
            poi_traj_graph="../../data/nyc/graph/nyc_traj_graph.csv",
            geohash_embedding="../../data/nyc/graph/nyc_geohash_gat_space_embedding.csv",
            poi_feat="../../data/nyc/graph/nyc_poi_feat.csv",
            user_neighbors="../../data/nyc/graph/nyc_neighbors.pt",
        )
    if dataset_key == "tky":
        return DataPaths(
            data_path="../../data/tky/raw/TKY.csv",
            poi_traj_graph="../../data/tky/graph/tky_traj_graph.csv",
            geohash_embedding="../../data/tky/graph/tky_geohash_gat_space_embedding.csv",
            poi_feat="../../data/tky/graph/tky_poi_feat.csv",
            user_neighbors="../../data/tky/graph/tky_neighbors.pt",
        )
    raise ValueError(f"Unknown dataset key: {dataset_key}")


# ---------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------
def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint to CPU/GPU."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "mappings" not in ckpt:
        raise KeyError("Checkpoint does not contain 'mappings'.")
    return ckpt


def parse_mappings(ckpt: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract mappings and derived meta.
    Returns:
      mappings: raw mapping dict
      meta: dict with num_users/num_pois/num_cats/user_set
    """
    mappings = ckpt["mappings"]

    user_id2idx = mappings["user_id2idx"]
    poi_id2idx = mappings["poi_id2idx"]
    cat_id2idx = mappings["cat_id2idx"]

    meta = {
        "num_users": len(user_id2idx) + 1,
        "num_pois": len(poi_id2idx) + 1,
        "num_cats": len(cat_id2idx) + 1,
        "user_set": set(user_id2idx.keys()),
    }
    return mappings, meta


# ---------------------------------------------------------------------
# Feature / graph builders
# ---------------------------------------------------------------------
def build_poi_feature_matrix(
    poi_feat_csv: str,
    poi_id2idx: Dict[int, int],
    num_pois_with_pad: int,
    device: torch.device,
) -> torch.Tensor:
    """Build POI feature table X with padding row at index 0."""
    poi_feats = pd.read_csv(poi_feat_csv)

    exclude_cols = {"PoiId", "PoiCategoryId"}
    feat_cols = [c for c in poi_feats.columns if c not in exclude_cols]

    feat_mat = np.zeros((num_pois_with_pad, len(feat_cols)), dtype=np.float32)

    idx_map = poi_feats["PoiId"].map(poi_id2idx)
    valid_mask = idx_map.notna()
    target_idx = idx_map[valid_mask].astype(int).values
    feat_vals = poi_feats.loc[valid_mask, feat_cols].to_numpy(dtype=np.float32)

    feat_mat[target_idx] = feat_vals
    return torch.tensor(feat_mat, dtype=torch.float32, device=device)


def build_poi_latlon_table(
    poi_idx2latlon: Dict[int, Tuple[float, float]],
    num_pois_with_pad: int,
    device: torch.device,
) -> torch.Tensor:
    """Build POI lat/lon table (degrees) with padding row at 0."""
    tab = torch.zeros(num_pois_with_pad, 2, dtype=torch.float32, device=device)
    for poi_idx, (lat, lon) in poi_idx2latlon.items():
        tab[poi_idx] = torch.tensor([lat, lon], dtype=torch.float32, device=device)
    return tab


def build_poi_cat_index(
    poi_idx2cat_idx: Dict[int, int],
    num_pois_with_pad: int,
    device: torch.device,
) -> torch.Tensor:
    """Build POI->category index tensor for all POIs (including padding)."""
    poi_cat_idx_full = torch.zeros(num_pois_with_pad, dtype=torch.long, device=device)
    for poi_idx, cat_idx in poi_idx2cat_idx.items():
        if 0 <= poi_idx < num_pois_with_pad:
            poi_cat_idx_full[poi_idx] = int(cat_idx)
    return poi_cat_idx_full


# ---------------------------------------------------------------------
# Diagnostics (optional)
# ---------------------------------------------------------------------
def run_coverage_and_baselines(
    train_dataset: TrajectoryDataset,
    test_dataset: TrajectoryDataset,
) -> None:
    """Optional: report test coverage vs train and baseline Acc@1."""
    logging.info("=== DIAG: coverage on TEST against TRAIN ===")

    train_poi_seen = set()
    train_trans_seen = set()
    for u, prev, nxt in _iter_pairs_from_dataset(train_dataset):
        train_poi_seen.add(prev)
        train_poi_seen.add(nxt)
        train_trans_seen.add((u, prev, nxt))

    test_poi_total = 0
    test_poi_unseen = 0
    test_trans_total = 0
    test_trans_unseen = 0

    for u, prev, nxt in _iter_pairs_from_dataset(test_dataset):
        test_poi_total += 1
        if nxt not in train_poi_seen:
            test_poi_unseen += 1
        test_trans_total += 1
        if (u, prev, nxt) not in train_trans_seen:
            test_trans_unseen += 1

    poi_unseen_rate = test_poi_unseen / max(1, test_poi_total)
    trans_unseen_rate = test_trans_unseen / max(1, test_trans_total)

    logging.info(f"[COVERAGE] TEST labels unseen-POI rate: {poi_unseen_rate:.3%} ({test_poi_unseen}/{test_poi_total})")
    logging.info(
        f"[COVERAGE] TEST (u,prev->next) unseen-transition rate: {trans_unseen_rate:.3%} "
        f"({test_trans_unseen}/{test_trans_total})"
    )

    global_next, user_next, user_prev_next = _build_freq_tables(train_dataset)
    global_top1 = _top1_from_counter(global_next)
    user_top1 = {u: _top1_from_counter(cnt) for u, cnt in user_next.items()}
    userprev_top1 = {k: _top1_from_counter(cnt) for k, cnt in user_prev_next.items()}

    base_acc = {"global": 0.0, "user": 0.0, "markov": 0.0}
    cnt = 0
    for sample in test_dataset:
        _, user_idx, input_seq, label_seq, _ = sample
        if len(label_seq) == 0:
            continue

        cnt += 1
        gold = label_seq[-1][0]
        prev = input_seq[-1][0]
        u = int(user_idx)

        pred_g = global_top1
        base_acc["global"] += float(pred_g == gold) if pred_g is not None else 0.0

        pred_u = user_top1.get(u, None)
        base_acc["user"] += float(pred_u == gold) if pred_u is not None else 0.0

        pred_m = userprev_top1.get((u, prev), None)
        base_acc["markov"] += float(pred_m == gold) if pred_m is not None else 0.0

    for k in base_acc:
        base_acc[k] = base_acc[k] / max(1, cnt)

    logging.info(
        f"[BASELINES|TEST Acc@1] global={base_acc['global']:.4f} "
        f"user={base_acc['user']:.4f} markov={base_acc['markov']:.4f} (N={cnt})"
    )


# ---------------------------------------------------------------------
# Model builder (must match training)
# ---------------------------------------------------------------------
def build_models(
    *,
    args,
    num_users: int,
    num_pois: int,
    num_cats: int,
    user_pad_idx: torch.Tensor,
    user_pad_w: torch.Tensor,
    poi_X: torch.Tensor,
) -> Tuple[Any, ...]:
    """Instantiate all model components (same as training)."""
    user_emb = UserEmbedding(
        num_users=num_users,
        dim=args.user_dim,
        user_pad_idx=user_pad_idx,
        user_pad_w=user_pad_w,
        padding_idx=0,
    ).to(args.device)

    cat_emb = CatEmbedding(num_cats=num_cats, dim=args.cat_dim, padding_idx=0).to(args.device)

    poi_encoder = POIEncoderGCN(
        in_dim=poi_X.shape[1] + args.cat_dim,
        hid_dim=args.poi_dim,
        out_dim=args.poi_dim,
    ).to(args.device)

    time_embed_model = TimeEmbedding(dim=args.time_dim).to(args.device)

    check_in_fusion_model = CheckInFusion(
        d_user=args.user_dim,
        d_poi=args.poi_dim,
        d_time=args.time_dim,
        d_space=args.space_dim,
        d_cat=args.cat_dim,
        out_dim=args.input_tok_dim,
        gate=args.fusion_gate,
    ).to(args.device)

    main_model = NextPOIWithPFA(
        pfa=PFA(),
        num_pois=num_pois,
        num_cats=num_cats,
        logit_mode=args.logit_mode,
        poi_proj_dim=args.poi_dim,
        cat_proj_dim=args.cat_dim,
        label_ignore_index=PAD_LABEL,
        lambda_poi=args.lambda_poi,
        lambda_time=args.lambda_time,
        lambda_cat=args.lambda_cat,
        lambda_loc=args.lambda_loc,
    ).to(args.device)

    return user_emb, cat_emb, poi_encoder, time_embed_model, check_in_fusion_model, main_model


def load_weights_from_checkpoint(
    *,
    ckpt: Dict[str, Any],
    user_emb,
    cat_emb,
    poi_encoder,
    time_embed_model,
    check_in_fusion_model,
    main_model,
) -> None:
    """Load trained weights (keys follow your current checkpoint format)."""
    main_model.load_state_dict(ckpt["main_model_state_dict"])
    poi_encoder.load_state_dict(ckpt["poi_encoder_state_dict"])
    check_in_fusion_model.load_state_dict(ckpt["check_in_fusion_state_dict"])
    cat_emb.load_state_dict(ckpt["cat_emb_state_dict"])
    user_emb.load_state_dict(ckpt["user_emb_state_dict"])

    # NOTE: your checkpoint key is 'time_embed_model' in the original code.
    # If you later rename it to 'time_embed_model_state_dict', change it here only.
    time_embed_model.load_state_dict(ckpt["time_embed_model_state_dict"])


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate(
    *,
    args,
    test_loader: DataLoader,
    user_emb,
    cat_emb,
    poi_encoder,
    time_embed_model,
    check_in_fusion_model,
    main_model,
    space_emb,
    poi_X: torch.Tensor,
    poi_traj_data,
    poi_cat_idx_full: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate on test set using last-step metrics."""
    for m in [main_model, user_emb, poi_encoder, cat_emb, time_embed_model, check_in_fusion_model]:
        m.eval()

    batches_top1, batches_top5, batches_top10, batches_top20 = [], [], [], []
    batches_map20, batches_mrr = [], []

    with torch.no_grad():
        # Compute POI embeddings once for testing
        cat_plus = cat_emb(poi_cat_idx_full)
        poi_input = torch.cat([poi_X, cat_plus], dim=1)
        poi_final_emb = poi_encoder(poi_input, poi_traj_data)

        for batch in tqdm(test_loader, desc="Testing"):
            batch_gpu = {
                k: v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            x = get_batch_inputs_embeds(
                user_idxs=batch_gpu["user_idxs"],
                x_poi_idxs=batch_gpu["x_poi_idxs"],
                x_cat_idxs=batch_gpu["x_cat_idxs"],
                x_geo_idxs=batch_gpu["x_geo_idxs"],
                x_time_feats=batch_gpu["x_time_feats"],
                user_final_emb=user_emb,
                poi_final_emb=poi_final_emb,
                cat_emb=cat_emb,
                space_emb=space_emb,
                time_embed_model=time_embed_model,
                check_in_fusion_model=check_in_fusion_model,
            )

            y_poi = batch_gpu["y_poi"]
            attention_mask = batch_gpu["attention_mask"]
            tgt_time_feats = batch_gpu["tgt_time_feats"]  # (B, L, 3)
            tgt_time_emb = time_embed_model(
                tgt_time_feats[..., 0],
                tgt_time_feats[..., 1],
                tgt_time_feats[..., 2],
            )

            if args.logit_mode == "cos":
                out = main_model(
                    inputs_embeds=x,
                    attention_mask=attention_mask,
                    time_prompt=tgt_time_emb,
                    poi_final_emb=poi_final_emb,
                    cat_emb_weight=cat_emb.weight,
                )
            else:
                out = main_model(inputs_embeds=x, attention_mask=attention_mask, time_prompt=tgt_time_emb)

            logits_poi = out["logits_poi"]
            logits_poi[..., 0] = -1e9  # mask padding

            metrics = compute_last_step_metrics(
                logits_poi=logits_poi,
                y_poi=y_poi,
                lengths=attention_mask.sum(dim=1),
                topk=(1, 5, 10, 20),
            )
            batches_top1.append(metrics["top1"])
            batches_top5.append(metrics["top5"])
            batches_top10.append(metrics["top10"])
            batches_top20.append(metrics["top20"])
            batches_map20.append(metrics["map20"])
            batches_mrr.append(metrics["mrr"])

    return {
        "top1": float(np.mean(batches_top1)) if batches_top1 else 0.0,
        "top5": float(np.mean(batches_top5)) if batches_top5 else 0.0,
        "top10": float(np.mean(batches_top10)) if batches_top10 else 0.0,
        "top20": float(np.mean(batches_top20)) if batches_top20 else 0.0,
        "map20": float(np.mean(batches_map20)) if batches_map20 else 0.0,
        "mrr": float(np.mean(batches_mrr)) if batches_mrr else 0.0,
    }


# ---------------------------------------------------------------------
# Main test routine
# ---------------------------------------------------------------------
def test(args) -> None:
    logging.info("Preparing test environment...")
    args = prepare_test_run(args)

    device = torch.device(args.device) if isinstance(args.device, str) else args.device
    args.device = device

    # Load full data
    df = pd.read_csv(args.data_path)
    test_df = df[df["SplitTag"] == "test"].reset_index(drop=True)
    if len(test_df) == 0:
        logging.warning("No test data found (SplitTag == 'test').")

    # Load checkpoint & mappings
    ckpt = load_checkpoint(args.checkpoint, device)
    mappings, meta = parse_mappings(ckpt)

    user_id2idx = mappings["user_id2idx"]
    poi_id2idx = mappings["poi_id2idx"]
    poi_idx2cat_idx = mappings["poi_idx2cat_idx"]
    poi_idx2geo_idx = mappings["poi_idx2geo_idx"]
    poi_idx2latlon = mappings["poi_idx2latlon"]

    num_users = meta["num_users"]
    num_pois = meta["num_pois"]
    num_cats = meta["num_cats"]
    user_set = meta["user_set"]

    # Load POI features
    poi_X = build_poi_feature_matrix(args.poi_feat, poi_id2idx, num_pois, device)

    # Load POI graph and transition prior
    poi_traj_G = pd.read_csv(args.poi_traj_graph)
    poi_traj_data = df_to_pyg_data(poi_traj_G, poi_id2idx, device=device)
    T_prior = build_transition_prior_dense(poi_traj_data, V=num_pois, device=device)

    # Load user neighbor info
    user_neighbors_data = torch.load(args.user_neighbors)
    user_pad_idx = user_neighbors_data["pad_idx"].to(device)
    user_pad_w = user_neighbors_data["pad_w"].to(device)

    # Load geohash embeddings (frozen)
    space_table, _ = build_space_table(args.geohash_embedding)
    space_emb = torch.nn.Embedding.from_pretrained(space_table, freeze=True, padding_idx=0).to(device)

    # Build models and load weights
    user_emb, cat_emb, poi_encoder, time_embed_model, check_in_fusion_model, main_model = build_models(
        args=args,
        num_users=num_users,
        num_pois=num_pois,
        num_cats=num_cats,
        user_pad_idx=user_pad_idx,
        user_pad_w=user_pad_w,
        poi_X=poi_X,
    )
    load_weights_from_checkpoint(
        ckpt=ckpt,
        user_emb=user_emb,
        cat_emb=cat_emb,
        poi_encoder=poi_encoder,
        time_embed_model=time_embed_model,
        check_in_fusion_model=check_in_fusion_model,
        main_model=main_model,
    )

    # Set transition prior and lat/lon table
    main_model.set_transition_prior(T_prior)
    poi_latlon_deg = build_poi_latlon_table(poi_idx2latlon, num_pois, device)
    main_model.set_poi_latlon(poi_latlon_deg)

    # Dataset / loader
    train_dataset = TrajectoryDataset(
        df=df,
        split_tag="train",
        user_set=user_set,
        user_id2idx=user_id2idx,
        poi_id2idx=poi_id2idx,
        short_traj_thres=args.short_traj_thres,
    )
    test_dataset = TrajectoryDataset(
        df=df,
        split_tag="test",
        user_set=user_set,
        user_id2idx=user_id2idx,
        poi_id2idx=poi_id2idx,
        short_traj_thres=args.short_traj_thres,
    )

    if getattr(args, "run_diagnostics", True):
        run_coverage_and_baselines(train_dataset, test_dataset)

    collator = TrajectoryCollator2(poi_id2cat_idx=poi_idx2cat_idx, poi_id2geo_idx=poi_idx2geo_idx, pad_label=PAD_LABEL)

    g = torch.Generator()
    g.manual_seed(args.seed)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=g,
    )

    poi_cat_idx_full = build_poi_cat_index(poi_idx2cat_idx, num_pois, device)

    # Evaluate
    logging.info("Running evaluation on test set...")
    results = evaluate(
        args=args,
        test_loader=test_loader,
        user_emb=user_emb,
        cat_emb=cat_emb,
        poi_encoder=poi_encoder,
        time_embed_model=time_embed_model,
        check_in_fusion_model=check_in_fusion_model,
        main_model=main_model,
        space_emb=space_emb,
        poi_X=poi_X,
        poi_traj_data=poi_traj_data,
        poi_cat_idx_full=poi_cat_idx_full,
    )

    logging.info(
        "Test Results | "
        f"Top1={results['top1']:.4f}, Top5={results['top5']:.4f}, Top10={results['top10']:.4f}, "
        f"Top20={results['top20']:.4f}, mAP@20={results['map20']:.4f}, MRR={results['mrr']:.4f}"
    )

    # Save summary CSV
    out_csv = Path(args.save_dir) / "test_results.csv"
    pd.DataFrame([{
        "Top1": results["top1"],
        "Top5": results["top5"],
        "Top10": results["top10"],
        "Top20": results["top20"],
        "mAP20": results["map20"],
        "MRR": results["mrr"],
    }]).to_csv(out_csv, index=False)

    logging.info(f"Saved test results to: {out_csv}")


def main() -> None:
    args = parameter_parser()

    # Resolve dataset paths
    paths = resolve_data_paths(args.data)
    args.data_path = paths.data_path
    args.poi_traj_graph = paths.poi_traj_graph
    args.geohash_embedding = paths.geohash_embedding
    args.poi_feat = paths.poi_feat
    args.user_neighbors = paths.user_neighbors

    test(args)


if __name__ == "__main__":
    main()
