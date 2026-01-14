"""
Training script for STELLAR next-POI prediction.

This script:
  1) Loads datasets & precomputed embeddings/graphs
  2) Builds model components (user, POI, time, space, fusion, main predictor)
  3) Trains with deterministic settings
  4) Evaluates on validation set (last-timestep metrics)
  5) Saves best checkpoint and per-epoch metrics

Recommended for paper-code release: keep this file as a single entry point.
"""

from __future__ import annotations

import logging
import math
import os
import random
import pathlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from param_parser import parameter_parser

# Project imports
from STELLAR.Model.datasets import TrajectoryDataset
from STELLAR.Model.utils import (
    TrajectoryCollator2,
    build_adamw_param_groups,
    build_mappings,
    build_space_table,
    compute_last_step_metrics,
    count_params_from_modules,
    df_to_pyg_data,
    increment_path,
    zipdir,
    _iter_pairs_from_dataset,
    _build_freq_tables,
    _top1_from_counter,
    _log_batch_predictions_new,
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
)

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
PAD_LABEL = -100

# Determinism for CUDA (must be set before heavy CUDA ops)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Set random seeds for Python/NumPy/PyTorch and enable deterministic behavior."""
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
# Logging / run directory management
# ---------------------------------------------------------------------
def setup_logger(save_dir: Path) -> None:
    """Configure both file and console logging."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Remove previous handlers (important in notebooks / repeated runs)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=str(save_dir / "log_training.txt"),
        filemode="w",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger("").addHandler(console)

    # Reduce noisy logs
    logging.getLogger("matplotlib.font_manager").disabled = True


def snapshot_code(save_dir: Path, include_ext: Tuple[str, ...] = (".py",)) -> None:
    """Zip code snapshot for reproducibility."""
    zip_path = save_dir / "code.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipdir(pathlib.Path().absolute(), zipf, include_format=list(include_ext))


def prepare_run(args) -> Any:
    """Create run directory, set seed, set logger, and dump args."""
    set_seed(args.seed)

    save_dir = Path(increment_path(Path(args.project) / args.data / args.logit_mode / args.name, exist_ok=args.exist_ok))
    args.save_dir = str(save_dir)
    setup_logger(save_dir)

    logging.info("Run configuration:")
    logging.info(args)

    with open(save_dir / "args.yaml", "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    snapshot_code(save_dir)

    logging.info(f"Seed: {args.seed}")
    logging.info(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
    logging.info(f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")

    return args


# ---------------------------------------------------------------------
# Path resolution (paper-code friendly)
# ---------------------------------------------------------------------
@dataclass
class DataPaths:
    data_path: str
    poi_traj_graph: str
    geohash_embedding: str
    poi_feat: str
    user_neighbors: str


def resolve_data_paths(dataset_key: str) -> DataPaths:
    """
    Resolve dataset-dependent paths.

    NOTE:
      - Keep these as relative paths for release
      - Make sure files exist in your repository / supplementary data
    """
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
# Diagnostics (optional)
# ---------------------------------------------------------------------
def log_val_coverage(train_dataset: TrajectoryDataset, val_dataset: TrajectoryDataset) -> None:
    """Coverage diagnostics: unseen POIs / unseen (u,prev->next) transitions on validation."""
    logging.info("=== DIAG: coverage on VAL against TRAIN ===")

    train_poi_seen = set()
    train_trans_seen = set()
    for u, prev, nxt in _iter_pairs_from_dataset(train_dataset):
        train_poi_seen.add(prev)
        train_poi_seen.add(nxt)
        train_trans_seen.add((u, prev, nxt))

    val_poi_total = 0
    val_poi_unseen = 0
    val_trans_total = 0
    val_trans_unseen = 0

    for u, prev, nxt in _iter_pairs_from_dataset(val_dataset):
        val_poi_total += 1
        if nxt not in train_poi_seen:
            val_poi_unseen += 1

        val_trans_total += 1
        if (u, prev, nxt) not in train_trans_seen:
            val_trans_unseen += 1

    poi_unseen_rate = val_poi_unseen / max(1, val_poi_total)
    trans_unseen_rate = val_trans_unseen / max(1, val_trans_total)

    logging.info(f"[COVERAGE] VAL labels unseen-POI rate: {poi_unseen_rate:.3%} ({val_poi_unseen}/{val_poi_total})")
    logging.info(
        f"[COVERAGE] VAL (u,prev->next) unseen-transition rate: {trans_unseen_rate:.3%} "
        f"({val_trans_unseen}/{val_trans_total})"
    )


def log_simple_baselines(train_dataset: TrajectoryDataset, val_dataset: TrajectoryDataset) -> None:
    """Compute simple baselines (global/user/Markov top-1) on validation last timestep."""
    global_next, user_next, user_prev_next = _build_freq_tables(train_dataset)
    global_top1 = _top1_from_counter(global_next)
    user_top1 = {u: _top1_from_counter(cnt) for u, cnt in user_next.items()}
    userprev_top1 = {k: _top1_from_counter(cnt) for k, cnt in user_prev_next.items()}

    base_acc = {"global": 0.0, "user": 0.0, "markov": 0.0}
    cnt = 0

    for sample in val_dataset:
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
        f"[BASELINES|VAL Acc@1] global={base_acc['global']:.4f} "
        f"user={base_acc['user']:.4f} markov={base_acc['markov']:.4f} (N={cnt})"
    )


# ---------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------
def build_poi_feature_matrix(
    poi_feat_csv: str,
    poi_id2idx: Dict[int, int],
    num_pois_with_pad: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build POI feature matrix X with padding row at index 0.

    Returns:
        poi_X: (V, F) float tensor on device (V includes padding).
    """
    poi_feats = pd.read_csv(poi_feat_csv)
    exclude_cols = {"PoiId", "PoiCategoryId"}
    feat_cols = [c for c in poi_feats.columns if c not in exclude_cols]

    feat_mat = np.zeros((num_pois_with_pad, len(feat_cols)), dtype=np.float32)

    idx_map = poi_feats["PoiId"].map(poi_id2idx)
    valid_mask = idx_map.notna()
    target_idx = idx_map[valid_mask].astype(int).values
    feat_vals = poi_feats.loc[valid_mask, feat_cols].to_numpy(dtype=np.float32)

    feat_mat[target_idx] = feat_vals
    poi_X = torch.tensor(feat_mat, dtype=torch.float32, device=device)

    logging.info(f"POI feature CSV shape: {poi_feats.shape}")
    logging.info(f"POI feature dim (excluding PoiId/PoiCategoryId): {len(feat_cols)}")
    return poi_X


def build_models_and_optim(
    *,
    args,
    num_users: int,
    num_pois: int,
    num_cats: int,
    poi_X: torch.Tensor,
    poi_traj_data,
    poi_idx2cat_idx: Dict[int, int],
    space_table: torch.Tensor,
    user_pad_idx: torch.Tensor,
    user_pad_w: torch.Tensor,
    poi_latlon_deg: torch.Tensor,
):
    """Build all modules and optimizer/scheduler."""
    device = args.device

    # 1) User embedding (long-term preference; includes social neighbors)
    user_emb = UserEmbedding(
        num_users=num_users,
        dim=args.user_dim,
        user_pad_idx=user_pad_idx,
        user_pad_w=user_pad_w,
        padding_idx=0,
    ).to(device)

    # 2) Category embedding
    cat_emb = CatEmbedding(num_cats=num_cats, dim=args.cat_dim, padding_idx=0).to(device)

    # 3) POI encoder (POI features + category embedding -> GCN)
    poi_cat_idx_full = torch.zeros(num_pois, dtype=torch.long, device=device)
    for poi_idx, cat_idx in poi_idx2cat_idx.items():
        if 0 <= poi_idx < num_pois:
            poi_cat_idx_full[poi_idx] = int(cat_idx)

    poi_encoder = POIEncoderGCN(
        in_dim=poi_X.shape[1] + args.cat_dim,
        hid_dim=args.poi_dim,
        out_dim=args.poi_dim,
    ).to(device)

    # 4) Time embedding
    time_emb_model = TimeEmbedding(dim=args.time_dim).to(device)

    # 5) Space embedding (pretrained; frozen)
    space_emb = torch.nn.Embedding.from_pretrained(space_table, freeze=True, padding_idx=0).to(device)

    # 6) Fusion model (user/poi/time/space/cat -> token embedding)
    check_in_fusion_model = CheckInFusion(
        d_user=args.user_dim,
        d_poi=args.poi_dim,
        d_time=args.time_dim,
        d_space=args.space_dim,
        d_cat=args.cat_dim,
        out_dim=args.input_tok_dim,
        gate=args.fusion_gate,
    ).to(device)

    # 7) Main predictor
    pfa = PFA()
    main_model = NextPOIWithPFA(
        pfa=pfa,
        num_pois=num_pois,
        num_cats=num_cats,
        logit_mode=args.logit_mode,
        poi_proj_dim=args.poi_dim,
        cat_proj_dim=args.cat_dim,
        learnable_scale=getattr(args, "learnable_scale", False),
        tail_gamma=getattr(args, "tail_gamma", 1.0),
        label_ignore_index=PAD_LABEL,
        temperature=getattr(args, "temperature", 1.0),
        lambda_poi=getattr(args, "lambda_poi", 1.0),
        lambda_time=getattr(args, "lambda_time", 0.2),
        lambda_cat=getattr(args, "lambda_cat", 0.2),
        lambda_loc=getattr(args, "lambda_loc", 0.2),
    ).to(device)

    main_model.set_poi_latlon(poi_latlon_deg.to(device))

    modules_for_optim = [user_emb, cat_emb, poi_encoder, time_emb_model, check_in_fusion_model, main_model]
    tr, al, per_mod, mem = count_params_from_modules(modules_for_optim)
    logging.info(f"Params (unique): trainable={tr:,} / total={al:,}")
    for row in per_mod:
        logging.info(f"{row['name']}: trainable={row['trainable']:,} / total={row['total']:,}")
    logging.info(f"Memory estimate (params only): {mem}")

    param_groups = build_adamw_param_groups(modules_for_optim, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.lr_scheduler_factor)

    return (
        user_emb,
        cat_emb,
        poi_encoder,
        time_emb_model,
        space_emb,
        check_in_fusion_model,
        main_model,
        poi_cat_idx_full,
        optimizer,
        scheduler,
    )


def run_one_epoch(
    *,
    phase: str,
    loader: DataLoader,
    args,
    user_emb,
    cat_emb,
    poi_encoder,
    time_emb_model,
    space_emb,
    check_in_fusion_model,
    main_model,
    poi_X: torch.Tensor,
    poi_traj_data,
    poi_cat_idx_full: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    """
    Run one epoch of training or validation.

    Returns:
        Dictionary of aggregated metrics for the epoch.
    """
    is_train = (optimizer is not None)
    device = args.device

    if is_train:
        user_emb.train()
        cat_emb.train()
        poi_encoder.train()
        time_emb_model.train()
        check_in_fusion_model.train()
        main_model.train()
    else:
        user_emb.eval()
        cat_emb.eval()
        poi_encoder.eval()
        time_emb_model.eval()
        check_in_fusion_model.eval()
        main_model.eval()

    # Accumulators
    losses, poi_losses, time_losses, cat_losses, loc_losses = [], [], [], [], []
    top1s, top5s, top10s, top20s, map20s, mrrs = [], [], [], [], [], []

    log_every = getattr(args, "log_every_n_batches", 200)
    log_sample_idx = getattr(args, "log_sample_idx", 0)
    log_topk = getattr(args, "log_topk", 10)
    log_preview_steps = getattr(args, "log_preview_steps", 5)

    iterator = tqdm(loader, desc=f"{phase}", leave=False)
    for b_idx, batch in enumerate(iterator):
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        # Move tensors to device
        batch_gpu = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        # --- POI embedding update (current encoder state) ---
        cat_plus = cat_emb(poi_cat_idx_full)                     # (V, cat_dim)
        poi_input = torch.cat([poi_X, cat_plus], dim=1).to(device)  # (V, F+cat_dim)
        poi_final_emb = poi_encoder(poi_input, poi_traj_data)     # (V, poi_dim)

        # --- Token embeddings for trajectory sequence ---
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
            time_embed_model=time_emb_model,
            check_in_fusion_model=check_in_fusion_model,
        )  # (B, L, H)

        # Labels / masks
        y_poi = batch_gpu["y_poi"]
        y_cat = batch_gpu["y_cat"]
        y_time = batch_gpu["y_time"]
        attention_mask = batch_gpu["attention_mask"]
        loss_mask = attention_mask.bool()

        tgt_time_feats = batch_gpu["tgt_time_feats"]  # (B, L, 3)
        tgt_time_emb = time_emb_model(
            tgt_time_feats[..., 0],  # weekday_frac
            tgt_time_feats[..., 1],  # time_frac
            tgt_time_feats[..., 2],  # holiday_flag
        )

        # Forward
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
        logits_cat = out.get("logits_cat", None)
        logits_time = out.get("pred_time", None)
        logits_loc = out.get("pred_loc", None)

        # Always mask padding index 0
        logits_poi[..., 0] = -1e9
        if logits_cat is not None:
            logits_cat[..., 0] = -1e9

        # Loss
        loss = torch.zeros((), device=device)

        poi_ce = main_model._shifted_ce_loss(logits=logits_poi, labels=y_poi, mask=loss_mask)
        loss = loss + args.lambda_poi * poi_ce

        time_loss_val = None
        if args.lambda_time > 0.0 and logits_time is not None:
            time_loss_val = main_model._shifted_time_cos_loss(pred_time=logits_time, y_time=y_time, mask=loss_mask)
            loss = loss + args.lambda_time * time_loss_val

        cat_ce = None
        if args.lambda_cat > 0.0 and logits_cat is not None:
            cat_ce = main_model._shifted_ce_loss(logits=logits_cat, labels=y_cat, mask=loss_mask)
            loss = loss + args.lambda_cat * cat_ce

        loc_loss_val = None
        if args.lambda_loc > 0.0 and logits_loc is not None:
            lat_rad = (math.pi / 2) * torch.tanh(logits_loc[..., 0])
            lon_rad = math.pi * torch.tanh(logits_loc[..., 1])
            pred_loc_rad = torch.stack([lat_rad, lon_rad], dim=-1)
            loc_loss_val = main_model._shifted_loc_haversine_loss(
                pred_loc_rad=pred_loc_rad, labels_poi=y_poi, mask=loss_mask
            )
            loss = loss + args.lambda_loc * loc_loss_val

        if is_train:
            loss.backward()
            optimizer.step()

        # Metrics @ last valid timestep
        with torch.no_grad():
            metrics = compute_last_step_metrics(
                logits_poi=logits_poi,
                y_poi=y_poi,
                lengths=attention_mask.sum(dim=1),
                topk=(1, 5, 10, 20),
                pad_label=PAD_LABEL,
            )

        # Accumulate
        losses.append(float(loss.detach().cpu()))
        poi_losses.append(float(poi_ce.detach().cpu()))
        if time_loss_val is not None:
            time_losses.append(float(time_loss_val.detach().cpu()))
        if cat_ce is not None:
            cat_losses.append(float(cat_ce.detach().cpu()))
        if loc_loss_val is not None:
            loc_losses.append(float(loc_loss_val.detach().cpu()))

        top1s.append(metrics["top1"])
        top5s.append(metrics["top5"])
        top10s.append(metrics["top10"])
        top20s.append(metrics["top20"])
        map20s.append(metrics["map20"])
        mrrs.append(metrics["mrr"])

        iterator.set_postfix({"loss": np.mean(losses), "top1": np.mean(top1s)})

        # Optional batch-level prediction logging
        if (b_idx % log_every) == 0 and getattr(args, "enable_batch_logging", False):
            _log_batch_predictions_new(
                batch=batch,
                logits_poi=logits_poi,
                y_poi=y_poi,
                attention_mask=attention_mask,
                phase=f"{phase}-b{b_idx}",
                sample_idx=log_sample_idx,
                topk=log_topk,
                preview_steps=log_preview_steps,
                pad_label=PAD_LABEL,
            )

    # Aggregate epoch stats
    def safe_mean(xs):
        return float(np.mean(xs)) if len(xs) else 0.0

    return {
        "loss": safe_mean(losses),
        "poi_loss": safe_mean(poi_losses),
        "time_loss": safe_mean(time_losses),
        "cat_loss": safe_mean(cat_losses),
        "loc_loss": safe_mean(loc_losses),
        "top1": safe_mean(top1s),
        "top5": safe_mean(top5s),
        "top10": safe_mean(top10s),
        "top20": safe_mean(top20s),
        "map20": safe_mean(map20s),
        "mrr": safe_mean(mrrs),
    }


def train(args) -> None:
    """Main training routine."""
    logging.info("Preparing run...")
    args = prepare_run(args)

    device = torch.device(args.device) if isinstance(args.device, str) else args.device
    args.device = device

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    logging.info("Loading data...")
    df = pd.read_csv(args.data_path)

    train_df = df[df["SplitTag"] == "train"].reset_index(drop=True)

    logging.info("Loading geohash embeddings...")
    space_table, geohash2idx = build_space_table(args.geohash_embedding)
    logging.info(f"Geohash space table: {space_table.shape}")

    logging.info("Building ID mappings...")
    sizes, mappings, user_set = build_mappings(train_df, geohash2idx)
    num_pois = sizes["num_pois"] + 1
    num_users = sizes["num_users"] + 1
    num_cats = sizes["num_cats"] + 1

    user_id2idx = mappings["user_id2idx"]
    poi_id2idx = mappings["poi_id2idx"]
    poi_idx2cat_idx = mappings["poi_idx2cat_idx"]
    poi_idx2geo_idx = mappings["poi_idx2geo_idx"]
    poi_idx2latlon = mappings["poi_idx2latlon"]

    logging.info(f"#POIs={sizes['num_pois']}, #Users={sizes['num_users']}, #Cats={sizes['num_cats']} (padding included)")

    logging.info("Building POI feature matrix...")
    poi_X = build_poi_feature_matrix(args.poi_feat, poi_id2idx, num_pois, device)

    logging.info("Loading POI transition graph...")
    poi_traj_G = pd.read_csv(args.poi_traj_graph)
    poi_traj_data = df_to_pyg_data(poi_traj_G, poi_id2idx, device=device)

    logging.info("Loading user neighbor graph...")
    user_neighbors_data = torch.load(args.user_neighbors)
    user_pad_idx = user_neighbors_data["pad_idx"].to(device)
    user_pad_w = user_neighbors_data["pad_w"].to(device)

    # POI lat/lon table for location loss
    poi_latlon_deg = torch.zeros(num_pois, 2, dtype=torch.float32)
    for poi_idx, (lat, lon) in poi_idx2latlon.items():
        poi_latlon_deg[poi_idx] = torch.tensor([lat, lon], dtype=torch.float32)
    poi_latlon_deg = poi_latlon_deg.to(device)

    # ------------------------------------------------------------
    # Datasets / loaders
    # ------------------------------------------------------------
    logging.info("Building datasets...")
    train_dataset = TrajectoryDataset(
        df=df,
        split_tag="train",
        user_set=user_set,
        user_id2idx=user_id2idx,
        poi_id2idx=poi_id2idx,
        short_traj_thres=args.short_traj_thres,
    )
    val_dataset = TrajectoryDataset(
        df=df,
        split_tag="validation",
        user_set=user_set,
        user_id2idx=user_id2idx,
        poi_id2idx=poi_id2idx,
        short_traj_thres=args.short_traj_thres,
    )

    logging.info(f"len(train_dataset)={len(train_dataset)}, len(val_dataset)={len(val_dataset)}")
    if len(val_dataset) == 0:
        logging.warning("Validation dataset is empty after filtering. Check user mapping and short_traj_thres.")

    if getattr(args, "enable_diagnostics", True):
        log_val_coverage(train_dataset, val_dataset)
        log_simple_baselines(train_dataset, val_dataset)

    collator = TrajectoryCollator2(
        poi_id2cat_idx=poi_idx2cat_idx,
        poi_id2geo_idx=poi_idx2geo_idx,
        pad_label=PAD_LABEL,
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=collator,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # ------------------------------------------------------------
    # Models / optimizer
    # ------------------------------------------------------------
    logging.info("Building models...")
    (
        user_emb,
        cat_emb,
        poi_encoder,
        time_emb_model,
        space_emb,
        check_in_fusion_model,
        main_model,
        poi_cat_idx_full,
        optimizer,
        scheduler,
    ) = build_models_and_optim(
        args=args,
        num_users=num_users,
        num_pois=num_pois,
        num_cats=num_cats,
        poi_X=poi_X,
        poi_traj_data=poi_traj_data,
        poi_idx2cat_idx=poi_idx2cat_idx,
        space_table=space_table,
        user_pad_idx=user_pad_idx,
        user_pad_w=user_pad_w,
        poi_latlon_deg=poi_latlon_deg,
    )

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_score = -np.inf
    best_epoch = -1
    epochs_wo_improve = 0

    train_hist = []
    val_hist = []

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 40} Epoch {epoch:03d} {'*' * 40}")

        train_stats = run_one_epoch(
            phase=f"train@e{epoch}",
            loader=train_loader,
            args=args,
            user_emb=user_emb,
            cat_emb=cat_emb,
            poi_encoder=poi_encoder,
            time_emb_model=time_emb_model,
            space_emb=space_emb,
            check_in_fusion_model=check_in_fusion_model,
            main_model=main_model,
            poi_X=poi_X,
            poi_traj_data=poi_traj_data,
            poi_cat_idx_full=poi_cat_idx_full,
            optimizer=optimizer,
        )

        val_stats = run_one_epoch(
            phase=f"val@e{epoch}",
            loader=val_loader,
            args=args,
            user_emb=user_emb,
            cat_emb=cat_emb,
            poi_encoder=poi_encoder,
            time_emb_model=time_emb_model,
            space_emb=space_emb,
            check_in_fusion_model=check_in_fusion_model,
            main_model=main_model,
            poi_X=poi_X,
            poi_traj_data=poi_traj_data,
            poi_cat_idx_full=poi_cat_idx_full,
            optimizer=None,
        )

        train_hist.append(train_stats)
        val_hist.append(val_stats)

        # Monitor score (same formula as your original code)
        monitor_score = (
            0.4 * val_stats["top1"]
            + 0.2 * val_stats["mrr"]
            + 0.2 * val_stats["top5"]
            + 0.2 * val_stats["top10"]
        )

        scheduler.step(monitor_score)

        logging.info(
            "Train | "
            f"loss={train_stats['loss']:.4f} poi={train_stats['poi_loss']:.4f} "
            f"time={train_stats['time_loss']:.4f} cat={train_stats['cat_loss']:.4f} loc={train_stats['loc_loss']:.4f} | "
            f"top1={train_stats['top1']:.4f} top5={train_stats['top5']:.4f} top10={train_stats['top10']:.4f} "
            f"top20={train_stats['top20']:.4f} map20={train_stats['map20']:.4f} mrr={train_stats['mrr']:.4f}"
        )
        logging.info(
            "Val   | "
            f"loss={val_stats['loss']:.4f} poi={val_stats['poi_loss']:.4f} "
            f"time={val_stats['time_loss']:.4f} cat={val_stats['cat_loss']:.4f} loc={val_stats['loc_loss']:.4f} | "
            f"top1={val_stats['top1']:.4f} top5={val_stats['top5']:.4f} top10={val_stats['top10']:.4f} "
            f"top20={val_stats['top20']:.4f} map20={val_stats['map20']:.4f} mrr={val_stats['mrr']:.4f}"
        )

        # Save best checkpoint
        is_best = monitor_score > best_score
        if is_best:
            logging.info("Saving best checkpoint...")
            best_score = monitor_score
            best_epoch = epoch
            epochs_wo_improve = 0

            if args.save_weights:
                state = {
                    "epoch": epoch,
                    "best_score": best_score,
                    "main_model_state_dict": main_model.state_dict(),
                    "poi_encoder_state_dict": poi_encoder.state_dict(),
                    "user_emb_state_dict": user_emb.state_dict(),
                    "cat_emb_state_dict": cat_emb.state_dict(),
                    "time_embed_model_state_dict": time_emb_model.state_dict(),
                    "check_in_fusion_state_dict": check_in_fusion_model.state_dict(),
                    # For reproducibility / re-loading
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "mappings": mappings,
                    "args": vars(args),
                }
                torch.save(state, ckpt_dir / "best_epoch.state.pt")
                with open(ckpt_dir / "best_epoch.txt", "w") as f:
                    f.write(f"epoch={epoch}, score={best_score:.6f}\n")
        else:
            epochs_wo_improve += 1
            logging.info(
                f"No improvement for {epochs_wo_improve} epoch(s) "
                f"(best@{best_epoch}, best_score={best_score:.4f})"
            )

        if epochs_wo_improve >= args.patience:
            logging.info(f"Early stopping at epoch {epoch}. Best epoch={best_epoch}, best_score={best_score:.4f}")
            break

    # ------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------
    logging.info("Training finished.")
    logging.info(f"Best epoch={best_epoch}, best_score={best_score:.4f}")

    train_df = pd.DataFrame(train_hist)
    val_df = pd.DataFrame(val_hist)
    train_df.to_csv(save_dir / "metrics-train.csv", index_label="epoch")
    val_df.to_csv(save_dir / "metrics-val.csv", index_label="epoch")


def main() -> None:
    args = parameter_parser()

    # Enforce device override
    if getattr(args, "no_cuda", False):
        args.device = "cpu"

    # Resolve dataset paths
    paths = resolve_data_paths(args.data)
    args.data_path = paths.data_path
    args.poi_traj_graph = paths.poi_traj_graph
    args.geohash_embedding = paths.geohash_embedding
    args.poi_feat = paths.poi_feat
    args.user_neighbors = paths.user_neighbors

    train(args)


if __name__ == "__main__":
    main()
