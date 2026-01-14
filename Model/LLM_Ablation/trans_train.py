"""
transformer_train.py

Paper-ready training script for STELLAR Next-POI (Transformer backbone ablation).

This script is updated to match the new:
  - param_parser.py  (unified backbone presets + transformer knobs: tf_*)
  - model.py         (PFA_Transformer / NextPOIWithPFA / get_batch_inputs_embeds)

Key correctness note (important for PyTorch autograd):
  - DO NOT reuse a grad-tracked POI embedding tensor across multiple backward() calls.
  - Therefore:
      * TRAIN: compute poi_final_emb inside each batch
      * VAL:   compute poi_final_emb once per epoch under torch.no_grad()
"""

from __future__ import annotations

import logging
import math
import os
import random
import pathlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from param_parser import parameter_parser
from STELLAR.Model.datasets import TrajectoryDataset
from model import (
    UserEmbedding,
    POIEncoderGCN,
    CatEmbedding,
    TimeEmbedding,
    CheckInFusion,
    PFA_Transformer,
    NextPOIWithPFA,
    get_batch_inputs_embeds,
)
from STELLAR.Model.utils import (
    TrajectoryCollator2,
    increment_path,
    zipdir,
    compute_last_step_metrics,
    build_adamw_param_groups,
    build_space_table,
    build_mappings,
    df_to_pyg_data,
    _iter_pairs_from_dataset,
    _build_freq_tables,
    _top1_from_counter,
    _log_batch_predictions_new,
    count_params_from_modules,
)

PAD_LABEL = -100


# -----------------------------
# Reproducibility helpers
# -----------------------------
def set_seed(seed: int = 42) -> None:
    """Set all relevant random seeds and enforce deterministic algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Deterministic dataloader worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------
# YAML-safe args serializer
# -----------------------------
def _yaml_safe(obj):
    """
    Convert objects that yaml.safe_dump cannot serialize (e.g., Path, torch.device, tensors)
    into simple Python types.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [ _yaml_safe(x) for x in obj ]
    if isinstance(obj, dict):
        return { str(k): _yaml_safe(v) for k, v in obj.items() }
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        # Do not dump tensors into YAML; just store shape/dtype/device for debugging.
        return {
            "_type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
    # Fallback: stringify unknown objects
    return str(obj)


def _dump_args_yaml(args, save_dir: str) -> None:
    """Dump args to YAML safely."""
    args_dict = {k: _yaml_safe(v) for k, v in vars(args).items()}
    with open(os.path.join(save_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(args_dict, f, sort_keys=False, allow_unicode=True)


# -----------------------------
# Run preparation
# -----------------------------
def prepare_run(args):
    """Create save directory, configure logger, snapshot code, and dump args."""
    set_seed(args.seed)

    # Save directory: runs/train/{data}/{logit_mode}/{name or incremented}
    args.save_dir = increment_path(
        Path(args.project) / args.data / args.logit_mode / args.name,
        exist_ok=args.exist_ok,
    )
    os.makedirs(args.save_dir, exist_ok=True)

    # Reset logging handlers
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(args.save_dir, "log_training.txt"),
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger("").addHandler(console)
    logging.getLogger("matplotlib.font_manager").disabled = True

    # Save run settings (YAML-safe)
    logging.info(args)
    _dump_args_yaml(args, args.save_dir)

    # Snapshot code (only .py)
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, "code.zip"), "w", zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=[".py"])
    zipf.close()

    logging.info(f"Seed: {args.seed}")
    logging.info(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}")
    logging.info(f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")

    return args


# -----------------------------
# Training loop
# -----------------------------
def train(args):
    logging.info("Preparing run...")
    prepare_run(args)

    # Enforce transformer backbone for this script (ablation #1)
    if getattr(args, "backbone", "transformer") != "transformer":
        logging.warning(f"This script is for transformer backbone. Overriding args.backbone -> 'transformer'.")
        args.backbone = "transformer"

    # ---------------------- Load data ----------------------
    logging.info("Loading data...")
    df = pd.read_csv(args.data_path)
    train_df = df[df["SplitTag"] == "train"].reset_index(drop=True)

    # ---------------------- Load geohash embeddings ----------------------
    logging.info("Loading geohash embeddings...")
    space_table, geohash2idx = build_space_table(args.geohash_embedding)
    logging.info(f"Geohash space table shape: {space_table.shape}")

    # ---------------------- Build ID mappings (TRAIN-only) ----------------------
    logging.info("Building ID mappings...")
    sizes, mappings, user_set = build_mappings(train_df, geohash2idx)
    num_pois = sizes["num_pois"] + 1
    num_users = sizes["num_users"] + 1
    num_cats = sizes["num_cats"] + 1

    user_id2idx = mappings["user_id2idx"]
    poi_id2idx = mappings["poi_id2idx"]
    cat_id2idx = mappings["cat_id2idx"]
    poi_idx2cat_idx = mappings["poi_idx2cat_idx"]
    poi_idx2geo_idx = mappings["poi_idx2geo_idx"]
    poi_idx2latlon = mappings["poi_idx2latlon"]

    logging.info(
        f"Number of POIs: {sizes['num_pois']}, "
        f"Number of users: {sizes['num_users']}, "
        f"Number of categories: {sizes['num_cats']}"
    )

    # ---------------------- Load POI feature table ----------------------
    logging.info("Loading POI features...")
    poi_feats = pd.read_csv(args.poi_feat)  # columns include PoiId, PoiCategoryId, others...
    exclude_cols = {"PoiId", "PoiCategoryId"}
    poi_feat_cols = [c for c in poi_feats.columns if c not in exclude_cols]
    poi_feats_dim = len(poi_feat_cols)

    feat_mat = np.zeros((num_pois, poi_feats_dim), dtype=np.float32)  # include PAD row
    idx_map = poi_feats["PoiId"].map(poi_id2idx)
    valid_mask = idx_map.notna()
    target_idx = idx_map[valid_mask].astype(int).values
    feat_vals = poi_feats.loc[valid_mask, poi_feat_cols].to_numpy(dtype=np.float32)
    feat_mat[target_idx] = feat_vals

    poi_X = torch.tensor(feat_mat, dtype=torch.float32, device=args.device)
    logging.info(f"POI feature dim (excluding PoiId & PoiCategoryId): {poi_feats_dim}")

    # ---------------------- Load POI transition graph ----------------------
    logging.info("Loading POI transition graph...")
    poi_traj_G = pd.read_csv(args.poi_traj_graph)
    poi_traj_data = df_to_pyg_data(poi_traj_G, poi_id2idx, device=args.device)

    # ---------------------- Load user social neighbors ----------------------
    logging.info("Loading user social neighbors...")
    user_neighbors_data = torch.load(args.user_neighbors)
    user_pad_idx = user_neighbors_data["pad_idx"].to(args.device)
    user_pad_w = user_neighbors_data["pad_w"].to(args.device)

    # ---------------------- POI lat/lon table (for haversine loss) ----------------------
    poi_latlon_deg = torch.zeros(num_pois, 2, dtype=torch.float32)
    for poi_id, (lat, lon) in poi_idx2latlon.items():
        poi_latlon_deg[poi_id] = torch.tensor([lat, lon], dtype=torch.float32)
    poi_latlon_deg = poi_latlon_deg.to(args.device)

    # Batch logging controls
    log_every_n_batches = getattr(args, "log_every_n_batches", 200)
    log_sample_idx = getattr(args, "log_sample_idx", 0)
    log_topk = getattr(args, "log_topk", 10)
    log_preview_steps = getattr(args, "log_preview_steps", 5)

    # ---------------------- Datasets / Dataloaders ----------------------
    logging.info("Preparing datasets/dataloaders...")
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
        logging.warning(
            "Validation dataset is empty after filtering (user/type/length). "
            "Check UserId matching and short_traj_thres."
        )

    # ---------------------- Optional diagnostics: VAL coverage vs TRAIN ----------------------
    if getattr(args, "enable_diagnostics", True):
        logging.info("=== DIAG: coverage on VAL against TRAIN ===")
        train_poi_seen = set()
        train_trans_seen = set()  # (u, prev, next)
        for u, prev, nxt in _iter_pairs_from_dataset(train_dataset):
            train_poi_seen.add(nxt)
            train_poi_seen.add(prev)
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
        logging.info(f"[COVERAGE] VAL unseen-POI rate: {poi_unseen_rate:.3%} ({val_poi_unseen}/{val_poi_total})")
        logging.info(
            f"[COVERAGE] VAL unseen-transition rate: {trans_unseen_rate:.3%} "
            f"({val_trans_unseen}/{val_trans_total})"
        )

        # Simple baselines for reference
        global_next, user_next, user_prev_next = _build_freq_tables(train_dataset)
        global_top1 = _top1_from_counter(global_next)
        user_top1 = {u: _top1_from_counter(cnt) for u, cnt in user_next.items()}
        userprev_top1 = {k: _top1_from_counter(cnt) for k, cnt in user_prev_next.items()}

        base_acc = {"global": 0, "user": 0, "markov": 0}
        cnt = 0
        for sample in val_dataset:
            traj_id, user_idx, input_seq, label_seq, prev_poi_seq = sample
            if len(label_seq) == 0:
                continue
            cnt += 1
            gold = label_seq[-1][0]
            prev = input_seq[-1][0]
            u = int(user_idx)

            pred_g = global_top1
            base_acc["global"] += int(pred_g == gold) if pred_g is not None else 0

            pred_u = user_top1.get(u, None)
            base_acc["user"] += int(pred_u == gold) if pred_u is not None else 0

            pred_m = userprev_top1.get((u, prev), None)
            base_acc["markov"] += int(pred_m == gold) if pred_m is not None else 0

        for k in base_acc:
            base_acc[k] = base_acc[k] / max(1, cnt)

        logging.info(
            f"[BASELINES|VAL Acc@1] global={base_acc['global']:.4f} "
            f"user={base_acc['user']:.4f} markov={base_acc['markov']:.4f} (N={cnt})"
        )

    # Collator
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

    # ---------------------- Build models ----------------------
    logging.info("Building models...")

    # (1) User embedding
    user_emb = UserEmbedding(
        num_users=num_users,
        dim=args.user_dim,
        user_pad_idx=user_pad_idx,
        user_pad_w=user_pad_w,
        padding_idx=0,
    ).to(args.device)

    # (2) Category embedding
    cat_emb = CatEmbedding(num_cats=num_cats, dim=args.cat_dim, padding_idx=0).to(args.device)

    # (3) POI encoder (GCN) input = poi_features + cat_embedding(poi)
    poi_cat_idx_full = torch.zeros(num_pois, dtype=torch.long, device=args.device)
    for poi_idx, cat_idx in poi_idx2cat_idx.items():
        if 0 <= poi_idx < num_pois:
            poi_cat_idx_full[poi_idx] = int(cat_idx)

    poi_encoder = POIEncoderGCN(
        in_dim=poi_X.shape[1] + args.cat_dim,
        hid_dim=args.gcn_nhid,
        out_dim=args.poi_dim,
        dropout=args.gcn_dropout,
    ).to(args.device)

    # (4) Time embedding module
    time_emb_model = TimeEmbedding(dim=args.time_dim).to(args.device)

    # (5) Frozen space embedding (pretrained geohash embeddings)
    space_emb = torch.nn.Embedding.from_pretrained(space_table, freeze=True, padding_idx=0).to(args.device)

    # (6) Check-in fusion -> token embedding
    check_in_fusion_model = CheckInFusion(
        d_user=args.user_dim,
        d_poi=args.poi_dim,
        d_time=args.time_dim,
        d_space=args.space_dim,
        d_cat=args.cat_dim,
        out_dim=args.input_tok_dim,
        gate=args.fusion_gate,  # already normalized to None if "none"
    ).to(args.device)

    # (7) Transformer backbone (full fine-tune baseline)
    pfa = PFA_Transformer(
        d_model=args.input_tok_dim,
        nhead=args.tf_nhead,
        num_layers=args.tf_layers,
        dim_feedforward=args.tf_ffn_dim,
        dropout=args.tf_dropout,
    )

    # (8) Main predictor
    main_model = NextPOIWithPFA(
        pfa=pfa,
        num_pois=num_pois,
        num_cats=num_cats,
        logit_mode=args.logit_mode,
        poi_proj_dim=args.poi_dim,
        cat_proj_dim=args.cat_dim,
        learnable_scale=getattr(args, "learnable_scale", True),
        tail_gamma=args.tail_gamma,
        label_ignore_index=PAD_LABEL,
        temperature=args.temperature,
        lambda_poi=args.lambda_poi,
        lambda_time=args.lambda_time,
        lambda_cat=args.lambda_cat,
        lambda_loc=args.lambda_loc,
        opt="transformer",
    ).to(args.device)

    # Set POI lat/lon for haversine loss (once)
    main_model.set_poi_latlon(poi_latlon_deg)

    # ---------------------- Optimizer / Scheduler ----------------------
    modules_for_optim = [user_emb, cat_emb, poi_encoder, time_emb_model, check_in_fusion_model, main_model]
    tr, al, per_mod, mem = count_params_from_modules(modules_for_optim)

    logging.info(f"Params (unique): trainable={tr:,} / total={al:,}")
    for row in per_mod:
        logging.info(f"{row['name']}: trainable={row['trainable']:,} / total={row['total']:,}")
    logging.info(f"Mem estimate: {mem}")

    param_groups = build_adamw_param_groups(modules_for_optim, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_scheduler_factor
    )

    # ---------------------- Train ----------------------
    logging.info("Start training...")

    # Epoch-level logs
    train_epochs_top1_acc_list, train_epochs_top5_acc_list = [], []
    train_epochs_top10_acc_list, train_epochs_top20_acc_list = [], []
    train_epochs_mAP20_list, train_epochs_mrr_list, train_epochs_loss_list = [], [], []
    val_epochs_top1_acc_list, val_epochs_top5_acc_list = [], []
    val_epochs_top10_acc_list, val_epochs_top20_acc_list = [], []
    val_epochs_mAP20_list, val_epochs_mrr_list, val_epochs_loss_list = [], [], []

    max_val_score = -np.inf
    best_score = -np.inf
    best_epoch = -1
    epochs_without_improve = 0

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50} Epoch: {epoch:03d} {'*' * 50}")

        # Set train mode
        for m in [user_emb, cat_emb, poi_encoder, time_emb_model, check_in_fusion_model, main_model]:
            m.train()

        train_batches_top1, train_batches_top5 = [], []
        train_batches_top10, train_batches_top20 = [], []
        train_batches_map20, train_batches_mrr, train_batches_loss = [], [], []

        # Component losses (for reporting)
        train_batches_poi_loss, train_batches_time_loss = [], []
        train_batches_cat_loss, train_batches_loc_loss = [], []

        for b_idx, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}")):
            optimizer.zero_grad(set_to_none=True)

            # Move tensors to device
            batch_gpu = {
                k: v.to(args.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # --------- (TRAIN) compute poi_final_emb INSIDE the batch ---------
            cat_plus = cat_emb(poi_cat_idx_full)                     # (V, cat_dim)
            poi_in = torch.cat([poi_X, cat_plus], dim=1)             # (V, poi_feat_dim+cat_dim)
            poi_final_emb = poi_encoder(poi_in, poi_traj_data)       # (V, poi_dim)

            # Build token embeddings (B, L, H)
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
            )

            # Labels / masks
            y_poi = batch_gpu["y_poi"]
            y_cat = batch_gpu["y_cat"]
            y_time = batch_gpu["y_time"]
            attention_mask = batch_gpu["attention_mask"]
            tgt_time_feats = batch_gpu["tgt_time_feats"]

            tgt_time_emb = time_emb_model(
                tgt_time_feats[..., 0],
                tgt_time_feats[..., 1],
                tgt_time_feats[..., 2],
            )
            loss_mask = attention_mask.bool()

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
                out = main_model(
                    inputs_embeds=x,
                    attention_mask=attention_mask,
                    time_prompt=tgt_time_emb,
                )

            logits_poi = out["logits_poi"]
            logits_cat = out.get("logits_cat", None)
            logits_time = out.get("pred_time", None)
            logits_loc = out.get("pred_loc", None)

            # Always mask PAD(0)
            logits_poi[..., 0] = -1e9
            if logits_cat is not None:
                logits_cat[..., 0] = -1e9

            # Loss
            loss = torch.zeros((), device=args.device)

            poi_ce = main_model._shifted_ce_loss(logits=logits_poi, labels=y_poi, mask=loss_mask)
            train_batches_poi_loss.append(poi_ce.detach().cpu())
            loss = loss + args.lambda_poi * poi_ce

            if args.lambda_time > 0.0 and logits_time is not None:
                time_loss = main_model._shifted_time_cos_loss(pred_time=logits_time, y_time=y_time, mask=loss_mask)
                train_batches_time_loss.append(time_loss.detach().cpu())
                loss = loss + args.lambda_time * time_loss

            if args.lambda_cat > 0.0 and logits_cat is not None:
                cat_ce = main_model._shifted_ce_loss(logits=logits_cat, labels=y_cat, mask=loss_mask)
                train_batches_cat_loss.append(cat_ce.detach().cpu())
                loss = loss + args.lambda_cat * cat_ce

            if args.lambda_loc > 0.0 and logits_loc is not None:
                lat_rad = (math.pi / 2) * torch.tanh(logits_loc[..., 0])
                lon_rad = math.pi * torch.tanh(logits_loc[..., 1])
                pred_loc_rad = torch.stack([lat_rad, lon_rad], dim=-1)
                loc_loss = main_model._shifted_loc_haversine_loss(
                    pred_loc_rad=pred_loc_rad, labels_poi=y_poi, mask=loss_mask
                )
                train_batches_loc_loss.append(loc_loss.detach().cpu())
                loss = loss + args.lambda_loc * loc_loss

            # Backprop
            loss.backward()
            optimizer.step()

            # Metrics (last valid step)
            with torch.no_grad():
                metrics = compute_last_step_metrics(
                    logits_poi=logits_poi,
                    y_poi=y_poi,
                    lengths=attention_mask.sum(dim=1),
                    topk=(1, 5, 10, 20),
                )
                train_batches_top1.append(metrics["top1"])
                train_batches_top5.append(metrics["top5"])
                train_batches_top10.append(metrics["top10"])
                train_batches_top20.append(metrics["top20"])
                train_batches_mrr.append(metrics["mrr"])
                train_batches_map20.append(metrics["map20"])
                train_batches_loss.append(float(loss.detach().cpu()))

            # Optional batch logging
            if getattr(args, "enable_batch_logging", False) and (b_idx % log_every_n_batches == 0) and epoch > 5:
                _log_batch_predictions_new(
                    batch=batch,
                    logits_poi=logits_poi,
                    y_poi=y_poi,
                    attention_mask=attention_mask,
                    phase=f"train@epoch{epoch}-b{b_idx}",
                    sample_idx=log_sample_idx,
                    topk=log_topk,
                    preview_steps=log_preview_steps,
                    pad_label=PAD_LABEL,
                )

        # ---------------------- Validation ----------------------
        for m in [user_emb, cat_emb, poi_encoder, time_emb_model, check_in_fusion_model, main_model]:
            m.eval()

        val_batches_top1, val_batches_top5 = [], []
        val_batches_top10, val_batches_top20 = [], []
        val_batches_map20, val_batches_mrr, val_batches_loss = [], [], []

        val_batches_poi_loss, val_batches_time_loss = [], []
        val_batches_cat_loss, val_batches_loc_loss = [], []

        with torch.no_grad():
            # --------- (VAL) compute poi_final_emb ONCE per epoch ---------
            cat_plus_val = cat_emb(poi_cat_idx_full)
            poi_in_val = torch.cat([poi_X, cat_plus_val], dim=1)
            poi_final_emb_val = poi_encoder(poi_in_val, poi_traj_data)

            for vb_idx, batch in enumerate(tqdm(val_loader, desc=f"Val {epoch}")):
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
                    poi_final_emb=poi_final_emb_val,
                    cat_emb=cat_emb,
                    space_emb=space_emb,
                    time_embed_model=time_emb_model,
                    check_in_fusion_model=check_in_fusion_model,
                )

                y_poi = batch_gpu["y_poi"]
                y_cat = batch_gpu["y_cat"]
                y_time = batch_gpu["y_time"]
                attention_mask = batch_gpu["attention_mask"]
                tgt_time_feats = batch_gpu["tgt_time_feats"]

                tgt_time_emb = time_emb_model(
                    tgt_time_feats[..., 0],
                    tgt_time_feats[..., 1],
                    tgt_time_feats[..., 2],
                )
                loss_mask = attention_mask.bool()

                if args.logit_mode == "cos":
                    out = main_model(
                        inputs_embeds=x,
                        attention_mask=attention_mask,
                        time_prompt=tgt_time_emb,
                        poi_final_emb=poi_final_emb_val,
                        cat_emb_weight=cat_emb.weight,
                    )
                else:
                    out = main_model(
                        inputs_embeds=x,
                        attention_mask=attention_mask,
                        time_prompt=tgt_time_emb,
                    )

                logits_poi = out["logits_poi"]
                logits_cat = out.get("logits_cat", None)
                logits_time = out.get("pred_time", None)
                logits_loc = out.get("pred_loc", None)

                logits_poi[..., 0] = -1e9
                if logits_cat is not None:
                    logits_cat[..., 0] = -1e9

                loss = torch.zeros((), device=args.device)

                poi_ce = main_model._shifted_ce_loss(logits=logits_poi, labels=y_poi, mask=loss_mask)
                val_batches_poi_loss.append(poi_ce.detach().cpu())
                loss = loss + args.lambda_poi * poi_ce

                if args.lambda_time > 0.0 and logits_time is not None:
                    time_loss = main_model._shifted_time_cos_loss(pred_time=logits_time, y_time=y_time, mask=loss_mask)
                    val_batches_time_loss.append(time_loss.detach().cpu())
                    loss = loss + args.lambda_time * time_loss

                if args.lambda_cat > 0.0 and logits_cat is not None:
                    cat_ce = main_model._shifted_ce_loss(logits=logits_cat, labels=y_cat, mask=loss_mask)
                    val_batches_cat_loss.append(cat_ce.detach().cpu())
                    loss = loss + args.lambda_cat * cat_ce

                if args.lambda_loc > 0.0 and logits_loc is not None:
                    lat_rad = (math.pi / 2) * torch.tanh(logits_loc[..., 0])
                    lon_rad = math.pi * torch.tanh(logits_loc[..., 1])
                    pred_loc_rad = torch.stack([lat_rad, lon_rad], dim=-1)
                    loc_loss = main_model._shifted_loc_haversine_loss(
                        pred_loc_rad=pred_loc_rad, labels_poi=y_poi, mask=loss_mask
                    )
                    val_batches_loc_loss.append(loc_loss.detach().cpu())
                    loss = loss + args.lambda_loc * loc_loss

                val_batches_loss.append(float(loss.detach().cpu()))

                metrics = compute_last_step_metrics(
                    logits_poi=logits_poi,
                    y_poi=y_poi,
                    lengths=attention_mask.sum(dim=1),
                    topk=(1, 5, 10, 20),
                )
                val_batches_top1.append(metrics["top1"])
                val_batches_top5.append(metrics["top5"])
                val_batches_top10.append(metrics["top10"])
                val_batches_top20.append(metrics["top20"])
                val_batches_map20.append(metrics["map20"])
                val_batches_mrr.append(metrics["mrr"])

                if getattr(args, "enable_batch_logging", False) and (vb_idx % 20 == 0) and epoch > 4:
                    _log_batch_predictions_new(
                        batch=batch,
                        logits_poi=logits_poi,
                        y_poi=y_poi,
                        attention_mask=attention_mask,
                        phase=f"val@epoch{epoch}-b{vb_idx}",
                        sample_idx=log_sample_idx,
                        topk=log_topk,
                        preview_steps=log_preview_steps,
                        pad_label=PAD_LABEL,
                    )

        # ---------------------- Epoch summary ----------------------
        epoch_train_top1 = float(np.mean(train_batches_top1)) if train_batches_top1 else 0.0
        epoch_train_top5 = float(np.mean(train_batches_top5)) if train_batches_top5 else 0.0
        epoch_train_top10 = float(np.mean(train_batches_top10)) if train_batches_top10 else 0.0
        epoch_train_top20 = float(np.mean(train_batches_top20)) if train_batches_top20 else 0.0
        epoch_train_map20 = float(np.mean(train_batches_map20)) if train_batches_map20 else 0.0
        epoch_train_mrr = float(np.mean(train_batches_mrr)) if train_batches_mrr else 0.0
        epoch_train_loss = float(np.mean(train_batches_loss)) if train_batches_loss else 0.0

        epoch_train_poi_loss = float(np.mean(train_batches_poi_loss)) if train_batches_poi_loss else 0.0
        epoch_train_time_loss = float(np.mean(train_batches_time_loss)) if train_batches_time_loss else 0.0
        epoch_train_cat_loss = float(np.mean(train_batches_cat_loss)) if train_batches_cat_loss else 0.0
        epoch_train_loc_loss = float(np.mean(train_batches_loc_loss)) if train_batches_loc_loss else 0.0

        epoch_val_top1 = float(np.mean(val_batches_top1)) if val_batches_top1 else 0.0
        epoch_val_top5 = float(np.mean(val_batches_top5)) if val_batches_top5 else 0.0
        epoch_val_top10 = float(np.mean(val_batches_top10)) if val_batches_top10 else 0.0
        epoch_val_top20 = float(np.mean(val_batches_top20)) if val_batches_top20 else 0.0
        epoch_val_map20 = float(np.mean(val_batches_map20)) if val_batches_map20 else 0.0
        epoch_val_mrr = float(np.mean(val_batches_mrr)) if val_batches_mrr else 0.0
        epoch_val_loss = float(np.mean(val_batches_loss)) if val_batches_loss else 0.0

        epoch_val_poi_loss = float(np.mean(val_batches_poi_loss)) if val_batches_poi_loss else 0.0
        epoch_val_time_loss = float(np.mean(val_batches_time_loss)) if val_batches_time_loss else 0.0
        epoch_val_cat_loss = float(np.mean(val_batches_cat_loss)) if val_batches_cat_loss else 0.0
        epoch_val_loc_loss = float(np.mean(val_batches_loc_loss)) if val_batches_loc_loss else 0.0

        # Store for CSV
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1)
        train_epochs_top5_acc_list.append(epoch_train_top5)
        train_epochs_top10_acc_list.append(epoch_train_top10)
        train_epochs_top20_acc_list.append(epoch_train_top20)
        train_epochs_mAP20_list.append(epoch_train_map20)
        train_epochs_mrr_list.append(epoch_train_mrr)

        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1)
        val_epochs_top5_acc_list.append(epoch_val_top5)
        val_epochs_top10_acc_list.append(epoch_val_top10)
        val_epochs_top20_acc_list.append(epoch_val_top20)
        val_epochs_mAP20_list.append(epoch_val_map20)
        val_epochs_mrr_list.append(epoch_val_mrr)

        # Monitoring score for scheduler + best saving
        monitor_score = (
            0.4 * epoch_val_top1
            + 0.2 * epoch_val_mrr
            + 0.2 * epoch_val_top5
            + 0.2 * epoch_val_top10
        )
        lr_scheduler.step(monitor_score)

        logging.info(
            f"Epoch {epoch}/{args.epochs}\n"
            f"train_loss={epoch_train_loss:.4f} "
            f"(poi={epoch_train_poi_loss:.4f}, time={epoch_train_time_loss:.4f}, "
            f"cat={epoch_train_cat_loss:.4f}, loc={epoch_train_loc_loss:.4f}) | "
            f"train@1={epoch_train_top1:.4f} @5={epoch_train_top5:.4f} @10={epoch_train_top10:.4f} "
            f"@20={epoch_train_top20:.4f} mAP@20={epoch_train_map20:.4f} mrr={epoch_train_mrr:.4f}\n"
            f"val_loss={epoch_val_loss:.4f} "
            f"(poi={epoch_val_poi_loss:.4f}, time={epoch_val_time_loss:.4f}, "
            f"cat={epoch_val_cat_loss:.4f}, loc={epoch_val_loc_loss:.4f}) | "
            f"val@1={epoch_val_top1:.4f} @5={epoch_val_top5:.4f} @10={epoch_val_top10:.4f} "
            f"@20={epoch_val_top20:.4f} mAP@20={epoch_val_map20:.4f} mrr={epoch_val_mrr:.4f} | "
            f"score={monitor_score:.6f}"
        )

        # ---------------------- Save best checkpoint ----------------------
        model_save_dir = os.path.join(args.save_dir, "checkpoints")
        os.makedirs(model_save_dir, exist_ok=True)

        is_best = monitor_score >= max_val_score
        if is_best:
            logging.info("Saving best model checkpoint...")

            if args.save_weights:
                state_dict = {
                    "epoch": epoch,
                    "main_model_state_dict": main_model.state_dict(),
                    "poi_encoder_state_dict": poi_encoder.state_dict(),
                    "user_emb_state_dict": user_emb.state_dict(),
                    "cat_emb_state_dict": cat_emb.state_dict(),
                    "time_embed_model_state_dict": time_emb_model.state_dict(),
                    "check_in_fusion_state_dict": check_in_fusion_model.state_dict(),
                    # NOTE: do NOT save poi_final_emb tensors as "state"; they are derived each run.
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": lr_scheduler.state_dict(),
                    "mappings": mappings,
                    "args": {k: _yaml_safe(v) for k, v in vars(args).items()},
                }
                torch.save(state_dict, os.path.join(model_save_dir, "best_epoch.state.pt"))
                with open(os.path.join(model_save_dir, "best_epoch.txt"), "w") as f:
                    f.write(f"epoch={epoch}, score={monitor_score:.6f}\n")
                    f.write(
                        f"val: loss={epoch_val_loss:.6f}, top1={epoch_val_top1:.6f}, top5={epoch_val_top5:.6f}, "
                        f"top10={epoch_val_top10:.6f}, top20={epoch_val_top20:.6f}, "
                        f"mAP20={epoch_val_map20:.6f}, mrr={epoch_val_mrr:.6f}\n"
                    )

            max_val_score = monitor_score
            best_score = monitor_score
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            logging.info(
                f"No improvement for {epochs_without_improve} epoch(s) "
                f"(best @ {best_epoch}, best_score={best_score:.4f})"
            )

        if epochs_without_improve >= args.patience:
            logging.info(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch} (score={best_score:.4f})"
            )
            break

    # ---------------------- Save metrics CSV ----------------------
    logging.info("Training finished.")
    logging.info(f"Best val score: {max_val_score:.4f}")

    train_metrics_df = pd.DataFrame(
        {
            "loss": train_epochs_loss_list,
            "top1_acc": train_epochs_top1_acc_list,
            "top5_acc": train_epochs_top5_acc_list,
            "top10_acc": train_epochs_top10_acc_list,
            "top20_acc": train_epochs_top20_acc_list,
            "mAP20": train_epochs_mAP20_list,
            "MRR": train_epochs_mrr_list,
        }
    )
    train_metrics_df.to_csv(os.path.join(args.save_dir, "metrics-train.csv"), index_label="epoch")

    val_metrics_df = pd.DataFrame(
        {
            "loss": val_epochs_loss_list,
            "top1_acc": val_epochs_top1_acc_list,
            "top5_acc": val_epochs_top5_acc_list,
            "top10_acc": val_epochs_top10_acc_list,
            "top20_acc": val_epochs_top20_acc_list,
            "mAP20": val_epochs_mAP20_list,
            "MRR": val_epochs_mrr_list,
        }
    )
    val_metrics_df.to_csv(os.path.join(args.save_dir, "metrics-val.csv"), index_label="epoch")


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # Deterministic behavior for cuBLAS (CUDA >= 10.2)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args = parameter_parser()
    set_seed(args.seed)

    # Data path presets (keep your original relative structure)
    if args.data == "nyc":
        args.data_path = "../../data/nyc/raw/NYC.csv"
        args.poi_traj_graph = "../../data/nyc/graph/nyc_traj_graph.csv"
        args.geohash_embedding = "../../data/nyc/graph/nyc_geohash_gat_space_embedding.csv"
        args.poi_feat = "../../data/nyc/graph/nyc_poi_feat.csv"
        args.user_neighbors = "../../data/nyc/graph/nyc_neighbors.pt"
    elif args.data == "tky":
        args.data_path = "../../data/tky/raw/TKY.csv"
        args.poi_traj_graph = "../../data/tky/graph/tky_traj_graph.csv"
        args.geohash_embedding = "../../data/tky/graph/tky_geohash_gat_space_embedding.csv"
        args.poi_feat = "../../data/tky/graph/tky_poi_feat.csv"
        args.user_neighbors = "../../data/tky/graph/tky_neighbors.pt"
    else:
        raise ValueError(f"Unknown dataset key: {args.data}")

    # Respect --no_cuda
    if getattr(args, "no_cuda", False):
        args.device = "cpu"

    train(args)
