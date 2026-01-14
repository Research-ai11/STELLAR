"""
transformer_test.py

Paper-ready evaluation script for STELLAR Next-POI (Transformer backbone ablation).

This script is designed to be compatible with the updated:
  - param_parser.py  (unified presets + transformer knobs: tf_*)
  - model.py         (PFA_Transformer / NextPOIWithPFA / get_batch_inputs_embeds)
  - transformer_train.py (the updated training script you now use)

What this script does:
  1) Loads the dataset CSV and selects SplitTag == 'test'
  2) Loads a trained checkpoint (best_epoch.state.pt)
  3) Rebuilds model components with the SAME architecture settings as training
     - Prefer checkpoint-stored args when available
     - Fallback to current CLI args otherwise
  4) Runs last-timestep ranking metrics: Acc@{1,5,10,20}, mAP@20, MRR
  5) Writes a CSV summary to runs/test/{data}/...

Important notes:
  - YAML serialization issues are avoided by only logging args as strings when needed.
  - We compute poi_final_emb ONCE under torch.no_grad() (safe and fast).
  - PAD index (0) is always masked in logits before metrics.
"""

from __future__ import annotations

import logging
import os
import random
import pathlib
import zipfile
from pathlib import Path
from typing import Any, Dict

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
    build_space_table,
    df_to_pyg_data,
    _iter_pairs_from_dataset,
    _build_freq_tables,
    _top1_from_counter,
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
# YAML-safe args serializer (for test logs only)
# -----------------------------
def _yaml_safe(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, torch.device):
        return str(obj)
    return str(obj)


def prepare_test_run(args):
    """
    Create a test save directory, configure logger, and snapshot code.
    We intentionally keep this lightweight and paper-friendly.
    """
    set_seed(args.seed)

    # runs/test/{data}/{name}
    args.save_dir = increment_path(
        Path(args.project) / args.data / ("test_" + args.name),
        exist_ok=args.exist_ok,
    )
    os.makedirs(args.save_dir, exist_ok=True)

    # Logger
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(args.save_dir, "log_testing.txt"),
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    logging.getLogger("").addHandler(console)
    logging.getLogger("matplotlib.font_manager").disabled = True

    # Dump args (YAML-safe)
    with open(os.path.join(args.save_dir, "args_test.yaml"), "w") as f:
        yaml.safe_dump({k: _yaml_safe(v) for k, v in vars(args).items()}, f, sort_keys=False, allow_unicode=True)

    # Snapshot code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, "code.zip"), "w", zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=[".py"])
    zipf.close()

    logging.info(args)
    return args


# -----------------------------
# Checkpoint arg handling
# -----------------------------
def _coalesce_arg(ckpt_args: Any, cli_args: Any, key: str, default: Any) -> Any:
    """
    Priority: checkpoint args -> cli args -> default.
    Supports both Namespace-like objects and dicts.
    """
    if ckpt_args is not None:
        if isinstance(ckpt_args, dict) and key in ckpt_args:
            return ckpt_args[key]
        if hasattr(ckpt_args, key):
            return getattr(ckpt_args, key)
    if hasattr(cli_args, key):
        return getattr(cli_args, key)
    return default


def _build_transformer_backbone_args(ckpt_args: Any, cli_args: Any) -> Dict[str, Any]:
    """
    Build transformer backbone hyperparameters in a version-tolerant way.

    New parser uses:
      - tf_nhead, tf_layers, tf_ffn_dim, tf_dropout

    Old runs may have:
      - nhead, enc_layers, ffn_dim, enc_dropout
    """
    d = {}
    d["nhead"] = _coalesce_arg(ckpt_args, cli_args, "tf_nhead", None)
    d["num_layers"] = _coalesce_arg(ckpt_args, cli_args, "tf_layers", None)
    d["dim_feedforward"] = _coalesce_arg(ckpt_args, cli_args, "tf_ffn_dim", None)
    d["dropout"] = _coalesce_arg(ckpt_args, cli_args, "tf_dropout", None)

    # Backward-compat fallback for older checkpoints
    if d["nhead"] is None:
        d["nhead"] = _coalesce_arg(ckpt_args, cli_args, "nhead", 8)
    if d["num_layers"] is None:
        d["num_layers"] = _coalesce_arg(ckpt_args, cli_args, "enc_layers", 6)
    if d["dim_feedforward"] is None:
        d["dim_feedforward"] = _coalesce_arg(ckpt_args, cli_args, "ffn_dim", 2048)
    if d["dropout"] is None:
        d["dropout"] = _coalesce_arg(ckpt_args, cli_args, "enc_dropout", 0.1)

    return d


# -----------------------------
# Main test
# -----------------------------
def test(args):
    logging.info("Preparing test environment...")
    prepare_test_run(args)

    # ---------------------- Load data ----------------------
    df = pd.read_csv(args.data_path)
    test_df = df[df["SplitTag"] == "test"].reset_index(drop=True)
    if len(test_df) == 0:
        logging.warning("No test data found (SplitTag == 'test').")

    # ---------------------- Load checkpoint ----------------------
    logging.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    ckpt_args = ckpt.get("args", None)  # could be dict or Namespace
    mappings = ckpt["mappings"]

    user_id2idx = mappings["user_id2idx"]
    poi_id2idx = mappings["poi_id2idx"]
    cat_id2idx = mappings["cat_id2idx"]
    poi_idx2cat_idx = mappings["poi_idx2cat_idx"]
    poi_idx2geo_idx = mappings["poi_idx2geo_idx"]
    poi_idx2latlon = mappings["poi_idx2latlon"]

    num_users = len(user_id2idx) + 1
    num_pois = len(poi_id2idx) + 1
    num_cats = len(cat_id2idx) + 1
    user_set = set(user_id2idx.keys())

    # ---------------------- Load POI features (same as training) ----------------------
    poi_feats = pd.read_csv(args.poi_feat)
    exclude_cols = {"PoiId", "PoiCategoryId"}
    poi_feat_cols = [c for c in poi_feats.columns if c not in exclude_cols]
    poi_feats_dim = len(poi_feat_cols)

    feat_mat = np.zeros((num_pois, poi_feats_dim), dtype=np.float32)
    idx_map = poi_feats["PoiId"].map(poi_id2idx)
    valid_mask = idx_map.notna()
    target_idx = idx_map[valid_mask].astype(int).values
    feat_vals = poi_feats.loc[valid_mask, poi_feat_cols].to_numpy(dtype=np.float32)
    feat_mat[target_idx] = feat_vals
    poi_X = torch.tensor(feat_mat, dtype=torch.float32, device=args.device)

    # ---------------------- Load POI transition graph (PyG Data) ----------------------
    poi_traj_G = pd.read_csv(args.poi_traj_graph)
    poi_traj_data = df_to_pyg_data(poi_traj_G, poi_id2idx, device=args.device)

    # ---------------------- Load user neighbors (for UserEmbedding) ----------------------
    user_neighbors_data = torch.load(args.user_neighbors)
    user_pad_idx = user_neighbors_data["pad_idx"].to(args.device)
    user_pad_w = user_neighbors_data["pad_w"].to(args.device)

    # ---------------------- Load geohash embeddings ----------------------
    space_table, geohash2idx = build_space_table(args.geohash_embedding)
    space_emb = torch.nn.Embedding.from_pretrained(space_table, freeze=True, padding_idx=0).to(args.device)

    # ---------------------- Build models (mirror training) ----------------------
    logging.info("Building model components...")
    user_emb = UserEmbedding(
        num_users=num_users,
        dim=_coalesce_arg(ckpt_args, args, "user_dim", args.user_dim),
        user_pad_idx=user_pad_idx,
        user_pad_w=user_pad_w,
        padding_idx=0,
    ).to(args.device)

    cat_emb = CatEmbedding(
        num_cats=num_cats,
        dim=_coalesce_arg(ckpt_args, args, "cat_dim", args.cat_dim),
        padding_idx=0,
    ).to(args.device)

    # GCN encoder dims (use ckpt args when present)
    poi_dim = _coalesce_arg(ckpt_args, args, "poi_dim", args.poi_dim)
    cat_dim = _coalesce_arg(ckpt_args, args, "cat_dim", args.cat_dim)
    gcn_nhid = _coalesce_arg(ckpt_args, args, "gcn_nhid", getattr(args, "gcn_nhid", poi_dim))
    gcn_dropout = _coalesce_arg(ckpt_args, args, "gcn_dropout", getattr(args, "gcn_dropout", 0.0))

    poi_encoder = POIEncoderGCN(
        in_dim=poi_X.shape[1] + cat_dim,
        hid_dim=gcn_nhid,
        out_dim=poi_dim,
        dropout=gcn_dropout,
    ).to(args.device)

    time_dim = _coalesce_arg(ckpt_args, args, "time_dim", args.time_dim)
    time_embed_model = TimeEmbedding(dim=time_dim).to(args.device)

    input_tok_dim = _coalesce_arg(ckpt_args, args, "input_tok_dim", args.input_tok_dim)
    space_dim = _coalesce_arg(ckpt_args, args, "space_dim", args.space_dim)
    user_dim = _coalesce_arg(ckpt_args, args, "user_dim", args.user_dim)

    check_in_fusion_model = CheckInFusion(
        d_user=user_dim,
        d_poi=poi_dim,
        d_time=time_dim,
        d_space=space_dim,
        d_cat=cat_dim,
        out_dim=input_tok_dim,
        gate=_coalesce_arg(ckpt_args, args, "fusion_gate", getattr(args, "fusion_gate", "none")),
    ).to(args.device)

    # Transformer backbone (version tolerant)
    tf_kwargs = _build_transformer_backbone_args(ckpt_args, args)
    pfa = PFA_Transformer(
        d_model=input_tok_dim,
        nhead=tf_kwargs["nhead"],
        num_layers=tf_kwargs["num_layers"],
        dim_feedforward=tf_kwargs["dim_feedforward"],
        dropout=tf_kwargs["dropout"],
    )

    # Main predictor
    main_model = NextPOIWithPFA(
        pfa=pfa,
        num_pois=num_pois,
        num_cats=num_cats,
        logit_mode=_coalesce_arg(ckpt_args, args, "logit_mode", args.logit_mode),
        poi_proj_dim=poi_dim,
        cat_proj_dim=cat_dim,
        learnable_scale=_coalesce_arg(ckpt_args, args, "learnable_scale", True),
        tail_gamma=_coalesce_arg(ckpt_args, args, "tail_gamma", getattr(args, "tail_gamma", 1.0)),
        label_ignore_index=PAD_LABEL,
        temperature=_coalesce_arg(ckpt_args, args, "temperature", getattr(args, "temperature", 1.0)),
        lambda_poi=_coalesce_arg(ckpt_args, args, "lambda_poi", args.lambda_poi),
        lambda_time=_coalesce_arg(ckpt_args, args, "lambda_time", args.lambda_time),
        lambda_cat=_coalesce_arg(ckpt_args, args, "lambda_cat", args.lambda_cat),
        lambda_loc=_coalesce_arg(ckpt_args, args, "lambda_loc", args.lambda_loc),
        opt="transformer",
    ).to(args.device)

    # ---------------------- Load trained weights ----------------------
    logging.info("Loading model weights from checkpoint...")
    main_model.load_state_dict(ckpt["main_model_state_dict"], strict=False)
    poi_encoder.load_state_dict(ckpt["poi_encoder_state_dict"], strict=True)

    # Support both legacy key names and new key names in checkpoint
    if "time_embed_model_state_dict" in ckpt:
        time_embed_model.load_state_dict(ckpt["time_embed_model_state_dict"], strict=True)
    else:
        time_embed_model.load_state_dict(ckpt["time_embed_model"], strict=True)

    check_in_fusion_model.load_state_dict(ckpt["check_in_fusion_state_dict"], strict=True)
    cat_emb.load_state_dict(ckpt["cat_emb_state_dict"], strict=True)
    user_emb.load_state_dict(ckpt["user_emb_state_dict"], strict=True)

    # ---------------------- Set POI lat/lon (for optional loc loss; safe even if unused) ----------------------
    poi_latlon_deg = torch.zeros(num_pois, 2, dtype=torch.float32, device=args.device)
    for poi_id, (lat, lon) in poi_idx2latlon.items():
        poi_latlon_deg[poi_id] = torch.tensor([lat, lon], dtype=torch.float32, device=args.device)
    main_model.set_poi_latlon(poi_latlon_deg)

    # ---------------------- Dataset / Dataloader ----------------------
    logging.info("Preparing test dataloader...")
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

    # ---------------------- Optional diagnostics: TEST coverage vs TRAIN ----------------------
    if getattr(args, "enable_diagnostics", True):
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
        logging.info(f"[COVERAGE] TEST unseen-POI rate: {poi_unseen_rate:.3%} ({test_poi_unseen}/{test_poi_total})")
        logging.info(
            f"[COVERAGE] TEST unseen-transition rate: {trans_unseen_rate:.3%} "
            f"({test_trans_unseen}/{test_trans_total})"
        )

        # Simple baselines computed from TRAIN
        global_next, user_next, user_prev_next = _build_freq_tables(train_dataset)
        global_top1 = _top1_from_counter(global_next)
        user_top1 = {u: _top1_from_counter(cnt) for u, cnt in user_next.items()}
        userprev_top1 = {k: _top1_from_counter(cnt) for k, cnt in user_prev_next.items()}

        base_acc = {"global": 0, "user": 0, "markov": 0}
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
            base_acc["global"] += int(pred_g == gold) if pred_g is not None else 0

            pred_u = user_top1.get(u, None)
            base_acc["user"] += int(pred_u == gold) if pred_u is not None else 0

            pred_m = userprev_top1.get((u, prev), None)
            base_acc["markov"] += int(pred_m == gold) if pred_m is not None else 0

        for k in base_acc:
            base_acc[k] = base_acc[k] / max(1, cnt)

        logging.info(
            f"[BASELINES|TEST Acc@1] global={base_acc['global']:.4f} "
            f"user={base_acc['user']:.4f} markov={base_acc['markov']:.4f} (N={cnt})"
        )

    collator = TrajectoryCollator2(
        poi_id2cat_idx=poi_idx2cat_idx,
        poi_id2geo_idx=poi_idx2geo_idx,
        pad_label=PAD_LABEL,
    )

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

    # ---------------------- Eval ----------------------
    logging.info("Running evaluation...")
    for m in [main_model, user_emb, poi_encoder, cat_emb, time_embed_model, check_in_fusion_model]:
        m.eval()

    # POI category index tensor (V,)
    poi_cat_idx_full = torch.zeros(num_pois, dtype=torch.long, device=args.device)
    for poi_idx, cat_idx in poi_idx2cat_idx.items():
        if 0 <= poi_idx < num_pois:
            poi_cat_idx_full[poi_idx] = int(cat_idx)

    batches_top1, batches_top5, batches_top10, batches_top20 = [], [], [], []
    batches_map20, batches_mrr = [], []

    with torch.no_grad():
        # Compute POI final embeddings once for TEST
        cat_plus_test = cat_emb(poi_cat_idx_full)
        poi_in_test = torch.cat([poi_X, cat_plus_test], dim=1)
        poi_final_emb_test = poi_encoder(poi_in_test, poi_traj_data)

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
                poi_final_emb=poi_final_emb_test,
                cat_emb=cat_emb,
                space_emb=space_emb,
                time_embed_model=time_embed_model,
                check_in_fusion_model=check_in_fusion_model,
            )

            y_poi = batch_gpu["y_poi"]
            attention_mask = batch_gpu["attention_mask"]
            tgt_time_feats = batch_gpu["tgt_time_feats"]

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
                    poi_final_emb=poi_final_emb_test,
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

            # Mask PAD(0)
            logits_poi[..., 0] = -1e9
            if logits_cat is not None:
                logits_cat[..., 0] = -1e9

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

    # ---------------------- Report / Save ----------------------
    top1 = float(np.mean(batches_top1)) if batches_top1 else 0.0
    top5 = float(np.mean(batches_top5)) if batches_top5 else 0.0
    top10 = float(np.mean(batches_top10)) if batches_top10 else 0.0
    top20 = float(np.mean(batches_top20)) if batches_top20 else 0.0
    map20 = float(np.mean(batches_map20)) if batches_map20 else 0.0
    mrr = float(np.mean(batches_mrr)) if batches_mrr else 0.0

    logging.info(
        "Test Results:\n"
        f"  Top1  = {top1:.4f}\n"
        f"  Top5  = {top5:.4f}\n"
        f"  Top10 = {top10:.4f}\n"
        f"  Top20 = {top20:.4f}\n"
        f"  mAP@20= {map20:.4f}\n"
        f"  MRR   = {mrr:.4f}"
    )

    results_df = pd.DataFrame(
        {
            "Top1": [top1],
            "Top5": [top5],
            "Top10": [top10],
            "Top20": [top20],
            "mAP20": [map20],
            "MRR": [mrr],
        }
    )
    results_df.to_csv(os.path.join(args.save_dir, "test_results.csv"), index=False)


if __name__ == "__main__":
    # Deterministic behavior for cuBLAS (CUDA >= 10.2)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args = parameter_parser()
    set_seed(args.seed)

    # Data path presets (keep consistent with training)
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

    test(args)
