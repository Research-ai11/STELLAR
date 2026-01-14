# llama_test.py
# Test script compatible with the (refactored) llama_train.py pipeline.
# - Loads mappings + model weights from the train checkpoint
# - Rebuilds the same modules/config
# - Evaluates last-step TopK / mAP@20 / MRR on SplitTag == "test"
#
# Notes:
# 1) This script assumes your unified param_parser has at least:
#    --data, --seed, --device, --batch, --workers, --short-traj-thres,
#    --project, --name, --exist-ok, --checkpoint,
#    embedding dims (poi_dim, cat_dim, time_dim, space_dim, user_dim, input_tok_dim),
#    --logit_mode, --fusion_gate,
#    --lambda_poi, --lambda_time, --lambda_cat, --lambda_loc,
#    and llama config args used by train (if any).
# 2) For strict reproducibility, keep the same seed setup as train.

import logging
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
from Next_POI_Recommendation.Model.datasets import TrajectoryDataset
from model import (
    UserEmbedding,
    POIEncoderGCN,
    CatEmbedding,
    TimeEmbedding,
    CheckInFusion,
    PFA_LLAMA,
    NextPOIWithPFA,
    get_batch_inputs_embeds,
)
from Next_POI_Recommendation.Model.utils import (
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


# -----------------------
# Reproducibility helpers
# -----------------------
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # If you enabled this in train, enable here too.
    # torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------
# Logging / run directory
# -----------------------
def prepare_test_run(args):
    set_seed(args.seed)

    # Save under runs/train/<data>/test_<name>/expN
    args.save_dir = increment_path(
        Path(args.project) / args.data / ("test_" + args.name),
        exist_ok=args.exist_ok,
    )
    os.makedirs(args.save_dir, exist_ok=True)

    # Reset root handlers
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

    # Save args
    with open(os.path.join(args.save_dir, "args_test.yaml"), "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Snapshot code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, "code.zip"), "w", zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=[".py"])
    zipf.close()

    logging.info(args)
    return args


# -----------------------
# Build dataset paths
# -----------------------
def attach_paths_by_dataset(args):
    """
    Keep this consistent with llama_train.py.
    (Your train code used: NYC.csv / TKY.csv in ../../data/.../raw/)
    """
    if args.data == "nyc":
        args.data_path = "../../data/nyc/raw/NYC.csv"
        args.poi_traj_graph = "../../data/nyc/graph/nyc_traj_graph.csv"
        args.geohash_embedding = "../../data/nyc/graph/nyc_geohash_gat_space_embedding_llama.csv"
        args.poi_feat = "../../data/nyc/graph/nyc_poi_feat.csv"
        args.user_neighbors = "../../data/nyc/graph/nyc_neighbors.pt"
    elif args.data == "tky":
        args.data_path = "../../data/tky/raw/TKY.csv"
        args.poi_traj_graph = "../../data/tky/graph/tky_traj_graph.csv"
        args.geohash_embedding = "../../data/tky/graph/tky_geohash_gat_space_embedding_llama.csv"
        args.poi_feat = "../../data/tky/graph/tky_poi_feat.csv"
        args.user_neighbors = "../../data/tky/graph/tky_neighbors.pt"
    else:
        raise ValueError(f"Unknown dataset key: {args.data}")
    return args


# -----------------------
# Main testing
# -----------------------
@torch.no_grad()
def test(args):
    prepare_test_run(args)

    if not args.checkpoint:
        raise ValueError("--checkpoint must be provided for testing.")

    # -------- Load checkpoint --------
    logging.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    # Use mappings saved in checkpoint (same as train)
    mappings = ckpt["mappings"]
    user_id2idx = mappings["user_id2idx"]
    poi_id2idx = mappings["poi_id2idx"]
    cat_id2idx = mappings["cat_id2idx"]
    poi_idx2cat_idx = mappings["poi_idx2cat_idx"]
    poi_idx2geo_idx = mappings["poi_idx2geo_idx"]
    poi_idx2latlon = mappings["poi_idx2latlon"]
    user_set = set(user_id2idx.keys())

    num_users = len(user_id2idx) + 1
    num_pois = len(poi_id2idx) + 1
    num_cats = len(cat_id2idx) + 1

    # Optional: read args from checkpoint for verification/debug
    ckpt_args = ckpt.get("args", None)
    if ckpt_args is not None:
        logging.info("Checkpoint contains training args. (Not overriding test args.)")

    # -------- Load data --------
    logging.info(f"Loading data: {args.data_path}")
    df = pd.read_csv(args.data_path)

    # -------- Load geohash embeddings (same as train) --------
    logging.info("Loading geohash embeddings...")
    space_table, _ = build_space_table(args.geohash_embedding)
    space_emb = torch.nn.Embedding.from_pretrained(space_table, freeze=True, padding_idx=0).to(args.device)

    # -------- Load POI features (same as train) --------
    logging.info("Loading POI features...")
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

    # -------- Load POI transition graph --------
    logging.info("Loading POI transition graph...")
    poi_traj_G = pd.read_csv(args.poi_traj_graph)
    poi_traj_data = df_to_pyg_data(poi_traj_G, poi_id2idx, device=args.device)

    # -------- Load user neighbors --------
    logging.info("Loading user neighbors...")
    user_neighbors_data = torch.load(args.user_neighbors, map_location=args.device)
    user_pad_idx = user_neighbors_data["pad_idx"].to(args.device)
    user_pad_w = user_neighbors_data["pad_w"].to(args.device)

    # -------- Build modules (same structure as train) --------
    logging.info("Building models...")
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
        dropout=getattr(args, "gcn_dropout", 0.3),
    ).to(args.device)

    time_embed_model = TimeEmbedding(dim=args.time_dim).to(args.device)

    check_in_fusion_model = CheckInFusion(
        d_user=args.user_dim,
        d_poi=args.poi_dim,
        d_time=args.time_dim,
        d_space=args.space_dim,
        d_cat=args.cat_dim,
        out_dim=args.input_tok_dim,
        gate=getattr(args, "fusion_gate", None),
    ).to(args.device)

    # IMPORTANT:
    # Make sure PFA_LLAMA constructor args match what your refactored train uses.
    # If your unified parser provides llama-specific args, plug them in here.
    pfa = PFA_LLAMA(
        model_name=getattr(args, "llama_model_name", "meta-llama/Llama-2-7b-hf"),
        num_layers=getattr(args, "llama_num_layers", 16),
        U=getattr(args, "llama_last_u", 4),
        lora_r=getattr(args, "lora_r", 8),
        lora_alpha=getattr(args, "lora_alpha", 16),
        lora_dropout=getattr(args, "lora_dropout", 0.1),
    )

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
        lambda_time=getattr(args, "lambda_time", 0.0),
        lambda_cat=getattr(args, "lambda_cat", 0.1),
        lambda_loc=getattr(args, "lambda_loc", 0.1),
        opt="llama",
    ).to(args.device)

    # -------- Load weights from checkpoint (same keys as train save) --------
    logging.info("Loading trained weights...")
    user_emb.load_state_dict(ckpt["user_emb_state_dict"])
    cat_emb.load_state_dict(ckpt["cat_emb_state_dict"])
    poi_encoder.load_state_dict(ckpt["poi_encoder_state_dict"])
    time_embed_model.load_state_dict(ckpt["time_embed_model"])
    check_in_fusion_model.load_state_dict(ckpt["check_in_fusion_state_dict"])

    # main_model strict=False because LoRA wrappers can change key names slightly
    main_model.load_state_dict(ckpt["main_model_state_dict"], strict=False)

    # -------- Set lat/lon table for location loss (if needed) --------
    poi_latlon_deg = torch.zeros(num_pois, 2, dtype=torch.float32, device=args.device)
    for poi_idx, (lat, lon) in poi_idx2latlon.items():
        if 0 <= int(poi_idx) < num_pois:
            poi_latlon_deg[int(poi_idx)] = torch.tensor([lat, lon], dtype=torch.float32, device=args.device)
    main_model.set_poi_latlon(poi_latlon_deg)

    # -------- Build datasets / loaders --------
    logging.info("Building datasets / dataloader...")
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

    logging.info(f"len(train_dataset)={len(train_dataset)}, len(test_dataset)={len(test_dataset)}")

    # Coverage diagnostics (train vs test)
    logging.info("=== DIAG: coverage on TEST against TRAIN ===")
    train_poi_seen, train_trans_seen = set(), set()
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

    logging.info(
        f"[COVERAGE] TEST labels unseen-POI rate: {test_poi_unseen / max(1, test_poi_total):.3%} "
        f"({test_poi_unseen}/{test_poi_total})"
    )
    logging.info(
        f"[COVERAGE] TEST (u,prev->next) unseen-transition rate: {test_trans_unseen / max(1, test_trans_total):.3%} "
        f"({test_trans_unseen}/{test_trans_total})"
    )

    # Simple popularity baselines computed on TRAIN, evaluated on TEST last step
    global_next, user_next, user_prev_next = _build_freq_tables(train_dataset)
    global_top1 = _top1_from_counter(global_next)
    user_top1 = {u: _top1_from_counter(cnt) for u, cnt in user_next.items()}
    userprev_top1 = {k: _top1_from_counter(cnt) for k, cnt in user_prev_next.items()}

    base_acc = {"global": 0, "user": 0, "markov": 0}
    cnt = 0
    for sample in test_dataset:
        traj_id, user_idx, input_seq, label_seq, prev_poi_seq = sample
        if len(label_seq) == 0:
            continue
        cnt += 1
        gold = label_seq[-1][0]
        prev = input_seq[-1][0]
        u = int(user_idx)

        pred_g = global_top1
        pred_u = user_top1.get(u, None)
        pred_m = userprev_top1.get((u, prev), None)

        base_acc["global"] += int(pred_g == gold) if pred_g is not None else 0
        base_acc["user"] += int(pred_u == gold) if pred_u is not None else 0
        base_acc["markov"] += int(pred_m == gold) if pred_m is not None else 0

    for k in base_acc:
        base_acc[k] = base_acc[k] / max(1, cnt)

    logging.info(
        f"[BASELINES|TEST Acc@1] global={base_acc['global']:.4f} "
        f"user={base_acc['user']:.4f} markov={base_acc['markov']:.4f} (N={cnt})"
    )

    # Collator (same as train)
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

    # -------- Switch to eval --------
    for m in [main_model, user_emb, poi_encoder, cat_emb, time_embed_model, check_in_fusion_model]:
        m.eval()

    # Precompute poi_cat_idx_full (same as train)
    poi_cat_idx_full = torch.zeros(num_pois, dtype=torch.long, device=args.device)
    for poi_idx, cat_idx in poi_idx2cat_idx.items():
        pi = int(poi_idx)
        if 0 <= pi < num_pois:
            poi_cat_idx_full[pi] = int(cat_idx)

    # Precompute POI embeddings once (same as val/test in train)
    logging.info("Precomputing POI final embeddings...")
    cat_plus = cat_emb(poi_cat_idx_full)                 # (V, d_cat)
    poi_emb = torch.cat([poi_X, cat_plus], dim=1)        # (V, feat + d_cat)
    poi_final_emb = poi_encoder(poi_emb, poi_traj_data)  # (V, d_poi)

    # -------- Evaluation loop --------
    logging.info("Testing...")
    batches_top1, batches_top5, batches_top10, batches_top20 = [], [], [], []
    batches_map20, batches_mrr = [], []

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
        logits_poi[..., 0] = -1e9  # mask PAD id

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

    # -------- Save results --------
    top1 = float(np.mean(batches_top1)) if batches_top1 else 0.0
    top5 = float(np.mean(batches_top5)) if batches_top5 else 0.0
    top10 = float(np.mean(batches_top10)) if batches_top10 else 0.0
    top20 = float(np.mean(batches_top20)) if batches_top20 else 0.0
    map20 = float(np.mean(batches_map20)) if batches_map20 else 0.0
    mrr = float(np.mean(batches_mrr)) if batches_mrr else 0.0

    logging.info(
        "Test Results:\n"
        f" Top1={top1:.4f}\n"
        f" Top5={top5:.4f}\n"
        f" Top10={top10:.4f}\n"
        f" Top20={top20:.4f}\n"
        f" mAP@20={map20:.4f}\n"
        f" MRR={mrr:.4f}"
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
    results_path = os.path.join(args.save_dir, "test_results.csv")
    results_df.to_csv(results_path, index=False)
    logging.info(f"Saved: {results_path}")


if __name__ == "__main__":
    # Optional deterministic env var (same as your code)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args = parameter_parser()

    # Ensure args.device is torch.device
    if isinstance(args.device, str):
        # allow 'cuda'/'cpu' strings from parser
        args.device = torch.device(args.device)

    # Attach dataset-specific file paths consistent with train
    args = attach_paths_by_dataset(args)

    test(args)
