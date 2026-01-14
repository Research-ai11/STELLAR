"""
Parameter parser for Embedding Ablation experiments.

Design goals (paper-ready):
- Reproducible defaults
- Clear argument grouping
- Robust CLI behavior (no torch.device defaults in argparse)
- Explicit experiment toggles for ablations
"""

from __future__ import annotations

import argparse
import os
from typing import Optional


def str2bool(x) -> bool:
    """Robust boolean parser for CLI."""
    if isinstance(x, bool):
        return x
    x = str(x).strip().lower()
    if x in ("1", "true", "t", "yes", "y"):
        return True
    if x in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean (true/false).")


def _default_device() -> str:
    """
    Default device string.
    Prefer CUDA if available; otherwise CPU.
    We keep it as a string for argparse stability.
    """
    # NOTE: do not import torch here if you want ultra-light parsing;
    # but importing torch is usually fine. If you want, you can set
    # DEVICE env var instead.
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parameter_parser(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embedding Ablation: Next-POI prediction with token-level fusion + PFA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------------------------------------------------
    # Reproducibility / runtime
    # ---------------------------------------------------------------------
    g = parser.add_argument_group("Runtime & Reproducibility")
    g.add_argument("--seed", type=int, default=42, help="Random seed.")
    g.add_argument(
        "--device",
        type=str,
        default=_default_device(),
        choices=["cpu", "cuda"],
        help="Device string. Use 'cuda' when available.",
    )
    g.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    g.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    g = parser.add_argument_group("Data")
    g.add_argument(
        "--data",
        type=str,
        default="nyc",
        choices=["nyc", "tky"],
        help="Dataset key.",
    )
    g.add_argument(
        "--short_traj_thres",
        type=int,
        default=2,
        help="Filter trajectories shorter than this threshold.",
    )

    # ---------------------------------------------------------------------
    # Model: embedding dimensions
    # ---------------------------------------------------------------------
    g = parser.add_argument_group("Model: Embedding Dimensions")
    g.add_argument("--LLM_model", type=str, default="gpt2", help="Backbone LLM name (e.g., gpt2).")

    g.add_argument("--poi_dim", type=int, default=256, help="POI embedding dimension.")
    g.add_argument("--space_dim", type=int, default=128, help="Space (geohash) embedding dimension.")
    g.add_argument("--time_dim", type=int, default=32, help="Time embedding dimension.")
    g.add_argument("--cat_dim", type=int, default=64, help="Category embedding dimension.")
    g.add_argument("--user_dim", type=int, default=144, help="User embedding dimension.")
    g.add_argument("--input_tok_dim", type=int, default=768, help="Input token dimension to the LLM.")

    # ---------------------------------------------------------------------
    # Graph encoder (GCN)
    # ---------------------------------------------------------------------
    g = parser.add_argument_group("Graph Encoder (GCN)")
    g.add_argument("--gcn_dropout", type=float, default=0.3, help="Dropout rate for GCN encoder.")
    g.add_argument("--gcn_nhid", type=int, default=128, help="Hidden dimension for GCN encoder.")

    # ---------------------------------------------------------------------
    # Fusion / logits
    # ---------------------------------------------------------------------
    g = parser.add_argument_group("Fusion & Logits")
    g.add_argument(
        "--logit_mode",
        type=str,
        default="index",
        choices=["cos", "index"],
        help="Logit computation mode.",
    )
    g.add_argument(
        "--fusion_gate",
        type=str,
        default="softmax",
        choices=["softmax", "none"],
        help="Gating strategy in CheckInFusion. Use 'none' to disable.",
    )

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    g = parser.add_argument_group("Training")
    g.add_argument("--batch", type=int, default=32, help="Batch size.")
    g.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    g.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    g.add_argument("--lr_scheduler_factor", type=float, default=0.5, help="LR scheduler factor.")
    g.add_argument("--weight_decay", type=float, default=5e-3, help="Weight decay.")
    g.add_argument("--patience", type=int, default=5, help="Early stopping patience.")

    # ---------------------------------------------------------------------
    # Loss weights / scaling
    # ---------------------------------------------------------------------
    g = parser.add_argument_group("Loss & Scaling")
    g.add_argument("--lambda_poi", type=float, default=1.0, help="POI loss weight.")
    g.add_argument("--lambda_time", type=float, default=0.0, help="Time loss weight.")
    g.add_argument("--lambda_cat", type=float, default=0.0, help="Category loss weight.")
    g.add_argument("--lambda_loc", type=float, default=0.0, help="Location loss weight.")

    # NOTE: 'store_true' should default to False by design.
    # If you want default True behavior, use an explicit flag pair.
    g.add_argument(
        "--learnable_scale",
        type=str2bool,
        default=True,
        help="Whether to learn a scale parameter for cosine logits.",
    )
    g.add_argument("--use_margin", type=str2bool, default=True, help="Whether to use margin loss.")
    g.add_argument("--margin", type=float, default=0.4, help="Margin value for margin loss.")
    g.add_argument("--temperature", type=float, default=0.7, help="Temperature for logits scaling.")
    g.add_argument("--tail_gamma", type=float, default=2.0, help="Tail weighting gamma.")

    # ---------------------------------------------------------------------
    # Experiment I/O
    # ---------------------------------------------------------------------
    g = parser.add_argument_group("Experiment I/O")
    g.add_argument(
        "--save_weights",
        type=str2bool,
        default=True,
        help="Whether to save checkpoints.",
    )
    g.add_argument("--project", type=str, default="runs/train", help="Base directory for runs.")
    g.add_argument("--name", type=str, default="exp", help="Experiment name.")
    g.add_argument("--m_type", type=str, default="basic", help="Experiment type folder.")
    g.add_argument(
        "--exist_ok",
        action="store_true",
        help="If set, do not increment the run directory when it exists.",
    )

    # test-only paths
    g.add_argument(
        "--checkpoint",
        type=str,
        default="runs/train/nyc/exp/checkpoints/best_epoch.state.pt",
        help="Checkpoint path for evaluation.",
    )
    g.add_argument(
        "--saved_emb_path",
        type=str,
        default="runs/train/nyc/exp/embeddings",
        help="Path to load trained embeddings (if used).",
    )

    # ---------------------------------------------------------------------
    # Ablation toggles
    # ---------------------------------------------------------------------
    g = parser.add_argument_group("Ablation Toggles")
    g.add_argument("--use_social", type=str2bool, default=True, help="Use social (neighbor) embedding.")
    g.add_argument("--use_user", type=str2bool, default=True, help="Use user embedding.")
    g.add_argument("--use_space", type=str2bool, default=True, help="Use space embedding.")
    g.add_argument("--use_poi", type=str2bool, default=True, help="Use POI embedding.")
    g.add_argument("--use_cat", type=str2bool, default=True, help="Use category embedding.")
    g.add_argument("--use_time", type=str2bool, default=True, help="Use time embedding.")
    g.add_argument("--use_pred_time", type=str2bool, default=True, help="Use predicted-time conditioning.")

    args = parser.parse_args(argv)

    # ---------------------------------------------------------------------
    # Post-processing / normalization
    # ---------------------------------------------------------------------
    # Handle --no-cuda override cleanly.
    if args.no_cuda:
        args.device = "cpu"

    # Normalize fusion_gate for downstream code (string -> None)
    if args.fusion_gate == "none":
        args.fusion_gate = None

    return args
