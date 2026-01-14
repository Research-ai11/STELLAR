"""Argument parsing utilities for training/testing the POI prediction model."""
from __future__ import annotations

import argparse

import torch


def _default_device() -> str:
    """Return the default device string based on CUDA availability."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def parameter_parser() -> argparse.Namespace:
    """Parse command-line arguments for the POI prediction model."""
    parser = argparse.ArgumentParser(description="Run POI Prediction Model")

    # ------------------------------------------------------------------
    # Reproducibility / environment
    # ------------------------------------------------------------------
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device(),
        help="Compute device string, e.g., 'cuda' or 'cpu'.",
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    parser.add_argument(
        "--data",
        type=str,
        default="nyc",
        choices=["nyc", "tky"],
        help="Dataset key. Choose from {'nyc', 'tky'}.",
    )

    # ------------------------------------------------------------------
    # Model hyper-parameters
    # ------------------------------------------------------------------
    parser.add_argument(
        "--LLM_model",
        type=str,
        default="gpt2",
        help="Backbone LLM identifier (used to configure default dimensions/settings).",
    )

    # Embedding dimensions
    # (POI, space, time, category) embeddings are used to form the final token representation.
    parser.add_argument(
        "--poi_dim",
        type=int,
        default=256,
        help="POI embedding dimension.",
    )
    parser.add_argument(
        "--space_dim",
        type=int,
        default=128,
        help="Space (geohash/region) embedding dimension.",
    )
    parser.add_argument(
        "--time_dim",
        type=int,
        default=32,
        help="Check-in time embedding dimension (typically smaller than POI/space).",
    )
    parser.add_argument(
        "--cat_dim",
        type=int,
        default=64,
        help="POI category embedding dimension.",
    )
    parser.add_argument(
        "--user_dim",
        type=int,
        default=144,
        help="User long-term preference embedding dimension.",
    )
    parser.add_argument(
        "--input_tok_dim",
        type=int,
        default=768,
        help="Input token embedding dimension (should match the backbone token size).",
    )

    # ------------------------------------------------------------------
    # GCN / Graph encoder hyper-parameters
    # ------------------------------------------------------------------
    parser.add_argument(
        "--gcn-dropout",
        type=float,
        default=0.3,
        help="Dropout rate for GCN layers.",
    )
    parser.add_argument(
        "--gcn-nhid",
        type=int,
        default=128,
        help="Hidden dimension for GCN layers.",
    )

    # ------------------------------------------------------------------
    # Training hyper-parameters
    # ------------------------------------------------------------------
    parser.add_argument(
        "--logit_mode",
        type=str,
        default="index",
        choices=["cos", "index"],
        help="Logit construction mode for next-POI prediction (cosine vs. index-based).",
    )
    parser.add_argument(
        "--fusion_gate",
        type=str,
        default="softmax",
        choices=["softmax", None],
        help="Gating mechanism for CheckInFusion (set to None to disable).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.5,
        help="Multiplicative factor for learning rate scheduler.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-3,
        help="Weight decay (L2 regularization).",
    )
    parser.add_argument(
        "--short-traj-thres",
        type=int,
        default=2,
        help="Drop trajectories whose number of transitions is below this threshold.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience for early stopping.",
    )

    # ------------------------------------------------------------------
    # Experiment / I/O configuration
    # ------------------------------------------------------------------
    parser.add_argument(
        "--save-weights",
        action="store_true",
        default=True,
        help="If set, save model checkpoints.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--project",
        default="runs/train",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--name",
        default="exp",
        help="Experiment name (used under --project).",
    )
    parser.add_argument(
        "--m_type",
        default="basic",
        help="Model type tag (used for organizing outputs).",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="If set, do not auto-increment experiment directory when it already exists.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disable CUDA even if it is available.",
    )

    # ------------------------------------------------------------------
    # Testing / inference configuration
    # ------------------------------------------------------------------
    parser.add_argument(
        "--checkpoint",
        default="runs/train/nyc/exp3/checkpoints/best_epoch.state.pt",
        help="Path to a saved model checkpoint for evaluation.",
    )
    parser.add_argument(
        "--saved_emb_path",
        default="runs/train/nyc/exp/embeddings",
        help="Path to the directory containing trained embeddings (if exported).",
    )

    # ------------------------------------------------------------------
    # Loss weights / auxiliary objectives
    # ------------------------------------------------------------------
    parser.add_argument(
        "--lambda_poi",
        type=float,
        default=1.0,
        help="Loss weight for POI prediction loss.",
    )
    parser.add_argument(
        "--lambda_time",
        type=float,
        default=0.0,
        help="Loss weight for time prediction/auxiliary loss.",
    )
    parser.add_argument(
        "--lambda_cat",
        type=float,
        default=0.0,
        help="Loss weight for category prediction/auxiliary loss.",
    )
    parser.add_argument(
        "--lambda_loc",
        type=float,
        default=0.0,
        help="Loss weight for location/auxiliary loss.",
    )

    # ------------------------------------------------------------------
    # Logit scaling / margin settings
    # ------------------------------------------------------------------
    parser.add_argument(
        "--learnable_scale",
        action="store_true",
        default=True,
        help="If set, use a learnable scaling factor for logits.",
    )
    parser.add_argument(
        "--use_margin",
        action="store_true",
        default=True,
        help="If set, enable margin-based loss.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.4,
        help="Margin value for margin-based loss.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature used to scale logits.",
    )
    parser.add_argument(
        "--tail_gamma",
        type=float,
        default=2.0,
        help="Gamma hyper-parameter for tail-aware loss weighting (if enabled).",
    )

    return parser.parse_args()
