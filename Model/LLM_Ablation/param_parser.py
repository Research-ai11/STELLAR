"""
Parameter parser for STELLAR next-POI experiments.

This single parser supports multiple backbones (Transformer / LLaMA / GPT-2) by:
  1) Defining a common set of arguments
  2) Applying backbone-specific default presets AFTER parsing, only when the user
     did not explicitly override the value via CLI.

Usage examples:
  - Transformer (full fine-tune):
      python train.py --backbone transformer
  - LLaMA2-7B (QLoRA + LoRA on last U blocks):
      python train.py --backbone llama --llama_model_name meta-llama/Llama-2-7b-hf
  - Override any preset:
      python train.py --backbone llama --batch 8 --lr 3e-5
"""

from __future__ import annotations

import argparse
from typing import Dict, Any

import torch


# ----------------------------
# Backbone presets (defaults)
# ----------------------------
BACKBONE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Full fine-tuning baseline with a vanilla TransformerEncoder backbone
    "transformer": dict(
        LLM_model="Transformer",
        poi_dim=256,
        space_dim=128,
        time_dim=32,
        cat_dim=64,
        user_dim=144,
        input_tok_dim=768,
        gcn_dropout=0.3,
        gcn_nhid=128,
        batch=32,
        lr=1e-4,
        weight_decay=5e-3,
        temperature=0.7,
        tail_gamma=2.0,
    ),
    # LLaMA2-7B QLoRA + LoRA adapters (last U blocks)
    "llama": dict(
        LLM_model="LLAMA2-7B",
        poi_dim=768,
        space_dim=256,
        time_dim=64,
        cat_dim=128,
        user_dim=768,
        input_tok_dim=4096,
        gcn_dropout=0.3,
        gcn_nhid=256,
        batch=16,
        lr=5e-5,
        weight_decay=5e-4,
        temperature=0.7,
        tail_gamma=2.0,
    ),
    # (Optional) Keep GPT-2 preset for backward compatibility / sanity checks
    "gpt2": dict(
        LLM_model="GPT2",
        poi_dim=256,
        space_dim=128,
        time_dim=32,
        cat_dim=64,
        user_dim=144,
        input_tok_dim=768,
        gcn_dropout=0.3,
        gcn_nhid=128,
        batch=32,
        lr=1e-4,
        weight_decay=5e-3,
        temperature=0.7,
        tail_gamma=2.0,
    ),
}


def _get_default_device_str() -> str:
    """Return a CLI-friendly device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run STELLAR Next-POI Prediction")

    # ------------------------------------------------------------------
    # Core settings
    # ------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=_get_default_device_str(),
        help="Device string, e.g., 'cuda', 'cuda:0', or 'cpu'.",
    )

    # Dataset key (used by train.py to resolve paths)
    parser.add_argument(
        "--data",
        type=str,
        default="nyc",
        choices=["nyc", "tky"],
        help="Dataset key. Choose from {'nyc', 'tky'}.",
    )

    # ------------------------------------------------------------------
    # Backbone selection
    # ------------------------------------------------------------------
    parser.add_argument(
        "--backbone",
        type=str,
        default="transformer",
        choices=["transformer", "llama", "gpt2"],
        help="Backbone preset to use.",
    )

    # Informational (kept for logging / paper tables)
    parser.add_argument(
        "--LLM_model",
        type=str,
        default=None,
        help="Model name label (auto-filled by preset unless overridden).",
    )

    # LLaMA-specific knobs (safe to keep even if unused)
    parser.add_argument(
        "--llama_model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF model ID for LLaMA backbone.",
    )
    parser.add_argument("--llama_num_layers", type=int, default=16, help="Number of LLaMA blocks to use.")
    parser.add_argument("--llama_last_u", type=int, default=4, help="Train LoRA only in the last U blocks.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")

    # Transformer-specific knobs (safe to keep even if unused)
    parser.add_argument("--tf_nhead", type=int, default=8, help="TransformerEncoder number of heads.")
    parser.add_argument("--tf_layers", type=int, default=4, help="TransformerEncoder number of layers.")
    parser.add_argument("--tf_ffn_dim", type=int, default=1024, help="TransformerEncoder FFN hidden dim.")
    parser.add_argument("--tf_dropout", type=float, default=0.1, help="TransformerEncoder dropout.")

    # ------------------------------------------------------------------
    # Model dimensions (these get preset depending on --backbone)
    # ------------------------------------------------------------------
    parser.add_argument("--poi_dim", type=int, default=None, help="POI embedding dim.")
    parser.add_argument("--space_dim", type=int, default=None, help="Space embedding dim.")
    parser.add_argument("--time_dim", type=int, default=None, help="Time embedding dim.")
    parser.add_argument("--cat_dim", type=int, default=None, help="Category embedding dim.")
    parser.add_argument("--user_dim", type=int, default=None, help="User preference embedding dim.")
    parser.add_argument("--input_tok_dim", type=int, default=None, help="Input token dim fed to the backbone.")

    # GCN hyper-parameters
    parser.add_argument("--gcn_dropout", type=float, default=None, help="Dropout rate for GCN.")
    parser.add_argument("--gcn_nhid", type=int, default=None, help="Hidden dim for GCN.")

    # ------------------------------------------------------------------
    # Training hyper-parameters
    # ------------------------------------------------------------------
    parser.add_argument(
        "--logit_mode",
        type=str,
        default="index",
        choices=["cos", "index"],
        help="Logit computation mode for next-POI prediction.",
    )
    parser.add_argument(
        "--fusion_gate",
        type=str,
        default="softmax",
        choices=["softmax", "none"],
        help="Gate type in CheckInFusion ('none' disables).",
    )

    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.5, help="ReduceLROnPlateau factor.")
    parser.add_argument("--weight_decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument("--short_traj_thres", type=int, default=2, help="Filter out over-short trajectories.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs).")

    # ------------------------------------------------------------------
    # Experiment config
    # ------------------------------------------------------------------
    parser.add_argument("--save_weights", action="store_true", default=True, help="Save best checkpoint.")
    parser.add_argument("--workers", type=int, default=0, help="Num workers for DataLoader.")
    parser.add_argument("--project", type=str, default="runs/train", help="Save to project directory.")
    parser.add_argument("--name", type=str, default="exp", help="Run name.")
    parser.add_argument("--m_type", type=str, default="basic", help="Experiment type (subfolder).")
    parser.add_argument("--exist_ok", action="store_true", help="Allow existing project/name (no increment).")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="Disable CUDA.")

    # For test/inference
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path to load.")
    parser.add_argument("--saved_emb_path", type=str, default="", help="Optional path for saved embeddings.")

    # Loss weights & logits calibration
    parser.add_argument("--lambda_poi", type=float, default=1.0, help="Loss weight for POI CE.")
    parser.add_argument("--lambda_time", type=float, default=0.0, help="Loss weight for time cosine loss.")
    parser.add_argument("--lambda_cat", type=float, default=0.1, help="Loss weight for category CE.")
    parser.add_argument("--lambda_loc", type=float, default=0.1, help="Loss weight for location haversine loss.")

    parser.add_argument("--learnable_scale", action="store_true", default=True, help="Use learnable scale in cos mode.")

    # Kept for backward compatibility (even if not used in current train.py)
    parser.add_argument("--use_margin", action="store_true", default=True, help="Enable margin loss (if implemented).")
    parser.add_argument("--margin", type=float, default=0.4, help="Margin for margin loss.")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for scaling logits.")
    parser.add_argument("--tail_gamma", type=float, default=None, help="Tail-weight exponent gamma.")

    # Optional logging toggles (train.py uses these if present)
    parser.add_argument("--enable_diagnostics", action="store_true", default=True, help="Run coverage/baseline logs.")
    parser.add_argument("--enable_batch_logging", action="store_true", default=False, help="Log batch predictions.")
    parser.add_argument("--log_every_n_batches", type=int, default=200, help="Batch logging frequency.")
    parser.add_argument("--log_sample_idx", type=int, default=0, help="Sample index for batch logging.")
    parser.add_argument("--log_topk", type=int, default=10, help="Top-K for batch logging preview.")
    parser.add_argument("--log_preview_steps", type=int, default=5, help="Steps to preview in batch logging.")

    args, unknown = parser.parse_known_args()

    # ------------------------------------------------------------------
    # Apply backbone preset defaults only when user did NOT override.
    # We detect overrides via "unknown" (parse_known_args) + raw sys.argv parsing.
    # Simpler/robust: just check if an arg is None and fill from preset.
    # ------------------------------------------------------------------
    preset = BACKBONE_PRESETS[args.backbone]

    # Fill label
    if args.LLM_model is None:
        args.LLM_model = preset["LLM_model"]

    # Fill dims / core hparams if not explicitly set
    for k in [
        "poi_dim",
        "space_dim",
        "time_dim",
        "cat_dim",
        "user_dim",
        "input_tok_dim",
        "gcn_dropout",
        "gcn_nhid",
        "batch",
        "lr",
        "weight_decay",
        "temperature",
        "tail_gamma",
    ]:
        if getattr(args, k) is None:
            setattr(args, k, preset[k])

    # Normalize fusion_gate: allow user to pass "none"
    if args.fusion_gate == "none":
        args.fusion_gate = None

    # Enforce device override
    if args.no_cuda:
        args.device = "cpu"

    return args
