"""
Model components for STELLAR next-POI prediction.

This file defines:
  - UserEmbedding: long-term user preference + social aggregation from fixed neighbors
  - CatEmbedding: categorical embedding table
  - POIEncoderGCN: GCN-based POI encoder over a transition graph
  - TimeEmbedding: Time2Vec-style embedding for (weekday, time-of-day, holiday)
  - CheckInFusion: fuse (user, social, poi, time, space, category) into token embeddings
  - PFA backbones:
      * PFA (GPT-2 partial fine-tuning / optional LoRA)
      * PFA_LLAMA (LLaMA-2 7B QLoRA + LoRA on last U blocks)
      * PFA_Transformer (full fine-tuning PyTorch TransformerEncoder baseline)
  - NextPOIWithPFA: main predictor producing POI/category logits + auxiliary time/location heads
  - Utilities: transition prior, haversine, prev-trajectory encoder, batch embedding builder
"""

from __future__ import annotations

import math
import types
from types import SimpleNamespace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import BitsAndBytesConfig, GPT2Model, LlamaModel
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

PAD = 0


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------
class UserEmbedding(nn.Module):
    """
    User embedding module with a simple 1-hop neighbor aggregation.

    Inputs:
      - user_ids: scalar int / (B,) tensor
    Outputs:
      - u_pref:   (B, d) or (d,)   user-specific long-term embedding
      - u_social: (B, d) or (d,)   neighbor-aggregated "social" embedding

    Neighbor indices/weights are fixed and registered as buffers (non-trainable).
    """

    def __init__(
        self,
        num_users: int,
        dim: int,
        user_pad_idx: torch.Tensor,  # (U, K) neighbor indices
        user_pad_w: torch.Tensor,    # (U, K) neighbor weights
        padding_idx: int = PAD,
    ):
        super().__init__()
        self.long_pref_emb = nn.Embedding(num_users, dim, padding_idx=padding_idx)

        # Fixed neighbor information (stored on-device with the module, not trainable)
        self.register_buffer("user_neighbor_idx", user_pad_idx)
        self.register_buffer("user_neighbor_w", user_pad_w)

    def forward(self, user_ids):
        # Convert scalar input to a tensor for unified handling
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor(
                [user_ids],
                device=self.long_pref_emb.weight.device,
                dtype=torch.long,
            )
            was_scalar = True
        else:
            was_scalar = (user_ids.ndim == 0)
            if was_scalar:
                user_ids = user_ids.view(1)

        # (B, d)
        u_pref = self.long_pref_emb(user_ids)

        # (B, K)
        neighbor_idx = self.user_neighbor_idx[user_ids]
        neighbor_w = self.user_neighbor_w[user_ids]

        # (B, K, d)
        neighbor_emb = self.long_pref_emb(neighbor_idx)

        # Weighted sum aggregation (B, d)
        u_social = (neighbor_emb * neighbor_w.unsqueeze(-1)).sum(dim=1)

        if was_scalar:
            return u_pref[0], u_social[0]
        return u_pref, u_social

    @property
    def weight(self):
        return self.long_pref_emb.weight


class CatEmbedding(nn.Module):
    """Category embedding table."""

    def __init__(self, num_cats: int, dim: int, padding_idx: int = PAD):
        super().__init__()
        self.cat_emb = nn.Embedding(num_cats, dim, padding_idx=padding_idx)

    def forward(self, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.cat_emb(cat_ids)

    @property
    def weight(self):
        return self.cat_emb.weight


# ---------------------------------------------------------------------
# POI encoder (GCN)
# ---------------------------------------------------------------------
class POIEncoderGCN(nn.Module):
    """
    GCN-based POI encoder over a transition graph.

    Args:
      - in_dim:  input feature dim (e.g., POI features + category embedding dim)
      - hid_dim: hidden dim
      - out_dim: output POI embedding dim
    """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hid_dim)

    def forward(self, x: torch.Tensor, data) -> torch.Tensor:
        """
        Inputs:
          - x:    (V, in_dim) node features
          - data: PyG Data with edge_index and optionally edge_weight
        Returns:
          - x:    (V, out_dim) final POI embeddings
        """
        x = self.conv1(x, data.edge_index, data.edge_weight)
        x = torch.relu(x)
        x = self.dropout(self.ln1(x))
        x = self.conv2(x, data.edge_index, data.edge_weight)
        return x


# ---------------------------------------------------------------------
# Time embedding (Time2Vec-style)
# ---------------------------------------------------------------------
class _T2V1D(nn.Module):
    """
    1D Time2Vec block that supports both (B,) and (B, L) inputs.

    Returns:
      - (B, 1+K) for (B,)
      - (B, L, 1+K) for (B, L)
    """

    def __init__(self, K: int, period: float | None = None):
        super().__init__()
        self.K = K
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(K))
        self.b = nn.Parameter(torch.zeros(K))

        if period is not None and K > 0:
            base = 2.0 * math.pi / period
            with torch.no_grad():
                for k in range(K):
                    self.w[k] = torch.tensor((k + 1) * base)
                self.b.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # (B, L) -> (B, L, 1), (B,) -> (B, 1, 1)
        orig_is_vector = (t.dim() == 1)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(1)
        else:
            raise ValueError(f"Expected t dim in {{1,2}}, got {t.dim()}")

        lin = self.w0 * t + self.b0                      # (B, L, 1)
        per = torch.sin(t * self.w + self.b)             # (B, L, K)
        out = torch.cat([lin, per], dim=-1)              # (B, L, 1+K)

        # Restore shape for (B,) input -> (B, 1+K)
        if orig_is_vector:
            out = out.squeeze(1)
        return out


class TimeEmbedding(nn.Module):
    """
    Time embedding from:
      - weekday in [0..6]
      - time fraction in [0..1]
      - holiday flag in {0,1}

    Supports inputs of shape (B,) or (B, L).
    Returns:
      - (B, dim) or (B, L, dim)
    """

    def __init__(self, dim: int, k_time: int = 3, k_weekday: int = 2, d_hol: int = 4):
        super().__init__()
        self.t2v_time = _T2V1D(K=k_time, period=1.0)
        self.t2v_weekday = _T2V1D(K=k_weekday, period=7.0)
        self.emb_holiday = nn.Embedding(2, d_hol)

        base_dim = (1 + k_time) + (1 + k_weekday) + d_hol
        self.proj = nn.Sequential(
            nn.LayerNorm(base_dim),
            nn.Linear(base_dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
        )

    def forward(self, weekday: torch.Tensor, time: torch.Tensor, is_holiday: torch.Tensor) -> torch.Tensor:
        t_rel = time.float().clamp(0.0, 1.0)
        wday = weekday.float().clamp(0.0, 6.0)
        hol_ix = is_holiday.long().clamp(0, 1)

        v_t = self.t2v_time(t_rel)
        v_w = self.t2v_weekday(wday)
        v_h = self.emb_holiday(hol_ix)

        base = torch.cat([v_t, v_w, v_h], dim=-1)
        return self.proj(base)


# ---------------------------------------------------------------------
# Check-in token fusion
# ---------------------------------------------------------------------
class CheckInFusion(nn.Module):
    """
    Fuse (user, social, poi, time, space, category) embeddings into a single token embedding.

    Default: concat -> projection MLP -> out_dim
    Optional gating (gate='softmax'): learn scalar weights per component for dynamic re-weighting.

    Inputs can be either:
      - (B, L, d) sequence tensors, or
      - (B, d) single-step tensors.
    """

    def __init__(
        self,
        d_user: int,
        d_poi: int,
        d_time: int,
        d_space: int,
        d_cat: int,
        out_dim: int = 768,
        dropout: float = 0.3,
        gate: str | None = None,   # None | 'softmax'
    ):
        super().__init__()
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        self.ln_u = nn.LayerNorm(d_user)
        self.ln_un = nn.LayerNorm(d_user)
        self.ln_p = nn.LayerNorm(d_poi)
        self.ln_t = nn.LayerNorm(d_time)
        self.ln_s = nn.LayerNorm(d_space)
        self.ln_c = nn.LayerNorm(d_cat)

        fused_in = d_user * 2 + d_poi + d_time + d_space + d_cat

        self.gate_type = gate
        if gate == "softmax":
            g_hid = max(64, fused_in // 4)
            self.gate_mlp = nn.Sequential(
                nn.Linear(fused_in, g_hid),
                nn.ReLU(),
                nn.Linear(g_hid, 6),  # user, social, poi, time, space, cat
            )

        self.proj = nn.Sequential(
            nn.Linear(fused_in, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """(B, d) -> (B, 1, d) for unified sequence handling."""
        if x.dim() == 2:
            return x.unsqueeze(1), True
        return x, False

    def forward(
        self,
        e_user: torch.Tensor,
        e_social: torch.Tensor,
        e_poi: torch.Tensor,
        e_time: torch.Tensor,
        e_space: torch.Tensor,
        e_cat: torch.Tensor,
    ) -> torch.Tensor:
        e_user, squeeze_u = self._ensure_3d(e_user)
        e_social, squeeze_n = self._ensure_3d(e_social)
        e_poi, squeeze_p = self._ensure_3d(e_poi)
        e_time, squeeze_t = self._ensure_3d(e_time)
        e_space, squeeze_s = self._ensure_3d(e_space)
        e_cat, squeeze_c = self._ensure_3d(e_cat)

        u = self.dropout(self.ln_u(e_user))
        u_n = self.dropout(self.ln_un(e_social))
        p = self.dropout(self.ln_p(e_poi))
        t = self.dropout(self.ln_t(e_time))
        s = self.dropout(self.ln_s(e_space))
        c = self.dropout(self.ln_c(e_cat))

        fused = torch.cat([u, u_n, p, t, s, c], dim=-1)

        if self.gate_type == "softmax":
            scores = self.gate_mlp(fused)          # (B, L, 6)
            alphas = F.softmax(scores, dim=-1)     # (B, L, 6)
            a_u, a_n, a_p, a_t, a_s, a_c = [alphas[..., i:i+1] for i in range(6)]
            fused = torch.cat([a_u*u, a_n*u_n, a_p*p, a_t*t, a_s*s, a_c*c], dim=-1)

        out = self.proj(fused)

        # Restore (B, out_dim) if original inputs were single-step
        if squeeze_u and squeeze_p and squeeze_t and squeeze_s and squeeze_c and squeeze_n:
            out = out.squeeze(1)
        return out


# ---------------------------------------------------------------------
# Optional: RoPE identity patch (debug / ablation utility)
# ---------------------------------------------------------------------
def _patch_rope_identity() -> None:
    """Patch LLaMA rotary embedding application to identity (q,k unchanged)."""
    try:
        from transformers.models.llama import modeling_llama as _llama_mod
    except Exception as e:
        raise RuntimeError("Check that transformers is installed and LLaMA is importable.") from e

    if getattr(_llama_mod, "_NEXTPOI_ROPE_PATCHED", False):
        return

    _llama_mod._NEXTPOI_ORIG_APPLY = getattr(_llama_mod, "apply_rotary_pos_emb", None)

    def apply_rotary_pos_emb_identity(q, k, cos, sin, position_ids=None):
        return (q, k)

    _llama_mod.apply_rotary_pos_emb = apply_rotary_pos_emb_identity
    _llama_mod._NEXTPOI_ROPE_PATCHED = True


def _unpatch_rope_identity() -> None:
    """Restore original RoPE behavior if patched."""
    from transformers.models.llama import modeling_llama as _llama_mod
    if getattr(_llama_mod, "_NEXTPOI_ROPE_PATCHED", False):
        if getattr(_llama_mod, "_NEXTPOI_ORIG_APPLY", None) is not None:
            _llama_mod.apply_rotary_pos_emb = _llama_mod._NEXTPOI_ORIG_APPLY
        _llama_mod._NEXTPOI_ROPE_PATCHED = False
        _llama_mod._NEXTPOI_ORIG_APPLY = None


# ---------------------------------------------------------------------
# PFA backbones
# ---------------------------------------------------------------------
class PFA_Transformer(nn.Module):
    """
    Pure PyTorch TransformerEncoder backbone (full fine-tuning baseline).

    Interface compatible with NextPOIWithPFA:
      - forward(inputs_embeds, attention_mask) -> (B, L, H)
      - provides self.gpt.config.hidden_size
      - trainable_parameters() method
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_pre_norm: bool = True,
        max_len: int = 200,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=use_pre_norm,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        # Mimic HF backbone interface: self.gpt.config.hidden_size
        self.gpt = SimpleNamespace(config=SimpleNamespace(hidden_size=d_model))

    @staticmethod
    def _causal_mask(L: int, device) -> torch.Tensor:
        # True means "mask out" for nn.Transformer
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = inputs_embeds.shape
        device = inputs_embeds.device

        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = inputs_embeds + self.pos_emb(pos_ids)
        x = self.dropout(x)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = self._causal_mask(L, device)

        h = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return h

    def trainable_parameters(self):
        return list(self.parameters())


class PFA_LLAMA(nn.Module):
    """
    LLaMA-2 7B backbone with QLoRA-style 4-bit loading + LoRA adapters.

    Design:
      - Load LlamaModel (not CausalLM) since we only need hidden states.
      - Truncate to `num_layers`.
      - Insert LoRA adapters on attention projections.
      - Train ONLY LoRA parameters in the last U blocks (plus optional final norm).
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        num_layers: int = 16,
        U: int = 4,
        train_final_ln: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_targets: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        super().__init__()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.gpt = LlamaModel.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            dtype=torch.bfloat16,
            device_map="auto",
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        self.gpt.config.use_cache = False

        assert num_layers <= self.gpt.config.num_hidden_layers, (
            f"num_layers({num_layers}) cannot exceed pretrained depth({self.gpt.config.num_hidden_layers})."
        )
        self.gpt.layers = self.gpt.layers[:num_layers]
        self.gpt.config.num_hidden_layers = num_layers
        self.last_start = num_layers - U

        # PEFT requires prepare_inputs_for_generation; add a dummy method for LlamaModel.
        def _dummy_prepare_inputs_for_generation(self_model, input_ids=None, inputs_embeds=None, **kwargs):
            if inputs_embeds is not None:
                model_inputs = {"inputs_embeds": inputs_embeds}
            else:
                model_inputs = {"input_ids": input_ids}
            model_inputs.update(kwargs)
            return model_inputs

        self.gpt.prepare_inputs_for_generation = types.MethodType(_dummy_prepare_inputs_for_generation, self.gpt)

        # Prepare for k-bit training (e.g., gradient checkpointing)
        self.gpt = prepare_model_for_kbit_training(self.gpt)
        if getattr(self.gpt, "supports_gradient_checkpointing", False):
            self.gpt.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_targets),
            bias="none",
        )
        self.gpt = get_peft_model(self.gpt, peft_cfg)

        # Optionally train the final norm layer
        if train_final_ln:
            norm_ref = getattr(self.gpt.base_model.model, "norm", None)
            if norm_ref is not None:
                for p in norm_ref.parameters():
                    p.requires_grad = True

        # Train only LoRA params in the last U blocks
        for name, p in self.gpt.named_parameters():
            if "lora_" in name:
                is_in_last_U = any(
                    (f".layers.{li}." in name) or (f"layers.{li}." in name)
                    for li in range(self.last_start, num_layers)
                )
                p.requires_grad = is_in_last_U

        # Print trainable parameter stats (helpful for sanity check)
        self.gpt.print_trainable_parameters()

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.last_hidden_state

    def trainable_parameters(self):
        return [p for p in self.gpt.parameters() if p.requires_grad]


class PFA(nn.Module):
    """
    GPT-2 backbone trained in a PFA-style partial fine-tuning setup.

    Strategy:
      - Freeze all parameters by default.
      - Optionally unfreeze layer norms in early blocks.
      - Train attention (and optionally MLP) in the last U blocks.
      - Optionally attach LoRA and train LoRA only in the last U blocks.
    """

    def __init__(
        self,
        gpt_name: str = "gpt2",
        gpt_layers: int = 6,
        U: int = 2,
        train_early_ln: bool = True,
        train_pos_emb_early: bool = True,
        train_final_ln: bool = True,
        train_last_mlp: bool = False,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.3,
        lora_targets: Tuple[str, ...] = ("c_attn", "c_proj"),
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        super().__init__()

        self.gpt = GPT2Model.from_pretrained(
            gpt_name,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        assert gpt_layers <= self.gpt.config.n_layer, "gpt_layers cannot exceed pretrained depth."
        self.gpt.h = self.gpt.h[:gpt_layers]
        self.gpt.config.n_layer = gpt_layers
        self.last_start = gpt_layers - U

        # Optional dropout overrides
        self.gpt.attn_pdrop = 0.3
        self.gpt.resid_pdrop = 0.3
        self.gpt.embd_pdrop = 0.3

        # Freeze everything by default
        for p in self.gpt.parameters():
            p.requires_grad = False

        # Unfreeze layer norms in early blocks (optional)
        if train_early_ln:
            for li in range(0, self.last_start):
                blk = self.gpt.h[li]
                for name, p in blk.named_parameters():
                    if name.startswith("ln_"):
                        p.requires_grad = True

        # Positional embedding (optional)
        if train_pos_emb_early:
            for p in self.gpt.wpe.parameters():
                p.requires_grad = True

        # Train last U blocks (attention by default; optionally MLP)
        for li in range(self.last_start, gpt_layers):
            blk = self.gpt.h[li]
            for name, p in blk.named_parameters():
                if ("attn" in name) or (train_last_mlp and "mlp" in name):
                    p.requires_grad = True

        # Final layer norm
        if train_final_ln:
            for p in self.gpt.ln_f.parameters():
                p.requires_grad = True

        # Optional LoRA
        self.use_lora = use_lora
        if use_lora:
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_targets),
            )
            self.gpt = get_peft_model(self.gpt, peft_cfg)

            # Train LoRA only in the last U blocks
            for name, p in self.gpt.named_parameters():
                if "lora_" in name:
                    keep = any(f"h.{li}." in name for li in range(self.last_start, gpt_layers))
                    p.requires_grad = keep

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.last_hidden_state

    def trainable_parameters(self):
        return [p for p in self.gpt.parameters() if p.requires_grad]


# ---------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------
class NextPOIWithPFA(nn.Module):
    """
    Next-POI predictor on top of a PFA backbone.

    Outputs:
      - logits_poi: (B, L, V_poi)
      - logits_cat: (B, L, V_cat)
      - pred_time : (B, L, 2)  (sin, cos)
      - pred_loc  : (B, L, 2)  (lat, lon) in unconstrained space; training converts to radians
    """

    def __init__(
        self,
        pfa: nn.Module,
        *,
        num_pois: Optional[int] = None,
        num_cats: Optional[int] = None,
        logit_mode: str = "cos",          # 'cos' | 'index'
        poi_proj_dim: int = 256,
        cat_proj_dim: int = 64,
        learnable_scale: bool = True,
        init_scale: float = 10.0,
        label_ignore_index: int = -100,
        tail_gamma: float = 1.0,
        temperature: float = 1.0,
        lambda_poi: float = 1.0,
        lambda_time: float = 0.2,
        lambda_cat: float = 0.2,
        lambda_loc: float = 0.2,
        opt: str = "llama",               # affects time prompt dims (kept for backward compat)
        eps: float = 1e-8,
    ):
        super().__init__()
        self.pfa = pfa
        H = self.pfa.gpt.config.hidden_size

        # Time prompt (FiLM) configuration
        if opt == "llama":
            d_time = 64
            r = 128
        else:
            d_time = 32
            r = 64

        self.use_time_prompt = True
        self.time_film_alpha = nn.Parameter(torch.tensor(0.10))

        self.time_norm = nn.LayerNorm(d_time)
        self.time_A = nn.Linear(d_time, r)
        self.time_act = nn.ReLU()
        self.time_B = nn.Linear(r, 2 * H)

        # Neutral init: gamma=0, beta=0 initially
        nn.init.zeros_(self.time_B.weight)
        nn.init.zeros_(self.time_B.bias)

        self.time_layernorm = nn.LayerNorm(H)

        # Heads
        if logit_mode == "cos":
            self.proj_poi = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, poi_proj_dim, bias=False))
            self.proj_cat = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, cat_proj_dim, bias=False))
        else:
            assert num_pois is not None and num_cats is not None, "index mode requires num_pois/num_cats"
            self.proj_poi = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, num_pois, bias=False))
            self.proj_cat = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, num_cats, bias=False))

        self.proj_time = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, 2, bias=False))
        self.proj_loc = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, 2, bias=False))

        self.num_pois = num_pois
        self.num_cats = num_cats
        self.logit_mode = logit_mode

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(init_scale)))
        self.learnable_scale = learnable_scale

        self.label_ignore_index = int(label_ignore_index)
        self.temperature = float(temperature)
        self.tail_gamma = float(tail_gamma)
        self.eps = float(eps)

        self.lambda_poi = float(lambda_poi)
        self.lambda_time = float(lambda_time)
        self.lambda_cat = float(lambda_cat)
        self.lambda_loc = float(lambda_loc)

        # For optional priors / geo losses
        self.T_prior_dense = None
        self.T_src_all = None
        self.T_dst_all = None
        self.T_val_all = None
        self.T_V = None
        self.poi_latlon_rad = None

    @torch.no_grad()
    def set_transition_prior(self, prior: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
        """Store dense (V,V) or sparse (src_all, dst_all, val_all, V) transition priors."""
        self.T_prior_dense = None
        self.T_src_all = None
        self.T_dst_all = None
        self.T_val_all = None

        if isinstance(prior, torch.Tensor):
            self.T_prior_dense = prior
            self.T_V = torch.tensor(prior.size(0), device=prior.device, dtype=torch.long)
        else:
            s, d, v, V = prior
            self.T_src_all, self.T_dst_all, self.T_val_all = s, d, v
            self.T_V = torch.tensor(V, device=v.device, dtype=torch.long)

    @torch.no_grad()
    def set_poi_latlon(self, poi_latlon_deg: torch.Tensor):
        """
        poi_latlon_deg: (V,2) [lat_deg, lon_deg], including PAD at row 0 (recommended).
        Stored internally in radians.
        """
        lat = torch.deg2rad(poi_latlon_deg[:, 0])
        lon = torch.deg2rad(poi_latlon_deg[:, 1])
        self.poi_latlon_rad = torch.stack([lat, lon], dim=-1)

    def trainable_parameters(self):
        params = []
        params += getattr(self.pfa, "trainable_parameters", lambda: list(self.pfa.parameters()))()
        params += list(self.proj_poi.parameters())
        params += list(self.proj_cat.parameters())
        params += list(self.proj_time.parameters())
        params += list(self.proj_loc.parameters())
        if self.learnable_scale:
            params += [self.scale]
        return params

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        time_prompt: Optional[torch.Tensor] = None,
        poi_final_emb: Optional[torch.Tensor] = None,      # (V_poi, poi_proj_dim) for cos mode
        cat_emb_weight: Optional[torch.Tensor] = None,     # (V_cat, cat_proj_dim) for cos mode
    ):
        """
        Returns:
          dict with logits_poi/logits_cat/pred_time/pred_loc (+ intermediate out_poi/out_cat).
        """
        h = self.pfa(inputs_embeds=inputs_embeds, attention_mask=attention_mask)  # (B, L, H)

        if self.use_time_prompt and (time_prompt is not None):
            z = self.time_norm(time_prompt)                        # (B, L, d_time)
            gb = self.time_B(self.time_act(self.time_A(z)))        # (B, L, 2H)
            gamma, beta = torch.chunk(gb, 2, dim=-1)               # (B, L, H), (B, L, H)
            gamma = torch.tanh(gamma)
            beta = torch.tanh(beta)
            h = self.time_layernorm(
                h * (1.0 + self.time_film_alpha * gamma) + self.time_film_alpha * beta
            )

        out_poi = self.proj_poi(h)
        out_cat = self.proj_cat(h)
        pred_time = self.proj_time(h)   # (B, L, 2)
        pred_loc = self.proj_loc(h)     # (B, L, 2)

        if self.logit_mode not in ("cos", "index"):
            raise ValueError(f"logit_mode must be 'cos' or 'index', got {self.logit_mode}")

        if self.logit_mode == "index":
            logits_poi = out_poi
            logits_cat = out_cat
        else:
            assert poi_final_emb is not None and cat_emb_weight is not None, \
                "cos mode requires poi_final_emb and cat_emb_weight."

            poi_out_norm = F.normalize(out_poi, dim=-1)
            cat_out_norm = F.normalize(out_cat, dim=-1)
            E_poi = F.normalize(poi_final_emb, dim=-1)
            E_cat = F.normalize(cat_emb_weight, dim=-1)

            logits_poi = torch.einsum("bld,vd->blv", poi_out_norm, E_poi) * self.scale
            logits_cat = torch.einsum("bld,vd->blv", cat_out_norm, E_cat)

        if self.temperature != 1.0:
            logits_poi = logits_poi / self.temperature
            logits_cat = logits_cat / self.temperature

        return {
            "logits_poi": logits_poi,
            "logits_cat": logits_cat,
            "pred_time": pred_time,
            "pred_loc": pred_loc,
            "out_poi": out_poi,
            "out_cat": out_cat,
        }

    # ---------------- losses ----------------
    def _shifted_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        B, L, V = logits.shape
        m = mask.float()
        ce = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=self.label_ignore_index,
            reduction="none",
            label_smoothing=label_smoothing,
        ).reshape(B, L)
        w = self._tail_weights(m)
        return (ce * w).sum() / w.sum().clamp_min(1e-8)

    def _shifted_time_cos_loss(self, pred_time: torch.Tensor, y_time: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        p = pred_time / pred_time.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        g = y_time / y_time.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        cos_sim = (p * g).sum(dim=-1)
        cos_loss = 1.0 - cos_sim
        return (cos_loss * m).sum() / m.sum().clamp_min(1.0)

    def _shifted_loc_haversine_loss(self, pred_loc_rad: torch.Tensor, labels_poi: torch.Tensor, mask: torch.Tensor):
        assert self.poi_latlon_rad is not None, "poi_latlon_rad is not set."
        m = mask.float()

        gold_ids = labels_poi.clamp_min(0)
        tgt = self.poi_latlon_rad[gold_ids]  # (B, L, 2)

        lat1, lon1 = pred_loc_rad[..., 0], pred_loc_rad[..., 1]
        lat2, lon2 = tgt[..., 0], tgt[..., 1]

        dlat = 0.5 * (lat2 - lat1)
        dlon = 0.5 * (lon2 - lon1)

        a = (
            torch.sin(dlat).pow(2)
            + torch.cos(lat1).clamp(-1, 1) * torch.cos(lat2).clamp(-1, 1) * torch.sin(dlon).pow(2)
        )
        c = 2.0 * torch.atan2(torch.sqrt(a.clamp_min(0)), torch.sqrt((1 - a).clamp_min(1e-12)))

        valid = (labels_poi != self.label_ignore_index).float()
        m = m * valid

        w = self._tail_weights(m)
        return (c * w).sum() / w.sum().clamp_min(1e-8)

    # ---------------- utils ----------------
    def _tail_weights(self, mask_bt: torch.Tensor) -> torch.Tensor:
        B, T = mask_bt.shape
        if T == 0:
            return mask_bt
        lengths = mask_bt.sum(dim=1, keepdim=True).clamp_min(1.0)
        pos = torch.arange(1, T + 1, device=mask_bt.device, dtype=mask_bt.dtype).unsqueeze(0)
        w = (pos / lengths) ** self.tail_gamma
        return w * mask_bt


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def build_transition_prior_dense(
    poi_traj_data,
    V: int,
    device,
    alpha: float = 0.95,
    pad_id: int = PAD,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Build a dense row-stochastic transition prior T[V, V] from a trajectory graph.

    T[row,:] = (1-alpha)/V + alpha * row_normalized_counts

    Note:
      - poi_traj_data should be constructed from TRAIN split only to avoid leakage.
      - padding row/column are zeroed and final row normalization is applied for stability.
    """
    ei = poi_traj_data.edge_index.to(device)  # (2, E)
    if not hasattr(poi_traj_data, "edge_weight") or poi_traj_data.edge_weight is None:
        w = torch.ones(ei.size(1), dtype=dtype, device=device)
    else:
        w = poi_traj_data.edge_weight.to(device).to(dtype)

    src = ei[0]
    dst = ei[1]

    keep = (src != pad_id) & (dst != pad_id)
    src, dst, w = src[keep], dst[keep], w[keep]

    idx = torch.stack([src, dst], dim=0)
    sp = torch.sparse_coo_tensor(idx, w, size=(V, V), dtype=dtype, device=device).coalesce()
    counts = sp.to_dense()

    T = torch.full((V, V), fill_value=(1.0 - alpha) / float(V), dtype=dtype, device=device)

    row_sum = counts.sum(dim=1, keepdim=True)
    has_out = (row_sum.squeeze(1) > 0)
    norm = torch.zeros_like(counts)
    norm[has_out] = counts[has_out] / row_sum[has_out].clamp_min(1e-12)

    T = T + alpha * norm
    T[pad_id, :] = 0.0
    T[:, pad_id] = 0.0
    T = T / T.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return T


def haversine_km(lat1, lon1, lat2, lon2, eps: float = 1e-12):
    """Haversine distance in kilometers. Inputs are radians."""
    dlat = (lat2 - lat1) * 0.5
    dlon = (lon2 - lon1) * 0.5
    a = torch.sin(dlat).pow(2) + torch.cos(lat1).clamp(-1, 1) * torch.cos(lat2).clamp(-1, 1) * torch.sin(dlon).pow(2)
    c = 2.0 * torch.atan2(torch.sqrt(torch.clamp(a, 0, 1)), torch.sqrt(torch.clamp(1 - a, eps, 1)))
    return 6371.0088 * c


class PrevTrajEncoder(nn.Module):
    """
    Encode a previous trajectory into a single summary token.

    Inputs:
      - prev_poi_idx:   (T_prev,)
      - prev_time_frac: (T_prev,) in [0, 1)
      - poi_final_emb:  (V, d_poi)
      - poi_latlon_deg: (V, 2) in degrees

    Outputs:
      - summary: (1, D) summary token
      - attn:    (T_prev,) attention weights (for debugging)
    """

    def __init__(
        self,
        d_poi: int,
        d_out: int,
        n_layers: int = 1,
        bidir: bool = False,
        recency_lambda: float = 1.0,
    ):
        super().__init__()
        self.d_poi = d_poi
        self.d_out = d_out
        self.recency_log_lambda = nn.Parameter(torch.tensor(float(recency_lambda)).log())

        # Feature vector: [sin(2πt), cos(2πt), log1p(Δt_hours), log1p(Δdist_km)] -> 4 dims
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_poi + 4),
            nn.Linear(d_poi + 4, d_poi + 100),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_poi + 100, d_poi),
        )

        self.gru = nn.GRU(
            input_size=d_poi,
            hidden_size=d_out,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        self.bidir = bidir
        self.q = nn.Parameter(torch.randn(d_out * (2 if bidir else 1)))

    @staticmethod
    def _time_deltas_hours(t_frac: torch.Tensor) -> torch.Tensor:
        """Δt in hours, wrap-around aware."""
        T = t_frac.size(0)
        if T == 1:
            return torch.zeros_like(t_frac)
        dif = t_frac[1:] - t_frac[:-1]
        dif = (dif % 1.0) * 24.0
        return torch.cat([torch.zeros(1, device=t_frac.device, dtype=t_frac.dtype), dif], dim=0)

    @torch.no_grad()
    def _poi_radians(self, poi_latlon_deg: torch.Tensor, idx: torch.Tensor):
        lat = torch.deg2rad(poi_latlon_deg[idx, 0].clamp(-90, 90))
        lon = torch.deg2rad(poi_latlon_deg[idx, 1].clamp(-180, 180))
        return lat, lon

    def forward(
        self,
        prev_poi_idx: torch.Tensor,
        prev_time_frac: torch.Tensor,
        poi_final_emb: torch.Tensor,
        poi_latlon_deg: torch.Tensor,
    ):
        device = poi_final_emb.device
        T = prev_poi_idx.numel()

        poi_emb = poi_final_emb[prev_poi_idx]  # (T, d_poi)

        t = prev_time_frac.clamp(0, 1)
        time2d = torch.stack([torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], dim=-1)  # (T, 2)
        dt_h = self._time_deltas_hours(t)

        lat, lon = self._poi_radians(poi_latlon_deg, prev_poi_idx)
        if T == 1:
            dd_km = torch.zeros_like(dt_h)
        else:
            dd = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
            dd_km = torch.cat([torch.zeros(1, device=device, dtype=dd.dtype), dd], dim=0)

        feat = torch.stack([time2d[:, 0], time2d[:, 1], torch.log1p(dt_h), torch.log1p(dd_km)], dim=-1)
        x = torch.cat([poi_emb, feat], dim=-1)  # (T, d_poi+4)
        x = self.fuse(x).unsqueeze(0)           # (1, T, d_poi)

        h, _ = self.gru(x)                      # (1, T, d_out)
        h = h.squeeze(0)                        # (T, d_out)

        D = h.size(-1)
        q = self.q[:D] / (D ** 0.5)
        score = (h @ q)

        age = torch.arange(T, device=device).float()
        lam = self.recency_log_lambda.exp().clamp_min(1e-4)
        score = score - lam * (T - 1 - age)

        attn = torch.softmax(score, dim=0)
        summary = (attn.unsqueeze(-1) * h).sum(dim=0, keepdim=True)  # (1, D)
        return summary, attn


# ---------------------------------------------------------------------
# Batch embedding builder
# ---------------------------------------------------------------------
def get_batch_inputs_embeds(
    user_idxs: torch.Tensor,        # (B,)
    x_poi_idxs: torch.Tensor,       # (B, L)
    x_cat_idxs: torch.Tensor,       # (B, L)
    x_geo_idxs: torch.Tensor,       # (B, L)
    x_time_feats: torch.Tensor,     # (B, L, 3)
    *,
    user_final_emb: nn.Module,      # UserEmbedding
    poi_final_emb: torch.Tensor,    # (V_poi, D_poi)
    cat_emb: nn.Module,             # CatEmbedding
    space_emb: nn.Module,           # nn.Embedding (pretrained, frozen)
    time_embed_model: nn.Module,    # TimeEmbedding
    check_in_fusion_model: nn.Module,  # CheckInFusion
) -> torch.Tensor:
    """
    Convert a collated trajectory batch into token embeddings for the backbone model.

    Returns:
      - fused_embeddings: (B, L, H)
    """
    B, L = x_poi_idxs.shape

    # User embeddings replicated across timesteps
    u_pref, u_social = user_final_emb(user_idxs)              # (B, d_user), (B, d_user)
    u_pref = u_pref.unsqueeze(1).expand(-1, L, -1)            # (B, L, d_user)
    u_social = u_social.unsqueeze(1).expand(-1, L, -1)        # (B, L, d_user)

    # Lookup per-step embeddings
    poi_emb = poi_final_emb[x_poi_idxs]                       # (B, L, d_poi)
    cat_e = cat_emb(x_cat_idxs)                               # (B, L, d_cat)
    space_e = space_emb(x_geo_idxs)                           # (B, L, d_space)

    # Time embedding: weekday_frac, time_frac, holiday_flag
    time_e = time_embed_model(
        x_time_feats[..., 0],
        x_time_feats[..., 1],
        x_time_feats[..., 2],
    )                                                         # (B, L, d_time)

    # Fuse into token embeddings
    fused_embeddings = check_in_fusion_model(
        u_pref,
        u_social,
        poi_emb,
        time_e,
        space_e,
        cat_e,
    )                                                         # (B, L, H)

    return fused_embeddings
