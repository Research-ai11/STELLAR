"""
Model components for STELLAR (Next-POI prediction).

This module implements:
  - UserEmbedding: long-term user preference + social neighbor aggregation
  - CatEmbedding: categorical embedding lookup
  - POIEncoderGCN: GCN-based POI encoder over the POI transition graph
  - TimeEmbedding: time feature encoding (weekday/time fraction/holiday flag)
  - CheckInFusion: fuse (user, social, poi, time, space, category) into token embeddings
  - PFA: parameter-efficient adaptation of GPT-2 as a sequence encoder
  - NextPOIWithPFA: prediction heads for next POI (+ optional auxiliary heads)

Utility functions:
  - build_transition_prior_dense: build dense transition prior matrix from a POI graph
  - haversine_km: compute distance in km (radians)
  - bias_logits_all_steps: inject transition prior into logits (optional)
  - get_batch_inputs_embeds: vectorized batch embedding construction

Notes for paper-code release:
  - Keep all shapes explicit.
  - Avoid data leakage: transition priors/graphs should be built from train split only.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from transformers import GPT2Model
from peft import get_peft_model, LoraConfig, TaskType

PAD = 0


# ---------------------------------------------------------------------
# User / Category Embeddings
# ---------------------------------------------------------------------
class UserEmbedding(nn.Module):
    """
    User representation = (long-term preference embedding) + (1-hop social aggregation).

    Parameters
    ----------
    num_users : int
        Number of users including padding index.
    dim : int
        Embedding dimension.
    user_pad_idx : torch.Tensor
        Neighbor index table of shape (U, K). Each row contains K neighbor user indices.
    user_pad_w : torch.Tensor
        Neighbor weight table of shape (U, K). Each row contains K neighbor weights.
    padding_idx : int
        Padding user index (default: 0).

    Forward
    -------
    Inputs:
        user_ids: Tensor of shape (B,) or scalar (0-dim).
    Returns:
        u_pref  : (B, dim) or (dim,)
        u_social: (B, dim) or (dim,)
    """
    def __init__(
        self,
        num_users: int,
        dim: int,
        user_pad_idx: torch.Tensor,
        user_pad_w: torch.Tensor,
        padding_idx: int = PAD,
    ):
        super().__init__()
        self.long_pref_emb = nn.Embedding(num_users, dim, padding_idx=padding_idx)

        # Store neighbor tables as buffers (moved with .to(device), but not trainable).
        self.register_buffer("user_neighbor_idx", user_pad_idx)
        self.register_buffer("user_neighbor_w", user_pad_w)

    def forward(self, user_ids: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize input to (B,)
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor([user_ids], device=self.long_pref_emb.weight.device, dtype=torch.long)
            was_scalar = True
        else:
            was_scalar = (user_ids.ndim == 0)
            if was_scalar:
                user_ids = user_ids.view(1)
            else:
                user_ids = user_ids.long()

        # (B, dim): long-term preference
        u_pref = self.long_pref_emb(user_ids)

        # (B, K): neighbor indices / weights
        n_idx = self.user_neighbor_idx[user_ids]
        n_w = self.user_neighbor_w[user_ids]  # already aligned with n_idx

        # (B, K, dim): neighbor embeddings (using the same table)
        n_emb = self.long_pref_emb(n_idx)

        # Weighted sum aggregation: (B, dim)
        u_social = (n_emb * n_w.unsqueeze(-1)).sum(dim=1)

        if was_scalar:
            return u_pref[0], u_social[0]
        return u_pref, u_social

    @property
    def weight(self) -> torch.Tensor:
        return self.long_pref_emb.weight


class CatEmbedding(nn.Module):
    """
    Category embedding lookup.

    Inputs:
      cat_ids: Long tensor of shape (...)  (e.g., (B,L) or (V,))
    Outputs:
      embeddings of shape (..., dim)
    """
    def __init__(self, num_cats: int, dim: int, padding_idx: int = PAD):
        super().__init__()
        self.cat_emb = nn.Embedding(num_cats, dim, padding_idx=padding_idx)

    def forward(self, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.cat_emb(cat_ids)

    @property
    def weight(self) -> torch.Tensor:
        return self.cat_emb.weight


# ---------------------------------------------------------------------
# POI Graph Encoder
# ---------------------------------------------------------------------
class POIEncoderGCN(nn.Module):
    """
    Two-layer GCN encoder for POI nodes.

    Inputs:
      x    : (V, in_dim) node features
      data : PyG Data with fields:
             - edge_index: (2, E)
             - edge_weight (optional): (E,)
    Outputs:
      h    : (V, out_dim) POI embeddings
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hid_dim)

    def forward(self, x: torch.Tensor, data) -> torch.Tensor:
        x = self.conv1(x, data.edge_index, getattr(data, "edge_weight", None))
        x = torch.relu(x)
        x = self.dropout(self.ln1(x))
        x = self.conv2(x, data.edge_index, getattr(data, "edge_weight", None))
        return x


# ---------------------------------------------------------------------
# Time Embedding (Time2Vec-style)
# ---------------------------------------------------------------------
class _T2V1D(nn.Module):
    """
    Lightweight 1D Time2Vec block.
    Given scalar time t, outputs [linear(t), sin(w_k t + b_k)].

    Inputs:
      t: (B,L) or (B,)
    Outputs:
      (B,L,1+K) or (B,1+K)
    """
    def __init__(self, K: int, period: Optional[float] = None):
        super().__init__()
        self.K = K
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(K))
        self.b = nn.Parameter(torch.zeros(K))

        # Optional periodic initialization
        if period is not None and K > 0:
            base = 2.0 * math.pi / period
            with torch.no_grad():
                for k in range(K):
                    self.w[k] = torch.tensor((k + 1) * base)
                self.b.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Promote to (B,L,1) for broadcasting
        if t.dim() == 2:
            t_ = t.unsqueeze(-1)  # (B,L,1)
            squeezed = False
        elif t.dim() == 1:
            t_ = t.unsqueeze(-1).unsqueeze(1)  # (B,1,1)
            squeezed = True
        else:
            raise ValueError(f"Invalid time tensor shape: {tuple(t.shape)}")

        lin = self.w0 * t_ + self.b0                 # (B,L,1)
        per = torch.sin(t_ * self.w + self.b)        # (B,L,K)
        out = torch.cat([lin, per], dim=-1)          # (B,L,1+K)

        if squeezed:
            out = out.squeeze(1)                     # (B,1+K)
        return out


class TimeEmbedding(nn.Module):
    """
    Time embedding from (weekday, time_fraction, holiday_flag).

    Inputs:
      weekday    : (B,L) or (B,)  in [0..6]
      time       : (B,L) or (B,)  in [0..1]
      is_holiday : (B,L) or (B,)  in {0,1}
    Outputs:
      (B,L,dim) or (B,dim)
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
        # Input normalization / clipping
        t_rel = time.float().clamp(0.0, 1.0)
        wday = weekday.float().clamp(0.0, 6.0)
        hol_ix = is_holiday.long().clamp(0, 1)

        v_t = self.t2v_time(t_rel)
        v_w = self.t2v_weekday(wday)
        v_h = self.emb_holiday(hol_ix)

        # Ensure shapes are compatible for concatenation
        if v_t.dim() == 2:  # (B, K)
            base = torch.cat([v_t, v_w, v_h], dim=-1)  # (B, base_dim)
        else:               # (B,L,K)
            base = torch.cat([v_t, v_w, v_h], dim=-1)  # (B,L,base_dim)

        return self.proj(base)


# ---------------------------------------------------------------------
# Check-in token fusion
# ---------------------------------------------------------------------
class CheckInFusion(nn.Module):
    """
    Fuse per-step embeddings into a token embedding fed to the LLM.

    Token = f(user_pref, user_social, poi, time, space, category).

    Supports:
      - batched sequences: (B,L,dim)
      - single-step tokens: (B,dim)

    If gate='softmax', learns scalar weights for each component.
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
        gate: Optional[str] = None,   # None | 'softmax'
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

        fused_in = (2 * d_user) + d_poi + d_time + d_space + d_cat

        self.gate_type = gate
        if gate == "softmax":
            g_hid = max(64, fused_in // 4)
            self.gate_mlp = nn.Sequential(
                nn.Linear(fused_in, g_hid),
                nn.ReLU(),
                nn.Linear(g_hid, 6),  # user_pref, user_social, poi, time, space, cat
            )

        self.proj = nn.Sequential(
            nn.Linear(fused_in, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    @staticmethod
    def _ensure_3d(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """Promote (B,d) to (B,1,d) for uniform sequence computation."""
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
        e_social, _ = self._ensure_3d(e_social)
        e_poi, squeeze_p = self._ensure_3d(e_poi)
        e_time, _ = self._ensure_3d(e_time)
        e_space, _ = self._ensure_3d(e_space)
        e_cat, _ = self._ensure_3d(e_cat)

        u = self.dropout(self.ln_u(e_user))
        u_n = self.dropout(self.ln_un(e_social))
        p = self.dropout(self.ln_p(e_poi))
        t = self.dropout(self.ln_t(e_time))
        s = self.dropout(self.ln_s(e_space))
        c = self.dropout(self.ln_c(e_cat))

        fused = torch.cat([u, u_n, p, t, s, c], dim=-1)

        if self.gate_type == "softmax":
            scores = self.gate_mlp(fused)                 # (B,L,6)
            alphas = F.softmax(scores, dim=-1)            # (B,L,6)
            a_u, a_n, a_p, a_t, a_s, a_c = [alphas[..., i:i+1] for i in range(6)]
            fused = torch.cat([a_u * u, a_n * u_n, a_p * p, a_t * t, a_s * s, a_c * c], dim=-1)

        out = self.proj(fused)                            # (B,L,out_dim)

        # If single-step input, restore (B,out_dim)
        if squeeze_u and squeeze_p:
            out = out.squeeze(1)
        return out


# ---------------------------------------------------------------------
# PFA: Parameter-Efficient GPT-2 Adaptation
# ---------------------------------------------------------------------
class PFA(nn.Module):
    """
    Parameter-efficient adaptation of GPT-2 as a sequence encoder.

    We use a truncated GPT-2 (first gpt_layers blocks).
    Only a small subset of parameters are trainable:
      - optionally LayerNorms in early blocks
      - attentions (and optionally MLP) in the last U blocks
      - optionally final LayerNorm ln_f
      - optional LoRA on attention projection modules

    Forward:
      inputs_embeds: (B,L,H)
      attention_mask: (B,L) or None
    Returns:
      hidden states: (B,L,H)
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
        assert gpt_layers <= self.gpt.config.n_layer, "gpt_layers must be <= pretrained depth."

        # Truncate blocks
        self.gpt.h = self.gpt.h[:gpt_layers]
        self.gpt.config.n_layer = gpt_layers

        self.last_start = gpt_layers - U

        # Set dropout (paper config)
        self.gpt.attn_pdrop = 0.3
        self.gpt.resid_pdrop = 0.3
        self.gpt.embd_pdrop = 0.3

        # Freeze all by default
        for p in self.gpt.parameters():
            p.requires_grad = False

        # Early blocks: optionally train LayerNorms
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

        # Last U blocks: train attention (and optionally MLP)
        for li in range(self.last_start, gpt_layers):
            blk = self.gpt.h[li]
            for name, p in blk.named_parameters():
                if ("attn" in name) or (train_last_mlp and "mlp" in name):
                    p.requires_grad = True

        # Final LayerNorm
        if train_final_ln:
            for p in self.gpt.ln_f.parameters():
                p.requires_grad = True

        # LoRA injection (optional)
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

            # Keep LoRA trainable only in the last U blocks
            for name, p in self.gpt.named_parameters():
                if "lora_" in name:
                    keep = any(f"h.{li}." in name for li in range(self.last_start, gpt_layers))
                    p.requires_grad = keep

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.last_hidden_state

    def trainable_parameters(self):
        """Return only parameters that require gradients."""
        return [p for p in self.gpt.parameters() if p.requires_grad]


# ---------------------------------------------------------------------
# Next POI predictor with PFA backbone
# ---------------------------------------------------------------------
class NextPOIWithPFA(nn.Module):
    """
    Predict next POI (and auxiliary targets) from PFA hidden states.

    logit_mode:
      - 'index': out_poi/out_cat are direct classification logits.
      - 'cos'  : out_poi/out_cat are projected vectors; logits computed via cosine similarity
                against embedding tables (poi_final_emb, cat_emb_weight).

    Forward Inputs:
      inputs_embeds   : (B,L,H) fused tokens
      attention_mask  : (B,L)
      time_prompt     : (B,L,d_time) optional
      poi_final_emb   : (V_poi,D_poi) required in 'cos'
      cat_emb_weight  : (V_cat,D_cat) required in 'cos'

    Forward Outputs:
      dict with:
        logits_poi : (B,L,V_poi)
        logits_cat : (B,L,V_cat)
        pred_time  : (B,L,2)  (sin, cos)
        pred_loc   : (B,L,2)  (lat, lon) in unconstrained range
    """
    def __init__(
        self,
        pfa: nn.Module,
        *,
        num_pois: Optional[int] = None,
        num_cats: Optional[int] = None,
        logit_mode: str = "cos",
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
        eps: float = 1e-8,
    ):
        super().__init__()
        self.pfa = pfa
        H = self.pfa.gpt.config.hidden_size  # GPT-2 hidden size

        # ---- Time FiLM prompt (low-rank; neutral init) ----
        self.use_time_prompt = True
        d_time = 32   # must match TimeEmbedding(dim)
        r = 64

        self.time_film_alpha = nn.Parameter(torch.tensor(0.10))
        self.time_norm = nn.LayerNorm(d_time)
        self.time_A = nn.Linear(d_time, r)
        self.time_act = nn.ReLU()
        self.time_B = nn.Linear(r, 2 * H)

        nn.init.zeros_(self.time_B.weight)
        nn.init.zeros_(self.time_B.bias)
        self.time_layernorm = nn.LayerNorm(H)

        # ---- Prediction heads ----
        self.logit_mode = logit_mode
        self.num_pois = num_pois
        self.num_cats = num_cats

        if logit_mode == "cos":
            self.proj_poi = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, poi_proj_dim, bias=False))
            self.proj_cat = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, cat_proj_dim, bias=False))
        elif logit_mode == "index":
            assert num_pois is not None and num_cats is not None, "index mode requires num_pois/num_cats."
            self.proj_poi = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, num_pois, bias=False))
            self.proj_cat = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, num_cats, bias=False))
        else:
            raise ValueError(f"logit_mode must be 'cos' or 'index', got: {logit_mode}")

        self.proj_time = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, 2, bias=False))
        self.proj_loc = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, 2, bias=False))

        # Scaling / loss config
        self.learnable_scale = bool(learnable_scale)
        if self.learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(init_scale)))

        self.label_ignore_index = int(label_ignore_index)
        self.tail_gamma = float(tail_gamma)
        self.temperature = float(temperature)
        self.lambda_poi = float(lambda_poi)
        self.lambda_time = float(lambda_time)
        self.lambda_cat = float(lambda_cat)
        self.lambda_loc = float(lambda_loc)
        self.eps = float(eps)

        # Optional priors / tables
        self.T_prior_dense = None
        self.T_src_all = self.T_dst_all = self.T_val_all = None
        self.T_V = None
        self.poi_latlon_rad = None

    # ---------------- settings ----------------
    @torch.no_grad()
    def set_transition_prior(self, prior: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
        """Set transition prior: dense (V,V) or sparse tuple (src,dst,val,V)."""
        self.T_prior_dense = None
        self.T_src_all = self.T_dst_all = self.T_val_all = None

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
        poi_latlon_deg: (V,2) with [lat_deg, lon_deg], including padding row at 0.
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

    # ---------------- internal helpers ----------------
    def _apply_time_film(self, h: torch.Tensor, time_prompt: torch.Tensor) -> torch.Tensor:
        """FiLM modulation: h <- LN(h*(1+a*gamma) + a*beta)."""
        z = self.time_norm(time_prompt)
        gb = self.time_B(self.time_act(self.time_A(z)))          # (B,L,2H)
        gamma, beta = torch.chunk(gb, 2, dim=-1)                 # (B,L,H), (B,L,H)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        a = self.time_film_alpha
        return self.time_layernorm(h * (1.0 + a * gamma) + a * beta)

    def _cosine_logits(
        self,
        out_poi: torch.Tensor,
        out_cat: torch.Tensor,
        poi_final_emb: torch.Tensor,
        cat_emb_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine-similarity logits against embedding tables."""
        poi_out_norm = F.normalize(out_poi, dim=-1)
        cat_out_norm = F.normalize(out_cat, dim=-1)
        E_poi = F.normalize(poi_final_emb, dim=-1)
        E_cat = F.normalize(cat_emb_weight, dim=-1)

        logits_poi = torch.einsum("bld,vd->blv", poi_out_norm, E_poi) * self.scale
        logits_cat = torch.einsum("bld,vd->blv", cat_out_norm, E_cat)
        return logits_poi, logits_cat

    # ---------------- forward ----------------
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        time_prompt: Optional[torch.Tensor] = None,
        poi_final_emb: Optional[torch.Tensor] = None,
        cat_emb_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.pfa(inputs_embeds=inputs_embeds, attention_mask=attention_mask)  # (B,L,H)

        if self.use_time_prompt and (time_prompt is not None):
            h = self._apply_time_film(h, time_prompt)

        out_poi = self.proj_poi(h)
        out_cat = self.proj_cat(h)
        pred_time = self.proj_time(h)
        pred_loc = self.proj_loc(h)

        if self.logit_mode == "index":
            logits_poi, logits_cat = out_poi, out_cat
        else:
            assert poi_final_emb is not None and cat_emb_weight is not None, \
                "cos mode requires poi_final_emb and cat_emb_weight."
            logits_poi, logits_cat = self._cosine_logits(out_poi, out_cat, poi_final_emb, cat_emb_weight)

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

    # -----------------------------------------------------------------
    # Losses (shifted; weighted towards later steps)
    # -----------------------------------------------------------------
    def _tail_weights(self, mask_bt: torch.Tensor) -> torch.Tensor:
        """Tail weighting w_t = (t / length)^gamma for valid positions."""
        B, T = mask_bt.shape
        if T == 0:
            return mask_bt
        lengths = mask_bt.sum(dim=1, keepdim=True).clamp_min(1.0)
        pos = torch.arange(1, T + 1, device=mask_bt.device, dtype=mask_bt.dtype).unsqueeze(0)
        w = (pos / lengths) ** self.tail_gamma
        return w * mask_bt

    def _shifted_ce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        B, L, V = logits.shape
        ce = F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=self.label_ignore_index,
            reduction="none",
            label_smoothing=label_smoothing,
        ).reshape(B, L)
        w = self._tail_weights(mask.float())
        return (ce * w).sum() / w.sum().clamp_min(1e-8)

    def _shifted_time_cos_loss(self, pred_time: torch.Tensor, y_time: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        p = pred_time / pred_time.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        g = y_time / y_time.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        cos_loss = 1.0 - (p * g).sum(dim=-1)
        return (cos_loss * m).sum() / m.sum().clamp_min(1.0)

    def _shifted_loc_haversine_loss(self, pred_loc_rad: torch.Tensor, labels_poi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        pred_loc_rad: (B,L,2) predicted (lat,lon) in radians
        labels_poi  : (B,L)   next-step POI indices
        """
        assert self.poi_latlon_rad is not None, "poi_latlon_rad is not set. Call set_poi_latlon() before training."
        m = mask.float()

        gold_ids = labels_poi.clamp_min(0)
        tgt = self.poi_latlon_rad[gold_ids]  # (B,L,2)

        lat1, lon1 = pred_loc_rad[..., 0], pred_loc_rad[..., 1]
        lat2, lon2 = tgt[..., 0], tgt[..., 1]

        dlat = 0.5 * (lat2 - lat1)
        dlon = 0.5 * (lon2 - lon1)

        a = (torch.sin(dlat).pow(2) +
             torch.cos(lat1).clamp(-1, 1) * torch.cos(lat2).clamp(-1, 1) * torch.sin(dlon).pow(2))
        c = 2.0 * torch.atan2(torch.sqrt(a.clamp_min(0)), torch.sqrt((1 - a).clamp_min(1e-12)))

        valid = (labels_poi != self.label_ignore_index).float()
        m = m * valid

        w = self._tail_weights(m)
        return (c * w).sum() / w.sum().clamp_min(1e-8)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def build_transition_prior_dense(
    poi_traj_data,
    V: int,
    device: torch.device,
    alpha: float = 0.95,
    pad_id: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a dense row-stochastic transition prior T \in R^{V x V} from a POI graph.

    T[row,:] = (1-alpha)/V + alpha * normalized_counts(row,:)

    - Padding row/col are set to 0.
    - Row normalization is applied again for numerical stability.

    IMPORTANT (no leakage):
      poi_traj_data must be built from TRAIN split only.
    """
    ei = poi_traj_data.edge_index.to(device)
    if not hasattr(poi_traj_data, "edge_weight") or poi_traj_data.edge_weight is None:
        w = torch.ones(ei.size(1), dtype=dtype, device=device)
    else:
        w = poi_traj_data.edge_weight.to(device).to(dtype)

    src, dst = ei[0], ei[1]
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
    """Haversine distance in km (inputs in radians)."""
    dlat = (lat2 - lat1) * 0.5
    dlon = (lon2 - lon1) * 0.5
    a = torch.sin(dlat).pow(2) + torch.cos(lat1).clamp(-1, 1) * torch.cos(lat2).clamp(-1, 1) * torch.sin(dlon).pow(2)
    c = 2.0 * torch.atan2(torch.sqrt(torch.clamp(a, 0, 1)), torch.sqrt(torch.clamp(1 - a, eps, 1)))
    return 6371.0088 * c


def bias_logits_all_steps(
    logits_poi: torch.Tensor,     # (B,L,V)
    attention_mask: torch.Tensor, # (B,L)
    prev_ids: torch.Tensor,       # (B,L)
    *,
    T_prior: torch.Tensor,        # (V,V)
    T_prior_topk: Optional[torch.Tensor] = None,  # (V,K)
    mode: str = "topk",           # "topk" | "full"
    beta_trans: float = 0.5,
    center_prior: Optional[str] = "zscore",
    pad_idx: int = 0,
    in_place: bool = True,
) -> torch.Tensor:
    """
    Add transition log-prior to logits at each time step.

    For each t, we retrieve log T_prior[prev_ids[:,t], :] and add it to logits_poi[:,t,:].
    Only valid positions (attention_mask=1 and prev_id>0) are updated.

    mode="full": add to all V entries.
    mode="topk": add only to top-K transitions via scatter_add_ (requires T_prior_topk).

    Returns:
      updated logits with the same shape (B,L,V).
    """
    assert mode in ("topk", "full")
    B, L, V = logits_poi.shape
    assert T_prior.shape == (V, V)

    if mode == "topk":
        assert T_prior_topk is not None and T_prior_topk.shape[0] == V, "T_prior_topk must be (V,K)."

    out = logits_poi if in_place else logits_poi.clone()

    prior_all = T_prior[prev_ids.clamp_min(0)].clamp_min(1e-12).log()  # (B,L,V)

    if center_prior == "zscore":
        mu = prior_all.mean(dim=-1, keepdim=True)
        sd = prior_all.std(dim=-1, keepdim=True).add_(1e-6)
        prior_all = (prior_all - mu) / sd

    valid = ((attention_mask > 0) & (prev_ids > 0)).unsqueeze(-1).float()

    if mode == "full":
        out += beta_trans * (prior_all * valid)
    else:
        idx_topk = T_prior_topk[prev_ids]                 # (B,L,K)
        vals_topk = prior_all.gather(-1, idx_topk)        # (B,L,K)
        vals_topk = vals_topk * valid
        out.scatter_add_(dim=2, index=idx_topk, src=beta_trans * vals_topk)

    out[..., pad_idx] = -1e9
    return out


def get_batch_inputs_embeds(
    user_idxs: torch.Tensor,      # (B,)
    x_poi_idxs: torch.Tensor,     # (B,L)
    x_cat_idxs: torch.Tensor,     # (B,L)
    x_geo_idxs: torch.Tensor,     # (B,L)
    x_time_feats: torch.Tensor,   # (B,L,3) => (weekday, time_frac, holiday_flag)
    *,
    user_final_emb: nn.Module,
    poi_final_emb: torch.Tensor,  # (V_poi,D_poi)
    cat_emb: nn.Module,
    space_emb: nn.Module,
    time_embed_model: nn.Module,
    check_in_fusion_model: nn.Module,
) -> torch.Tensor:
    """
    Vectorized embedding construction for an entire batch.

    Returns:
      fused_embeddings: (B,L,H) to be used as inputs_embeds for PFA/GPT2.
    """
    B, L = x_poi_idxs.shape

    u_pref, u_social = user_final_emb(user_idxs)               # (B,Du), (B,Du)
    u_pref = u_pref.unsqueeze(1).expand(-1, L, -1)             # (B,L,Du)
    u_social = u_social.unsqueeze(1).expand(-1, L, -1)         # (B,L,Du)

    poi_emb = poi_final_emb[x_poi_idxs]                        # (B,L,Dp)
    cat_e = cat_emb(x_cat_idxs)                                # (B,L,Dc)
    space_e = space_emb(x_geo_idxs)                            # (B,L,Ds)

    time_e = time_embed_model(
        x_time_feats[..., 0],
        x_time_feats[..., 1],
        x_time_feats[..., 2],
    )                                                         # (B,L,Dt)

    fused = check_in_fusion_model(
        u_pref,
        u_social,
        poi_emb,
        time_e,
        space_e,
        cat_e,
    )                                                         # (B,L,H)

    return fused
