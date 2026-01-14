# Embedding_Ablation/model.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from transformers import GPT2Model
from peft import get_peft_model, LoraConfig, TaskType

PAD = 0


# =============================================================================
# Embedding modules
# =============================================================================
class UserEmbedding(nn.Module):
    """
    User embedding with 1-hop neighbor aggregation (social embedding).

    - long_pref_emb: learned user embedding
    - user_neighbor_idx/user_neighbor_w: fixed neighbor graph buffers
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

        # Fixed neighbor structures as buffers
        self.register_buffer("user_neighbor_idx", user_pad_idx)
        self.register_buffer("user_neighbor_w", user_pad_w)

    def forward(self, user_ids: torch.Tensor | int):
        """
        Args:
            user_ids: (B,) or scalar int/tensor

        Returns:
            u_pref:   (B, d) or (d,)
            u_social: (B, d) or (d,)
        """
        if not isinstance(user_ids, torch.Tensor):
            user_ids = torch.tensor([user_ids], device=self.long_pref_emb.weight.device, dtype=torch.long)
            was_scalar = True
        else:
            was_scalar = (user_ids.ndim == 0)
            if was_scalar:
                user_ids = user_ids.view(1)

        # (B,d)
        u_pref = self.long_pref_emb(user_ids)

        # (B,K)
        neighbor_idx = self.user_neighbor_idx[user_ids]
        neighbor_w = self.user_neighbor_w[user_ids]

        # (B,K,d)
        neighbor_emb = self.long_pref_emb(neighbor_idx)

        # weighted sum -> (B,d)
        u_social = (neighbor_emb * neighbor_w.unsqueeze(-1)).sum(dim=1)

        if was_scalar:
            return u_pref[0], u_social[0]
        return u_pref, u_social

    @property
    def weight(self) -> torch.Tensor:
        return self.long_pref_emb.weight


class CatEmbedding(nn.Module):
    """Category embedding."""

    def __init__(self, num_cats: int, dim: int, padding_idx: int = PAD):
        super().__init__()
        self.cat_emb = nn.Embedding(num_cats, dim, padding_idx=padding_idx)

    def forward(self, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.cat_emb(cat_ids)

    @property
    def weight(self) -> torch.Tensor:
        return self.cat_emb.weight


class POIEncoderGCN(nn.Module):
    """
    2-layer GCN encoder for POI embeddings.

    x: (V, in_dim)
    data: PyG Data with edge_index, edge_weight
    return: (V, out_dim)
    """

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hid_dim)

    def forward(self, x: torch.Tensor, data) -> torch.Tensor:
        x = self.conv1(x, data.edge_index, data.edge_weight)
        x = torch.relu(x)
        x = self.dropout(self.ln1(x))
        x = self.conv2(x, data.edge_index, data.edge_weight)
        return x


# =============================================================================
# Time embedding (Time2Vec-like)
# =============================================================================
class _T2V1D(nn.Module):
    """
    Simple Time2Vec for one scalar input.

    Accepts:
      - t: (B,) or (B,L)
    Returns:
      - (B, 1+K) if input was (B,)
      - (B, L, 1+K) if input was (B,L)
    """

    def __init__(self, K: int, period: float | None = None):
        super().__init__()
        self.K = int(K)

        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.zeros(1))

        self.w = nn.Parameter(torch.randn(self.K))
        self.b = nn.Parameter(torch.zeros(self.K))

        if period is not None and self.K > 0:
            base = 2.0 * math.pi / float(period)
            with torch.no_grad():
                for k in range(self.K):
                    self.w[k] = torch.tensor((k + 1) * base)
                self.b.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            # (B,L) -> (B,L,1)
            t3 = t.unsqueeze(-1)
            squeeze_seq = False
        elif t.dim() == 1:
            # (B,) -> (B,1,1) for unified ops then squeeze back
            t3 = t.unsqueeze(-1).unsqueeze(1)
            squeeze_seq = True
        else:
            raise ValueError(f"_T2V1D expects (B,) or (B,L), got shape={tuple(t.shape)}")

        lin = self.w0 * t3 + self.b0                   # (B,L,1)
        per = torch.sin(t3 * self.w + self.b)          # (B,L,K)
        out = torch.cat([lin, per], dim=-1)            # (B,L,1+K)

        if squeeze_seq:
            out = out.squeeze(1)                       # (B,1+K)
        return out


class TimeEmbedding(nn.Module):
    """
    Embed time features (weekday, time_frac, holiday_flag).

    Inputs:
      - weekday: (B,) or (B,L)  [0..6] or normalized (0..6)
      - time:    (B,) or (B,L)  normalized [0..1]
      - is_holiday: (B,) or (B,L) 0/1

    Returns:
      - (B,dim) or (B,L,dim)
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
        # allow (B,) or (B,L)
        if weekday.dim() not in (1, 2) or time.dim() != weekday.dim() or is_holiday.dim() != weekday.dim():
            raise ValueError(
                f"TimeEmbedding expects weekday/time/is_holiday with same dim (1 or 2). "
                f"Got weekday={tuple(weekday.shape)}, time={tuple(time.shape)}, hol={tuple(is_holiday.shape)}"
            )

        # stabilize
        t_rel = time.float().clamp(0.0, 1.0)
        wday = weekday.float().clamp(0.0, 6.0)
        hol_ix = is_holiday.long().clamp(0, 1)

        v_t = self.t2v_time(t_rel)          # (B,1+K) or (B,L,1+K)
        v_w = self.t2v_weekday(wday)        # same
        v_h = self.emb_holiday(hol_ix)      # (B,d_hol) or (B,L,d_hol)

        # If (B,) case, make v_h align with (B, base) concat
        if weekday.dim() == 1 and v_h.dim() == 2:
            # (B,d_hol) is fine, concat expects (B, *)
            base = torch.cat([v_t, v_w, v_h], dim=-1)  # (B, base_dim)
        else:
            # (B,L,*) case
            base = torch.cat([v_t, v_w, v_h], dim=-1)  # (B, L, base_dim)

        return self.proj(base)


# =============================================================================
# Check-in fusion
# =============================================================================
class CheckInFusion(nn.Module):
    """
    Fuse (user, social, poi, time, space, cat) embeddings into token embedding.

    Supports:
      - inputs: (B,L,d) or (B,d)
      - output: (B,L,out_dim) or (B,out_dim)

    Optional gate='softmax':
      - learn per-part scalar weights (6 parts) per timestep
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
        gate: str | None = None,  # None | 'softmax'
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.dropout = nn.Dropout(dropout)

        self.ln_u = nn.LayerNorm(d_user)
        self.ln_un = nn.LayerNorm(d_user)
        self.ln_p = nn.LayerNorm(d_poi)
        self.ln_t = nn.LayerNorm(d_time)
        self.ln_s = nn.LayerNorm(d_space)
        self.ln_c = nn.LayerNorm(d_cat)

        fused_in = d_user * 2 + d_poi + d_time + d_space + d_cat
        self.fused_in = fused_in

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
    def _ensure_3d(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        # (B,d) -> (B,1,d)
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
        # unify dims
        e_user, squeeze_u = self._ensure_3d(e_user)
        e_social, squeeze_s = self._ensure_3d(e_social)
        e_poi, squeeze_p = self._ensure_3d(e_poi)
        e_time, squeeze_t = self._ensure_3d(e_time)
        e_space, squeeze_sp = self._ensure_3d(e_space)
        e_cat, squeeze_c = self._ensure_3d(e_cat)

        # normalize + dropout
        u = self.dropout(self.ln_u(e_user))
        un = self.dropout(self.ln_un(e_social))
        p = self.dropout(self.ln_p(e_poi))
        t = self.dropout(self.ln_t(e_time))
        s = self.dropout(self.ln_s(e_space))
        c = self.dropout(self.ln_c(e_cat))

        fused = torch.cat([u, un, p, t, s, c], dim=-1)  # (B,L,fused_in)

        if self.gate_type == "softmax":
            scores = self.gate_mlp(fused)               # (B,L,6)
            alphas = F.softmax(scores, dim=-1)          # (B,L,6)
            a_u, a_un, a_p, a_t, a_s, a_c = [alphas[..., i:i+1] for i in range(6)]
            fused = torch.cat([a_u * u, a_un * un, a_p * p, a_t * t, a_s * s, a_c * c], dim=-1)

        out = self.proj(fused)  # (B,L,out_dim)

        # squeeze back only if ALL inputs were non-sequence (safe rule)
        if squeeze_u and squeeze_s and squeeze_p and squeeze_t and squeeze_sp and squeeze_c:
            out = out.squeeze(1)  # (B,out_dim)
        return out


# =============================================================================
# PFA (GPT2 encoder)
# =============================================================================
class PFA(nn.Module):
    """
    GPT-2 encoder with partial fine-tuning (PFA-style).

    - freeze all by default
    - optionally train early block layer norms
    - train last U blocks attention (and optionally MLP)
    - optional LoRA only on last U blocks
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
        lora_targets: tuple[str, ...] = ("c_attn", "c_proj"),
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        super().__init__()

        self.gpt = GPT2Model.from_pretrained(
            gpt_name,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if gpt_layers > self.gpt.config.n_layer:
            raise ValueError("gpt_layers cannot exceed pretrained model depth.")

        # truncate blocks
        self.gpt.h = self.gpt.h[:gpt_layers]
        self.gpt.config.n_layer = gpt_layers

        self.last_start = max(0, gpt_layers - int(U))
        self.use_lora = bool(use_lora)

        # set dropouts (config-level)
        self.gpt.config.attn_pdrop = 0.3
        self.gpt.config.resid_pdrop = 0.3
        self.gpt.config.embd_pdrop = 0.3

        # freeze all
        for p in self.gpt.parameters():
            p.requires_grad = False

        # early blocks: LN train
        if train_early_ln:
            for li in range(0, self.last_start):
                blk = self.gpt.h[li]
                for name, p in blk.named_parameters():
                    if name.startswith("ln_"):
                        p.requires_grad = True

        # positional embedding optional train
        if train_pos_emb_early:
            for p in self.gpt.wpe.parameters():
                p.requires_grad = True

        # last U blocks: train attention (and optionally MLP)
        for li in range(self.last_start, gpt_layers):
            blk = self.gpt.h[li]
            for name, p in blk.named_parameters():
                if ("attn" in name) or (train_last_mlp and "mlp" in name):
                    p.requires_grad = True

        # final LN
        if train_final_ln:
            for p in self.gpt.ln_f.parameters():
                p.requires_grad = True

        # LoRA attachment (optional)
        if self.use_lora:
            peft_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_targets),
            )
            self.gpt = get_peft_model(self.gpt, peft_cfg)

            # Only keep LoRA params trainable for last U blocks
            for name, p in self.gpt.named_parameters():
                if "lora_" in name:
                    keep = any(f"h.{li}." in name for li in range(self.last_start, gpt_layers))
                    p.requires_grad = keep

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.gpt(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.last_hidden_state

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


# =============================================================================
# Next POI predictor
# =============================================================================
class NextPOIWithPFA(nn.Module):
    """
    Predict next POI / category / time / location from PFA hidden states.

    logit_mode:
      - 'index': proj heads output vocab-sized logits directly
      - 'cos'  : proj heads output embedding; logits computed by cosine to item embeddings
    """

    def __init__(
        self,
        pfa: PFA,
        use_time_prompt: bool = True,
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
        H = self.pfa.gpt.config.hidden_size

        self.use_time_prompt = bool(use_time_prompt)
        self.time_film_alpha = nn.Parameter(torch.tensor(0.10))

        d_time = 32   # must match TimeEmbedding(dim)
        r = 64

        self.time_norm = nn.LayerNorm(d_time)
        self.time_A = nn.Linear(d_time, r)
        self.time_act = nn.ReLU()
        self.time_B = nn.Linear(r, 2 * H)

        nn.init.zeros_(self.time_B.weight)
        nn.init.zeros_(self.time_B.bias)

        self.time_layernorm = nn.LayerNorm(H)

        self.logit_mode = logit_mode
        if self.logit_mode not in ("cos", "index"):
            raise ValueError(f"logit_mode must be 'cos' or 'index', got={self.logit_mode}")

        self.num_pois = num_pois
        self.num_cats = num_cats

        if self.logit_mode == "cos":
            self.proj_poi = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, poi_proj_dim, bias=False))
            self.proj_cat = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, cat_proj_dim, bias=False))
        else:
            if num_pois is None or num_cats is None:
                raise ValueError("index mode requires num_pois and num_cats.")
            self.proj_poi = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, num_pois, bias=False))
            self.proj_cat = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, num_cats, bias=False))

        self.proj_time = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, 2, bias=False))
        self.proj_loc = nn.Sequential(nn.Dropout(0.3), nn.Linear(H, 2, bias=False))

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(init_scale)))
        self.learnable_scale = bool(learnable_scale)

        self.label_ignore_index = int(label_ignore_index)
        self.temperature = float(temperature)
        self.tail_gamma = float(tail_gamma)
        self.eps = float(eps)

        self.lambda_poi = float(lambda_poi)
        self.lambda_time = float(lambda_time)
        self.lambda_cat = float(lambda_cat)
        self.lambda_loc = float(lambda_loc)

        self.poi_latlon_rad: Optional[torch.Tensor] = None

    @torch.no_grad()
    def set_poi_latlon(self, poi_latlon_deg: torch.Tensor):
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
        params += [self.time_film_alpha]
        params += list(self.time_norm.parameters())
        params += list(self.time_A.parameters())
        params += list(self.time_B.parameters())
        params += list(self.time_layernorm.parameters())
        return params

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        time_prompt: Optional[torch.Tensor] = None,
        poi_final_emb: Optional[torch.Tensor] = None,
        cat_emb_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # PFA encoding
        h = self.pfa(inputs_embeds=inputs_embeds, attention_mask=attention_mask)  # (B,L,H)

        # FiLM from time_prompt
        if self.use_time_prompt and (time_prompt is not None):
            z = self.time_norm(time_prompt)                          # (B,L,dt)
            gb = self.time_B(self.time_act(self.time_A(z)))          # (B,L,2H)
            gamma, beta = torch.chunk(gb, 2, dim=-1)                 # (B,L,H)
            gamma = torch.tanh(gamma)
            beta = torch.tanh(beta)
            h = self.time_layernorm(h * (1.0 + self.time_film_alpha * gamma) + self.time_film_alpha * beta)

        # heads
        out_poi = self.proj_poi(h)
        out_cat = self.proj_cat(h)
        pred_time = self.proj_time(h)
        pred_loc = self.proj_loc(h)

        # logits
        if self.logit_mode == "index":
            logits_poi = out_poi
            logits_cat = out_cat
        else:
            if poi_final_emb is None or cat_emb_weight is None:
                raise ValueError("cos mode requires poi_final_emb and cat_emb_weight.")

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
    def _tail_weights(self, mask_bt: torch.Tensor) -> torch.Tensor:
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

    def _shifted_loc_haversine_loss(self, pred_loc_rad: torch.Tensor, labels_poi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.poi_latlon_rad is None:
            raise ValueError("poi_latlon_rad is not set. Call set_poi_latlon() first.")

        m = mask.float()
        gold_ids = labels_poi.clamp_min(0)
        tgt = self.poi_latlon_rad[gold_ids]  # (B,L,2)

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


# =============================================================================
# Utility: batch -> inputs_embeds  (Ablation-aware)
# =============================================================================
def get_batch_inputs_embeds(
    user_idxs: torch.Tensor,      # (B,)
    x_poi_idxs: torch.Tensor,     # (B,L)
    x_cat_idxs: torch.Tensor,     # (B,L)
    x_geo_idxs: torch.Tensor,     # (B,L)
    x_time_feats: torch.Tensor,   # (B,L,3) (weekday_frac, time_frac, holiday_flag)
    *,
    user_final_emb: nn.Module,    # UserEmbedding
    poi_final_emb: torch.Tensor,  # (V_poi, D_poi)
    cat_emb: nn.Module,           # CatEmbedding
    space_emb: nn.Module,         # Embedding
    time_embed_model: nn.Module,  # TimeEmbedding
    check_in_fusion_model: nn.Module,  # CheckInFusion
    args,
) -> torch.Tensor:
    """
    Build (B,L,H) token embeddings for LLM input, with embedding ablation flags.

    Flags expected in args:
      - use_user, use_social, use_poi, use_cat, use_space, use_time
      - poi_dim, cat_dim, space_dim, time_dim, user_dim
    """
    device = x_poi_idxs.device
    B, L = x_poi_idxs.shape

    # ---- user/social (always define both) ----
    if getattr(args, "use_user", True) or getattr(args, "use_social", True):
        u_base, s_base = user_final_emb(user_idxs)  # (B,du), (B,du)
    else:
        u_base = torch.zeros(B, getattr(args, "user_dim"), device=device)
        s_base = torch.zeros(B, getattr(args, "user_dim"), device=device)

    if getattr(args, "use_user", True):
        u = u_base.unsqueeze(1).expand(-1, L, -1)
    else:
        u = torch.zeros(B, L, getattr(args, "user_dim"), device=device)

    if getattr(args, "use_social", True):
        s = s_base.unsqueeze(1).expand(-1, L, -1)
    else:
        s = torch.zeros(B, L, getattr(args, "user_dim"), device=device)

    # ---- poi ----
    if getattr(args, "use_poi", True):
        p = poi_final_emb[x_poi_idxs]
    else:
        p = torch.zeros(B, L, getattr(args, "poi_dim"), device=device)

    # ---- cat ----
    if getattr(args, "use_cat", True):
        c = cat_emb(x_cat_idxs)
    else:
        c = torch.zeros(B, L, getattr(args, "cat_dim"), device=device)

    # ---- space ----
    if getattr(args, "use_space", True):
        sp = space_emb(x_geo_idxs)
    else:
        sp = torch.zeros(B, L, getattr(args, "space_dim"), device=device)

    # ---- time ----
    if getattr(args, "use_time", True):
        t = time_embed_model(
            x_time_feats[..., 0],
            x_time_feats[..., 1],
            x_time_feats[..., 2],
        )
    else:
        t = torch.zeros(B, L, getattr(args, "time_dim"), device=device)

    # ---- fuse ----
    fused = check_in_fusion_model(
        u, s, p, t, sp, c
    )
    return fused
