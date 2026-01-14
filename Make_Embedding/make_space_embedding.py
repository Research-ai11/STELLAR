from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler

from make_geohash_init_embedding import make_geohash_timebase_node
from make_road_graph import make_road_graph


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class GATConfig:
    hidden_dim: int = 64
    out_dim: int = 128
    heads: int = 4
    dropout: float = 0.2


@dataclass(frozen=True)
class ContrastiveConfig:
    epochs: int = 100
    lr: float = 1e-3
    margin: float = 0.1
    weight_thresh: int = 1
    neg_sampling_attempts: int = 200
    seed: int = 42


DEFAULT_CAT_COLS = [
    "major_category_weekday",
    "major_category_holiday",
    "major_category_morning",
    "major_category_afternoon",
    "major_category_evening",
    "major_category_night",
]


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Small helpers
# -------------------------
def make_bidirectional_graph(graph: pd.DataFrame) -> pd.DataFrame:
    """Return a bidirectional version of an edge list DataFrame."""
    graph_bidirectional = pd.concat(
        [
            graph,
            graph.rename(columns={"geohash_start": "geohash_end", "geohash_end": "geohash_start"}),
        ],
        ignore_index=True,
    )
    return graph_bidirectional


def find_isolated_geohashes(graph: pd.DataFrame, emb: pd.DataFrame) -> Set[str]:
    """Find geohashes that never appear in the road graph edges."""
    connected_geos = set(graph["geohash_start"]).union(set(graph["geohash_end"]))
    all_geos = set(emb["geohash"])
    isolated_geos = all_geos - connected_geos
    print(f"💡 Isolated geohash count: {len(isolated_geos)}")
    return isolated_geos


def save_loss_curve(loss_arr: np.ndarray, save_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(loss_arr[:, 0], label="Total Loss")
    plt.plot(loss_arr[:, 1], label="Positive Loss")
    plt.plot(loss_arr[:, 2], label="Negative Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Contrastive Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------------
# Model
# -------------------------
class MultiHeadGAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

        self.act = nn.PReLU()
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gat1(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = self.norm(x)
        return x


# -------------------------
# Pair mining
# -------------------------
PosPair = Tuple[str, str, float, float]  # (g1, g2, weight, shared_cats)
NegPair = Tuple[str, str]                # (g1, g2)


def get_positive_pairs(
    graph: pd.DataFrame,
    main_cat_df: pd.DataFrame,
    *,
    weight_thresh: int = 1,
    cat_cols: Sequence[str] = DEFAULT_CAT_COLS,
) -> List[PosPair]:
    """
    Positive pairs: connected geohash pairs with edge weight >= threshold.
    Also compute shared major categories as an auxiliary signal.
    """
    cat_dict = main_cat_df.set_index("geohash")[list(cat_cols)].to_dict(orient="index")
    pos_pairs: List[PosPair] = []

    for _, row in graph.iterrows():
        g1, g2 = row["geohash_start"], row["geohash_end"]
        weight = row["weight"]

        if g1 in cat_dict and g2 in cat_dict:
            shared_cats = sum(cat_dict[g1][col] == cat_dict[g2][col] for col in cat_cols)
            if weight >= weight_thresh:
                pos_pairs.append((g1, g2, float(weight), float(shared_cats)))

    return pos_pairs


def get_isolated_positive_pairs(
    isolated_geos: Iterable[str],
    main_cat_df: pd.DataFrame,
    *,
    cat_cols: Sequence[str] = DEFAULT_CAT_COLS,
) -> List[PosPair]:
    """
    For isolated geohashes (no road connections), create a soft positive pair
    by matching the most similar geohash in terms of shared major categories.
    """
    cat_dict = main_cat_df.set_index("geohash")[list(cat_cols)].to_dict(orient="index")
    pos_pairs: List[PosPair] = []

    for g1 in isolated_geos:
        if g1 not in cat_dict:
            continue

        best_match = None
        best_score = -1

        for g2 in cat_dict.keys():
            if g1 == g2:
                continue
            shared_cats = sum(cat_dict[g1][col] == cat_dict[g2][col] for col in cat_cols)
            if shared_cats > best_score:
                best_score = shared_cats
                best_match = g2

        if best_match is not None and best_score > 0:
            pos_pairs.append(
                (
                    g1,
                    best_match,
                    0.1,                 # soft weight (kept as original)
                    best_score / 2.0,     # soft shared_cats (kept as original)
                )
            )

    return pos_pairs


def get_negative_pairs(
    pos_pairs: Sequence[PosPair],
    main_cat_df: pd.DataFrame,
    graph_df: pd.DataFrame,
    *,
    seed: int = 42,
    attempts: int = 200,
    cat_cols: Sequence[str] = DEFAULT_CAT_COLS,
) -> List[NegPair]:
    """
    For each geohash, generate as many negative pairs as its number of positives.
    Negatives must be non-connected and share zero major categories.
    """
    random.seed(seed)

    geos = main_cat_df["geohash"].tolist()
    cat_dict = main_cat_df.set_index("geohash")[list(cat_cols)].to_dict(orient="index")

    connected = set(zip(graph_df["geohash_start"], graph_df["geohash_end"]))

    pos_count = defaultdict(int)
    for g1, g2, _, _ in pos_pairs:
        pos_count[g1] += 1
        pos_count[g2] += 1

    neg_pairs: List[NegPair] = []
    for g1 in pos_count:
        count = 0
        tries = 0
        while count < pos_count[g1] and tries < attempts:
            g2 = random.choice(geos)
            if g1 == g2 or (g1, g2) in connected or (g2, g1) in connected:
                tries += 1
                continue

            shared_cats = sum(cat_dict[g1][col] == cat_dict[g2][col] for col in cat_cols)
            if shared_cats == 0:
                neg_pairs.append((g1, g2))
                count += 1

            tries += 1

    return neg_pairs


def count_pair_frequency(pos_pairs: Sequence[PosPair], neg_pairs: Sequence[NegPair]):
    """Check whether the number of negative pairs matches the number of positive pairs per geohash."""
    pos_count = defaultdict(int)
    neg_count = defaultdict(int)

    for g1, g2, _, _ in pos_pairs:
        pos_count[g1] += 1
        pos_count[g2] += 1

    for g1, _ in neg_pairs:
        neg_count[g1] += 1

    all_geos = set(pos_count.keys()).union(set(neg_count.keys()))
    mismatch = []
    for geo in sorted(all_geos):
        pos = pos_count.get(geo, 0)
        neg = neg_count.get(geo, 0)
        if pos != neg:
            mismatch.append((geo, pos, neg))

    if mismatch:
        print(f"⚠️ {len(mismatch)} geohashes have mismatched numbers of positive and negative pairs.")
        for geo, pos, neg in mismatch[:10]:
            print(f"  - {geo}: #pos={pos}, #neg={neg}")
    else:
        print("✅ All geohashes have perfectly matched numbers of positive and negative pairs.")

    return mismatch


# -------------------------
# Training
# -------------------------
def train_contrastive_gat(
    dataset_name: str,
    emb: pd.DataFrame,
    graph: pd.DataFrame,
    pos_pairs: Sequence[PosPair],
    neg_pairs: Sequence[NegPair],
    *,
    gat_cfg: GATConfig,
    cfg: ContrastiveConfig,
    llm_model: str = "GPT2",
) -> Tuple[MultiHeadGAT, torch.Tensor, Dict[str, int]]:
    """
    Train a GAT encoder with a margin-based contrastive objective and return:
      - trained model
      - normalized embeddings [N, D] on CPU
      - geo2idx mapping
    """
    # Keep original LLM-dependent dims (but map to configs)
    if llm_model == "GPT2":
        hidden_dim, output_dim = 64, 128
    elif llm_model == "LLAMA2":
        hidden_dim, output_dim = 128, 256
    else:
        # fallback to provided cfg
        hidden_dim, output_dim = gat_cfg.hidden_dim, gat_cfg.out_dim

    set_seed(cfg.seed)

    geo2idx = {geo: idx for idx, geo in enumerate(emb["geohash"])}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature scaling (same behavior)
    features = emb.drop(columns=["geohash"]).values
    scaler = StandardScaler().fit(features)
    X_np = scaler.transform(features)
    X = torch.from_numpy(X_np).float().to(device)

    # edge_index
    edge_list = [(geo2idx[r["geohash_start"]], geo2idx[r["geohash_end"]]) for _, r in graph.iterrows()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(device)

    model = MultiHeadGAT(
        in_channels=X.shape[1],
        hidden_channels=hidden_dim,
        out_channels=output_dim,
        heads=gat_cfg.heads,
        dropout=gat_cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Prebuild indices for vectorization (same logic)
    if len(pos_pairs) > 0:
        i_pos = torch.tensor([geo2idx[g1] for g1, _, _, _ in pos_pairs], device=device, dtype=torch.long)
        j_pos = torch.tensor([geo2idx[g2] for _, g2, _, _ in pos_pairs], device=device, dtype=torch.long)
        w_pos_vals = [max(np.log1p(w) + sc, 0.1) for _, _, w, sc in pos_pairs]
        w_pos = torch.tensor(w_pos_vals, device=device, dtype=torch.float)
    else:
        i_pos = j_pos = w_pos = None

    if len(neg_pairs) > 0:
        i_neg = torch.tensor([geo2idx[g1] for g1, _ in neg_pairs], device=device, dtype=torch.long)
        j_neg = torch.tensor([geo2idx[g2] for _, g2 in neg_pairs], device=device, dtype=torch.long)
    else:
        i_neg = j_neg = None

    loss_list = []

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()

        z = F.normalize(model(X, edge_index), p=2, dim=1)

        if i_pos is not None and len(i_pos) > 0:
            sim_pos = F.cosine_similarity(z[i_pos], z[j_pos], dim=1)
            loss_pos = ((1.0 - sim_pos) * (w_pos / (w_pos.mean() + 1e-12))).mean()
        else:
            loss_pos = torch.tensor(0.0, device=device)

        if i_neg is not None and len(i_neg) > 0:
            sim_neg = F.cosine_similarity(z[i_neg], z[j_neg], dim=1)
            loss_neg = F.relu(sim_neg - cfg.margin).mean()
        else:
            loss_neg = torch.tensor(0.0, device=device)

        loss = loss_pos + loss_neg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == cfg.epochs:
            print(
                f"Epoch {epoch+1:3d}/{cfg.epochs} - Loss: {loss.item():.4f} "
                f"(Pos: {loss_pos.item():.4f}, Neg: {loss_neg.item():.4f})"
            )

        loss_list.append((float(loss.item()), float(loss_pos.item()), float(loss_neg.item())))

    model.eval()
    with torch.no_grad():
        embeddings = F.normalize(model(X, edge_index), p=2, dim=1).detach().cpu()

    loss_arr = np.array(loss_list)
    #save_loss_curve(loss_arr, save_path=f"loss_curve_{dataset_name}_{llm_model}_epochs{cfg.epochs}.png")

    return model, embeddings, geo2idx


def summarize_pos_stats(pos_pairs: Sequence[PosPair]) -> None:
    """Print the same descriptive statistics as the original code."""
    weights = np.array([w for (_, _, w, _) in pos_pairs], dtype=float)
    shareds = np.array([s for (_, _, _, s) in pos_pairs], dtype=float)

    log_weights = np.log1p(weights)

    print("📊 [Weight]")
    print(f"Mean     : {weights.mean():.4f}")
    print(f"Min      : {weights.min():.4f}")
    print(f"Max      : {weights.max():.4f}")
    print(f"Std      : {weights.std():.4f}")

    print("📊 [Log Weight]")
    print(f"Mean     : {log_weights.mean():.4f}")
    print(f"Min      : {log_weights.min():.4f}")
    print(f"Max      : {log_weights.max():.4f}")
    print(f"Std      : {log_weights.std():.4f}")

    print("📊 [Shared Categories]")
    print(f"Mean     : {shareds.mean():.4f}")
    print(f"Min      : {shareds.min():.4f}")
    print(f"Max      : {shareds.max():.4f}")
    print(f"Std      : {shareds.std():.4f}")


def get_space_embedding(
    dataset_name: str,
    emb: pd.DataFrame,
    main_cat: pd.DataFrame,
    isolated_geos: Set[str],
    bigraph: pd.DataFrame,
    *,
    cfg: ContrastiveConfig,
    gat_cfg: Optional[GATConfig] = None,
    llm_model: str = "GPT2",
    cat_cols: Sequence[str] = DEFAULT_CAT_COLS,
):
    """
    Full space-embedding routine (pair mining + training). Kept close to original behavior.
    """
    if gat_cfg is None:
        gat_cfg = GATConfig()

    pos_pairs = get_positive_pairs(bigraph, main_cat, weight_thresh=cfg.weight_thresh, cat_cols=cat_cols)
    isolated_pos_pairs = get_isolated_positive_pairs(isolated_geos, main_cat, cat_cols=cat_cols)
    all_pos_pairs = pos_pairs + isolated_pos_pairs

    summarize_pos_stats(all_pos_pairs)

    neg_pairs = get_negative_pairs(
        all_pos_pairs, main_cat, bigraph, seed=cfg.seed, attempts=cfg.neg_sampling_attempts, cat_cols=cat_cols
    )

    count_pair_frequency(all_pos_pairs, neg_pairs)

    model, embeddings, geo2idx = train_contrastive_gat(
        dataset_name,
        emb,
        bigraph,
        all_pos_pairs,
        neg_pairs,
        gat_cfg=gat_cfg,
        cfg=cfg,
        llm_model=llm_model,
    )

    return model, embeddings, geo2idx, all_pos_pairs, neg_pairs


def space_embedding_pipeline(
    dataset_name: str,
    *,
    cfg: Optional[ContrastiveConfig] = None,
    gat_cfg: Optional[GATConfig] = None,
    llm_model: str = "GPT2",
):
    """
    Orchestration wrapper (still here for backward compatibility).
    In the final top-tier structure, this orchestration should move to run_space_embedding.py.
    """
    if cfg is None:
        cfg = ContrastiveConfig()
    if gat_cfg is None:
        gat_cfg = GATConfig()

    print(f"🚀 {dataset_name.upper()} Space Embedding Pipeline Start!")

    road_node, road_graph = make_road_graph(dataset_name)
    emb, main_cat = make_geohash_timebase_node(dataset_name)

    bigraph = make_bidirectional_graph(road_graph)
    print(f"Bigraph edge shape: {bigraph.shape[0]}개")

    isolated_geos = find_isolated_geohashes(bigraph, emb)

    model, embeddings, geo2idx, pos_pairs, neg_pairs = get_space_embedding(
        dataset_name,
        emb,
        main_cat,
        isolated_geos,
        bigraph,
        cfg=cfg,
        gat_cfg=gat_cfg,
        llm_model=llm_model,
    )

    return model, embeddings, geo2idx, pos_pairs, neg_pairs


if __name__ == "__main__":
    # Keep as a minimal smoke-test entry (final saving should be moved to run_space_embedding.py)
    nyc_model, nyc_embeddings, nyc_geo2idx, nyc_pos_pairs, nyc_neg_pairs = space_embedding_pipeline(
        "nyc", cfg=ContrastiveConfig(epochs=100, lr=0.001), llm_model="GPT2"
    )
    tky_model, tky_embeddings, tky_geo2idx, tky_pos_pairs, tky_neg_pairs = space_embedding_pipeline(
        "tky", cfg=ContrastiveConfig(epochs=100, lr=0.001), llm_model="GPT2"
    )
    print("✅ Space embedding pipeline completed for both NYC and TKY.")