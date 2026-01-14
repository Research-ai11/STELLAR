from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

# Import your modularized functions
from make_road_graph import make_road_graph
from make_geohash_init_embedding import make_geohash_timebase_node
from make_space_embedding import (
    GATConfig,
    ContrastiveConfig,
    make_bidirectional_graph,
    find_isolated_geohashes,
    get_space_embedding,
)


def get_project_root(explicit: Optional[str] = None) -> Path:
    """
    Resolve project root (STELLAR).
    Assumes this file is located at: STELLAR/Make_Embedding/run_space_embedding.py
    """
    if explicit is not None:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parents[1]


def save_space_embedding_csv(
    embeddings: torch.Tensor,
    geo2idx: Dict[str, int],
    out_path: Path,
) -> None:
    """
    Save [N, D] embedding tensor into CSV with geohash as index
    and feature_0..feature_{D-1} as columns.
    """
    emb_np = embeddings.detach().cpu().numpy()
    idx2geo = {idx: geo for geo, idx in geo2idx.items()}
    geo_list = [idx2geo[i] for i in range(len(idx2geo))]

    df = pd.DataFrame(
        data=emb_np,
        index=geo_list,
        columns=[f"feature_{i}" for i in range(emb_np.shape[1])],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run STELLAR space embedding pipeline (road -> init -> contrastive GAT).")

    p.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["nyc", "tky", "all"],
        help="Which dataset to run.",
    )
    p.add_argument("--llm_model", type=str, default="GPT2", choices=["GPT2", "LLAMA2"], help="Backbone setting.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--weight_thresh", type=int, default=1)
    p.add_argument("--neg_attempts", type=int, default=200)

    p.add_argument("--geohash_precision", type=int, default=6)
    p.add_argument("--plot_road_graph", action="store_true", help="If set, plot OSM road graph (slow / interactive).")

    p.add_argument(
        "--project_root",
        type=str,
        default=None,
        help="Explicit project root path (STELLAR). If omitted, inferred from file location.",
    )

    p.add_argument("--no_cache", action="store_true", help="If set, recompute even if output files exist.")
    return p


def run_one_dataset(
    dataset_name: str,
    *,
    project_root: Path,
    llm_model: str,
    cfg: ContrastiveConfig,
    gat_cfg: GATConfig,
    geohash_precision: int,
    plot_road_graph: bool,
    no_cache: bool,
) -> None:
    """
    Execute the full pipeline for a single dataset:
      1) build road graph (and save node/edge features)
      2) build initial geohash embedding (and save)
      3) contrastive GAT training -> final space embeddings (and save)
    """
    out_path = project_root / "data" / dataset_name / "graph" / f"{dataset_name}_geohash_gat_space_embedding.csv"

    if out_path.exists() and not no_cache:
        print(f"[CACHE] {dataset_name}: found {out_path.name}, skip. (use --no_cache to recompute)")
        return

    print(f"\n========== {dataset_name.upper()} | Space Embedding Pipeline ==========")

    # Step 1) Road graph
    # NOTE: Your make_road_graph has been updated to accept project_root / geohash_precision / plot_graph
    make_road_graph(
        dataset_name,
        project_root=project_root,
        geohash_precision=geohash_precision,
        plot_graph=plot_road_graph,
    )

    # Step 2) Initial embedding
    emb, main_cat = make_geohash_timebase_node(
        dataset_name,
        project_root=project_root,
        geohash_precision=geohash_precision,
    )

    # Step 3) Load edge list created in Step 1 (recommended: stable pipeline)
    # Your make_road_graph saves edge list to:
    #   data/{dataset}/graph/{dataset}_geohash_edge_features.csv
    edge_path = project_root / "data" / dataset_name / "graph" / f"{dataset_name}_geohash_edge_features.csv"
    road_graph = pd.read_csv(edge_path)

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

    # Save final embedding
    save_space_embedding_csv(embeddings, geo2idx, out_path)
    print(f"[SAVE] {dataset_name}: {out_path}")


def main():
    args = build_parser().parse_args()

    project_root = get_project_root(args.project_root)

    # LLM-dependent default dimensions (kept consistent with your original logic)
    if args.llm_model == "GPT2":
        gat_cfg = GATConfig(hidden_dim=64, out_dim=128, heads=4, dropout=0.2)
    else:  # LLAMA2
        gat_cfg = GATConfig(hidden_dim=128, out_dim=256, heads=4, dropout=0.2)

    cfg = ContrastiveConfig(
        epochs=args.epochs,
        lr=args.lr,
        margin=args.margin,
        weight_thresh=args.weight_thresh,
        neg_sampling_attempts=args.neg_attempts,
        seed=args.seed,
    )

    if args.dataset == "all":
        datasets = ["nyc", "tky"]
    else:
        datasets = [args.dataset]

    for ds in datasets:
        run_one_dataset(
            ds,
            project_root=project_root,
            llm_model=args.llm_model,
            cfg=cfg,
            gat_cfg=gat_cfg,
            geohash_precision=args.geohash_precision,
            plot_road_graph=args.plot_road_graph,
            no_cache=args.no_cache,
        )


if __name__ == "__main__":
    main()
