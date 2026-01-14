# Make_Embedding/run_user_neighbors.py
from __future__ import annotations

import torch
import pandas as pd

from make_user_neighbor import (
    build_space_table,
    build_mappings,
    build_user_checkin_vectors,
    build_user_timeslot_probs,
    build_user_geohash_probs,
    build_user_category_probs,
    quick_diag,
    print_view_similarity_reports,
    analyze_hybrid_similarity_stats,
    build_user_neighbors_hybrid,
    check_neighbors,
)


def run_for_dataset(
    dataset_name: str,
    raw_csv_path: str,
    geohash_emb_path: str,
    out_neighbors_path: str,
    w_poi: float,
    w_time: float,
    w_geo: float,
    w_cat: float,
    topk_candidates: int = 64,
    topk_min: int = 6,
    topk_max: int = 12,
    percentile: int = 80,
    shrink_m: float = 100.0,
):
    """
    End-to-end neighbor construction for a single dataset.

    Pipeline (behavior preserved):
      1) Load raw check-in data and keep train split only.
      2) Load space embedding table (geohash -> embedding) and create geohash2idx.
      3) Build ID mappings for users/POIs/categories and geohash indices.
      4) Build user feature views: POI / TIME(96) / GEO / CAT probability distributions.
      5) Compute hybrid similarity stats to set tau (= p50 as in the original code).
      6) Build user neighbors using hybrid similarity + thresholding + percentile + backfilling.
      7) Save neighbors in the same format as the original implementation.
    """
    # (1) Load full data and filter train split
    df = pd.read_csv(raw_csv_path)
    train_df = df[df["SplitTag"] == "train"]

    # (2) Load space embedding table + geohash2idx
    # NOTE: space_table is loaded for consistency with the original pipeline,
    # even if it is not directly used in neighbor construction here.
    space_table, geohash2idx = build_space_table(geohash_emb_path)

    # (3) Build ID mappings
    sizes, mappings, _user_set = build_mappings(train_df, geohash2idx)
    num_pois = sizes["num_pois"] + 1   # +1 for padding index 0
    num_users = sizes["num_users"] + 1
    num_cats = sizes["num_cats"] + 1
    num_geos = len(geohash2idx) + 1

    user_id2idx = mappings["user_id2idx"]
    poi_id2idx = mappings["poi_id2idx"]
    cat_id2idx = mappings["cat_id2idx"]
    poi_idx2geo_idx = mappings["poi_idx2geo_idx"]

    print(f"[{dataset_name.upper()}] Dataset sizes")
    print(f"  num_users={num_users}, num_pois={num_pois}, num_cats={num_cats}, num_geos={num_geos}")
    print("-" * 100)

    # (4) Build user feature views
    #  - POI view: user x POI visit distribution
    #  - TIME view: user x 96 timeslot distribution
    #  - GEO view: user x geohash distribution (via POI -> geohash mapping)
    #  - CAT view: user x category distribution
    checkin_prob, user_activity = build_user_checkin_vectors(
        train_df=train_df,
        user_id2idx=user_id2idx,
        poi_id2idx=poi_id2idx,
        num_users=num_users,
        num_pois=num_pois,
    )

    time_probs_96 = build_user_timeslot_probs(
        train_df=train_df,
        user_id2idx=user_id2idx,
        num_users=num_users,
        time_col="TimeSlot96",
        num_bins=96,
        device="cpu",
    )

    user_geoprob = build_user_geohash_probs(
        train_df=train_df,
        user_id2idx=user_id2idx,
        poi_id2idx=poi_id2idx,
        poi_idx2geo_idx=poi_idx2geo_idx,
        num_users=num_users,
        num_geohashes=num_geos,
        device="cpu",
    )

    user_catprob = build_user_category_probs(
        train_df=train_df,
        user_id2idx=user_id2idx,
        cat_id2idx=cat_id2idx,
        num_users=num_users,
        num_cats=num_cats,
        device="cpu",
    )

    print(f"[{dataset_name.upper()}] User feature tensors")
    print("  - user_activity:", user_activity)
    print("  - checkin_prob:", checkin_prob.shape)
    print("  - time_probs_96:", time_probs_96.shape)
    print("  - user_geoprob:", user_geoprob.shape)
    print("  - user_catprob:", user_catprob.shape)

    # Diagnostics (kept identical in behavior)
    quick_diag("POI", checkin_prob)
    quick_diag("TIME96", time_probs_96)
    quick_diag("GEO", user_geoprob)
    quick_diag("CAT", user_catprob)

    # View-wise similarity report (Top-K candidate pool)
    print_view_similarity_reports(
        checkin_prob, time_probs_96, user_geoprob, user_catprob,
        K0=topk_candidates,
        name_prefix=dataset_name.upper(),
    )

    # (5) Compute hybrid similarity stats and choose tau
    # Original behavior: use p50 as tau.
    hybrid_stats = analyze_hybrid_similarity_stats(
        checkin_prob,
        time_probs_96,
        user_geoprob,
        user_catprob,
        user_activity,
        w_poi=w_poi,
        w_time=w_time,
        w_geo=w_geo,
        w_cat=w_cat,
        shrink_m=shrink_m,
    )

    print(f"[{dataset_name.upper()}] Hybrid Similarity Stats:")
    print(hybrid_stats)

    tau = hybrid_stats["p50"]  # keep original logic unchanged

    # (6) Build neighbors
    pad_idx, pad_w, cold_users = build_user_neighbors_hybrid(
        poi_probs=checkin_prob,
        time_probs=time_probs_96,
        geo_probs=user_geoprob,
        cat_probs=user_catprob,
        user_activity=user_activity,
        topk_candidates=topk_candidates,
        topk_min=topk_min,
        topk_max=topk_max,
        w_poi=w_poi,
        w_time=w_time,
        w_geo=w_geo,
        w_cat=w_cat,
        tau=tau,
        percentile=percentile,
        shrink_m=shrink_m,
    )

    # Neighbor count summary
    check_neighbors(pad_idx)

    # (7) Save neighbors (same schema as original)
    torch.save(
        {
            "pad_idx": pad_idx.cpu(),
            "pad_w": pad_w.cpu(),
            "cold_users": cold_users,
        },
        out_neighbors_path,
    )

    print(f"[{dataset_name.upper()}] Saved neighbors to: {out_neighbors_path}")
    return pad_idx, pad_w, cold_users


def main():
    # NYC
    run_for_dataset(
        dataset_name="nyc",
        raw_csv_path="../data/nyc/raw/NYC.csv",
        geohash_emb_path="../data/nyc/graph/nyc_geohash_gat_space_embedding.csv",
        out_neighbors_path="../data/nyc/graph/nyc_neighbors.pt",
        # NYC weights (kept identical to the original code)
        w_poi=0.7,
        w_time=0.05,
        w_geo=0.1,
        w_cat=0.05,
        topk_candidates=64,
        topk_min=6,
        topk_max=12,
        percentile=80,
        shrink_m=100.0,
    )

    # TKY
    run_for_dataset(
        dataset_name="tky",
        raw_csv_path="../data/tky/raw/TKY.csv",
        geohash_emb_path="../data/tky/graph/tky_geohash_gat_space_embedding.csv",
        out_neighbors_path="../data/tky/graph/tky_neighbors.pt",
        # TKY weights (kept identical to the original code)
        w_poi=0.7,
        w_time=0.1,
        w_geo=0.15,
        w_cat=0.05,
        topk_candidates=64,
        topk_min=6,
        topk_max=12,
        percentile=80,
        shrink_m=100.0,
    )


if __name__ == "__main__":
    main()
