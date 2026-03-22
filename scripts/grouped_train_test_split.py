#!/usr/bin/env python3
"""
Grouped train-test split for TSS sequence datasets using per-class MMseqs2 clusters.

- Clusters are provided as two-column TSVs with NO header:
    col0 = representative (rep)
    col1 = member
- FASTA headers (up to first whitespace) must exactly match the TSV IDs.

This script:
    1) Loads positive and negative FASTA files.
    2) Loads their respective cluster tables and builds member→cluster_id mappings
       (cluster_id is the representative ID).
    3) Treats any FASTA IDs not present in the TSV as singleton clusters (cluster_id = itself).
    4) Performs grouped splitting PER CLASS with GroupShuffleSplit to avoid within-class
       cluster leakage, then merges pos/neg train and test sets.
    5) Writes combined `train.fasta` and `test.fasta` (both classes together).
    6) Prints a concise summary to stdout.

Usage (defaults match repo layout):
    python scripts/grouped_train_test_split.py \
        --pos-fasta data/interim/tss_seq_range.fasta \
        --pos-clusters data/interim/cluster_tss_pos_clusters.tsv \
        --neg-fasta data/interim/tss_neg_seq_range.fasta \
        --neg-clusters data/interim/cluster_tss_neg_clusters.tsv \
        --test-size 0.2 \
        --seed 42 \
        --out-dir data/processed/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import GroupShuffleSplit


def read_fasta_as_dict(fasta_path: Path):
    """
    Read FASTA as:
        - id_to_record: dict[str, SeqRecord]
        - id_order: list[str] preserving file order (deterministic output)

    The FASTA record ID is `record.id` (Biopython: first token of the header).
    """
    id_to_record: Dict[str, SeqIO.SeqRecord] = {}
    id_order: List[str] = []
    with fasta_path.open("r") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            # rec.id is header up to first whitespace, matching the TSV IDs per user confirmation.
            id_to_record[rec.id] = rec
            id_order.append(rec.id)
    return id_to_record, id_order


def read_clusters_tsv(tsv_path: Path) -> pd.DataFrame:
    """
    Read cluster table with schema: rep \t member (no header).
    Returns a DataFrame with columns ['rep', 'member'] (dtype=str), whitespace trimmed.
    """
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["rep", "member"],
        dtype=str,
        comment=None,
    )

    # Drop rows fully NaN (if any trailing newlines)
    df = df.dropna(how="all")
    # Strip potential whitespace artifacts
    for col in ["rep", "member"]:
        df[col] = df[col].astype(str).str.strip()
    return df


def build_member_to_cluster(
    id_to_record: Dict[str, SeqIO.SeqRecord],
    clusters_df: pd.DataFrame,
    label: str,
) -> Tuple[Dict[str, str], int]:
    """
    Build member -> cluster_id mapping (cluster_id is 'rep').
    - Ignores TSV rows whose 'member' is not present in the FASTA (count and report).
    - Ensures any FASTA ID absent in TSV is treated as singleton: cluster_id = itself.

    Returns:
        member_to_cluster: dict mapping FASTA ID -> cluster_id
        ignored_rows: count of TSV rows whose 'member' wasn't found in FASTA
    """
    fasta_ids: Set[str] = set(id_to_record.keys())

    # Filter to members present in FASTA; count ignored
    in_fasta_mask = clusters_df["member"].isin(fasta_ids)
    ignored_rows = int((~in_fasta_mask).sum())
    if ignored_rows > 0:
        # Keep only valid rows
        clusters_df = clusters_df.loc[in_fasta_mask].copy()

    # Build mapping from member -> rep (cluster_id)
    member_to_cluster: Dict[str, str] = {}
    # If duplicates exist, later entries simply overwrite with the same value (benign).
    for rep, member in zip(clusters_df["rep"].values, clusters_df["member"].values):
        member_to_cluster[member] = rep

    # Fill singletons for any FASTA ID not in TSV
    for seq_id in fasta_ids:
        if seq_id not in member_to_cluster:
            member_to_cluster[seq_id] = seq_id  # singleton cluster

    return member_to_cluster, ignored_rows


def grouped_split_ids(
    id_order: List[str],
    member_to_cluster: Dict[str, str],
    test_size: float,
    seed: int,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    Perform grouped split over the IDs in `id_order` using cluster IDs in `member_to_cluster`.
    Returns (train_ids, test_ids, stats) with cluster stats included.
    """
    # Prepare X as positions (stable with id_order), groups as cluster IDs
    X_idx = np.arange(len(id_order))
    groups = np.array([member_to_cluster[i] for i in id_order], dtype=object)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(X_idx, groups=groups))

    train_ids = [id_order[i] for i in train_idx]
    test_ids = [id_order[i] for i in test_idx]

    # Stats
    all_clusters = set(groups.tolist())
    train_clusters = {member_to_cluster[i] for i in train_ids}
    test_clusters = {member_to_cluster[i] for i in test_ids}
    stats = {
        "n_total": len(id_order),
        "n_clusters": len(all_clusters),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "n_train_clusters": len(train_clusters),
        "n_test_clusters": len(test_clusters),
    }
    return train_ids, test_ids, stats


def write_fasta(records: List[SeqIO.SeqRecord], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        SeqIO.write(records, handle, "fasta")


def main():
    parser = argparse.ArgumentParser(
        description="Grouped train-test split for TSS sequences using MMseqs2 clusters (per class)."
    )
    parser.add_argument(
        "--pos-fasta",
        default="data/interim/tss_seq_range.fasta",
        type=str,
        help="Path to positive FASTA file.",
    )
    parser.add_argument(
        "--pos-clusters",
        default="data/interim/cluster_tss_pos_clusters.tsv",
        type=str,
        help="Path to positive cluster TSV (rep<TAB>member, no header).",
    )
    parser.add_argument(
        "--neg-fasta",
        default="data/interim/tss_neg_seq_range.fasta",
        type=str,
        help="Path to negative FASTA file.",
    )
    parser.add_argument(
        "--neg-clusters",
        default="data/interim/cluster_tss_neg_clusters.tsv",
        type=str,
        help="Path to negative cluster TSV (rep<TAB>member, no header).",
    )
    parser.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        help="Fraction of samples for test split (applied per class). Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducibility. Default: 42",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/",
        type=str,
        help="Output directory for train.fasta and test.fasta. Default: data/processed/",
    )

    args = parser.parse_args()

    pos_fasta_path = Path(args.pos_fasta)
    pos_clusters_path = Path(args.pos_clusters)
    neg_fasta_path = Path(args.neg_fasta)
    neg_clusters_path = Path(args.neg_clusters)
    out_dir = Path(args.out_dir)
    test_size = float(args.test_size)
    seed = int(args.seed)

    # --- Positive class ---
    pos_id_to_record, pos_id_order = read_fasta_as_dict(pos_fasta_path)
    pos_clusters_df = read_clusters_tsv(pos_clusters_path)
    pos_member_to_cluster, pos_ignored = build_member_to_cluster(
        pos_id_to_record, pos_clusters_df, label="pos"
    )
    pos_train_ids, pos_test_ids, pos_stats = grouped_split_ids(
        pos_id_order, pos_member_to_cluster, test_size=test_size, seed=seed
    )

    # --- Negative class ---
    neg_id_to_record, neg_id_order = read_fasta_as_dict(neg_fasta_path)
    neg_clusters_df = read_clusters_tsv(neg_clusters_path)
    neg_member_to_cluster, neg_ignored = build_member_to_cluster(
        neg_id_to_record, neg_clusters_df, label="neg"
    )
    neg_train_ids, neg_test_ids, neg_stats = grouped_split_ids(
        neg_id_order, neg_member_to_cluster, test_size=test_size, seed=seed
    )

    # Merge train/test IDs across classes (keep per-file ordering for determinism)
    train_ids = pos_train_ids + neg_train_ids
    test_ids = pos_test_ids + neg_test_ids

    # Build combined records (preserve original sequences verbatim)
    train_records = [pos_id_to_record[i] for i in pos_train_ids] + [
        neg_id_to_record[i] for i in neg_train_ids
    ]
    test_records = [pos_id_to_record[i] for i in pos_test_ids] + [
        neg_id_to_record[i] for i in neg_test_ids
    ]

    # Write outputs
    write_fasta(train_records, out_dir / "train.fasta")
    write_fasta(test_records, out_dir / "test.fasta")

    # Summary
    print("\n=== Grouped Train/Test Split Summary ===")
    print(f"Seed: {seed} | Test size target: {test_size}")
    print(f"Output dir: {out_dir.resolve()}")
    print("\n-- Positive --")
    print(f"Total seqs: {pos_stats['n_total']} | Clusters: {pos_stats['n_clusters']}")
    print(
        f"Train seqs: {pos_stats['n_train']} (clusters: {pos_stats['n_train_clusters']}) | "
        f"Test seqs: {pos_stats['n_test']} (clusters: {pos_stats['n_test_clusters']})"
    )
    if pos_ignored > 0:
        print(f"Note: Ignored {pos_ignored} TSV rows (member not found in FASTA).")

    print("\n-- Negative --")
    print(f"Total seqs: {neg_stats['n_total']} | Clusters: {neg_stats['n_clusters']}")
    print(
        f"Train seqs: {neg_stats['n_train']} (clusters: {neg_stats['n_train_clusters']}) | "
        f"Test seqs: {neg_stats['n_test']} (clusters: {neg_stats['n_test_clusters']})"
    )
    if neg_ignored > 0:
        print(f"Note: Ignored {neg_ignored} TSV rows (member not found in FASTA).")

    print("\n-- Combined --")
    print(f"Train total: {len(train_ids)} | Test total: {len(test_ids)}")
    print("Wrote:")
    print(f"  - {out_dir / 'train.fasta'}")
    print(f"  - {out_dir / 'test.fasta'}")
    print("========================================\n")


if __name__ == "__main__":
    main()