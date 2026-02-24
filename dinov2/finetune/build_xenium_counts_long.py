#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def decode_if_bytes(x):
    # Xenium parquet often stores bytes for strings
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8")
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", type=str, required=True, help="NCBIxxx_transcripts.parquet")
    ap.add_argument("--slide-id", type=str, required=True, help="e.g. NCBI884")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--assigned-only", action="store_true",
                    help="Keep only transcripts with a non-null cell_id (recommended)")
    ap.add_argument("--use-overlaps-nucleus", action="store_true",
                    help="Keep only overlaps_nucleus==1 transcripts (optional)")
    args = ap.parse_args()

    in_path = Path(args.in_parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read only needed cols (keeps memory down)
    df = pd.read_parquet(in_path, columns=["cell_id", "feature_name", "overlaps_nucleus"])
    # Decode bytes -> str (vectorized via .map)
    df["cell_id"] = df["cell_id"].map(decode_if_bytes)
    df["feature_name"] = df["feature_name"].map(decode_if_bytes)

    if args.assigned_only:
        df = df[df["cell_id"].notna()]

    if args.use_overlaps_nucleus:
        df = df[df["overlaps_nucleus"] == 1]

    # Aggregate transcript counts per (cell_id, gene)
    # This is the key step; still "long", not wide.
    g = (
        df.groupby(["cell_id", "feature_name"], sort=False)
          .size()
          .rename("count")
          .reset_index()
    )

    # Add slide-aware cell_uid
    g["slide_id"] = args.slide_id
    g["cell_uid"] = g["slide_id"].astype(str) + "_" + g["cell_id"].astype(str)

    # Keep compact columns
    g = g[["cell_uid", "slide_id", "cell_id", "feature_name", "count"]]

    out_path = out_dir / f"{args.slide_id}_cell_gene_counts_long.parquet"
    g.to_parquet(out_path, index=False)
    print("Wrote:", out_path, "rows:", len(g))

if __name__ == "__main__":
    main()
