# scripts/make_wide_topk.py
from pathlib import Path
import pandas as pd

long_path = Path("/data/datasets/hest/expression_long/ALL_cell_gene_counts_long.parquet")
out_path  = Path("/data/datasets/hest/expression_matrices/cellxgene_top100_counts.parquet")
K = 100

# 1) pick top-K genes by total counts
df = pd.read_parquet(long_path, columns=["feature_name", "count"])
top_genes = (
    df.groupby("feature_name")["count"].sum()
      .sort_values(ascending=False)
      .head(K).index.tolist()
)
print("Top genes:", top_genes[:10])

# 2) build wide for only those genes
df = pd.read_parquet(long_path, columns=["cell_uid", "feature_name", "count"])
df = df[df["feature_name"].isin(top_genes)]

wide = df.pivot_table(
    index="cell_uid",
    columns="feature_name",
    values="count",
    aggfunc="sum",
    fill_value=0,
)
wide.columns.name = None

out_path.parent.mkdir(parents=True, exist_ok=True)
wide.to_parquet(out_path)
print("Wrote:", out_path, "shape:", wide.shape)
