# scripts/combine_long_parquets.py
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

in_dir = Path("/data/datasets/hest/expression_long")
out_path = in_dir / "ALL_cell_gene_counts_long.parquet"

paths = sorted(in_dir.glob("NCBI*_cell_gene_counts_long.parquet"))
assert paths, "No per-slide long parquets found."

writer = None
for p in paths:
    pf = pq.ParquetFile(p)
    for batch in pf.iter_batches(batch_size=2_000_000):  # tune
        table = pa.Table.from_batches([batch])
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)

if writer is not None:
    writer.close()

print("Wrote:", out_path)
