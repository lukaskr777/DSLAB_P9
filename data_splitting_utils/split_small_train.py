"""
Split `small_train.parquet` from the apertus-sft-mixture dataset into `num_parts` separate Parquet files 
using a round-robin distribution.
"""

from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "swiss-ai_apertus-sft-mixture"

input_path = DATA_DIR / "small_train.parquet"
num_parts = 10
output_prefix = DATA_DIR / "small_train_part"

# Open input
pf = pq.ParquetFile(input_path)

# Prepare writers (lazy initialization)
writers: list[pq.ParquetWriter | None] = [None] * num_parts

current_part = 0

for rg_idx in range(pf.num_row_groups):
    table = pf.read_row_group(rg_idx)
    batches = table.to_batches()

    for batch in batches:
        cols = batch.to_pydict()
        size = batch.num_rows

        part_tables = [[] for _ in range(num_parts)]

        # Round-robin distribution
        for i in range(size):
            row = {col: cols[col][i] for col in cols}
            part_tables[current_part].append(row)
            current_part = (current_part + 1) % num_parts

        # Write out each part
        for i in range(num_parts):
            if not part_tables[i]:
                continue

            tab = pa.Table.from_pylist(part_tables[i])
            out_path = f"{output_prefix}_{i}.parquet"

            if writers[i] is None:
                writers[i] = pq.ParquetWriter(out_path, tab.schema)
            writers[i].write_table(tab)

# Close writers
for w in writers:
    if w is not None:
        w.close()
