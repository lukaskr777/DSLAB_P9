import pyarrow.parquet as pq
import pyarrow as pa

DATA_DIR = "data/swiss-ai_apertus-sft-mixture"

input_path = f"{DATA_DIR}/small_train.parquet"
num_parts = 10
output_prefix = f"{DATA_DIR}/small_train_part"

# Open input
pf = pq.ParquetFile(input_path)

# Prepare writers (lazy initialization: only when first batch arrives)
writers: list[pq.ParquetWriter | None] = [None] * num_parts

# For round-robin distribution
current_part = 0

for rg_idx in range(pf.num_row_groups):
    table = pf.read_row_group(rg_idx)  # streaming: one row group at a time
    n = table.num_rows

    # Split this row group into num_parts slices
    # chunk_size = ceil(n / num_parts)
    # but a simple round-robin assignment works well and keeps memory small
    batches = table.to_batches()

    for batch in batches:
        cols = batch.to_pydict()
        size = batch.num_rows

        # Distribute rows one by one into parts
        # (efficient because rows are small; but we convert back to Arrow for writing)
        part_tables = [[] for _ in range(num_parts)]

        for i in range(size):
            row_idx = i
            part_tables[current_part].append({col: cols[col][row_idx] for col in cols})
            current_part = (current_part + 1) % num_parts

        # Write each non-empty table
        for i in range(num_parts):
            if not part_tables[i]:
                continue
            tab = pa.Table.from_pylist(part_tables[i])

            if writers[i] is None:
                writers[i] = pq.ParquetWriter(f"{output_prefix}_{i}.parquet", tab.schema)
            writers[i].write_table(tab)  # type: ignore

# Close all writers
for w in writers:
    if w is not None:
        w.close()
