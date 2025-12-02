import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np

DATA_DIR = "data/swiss-ai_apertus-sft-mixture"

input_path = f"{DATA_DIR}/train.parquet"
output_path = f"{DATA_DIR}/small_train.parquet"

# Probability for sampling
p = 0.05

# Open the writer lazily
writer = None

parquet_file = pq.ParquetFile(input_path)

for rg_index in range(parquet_file.num_row_groups):
    # Read one row group (fits in memory)
    table = parquet_file.read_row_group(rg_index)

    # Draw a Bernoulli mask for this row group
    mask = np.random.rand(table.num_rows) < p

    # Filter rows
    sub = table.filter(pa.array(mask))

    # Write only if non-empty
    if sub.num_rows > 0:
        if writer is None:
            writer = pq.ParquetWriter(output_path, sub.schema)
        writer.write_table(sub)

# Close writer
if writer is not None:
    writer.close()
else:
    print("No rows selected (rare with p=0.05)")
