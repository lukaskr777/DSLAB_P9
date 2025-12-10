"""
Create a smaller Parquet subset of the apertus-sft-mixture training data.

This script samples each row independently with probability `p` and writes the selected rows into a new Parquet file.
"""

from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "swiss-ai_apertus-sft-mixture"

input_path = DATA_DIR / "train.parquet"
output_path = DATA_DIR / "small_train.parquet"

# Probability for sampling
p = 0.05

writer = None
parquet_file = pq.ParquetFile(input_path)

for rg_index in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(rg_index)

    mask = np.random.rand(table.num_rows) < p
    sub = table.filter(pa.array(mask))

    if sub.num_rows > 0:
        if writer is None:
            writer = pq.ParquetWriter(output_path, sub.schema)
        writer.write_table(sub)

if writer is not None:
    writer.close()
else:
    print("No rows selected (rare with p=0.05)")
