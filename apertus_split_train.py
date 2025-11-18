"""
Split a large Parquet file (train.parquet) into 100 smaller parquet files, with nearly equal numbers of rows.

Output files will be named:
    train_part_00.parquet
    ...
    train_part_99.parquet

This script is memory-safe and does not load the whole file at once.
"""

from pathlib import Path
import math
import pandas as pd
import pyarrow.parquet as pq

from utility_scripts.file_utils import ensure_outdir


# ---------- CONFIG ----------
DATA_DIR = Path("data/swiss-ai_apertus-sft-mixture")
INPUT_PATH = DATA_DIR / "train.parquet"
OUT_DIR = DATA_DIR / "split_train"

N_PARTS = 100
BASENAME = "train_part"


def main() -> None:
    print(f"Opening dataset: {INPUT_PATH}")
    dataset = pq.ParquetFile(INPUT_PATH)

    ensure_outdir(OUT_DIR)

    total_rows = dataset.metadata.num_rows
    print(f"Total rows: {total_rows}")

    # Compute target rows per part (last part may be slightly smaller)
    rows_per_part = math.ceil(total_rows / N_PARTS)
    print(f"Rows per part ≈ {rows_per_part}")

    # --- Split by streaming row groups ---
    current_rows = 0
    part_id = 0
    batch_frames = []  # temporary list to store DataFrame pieces for the current part

    def write_part(frames, pid):
        """Write one parquet part to disk."""
        if not frames:
            return
        df_part = pd.concat(frames, ignore_index=True)
        out_path = OUT_DIR / f"{BASENAME}_{pid:02d}.parquet"
        df_part.to_parquet(out_path, index=False)
        print(f"Saved part {pid:02d}: shape={df_part.shape} → {out_path}")

    for rg in range(dataset.num_row_groups):
        table = dataset.read_row_group(rg)
        df = table.to_pandas()

        batch_frames.append(df)
        current_rows += len(df)

        # If we reached the desired part size, flush to disk
        if current_rows >= rows_per_part and part_id < N_PARTS - 1:
            write_part(batch_frames, part_id)

            part_id += 1
            current_rows = 0
            batch_frames = []

    # Write the final part (whatever remains)
    write_part(batch_frames, part_id)

    print("Done.")


if __name__ == "__main__":
    main()
