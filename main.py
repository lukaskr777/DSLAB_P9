import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob




if __name__ == "__main__":
    files = glob.glob("Datasets/litellm/public.LiteLLM_SpendLogs/1/*.parquet")  # adjust path
    first_file = files[0]                # pick the first file
    df = pd.read_parquet(first_file)

    print(df.describe())

    df.plot(x="total_tokens", y="total_spend", kind="line")
    print()


