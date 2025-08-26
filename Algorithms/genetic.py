from Utilities.SubsetContainer import SubsetContainer
import pandas as pd
from Utilities import Constants as util
import random
from datetime import datetime

def get_sub_dfs(df, anomaly, top_n=10, num_samples=1000):
    """
    Monte Carlo sampling algorithm for identifying top-N subsets
    most similar to the given anomaly instance.

    Args:
        df (pd.DataFrame): Input dataset.
        anomaly (pd.Series): The anomaly to compare subsets against.
        top_n (int): Number of top subsets to return.
        num_samples (int): Number of random subset samples to evaluate.

    Returns:
        list of SubsetContainer: Top-N subsets with lowest similarity (Euclidean distance).
    """
    print(f"▶️ Running Monte Carlo algorithm at {datetime.now().strftime('%H:%M:%S')}")
    rows = list(range(df.shape[0]))
    cols = list(df.columns)
    top_subsets = []
    seen_combinations = set()

    for i in range(num_samples):
        r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)))
        c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)))

        row_comb = tuple(sorted(random.sample(rows, r)))
        if row_comb in seen_combinations:
            continue
        seen_combinations.add(row_comb)

        sub_df = df.iloc[list(row_comb)]
        subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, sim_features_amount=c)
        distance = subset_container.get_euclidian_distance()

        if len(top_subsets) < top_n:
            top_subsets.append((distance, subset_container))
            top_subsets.sort(key=lambda x: x[0])
        elif distance < top_subsets[-1][0]:
            top_subsets[-1] = (distance, subset_container)
            top_subsets.sort(key=lambda x: x[0])

    print(f"✅ Monte Carlo completed at {datetime.now().strftime('%H:%M:%S')}")
    return [subset for _, subset in top_subsets]