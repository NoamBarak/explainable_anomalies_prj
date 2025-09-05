from Utilities.SubsetContainer import SubsetContainer
from Utilities import Constants as util
import random


def get_sub_dfs(df, anomaly, top_n=10, num_samples=1000):
    """
    FIXED Monte Carlo sampling algorithm for identifying top-N subsets
    with HIGHEST AFES scores (best explanations).
    """
    samples_counter = 0
    rows = list(range(df.shape[0]))
    cols = list(df.columns)
    top_subsets = []
    seen_combinations = set()

    # Calculate theoretical maximum combinations to avoid infinite loops
    max_possible = 1
    for r in range(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)) + 1):
        from math import comb
        max_possible += comb(len(rows), r)

    actual_samples = min(num_samples, max_possible)

    while samples_counter < actual_samples:
        r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)))
        c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)))

        # Keep trying until we find a unique combination
        max_attempts = 1000
        attempts = 0

        while attempts < max_attempts:
            row_comb = tuple(sorted(random.sample(rows, r)))

            if row_comb not in seen_combinations:
                seen_combinations.add(row_comb)
                break

            attempts += 1

        if attempts >= max_attempts:
            break

        sub_df = df.iloc[list(row_comb)]
        subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, sim_features_amount=c)

        # Use AFES score (higher is better)
        afes_score = subset_container.get_explanation_score()
        samples_counter += 1

        # Keep highest AFES scores
        if len(top_subsets) < top_n:
            top_subsets.append((afes_score, subset_container))
            top_subsets.sort(key=lambda x: x[0], reverse=True)  # Sort descending (highest first)
        elif afes_score > top_subsets[-1][0]:  # If better than worst in top_n
            top_subsets[-1] = (afes_score, subset_container)
            top_subsets.sort(key=lambda x: x[0], reverse=True)

    print(f"Actual samples evaluated: {samples_counter}")
    print(f"Best AFES score found: {top_subsets[0][0]:.4f}")

    return [subset for _, subset in top_subsets]
