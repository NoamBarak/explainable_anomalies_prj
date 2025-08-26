from Utilities.SubsetContainer import SubsetContainer
from Utilities import Constants as util
import random
from datetime import datetime
from collections import defaultdict
from itertools import combinations



def get_sub_dfs(df, anomaly, top_n=10, num_samples=1000, epsilon=0.1):
    """
    Generate and return the top N sub-DataFrames from the input DataFrame using
    a Multi-Armed Bandit approach.

    Args:
        df (pd.DataFrame): The input DataFrame.
        anomaly (pd.Series): The anomaly instance to compare against.
        top_n (int): The number of top subsets to return.
        num_samples (int): The number of bandit iterations.
        epsilon (float): The exploration rate for the epsilon-greedy strategy.

    Returns:
        list: A list of the top N SubsetContainers with the lowest similarity values.
    """

    rows = list(range(df.shape[0]))  # List of row indices
    cols = list(df.columns)  # List of column names
    top_subsets = []  # List to store the top N subsets
    seen_combinations = set()  # Set to track unique combinations
    arm_rewards = defaultdict(lambda: 0)  # Track rewards for each combination


    cur_row_num = 0

    for i in range(num_samples):

        # Epsilon-Greedy Strategy: Decide whether to explore or exploit
        if random.random() < epsilon:
            # Exploration: Randomly choose the number of rows and columns
            r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)))
            c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)))
        else:
            # Exploitation: Choose the best combination (min reward)
            best_arm = max(arm_rewards, key=arm_rewards.get, default=None)
            if best_arm:
                r, c = best_arm
            else:
                # If no arms have been tried yet, fall back to random sampling
                r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)))
                c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)))

        cur_row_num += 1

        # Iterate over row combinations with the current number of rows `r`
        for row_comb in combinations(rows, r):
            # Now instead of randomly sampling columns, just take `c` columns and pass them into SubsetContainer
            sub_df = df.iloc[list(row_comb)]  # Create sub-DataFrame with current row combination
            subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, sim_features_amount=c)

            euclidean_distance = subset_container.get_euclidian_distance()

            # If we have fewer than top_n subsets, just add the new one
            if len(top_subsets) < top_n:
                top_subsets.append((euclidean_distance, subset_container))
                top_subsets.sort(key=lambda x: x[0])  # Sort by similarity value
            else:
                # If the new subset has a lower similarity value, replace the worst one
                if euclidean_distance < top_subsets[-1][0]:
                    top_subsets[-1] = (euclidean_distance, subset_container)
                    top_subsets.sort(key=lambda x: x[0])  # Re-sort the list

            # Update reward for the arm (number of rows, columns combination)
            arm_rewards[(r, c)] = -euclidean_distance  # Reward is inverse of Euclidean distance

    return [subset for _, subset in top_subsets]