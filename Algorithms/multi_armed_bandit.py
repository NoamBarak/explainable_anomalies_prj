from Utilities.SubsetContainer import SubsetContainer
from Utilities import Constants as util
import random
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np


def get_sub_dfs(df, anomaly, top_n=10, num_samples=1000, epsilon=0.4):
    """
    Improved Multi-Armed Bandit sampling algorithm for identifying top-N subsets
    most similar to the given anomaly instance.

    This version uses row-based arms where each arm represents the inclusion/exclusion
    of specific rows, making the learning more meaningful than size-based arms.

    Args:
        df (pd.DataFrame): Input dataset.
        anomaly (pd.Series): The anomaly to compare subsets against.
        top_n (int): Number of top subsets to return.
        num_samples (int): Number of bandit iterations to evaluate.
        epsilon (float): The exploration rate for the epsilon-greedy strategy.

    Returns:
        list of SubsetContainer: Top-N subsets with lowest similarity (Euclidean distance).
    """
    samples_counter = 0
    rows = list(range(df.shape[0]))
    cols = list(df.columns)
    top_subsets = []
    seen_combinations = set()

    # ROW-BASED MAB: Each arm represents a specific row's inclusion probability
    row_rewards = defaultdict(list)  # Tracks rewards for including each row
    row_inclusion_prob = defaultdict(lambda: 0.5)  # Initial 50% inclusion probability

    # ADAPTIVE LEARNING RATE
    learning_rate = 0.1

    for i in range(num_samples):
        # Generate subset size (keep this random as it's less critical)
        r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)))
        c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)))

        # ROW SELECTION STRATEGY: Use learned probabilities with epsilon-greedy
        if random.random() < epsilon or i < 50:  # Explore more in beginning
            # EXPLORATION: Pure random selection
            selected_rows = random.sample(rows, r)
        else:
            # EXPLOITATION: Use learned row inclusion probabilities
            selected_rows = []

            # Sort rows by inclusion probability (descending)
            sorted_rows = sorted(rows, key=lambda row: row_inclusion_prob[row], reverse=True)

            # Probabilistic selection based on learned preferences
            for row in sorted_rows:
                if len(selected_rows) >= r:
                    break

                # Include row based on its learned probability (with some randomness)
                include_prob = row_inclusion_prob[row] * (1 - epsilon) + epsilon * 0.5
                if random.random() < include_prob and len(selected_rows) < r:
                    selected_rows.append(row)

            # Fill remaining slots randomly if needed
            while len(selected_rows) < r:
                remaining_rows = [row for row in rows if row not in selected_rows]
                if remaining_rows:
                    selected_rows.append(random.choice(remaining_rows))
                else:
                    break

            # Ensure we have exactly r rows
            selected_rows = selected_rows[:r]

        # Check for duplicates
        row_comb = tuple(sorted(selected_rows))
        if row_comb in seen_combinations:
            continue
        seen_combinations.add(row_comb)

        # Create subset and evaluate
        sub_df = df.iloc[selected_rows]
        subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, sim_features_amount=c)
        distance = subset_container.get_euclidian_distance()
        samples_counter += 1

        # UPDATE ROW-BASED LEARNING
        reward = -distance  # Lower distance = higher reward

        # Update each selected row's reward history
        for row in selected_rows:
            row_rewards[row].append(reward)

            # Update inclusion probability using exponential moving average
            if len(row_rewards[row]) > 1:
                avg_reward = np.mean(row_rewards[row][-10:])  # Use recent history

                # Positive reward increases inclusion probability
                if avg_reward > 0:
                    row_inclusion_prob[row] = min(0.95,
                                                  row_inclusion_prob[row] + learning_rate * abs(avg_reward))
                else:
                    row_inclusion_prob[row] = max(0.05,
                                                  row_inclusion_prob[row] - learning_rate * abs(avg_reward))

        # Also slightly decrease probability for non-selected rows (exploration)
        if not (random.random() < epsilon):  # Only during exploitation
            for row in rows:
                if row not in selected_rows and len(row_rewards[row]) > 0:
                    row_inclusion_prob[row] *= 0.999  # Very small decay

        # Manage top subsets (same as Monte Carlo)
        if len(top_subsets) < top_n:
            top_subsets.append((distance, subset_container))
            top_subsets.sort(key=lambda x: x[0])
        elif distance < top_subsets[-1][0]:
            top_subsets[-1] = (distance, subset_container)
            top_subsets.sort(key=lambda x: x[0])

    print(samples_counter)

    # DEBUG: Print some learned probabilities
    if row_rewards:
        best_rows = sorted(row_inclusion_prob.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5 learned row probabilities: {[(r, f'{p:.3f}') for r, p in best_rows]}")

    return [subset for _, subset in top_subsets]