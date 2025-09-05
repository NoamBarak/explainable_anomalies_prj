from Utilities.SubsetContainer import SubsetContainer
from Utilities import Constants as util
import random
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np


def get_sub_dfs(df, anomaly, top_n=10, num_samples=1000, epsilon=0.4):
    """
    FIXED Multi-Armed Bandit sampling algorithm for identifying top-N subsets
    with HIGHEST AFES scores (best explanations).
    """
    samples_counter = 0
    rows = list(range(df.shape[0]))
    cols = list(df.columns)
    top_subsets = []
    seen_combinations = set()

    # ROW-BASED MAB: Each arm represents a specific row's inclusion probability
    row_rewards = defaultdict(list)
    row_inclusion_prob = defaultdict(lambda: 0.5)

    learning_rate = 0.1

    for i in range(num_samples):
        r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)))
        c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)))

        # ROW SELECTION STRATEGY: Use learned probabilities with epsilon-greedy
        if random.random() < epsilon or i < 50:
            selected_rows = random.sample(rows, r)
        else:
            selected_rows = []
            sorted_rows = sorted(rows, key=lambda row: row_inclusion_prob[row], reverse=True)

            for row in sorted_rows:
                if len(selected_rows) >= r:
                    break

                include_prob = row_inclusion_prob[row] * (1 - epsilon) + epsilon * 0.5
                if random.random() < include_prob and len(selected_rows) < r:
                    selected_rows.append(row)

            while len(selected_rows) < r:
                remaining_rows = [row for row in rows if row not in selected_rows]
                if remaining_rows:
                    selected_rows.append(random.choice(remaining_rows))
                else:
                    break

            selected_rows = selected_rows[:r]

        row_comb = tuple(sorted(selected_rows))
        if row_comb in seen_combinations:
            continue
        seen_combinations.add(row_comb)

        # Create subset and evaluate
        sub_df = df.iloc[selected_rows]
        subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, sim_features_amount=c)

        # CRITICAL FIX: Use AFES score directly (higher is better)
        afes_score = subset_container.get_explanation_score()
        samples_counter += 1

        # CRITICAL FIX: Use AFES score as positive reward (higher AFES = higher reward)
        reward = afes_score  # NO NEGATION! Higher AFES = higher reward

        # Update each selected row's reward history
        for row in selected_rows:
            row_rewards[row].append(reward)

            # Update inclusion probability
            if len(row_rewards[row]) > 1:
                avg_reward = np.mean(row_rewards[row][-10:])

                # CRITICAL FIX: Higher AFES scores increase inclusion probability
                # Normalize reward to [0,1] range for probability updates
                normalized_reward = max(0, min(1, (avg_reward + 1) / 2))  # Rough normalization

                # Update probability based on normalized reward
                current_prob = row_inclusion_prob[row]
                target_prob = normalized_reward
                row_inclusion_prob[row] = current_prob + learning_rate * (target_prob - current_prob)

                # Keep probabilities in valid range
                row_inclusion_prob[row] = max(0.05, min(0.95, row_inclusion_prob[row]))

        # Slightly decrease probability for non-selected rows
        if not (random.random() < epsilon):
            for row in rows:
                if row not in selected_rows and len(row_rewards[row]) > 0:
                    row_inclusion_prob[row] *= 0.999

        # CRITICAL FIX: Keep highest AFES scores (not lowest distances)
        if len(top_subsets) < top_n:
            top_subsets.append((afes_score, subset_container))
            top_subsets.sort(key=lambda x: x[0], reverse=True)  # Sort descending
        elif afes_score > top_subsets[-1][0]:  # If better than worst
            top_subsets[-1] = (afes_score, subset_container)
            top_subsets.sort(key=lambda x: x[0], reverse=True)

    print(f"MAB samples evaluated: {samples_counter}")
    if top_subsets:
        print(f"Best AFES score found: {top_subsets[0][0]:.4f}")

    # DEBUG: Print some learned probabilities
    if row_rewards:
        best_rows = sorted(row_inclusion_prob.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"Top 5 learned row probabilities: {[(r, f'{p:.3f}') for r, p in best_rows]}")

    return [subset for _, subset in top_subsets]