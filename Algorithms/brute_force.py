from itertools import combinations
from Utilities.SubsetContainer import SubsetContainer
from datetime import datetime
import torch
import Utilities.Constants as constants
from Utilities import Constants as util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sub_dfs(df, anomaly, top_n=constants.SUBSETS_AMOUNT):
    """
        Generate and return the top N sub-DataFrames from the input DataFrame by
        considering all combinations of rows and columns. The top N subsets with
        the lowest similarity method values are kept in memory.
    """

    rows = list(range(df.shape[0]))  # List of row indices
    cols = list(df.columns)  # List of column names
    top_subsets = []  # List to store the top N subsets

    cur_row_num = 0

    # Generate combinations of rows with amount between MIN_ROWS_AMOUNT and MAX_ROWS_AMOUNT
    for r in range(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)) + 1):
        for row_comb in combinations(rows, r):
            # Generate combinations of columns with amount between MIN_COLS_AMOUNT and MAX_COLS_AMOUNT
            for c in range(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)) + 1):
                sub_df = df.iloc[list(row_comb)]  # Create sub-DataFrame
                subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, sim_features_amount=c)

                euclidian_distance = subset_container.get_euclidian_distance()

                # If we have fewer than top_n subsets, just add the new one
                if len(top_subsets) < top_n:
                    top_subsets.append((euclidian_distance, subset_container))
                    top_subsets.sort(key=lambda x: x[0])  # Sort by similarity value
                else:
                    # If the new subset has a lower similarity value, replace the worst one
                    if euclidian_distance < top_subsets[-1][0]:
                        top_subsets[-1] = (euclidian_distance, subset_container)
                        top_subsets.sort(key=lambda x: x[0])  # Re-sort the list

    return [subset for _, subset in top_subsets]

