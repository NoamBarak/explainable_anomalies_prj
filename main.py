from Utilities.DataFrameContainer import DataFrameContainer
from Utilities.SubsetContainer import SubsetContainer
from Utilities import Constants as constants
from Algorithms import brute_force, monte_carlo, genetic, multi_armed_bandit as mab
import time, math

import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

best_subsets = {
    "Brute Force": None,
    "Monte Carlo": None,
    "Genetic": None,
    "Multi Arm-Bandit": None
}


def find_oafe(dataframe_container, anomaly, algorithm_results):
    """
    Definition 3 (optimal anomaly feature explanation - OAFE)
    max(D', Fdiff) ω1*g(D, s) + ω2*|D'|
    such that AD'(s) = T ∧ ∀r ∈ R': AD'(r) = F

    This function evaluates and ranks results based on OAFE criteria
    """
    oafe_scores = {}

    for method_name, subset_containers in algorithm_results.items():
        if subset_containers is None:
            continue

        best_container = subset_containers[0]  # Top result from each method

        # Calculate g(D, s) using AFES score
        afes_score = best_container.calc_afes_score(anomaly)

        # Calculate subset size |D'|
        subset_size = len(best_container.subset)

        # OAFE objective: ω1*g(D, s) + ω2*|D'|
        # Using OMEGA_1 and OMEGA_2 as weights
        oafe_score = constants.OMEGA_1 * afes_score + constants.OMEGA_2 * subset_size

        oafe_scores[method_name] = {
            'oafe_score': oafe_score,
            'afes_score': afes_score,
            'subset_size': subset_size,
            'container': best_container
        }

        print(f"{method_name} - OAFE Score: {oafe_score:.4f} (AFES: {afes_score:.4f}, Size: {subset_size})")

    # Find the method with maximum OAFE score
    if oafe_scores:
        best_method = max(oafe_scores.keys(), key=lambda k: oafe_scores[k]['oafe_score'])
        print(f"\nOptimal Anomaly Feature Explanation (OAFE): {best_method}")
        return best_method, oafe_scores

    return None, oafe_scores


def save_anomaly_and_counter_examples(top_10_subsets, output_filename):
    with open(output_filename, 'w') as file:
        for i, subset_container in enumerate(top_10_subsets):
            sim_features = subset_container.sim_features
            diff_features = subset_container.diff_features

            sub_df = subset_container.subset
            distance = subset_container.distance
            afes_score = subset_container.calc_afes_score(anomaly)  # Add AFES score

            output_text = (f"Subset {i + 1}:\n"
                           f"Distance (sE): {distance:.4f}\n"
                           f"AFES Score: {afes_score:.4f}\n"
                           f"Similar Features: {sim_features}\n"
                           f"Different Features: {diff_features}\n"
                           f"Subset:\n{sub_df}\n"
                           + "-" * 50 + "\n")

            file.write(output_text)

def run_tsne(data, anomaly, counter_examples, output_file_name):
    # Reset index for the full data (but not for subsets) to ensure alignment with t-SNE results
    data_reset = data.reset_index(drop=True)

    n_samples = len(data_reset)
    perplexity_value = min(n_samples // 3, 30)

    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    tsne_results = tsne.fit_transform(data_reset)

    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE_1', 'TSNE_2'])

    plt.figure(figsize=(8, 6))

    # Plot the full data in blue
    plt.scatter(df_tsne['TSNE_1'], df_tsne['TSNE_2'], color='blue', label='Data', alpha=0.5)

    # Plot the anomaly subset in red
    plt.scatter(df_tsne.loc[anomaly.index, 'TSNE_1'], df_tsne.loc[anomaly.index, 'TSNE_2'],
                color='red', label='Anomaly', alpha=0.7)

    # Plot the counter_examples subset in green
    plt.scatter(df_tsne.loc[counter_examples.index, 'TSNE_1'], df_tsne.loc[counter_examples.index, 'TSNE_2'],
                color='green', label='Counter Examples', alpha=0.7)

    # Add title and labels
    plt.title(output_file_name)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    # Add legend to differentiate the subsets
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(output_file_name, format='png', dpi=300)

    # Show the plot
    # plt.show()


def analyze_counter_examples(counter_examples, method_name, anomaly):
    print(f"\n=== {method_name} Analysis ===")
    print(f"Similar features: {counter_examples.sim_features}")
    print(f"Different features: {counter_examples.diff_features}")

    # Calculate and display AFES score
    afes_score = counter_examples.calc_afes_score(anomaly)
    sE_score = counter_examples.calc_euclidian_distance_definition1(anomaly, counter_examples.all_features)
    print(f"AFES Score: {afes_score:.4f}")
    print(f"sE Distance (Definition 1): {sE_score:.4f}")

    # print("Counter examples:")
    # for index in counter_examples.subset.index:
    #     original_row = dataframe_container.original_df.loc[index]
    #     print(f"Row ID {index}:\n{original_row}\n")

    sim_features = counter_examples.sim_features
    diff_features = counter_examples.diff_features

    output_folder = "Results/" + method_name

    run_tsne(dataframe_container.full_data, anomaly, counter_examples.subset,
             output_folder + "/all_features.png")
    run_tsne(dataframe_container.full_data[sim_features], anomaly[sim_features],
             counter_examples.subset[sim_features], output_folder + "/sim_features.png")
    run_tsne(dataframe_container.full_data[diff_features], anomaly[diff_features],
             counter_examples.subset[diff_features], output_folder + "/diff_features.png")


# Main execution
dataframe_container = DataFrameContainer()
data = dataframe_container.normal_data
original_data = dataframe_container.original_df
anomaly = dataframe_container.anomalies.iloc[0]
anomaly = anomaly.to_frame().T
cols_names = dataframe_container.cols_names
num_samples = (dataframe_container.rows_amount * dataframe_container.cols_amount) // 3
pd.set_option('display.max_columns', None)
anomaly_vec = anomaly.to_numpy().flatten()

print("=== Anomaly Feature Explanation System ===")
print("Using Mathematical Definitions 1, 2, and 3")

# Start overall timer
overall_start = time.perf_counter()

# ------------------------------------ Brute Force ------------------------------------
# start = time.perf_counter()
# best_subsets["Brute Force"] = brute_force.get_sub_dfs(df=data, anomaly=anomaly, top_n=constants.SUBSETS_AMOUNT)
# brute_force_counter_examples = best_subsets["Brute Force"][0]
# analyze_counter_examples(counter_examples=brute_force_counter_examples, method_name="Brute Force", anomaly=anomaly)
# brute_force_idxs = brute_force_counter_examples.subset.index
# brute_force_original_values_subset = original_data.loc[brute_force_idxs]
# brute_force_original_values_subset.to_csv("Results/Brute Force/Counterexamples", index=False)
# end = time.perf_counter()
# brute_force_time = end - start
# print(f"Brute Force algorithm completed in {brute_force_time:.4f} seconds")

# ------------------------------------ Monte Carlo ------------------------------------
start = time.perf_counter()
best_subsets["Monte Carlo"] = monte_carlo.get_sub_dfs(df=data, anomaly=anomaly, top_n=constants.SUBSETS_AMOUNT,
                                                      num_samples=num_samples)
monte_carlo_counter_examples = best_subsets["Monte Carlo"][0]
analyze_counter_examples(counter_examples=monte_carlo_counter_examples, method_name="Monte Carlo", anomaly=anomaly)
monte_carlo_idxs = monte_carlo_counter_examples.subset.index
monte_carlo_original_values_subset = original_data.loc[monte_carlo_idxs]
monte_carlo_original_values_subset.to_csv("Results/Monte Carlo/Counterexamples", index=False)
end = time.perf_counter()
monte_carlo_time = end - start
print(f"Monte Carlo algorithm completed in {monte_carlo_time:.4f} seconds")

# ------------------------------------ Genetic Algorithm ------------------------------------
start = time.perf_counter()
best_subsets["Genetic"] = genetic.get_sub_dfs(df=data, anomaly=anomaly, top_n=constants.SUBSETS_AMOUNT)
genetic_counter_examples = best_subsets["Genetic"][0]
analyze_counter_examples(counter_examples=genetic_counter_examples, method_name="Genetic", anomaly=anomaly)
genetic_idxs = genetic_counter_examples.subset.index
genetic_original_values_subset = original_data.loc[genetic_idxs]
genetic_original_values_subset.to_csv("Results/Genetic/Counterexamples", index=False)
end = time.perf_counter()
genetic_time = end - start
print(f"Genetic algorithm completed in {genetic_time:.4f} seconds")

# ------------------------------------ Multi-Armed Bandit ------------------------------------
start = time.time()
best_subsets["Multi Arm-Bandit"] = mab.get_sub_dfs(df=data, anomaly=anomaly, top_n=constants.SUBSETS_AMOUNT,
                                                   num_samples=num_samples)
mab_counter_examples = best_subsets["Multi Arm-Bandit"][0]
analyze_counter_examples(counter_examples=mab_counter_examples, method_name="Multi Arm-Bandit", anomaly=anomaly)
mab_idxs = mab_counter_examples.subset.index
mab_original_values_subset = original_data.loc[mab_idxs]
mab_original_values_subset.to_csv("Results/Multi Arm-Bandit/Counterexamples", index=False)
end = time.time()
print(f"Multi Arm-Bandit algorithm completed in {end - start:.4f} seconds")

# ------------------------------------ OAFE Analysis ------------------------------------
print("\n" + "=" * 60)
print("OPTIMAL ANOMALY FEATURE EXPLANATION (OAFE) ANALYSIS")
print("=" * 60)

best_method, oafe_scores = find_oafe(dataframe_container, anomaly, best_subsets)

if best_method:
    optimal_container = oafe_scores[best_method]['container']
    print(f"\nThe optimal explanation is provided by: {best_method}")
    print(f"Final OAFE Score: {oafe_scores[best_method]['oafe_score']:.4f}")

    # Save the optimal result
    save_anomaly_and_counter_examples([optimal_container], "Results/optimal_oafe_explanation.txt")

overall_end = time.perf_counter()
print(f"\nTotal execution time: {overall_end - overall_start:.4f} seconds")