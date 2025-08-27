from Utilities.DataFrameContainer import DataFrameContainer
from Utilities.SubsetContainer import SubsetContainer
from Utilities import Constants as constants
from Algorithms import brute_force, monte_carlo, genetic, multi_armed_bandit as mab
import time, math


import pandas as pd
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.get_device_name(torch.cuda.current_device()))

best_subsets = {
    "Brute Force": None,
    "Monte Carlo": None,
    "Genetic": None,
    "Multi Arm-Bandit": None
}

def save_anomaly_and_counter_examples2(top_10_subsets, output_filename):
    with open(output_filename, 'w') as file:
        for i, subset_container in enumerate(top_10_subsets):
            sim_features = subset_container.sim_features
            diff_features = subset_container.diff_features

            sub_df = subset_container.subset
            distance = subset_container.distance

            output_text = (f"Subset {i+1}:\n"
                           f"Distance:{distance}\n"
                           f"Similar Features:{sim_features}\n"
                           f"Different Features:{diff_features}\n"
                           f"Subset:{sub_df}\n"
                           + "-" * 50 + "\n")

            file.write(output_text)


def save_anomaly_and_counter_examples(anomaly, counter_examples, dataframe_container,
                                      txt_output_path='Results/output.txt'):
    """Saves anomaly and counter examples rows to a text file."""

    with open(txt_output_path, 'w') as file:
        # Save the anomaly first
        file.write("Anomaly:\n")
        file.write("-" * 50 + "\n")

        for i, index in enumerate(anomaly.index):
            anomaly_row = dataframe_container.original_df.loc[index]
            output_text = (f"Anomaly {i + 1} (Row ID {index}):\n"
                           f"{anomaly_row}\n"
                           + "-" * 50 + "\n")
            file.write(output_text)

        # Save the counter examples
        file.write("Counter Examples:\n")
        file.write("-" * 50 + "\n")

        for i, index in enumerate(counter_examples.index):
            counter_example_row = dataframe_container.original_df.loc[index]
            output_text = (f"Counter Example {i + 1} (Row ID {index}):\n"
                           f"{counter_example_row}\n"
                           + "-" * 50 + "\n")
            file.write(output_text)

    print(f"Anomalies and counter examples saved to {txt_output_path}")


def run_tsne(data, anomaly, counter_examples, output_file_name):
    # Reset index for the full data (but not for subsets) to ensure alignment with t-SNE results
    data_reset = data.reset_index(drop=True)

    n_samples = len(data_reset)
    perplexity_value = min(n_samples // 3, 30)

    # TODO: what is n_components?
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


def run_tsne_new(data, anomaly, counter_examples):
        """
        Perform t-SNE on the data and visualize it with highlighted points.

        Parameters:
        - data: np.ndarray or pd.DataFrame
            All the data points to be reduced with t-SNE.
        - anomaly: np.ndarray or pd.Series
            A single point to be highlighted as an anomaly (red).
        - counter_examples: np.ndarray or pd.DataFrame
            Subset of data points to be highlighted as counter-examples (green).
        """
        # Convert input to numpy arrays if necessary
        data = np.asarray(data)
        anomaly = np.asarray(anomaly).reshape(1, -1)  # Ensure anomaly is a 2D array
        counter_examples = np.asarray(counter_examples)

        # Stack all data together (t-SNE works on the whole set)
        all_data = np.vstack([data, anomaly, counter_examples])

        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings = tsne.fit_transform(all_data)

        # Extract embeddings for each category
        data_embeddings = embeddings[:len(data)]
        anomaly_embedding = embeddings[len(data):len(data) + 1]
        counter_example_embeddings = embeddings[len(data) + 1:]

        # Plot the t-SNE results
        plt.figure(figsize=(10, 8))
        plt.scatter(data_embeddings[:, 0], data_embeddings[:, 1], c='blue', label='Data', alpha=0.6)
        plt.scatter(counter_example_embeddings[:, 0], counter_example_embeddings[:, 1], c='green',
                    label='Counter Examples', alpha=0.8)
        plt.scatter(anomaly_embedding[:, 0], anomaly_embedding[:, 1], c='red', label='Anomaly', edgecolor='black',
                    s=100, marker='x')

        # Add labels and legend
        plt.title("t-SNE Visualization!!!!!!!!!!!!!!!!!!!!!!!!")
        plt.legend()
        plt.show()

def analyze_counter_examples(counter_examples, method_name):
    print(method_name)
    print(f"Similar features:{counter_examples.sim_features}\n")
    print("Counter examples")
    for index in counter_examples.subset.index:
        original_row = dataframe_container.original_df.loc[index]
        print(f"Row ID {index}:\n{original_row}\n")

    brute_force_sim_features = counter_examples.sim_features
    brute_force_diff_features = counter_examples.diff_features

    output_folder = "Results/" + method_name

    run_tsne(dataframe_container.full_data, anomaly, counter_examples.subset,
             output_folder + "/all_features.png")
    run_tsne(dataframe_container.full_data[brute_force_sim_features], anomaly[brute_force_sim_features],
             counter_examples.subset[brute_force_sim_features], output_folder + "/sim_features.png")
    run_tsne(dataframe_container.full_data[brute_force_diff_features], anomaly[brute_force_diff_features],
             counter_examples.subset[brute_force_diff_features], output_folder + "/diff_features.png")



dataframe_container = DataFrameContainer()
data = dataframe_container.normal_data
original_data = dataframe_container.original_df
anomaly = dataframe_container.anomalies.iloc[0]
anomaly = anomaly.to_frame().T
cols_names = dataframe_container.cols_names
num_samples = (dataframe_container.rows_amount * dataframe_container.cols_amount) // 3
pd.set_option('display.max_columns', None)      # Ensure all columns are displayed
anomaly_vec = anomaly.to_numpy().flatten()  # Ensure anomaly is a 1D array

# Start overall timer
overall_start = time.perf_counter()
# ------------------------------------ Brute Force ------------------------------------
# start = time.perf_counter()
# best_subsets["Brute Force"] = brute_force.get_sub_dfs(df=data, anomaly=anomaly, top_n=constants.SUBSETS_AMOUNT)
# brute_force_counter_examples = best_subsets["Brute Force"][0]
# analyze_counter_examples(counter_examples=brute_force_counter_examples, method_name="Brute Force")
# brute_force_idxs = brute_force_counter_examples.subset.index
# brute_force_original_values_subset = original_data.loc[brute_force_idxs]
# brute_force_original_values_subset.to_csv("Results/Brute Force/Counterexamples", index=False)
# end = time.perf_counter()
# brute_force_time = end - start
# print(f"Brute Force algorithm completed in {brute_force_time:.4f} seconds")
#
# # Ensure all x values are also 1D arrays
# distances = [math.dist(anomaly_vec, x.to_numpy()) for _, x in brute_force_original_values_subset.iterrows()]
# avg_distance = sum(distances) / len(distances)
# print("Average Euclidean distance:", avg_distance)

# ------------------------------------ Monte Carlo ------------------------------------
start = time.perf_counter()
best_subsets["Monte Carlo"] = monte_carlo.get_sub_dfs(df=data, anomaly=anomaly, top_n=constants.SUBSETS_AMOUNT, num_samples=num_samples)
monte_carlo_counter_examples = best_subsets["Monte Carlo"][0]
analyze_counter_examples(counter_examples=monte_carlo_counter_examples, method_name="Monte Carlo")
monte_carlo_idxs = monte_carlo_counter_examples.subset.index
monte_carlo_original_values_subset = original_data.loc[monte_carlo_idxs]
monte_carlo_original_values_subset.to_csv("Results/Brute Force/Counterexamples", index=False)

end = time.perf_counter()
monte_carlo_time = end - start
print(f"Monte Carlo algorithm completed in {monte_carlo_time:.4f} seconds")

# Ensure all x values are also 1D arrays
distances = [math.dist(anomaly_vec, x.to_numpy()) for _, x in monte_carlo_original_values_subset.iterrows()]
avg_distance = sum(distances) / len(distances)
print("Average Euclidean distance:", avg_distance)

# ------------------------------------ Genetic Algorithm ------------------------------------
start = time.perf_counter()
best_subsets["Genetic"] = genetic.get_sub_dfs(df=data, anomaly=anomaly, top_n=constants.SUBSETS_AMOUNT)
genetic_counter_examples = best_subsets["Genetic"][0]
# counter_examples = pd.concat([container.subset for container in best_subsets["Genetic"]], ignore_index=True)
# genetic_subset_container = SubsetContainer(subset=counter_examples, anomaly=anomaly, sim_features_amount=constants.MAX_COLS_AMOUNT)
analyze_counter_examples(counter_examples=genetic_counter_examples, method_name="Genetic")
genetic_idxs = genetic_counter_examples.subset.index
genetic_original_values_subset = original_data.loc[genetic_idxs]
genetic_original_values_subset.to_csv("Results/Genetic/Counterexamples", index=False)

end = time.perf_counter()
genetic_time = end - start
print(f"Genetic algorithm completed in {genetic_time:.4f} seconds")
# Ensure all x values are also 1D arrays
distances = [math.dist(anomaly_vec, x.to_numpy()) for _, x in genetic_original_values_subset.iterrows()]
avg_distance = sum(distances) / len(distances)
print("Average Euclidean distance:", avg_distance)

# ------------------------------------ Multi-Armed Bandit ------------------------------------
start = time.time()
best_subsets["Multi Arm-Bandit"] = mab.get_sub_dfs(df=data, anomaly=anomaly, top_n=constants.SUBSETS_AMOUNT, num_samples=num_samples)
mab_counter_examples = best_subsets["Multi Arm-Bandit"][0]
analyze_counter_examples(counter_examples=mab_counter_examples, method_name="Multi Arm-Bandit")
end = time.time()
mab_idxs = mab_counter_examples.subset.index
mab_original_values_subset = original_data.loc[mab_idxs]
mab_original_values_subset.to_csv("Results/Multi Arm-Bandit/Counterexamples", index=False)
print("Execution time for Multi Arm-Bandit: ", end - start, "seconds")

# Ensure all x values are also 1D arrays
distances = [math.dist(anomaly_vec, x.to_numpy()) for _, x in mab_original_values_subset.iterrows()]
avg_distance = sum(distances) / len(distances)
print("Average Euclidean distance:", avg_distance)
