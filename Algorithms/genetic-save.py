import random
import torch
from itertools import combinations
from Utilities.SubsetContainer import SubsetContainer
import Utilities.Constants as constants
from Utilities import Constants as util

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_population(df, anomaly, population_size):
    population = []
    rows = list(range(df.shape[0]))
    cols = list(df.columns)

    for _ in range(population_size):
        row_count = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, len(rows)))
        col_count = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, len(cols)))
        selected_rows = random.sample(rows, row_count)
        sub_df = df.iloc[list(selected_rows)]
        subset_container = SubsetContainer(subset=sub_df, anomaly=anomaly, sim_features_amount=col_count)
        fitness = subset_container.get_euclidian_distance()
        population.append((fitness, subset_container))

    return sorted(population, key=lambda x: x[0])

def mutate(subset_container, df, anomaly):
    rows = list(range(df.shape[0]))
    cols = list(df.columns)
    subset_df = subset_container.subset.copy()
    sim_features_amount = len(subset_container.sim_features)

    # Ensure all columns are present
    missing_cols = set(cols) - set(subset_df.columns)
    for col in missing_cols:
        subset_df[col] = df[col]

    # Mutate rows by either adding or removing a row
    if random.random() < 0.5 and len(subset_df) > 1:
        new_row = random.choice(rows)
        if new_row not in subset_df.index:
            subset_df.loc[new_row] = df.loc[new_row]
        else:
            subset_df = subset_df.drop(index=random.choice(subset_df.index))

    mutated_container = SubsetContainer(subset=subset_df, anomaly=anomaly, sim_features_amount=sim_features_amount)
    fitness = mutated_container.get_euclidian_distance()
    return (fitness, mutated_container)

def crossover(parent1, parent2, df, anomaly):
    subset1 = parent1.subset.copy()
    subset2 = parent2.subset.copy()
    common_rows = list(set(subset1.index) & set(subset2.index))
    common_cols = list(set(subset1.columns) & set(subset2.columns))

    if not common_cols:
        return mutate(parent1, df, anomaly)

    child_df = df.loc[common_rows, common_cols] if common_rows else df[common_cols].sample(n=1)
    child_container = SubsetContainer(subset=child_df, anomaly=anomaly, sim_features_amount=len(parent1.sim_features))
    fitness = child_container.get_euclidian_distance()
    return (fitness, child_container)

def get_sub_dfs(df, anomaly, top_n=constants.SUBSETS_AMOUNT, population_size=50, generations=100):
    population = initialize_population(df, anomaly, population_size)

    for _ in range(generations):
        new_population = []
        population = sorted(population, key=lambda x: x[0])[:population_size // 2]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1[1], parent2[1], df, anomaly)
            new_population.append(child)

            if random.random() < 0.2:
                mutant = mutate(parent1[1], df, anomaly)
                new_population.append(mutant)

        population = list(set(sorted(new_population, key=lambda x: x[0])[:population_size]))

    return [subset for _, subset in population[:top_n]]
