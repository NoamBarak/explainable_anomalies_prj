from Utilities.SubsetContainer import SubsetContainer
import pandas as pd
from Utilities import Constants as util
import random
from datetime import datetime
import numpy as np


class Chromosome:
    """Represents a solution as a chromosome with row and column selections."""

    def __init__(self, rows, cols, total_rows, total_cols):
        self.rows = set(rows)  # Set of selected row indices
        self.cols = cols  # Number of columns to use
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.fitness = None
        self.distance = None

    def to_subset_container(self, df, anomaly):
        """Convert chromosome to SubsetContainer for evaluation."""
        row_list = sorted(list(self.rows))
        sub_df = df.iloc[row_list]
        return SubsetContainer(subset=sub_df, anomaly=anomaly, sim_features_amount=self.cols)


def create_random_chromosome(total_rows, total_cols):
    """Create a random valid chromosome."""
    # Random number of rows and columns within constraints
    r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, total_rows))
    c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, total_cols))

    # Random selection of rows
    rows = random.sample(range(total_rows), r)

    return Chromosome(rows, c, total_rows, total_cols)


def evaluate_chromosome(chromosome, df, anomaly):
    """Evaluate chromosome fitness (lower distance = higher fitness)."""
    if chromosome.fitness is not None:
        return chromosome.fitness

    container = chromosome.to_subset_container(df, anomaly)
    distance = container.get_euclidian_distance()

    chromosome.distance = distance
    chromosome.fitness = -distance  # Negative distance for maximization

    return chromosome.fitness


def tournament_selection(population, tournament_size=3):
    """Select parent using tournament selection."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.fitness)


def crossover(parent1, parent2):
    """Specialized crossover for subset selection."""
    # Combine row sets and randomly split
    all_rows = list(parent1.rows | parent2.rows)

    if len(all_rows) < util.MIN_ROWS_AMOUNT:
        # If not enough unique rows, pad with random rows
        available_rows = set(range(parent1.total_rows)) - set(all_rows)
        if available_rows:
            additional_needed = util.MIN_ROWS_AMOUNT - len(all_rows)
            additional_rows = random.sample(list(available_rows),
                                            min(additional_needed, len(available_rows)))
            all_rows.extend(additional_rows)

    # Determine child sizes
    max_rows = min(len(all_rows), util.MAX_ROWS_AMOUNT)
    child1_rows = random.randint(util.MIN_ROWS_AMOUNT, max_rows)
    child2_rows = random.randint(util.MIN_ROWS_AMOUNT, max_rows)

    # Split rows between children
    random.shuffle(all_rows)
    child1_row_set = set(all_rows[:child1_rows])
    child2_row_set = set(all_rows[-child2_rows:])

    # Average column counts with some randomness
    avg_cols = (parent1.cols + parent2.cols) // 2
    child1_cols = max(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT,
                                                avg_cols + random.randint(-1, 1)))
    child2_cols = max(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT,
                                                avg_cols + random.randint(-1, 1)))

    child1 = Chromosome(child1_row_set, child1_cols, parent1.total_rows, parent1.total_cols)
    child2 = Chromosome(child2_row_set, child2_cols, parent1.total_rows, parent1.total_cols)

    return child1, child2


def mutate(chromosome, mutation_rate=0.1):
    """Mutate chromosome by changing row selection or column count."""
    if random.random() < mutation_rate:
        if random.random() < 0.7:  # 70% chance to mutate rows
            # Row mutation: add/remove/replace a row
            available_rows = set(range(chromosome.total_rows)) - chromosome.rows

            if len(chromosome.rows) > util.MIN_ROWS_AMOUNT and available_rows:
                if random.random() < 0.5:  # Replace a row
                    old_row = random.choice(list(chromosome.rows))
                    new_row = random.choice(list(available_rows))
                    chromosome.rows.remove(old_row)
                    chromosome.rows.add(new_row)
                elif len(chromosome.rows) < util.MAX_ROWS_AMOUNT:  # Add a row
                    new_row = random.choice(list(available_rows))
                    chromosome.rows.add(new_row)
                else:  # Remove a row
                    old_row = random.choice(list(chromosome.rows))
                    chromosome.rows.remove(old_row)
        else:  # 30% chance to mutate column count
            chromosome.cols = max(util.MIN_COLS_AMOUNT,
                                  min(util.MAX_COLS_AMOUNT,
                                      chromosome.cols + random.randint(-1, 1)))

    # Reset fitness after mutation
    chromosome.fitness = None
    chromosome.distance = None

    return chromosome


def get_sub_dfs(df, anomaly, top_n=10, num_samples=1000):
    """
    Genetic Algorithm for identifying top-N subsets most similar to the given anomaly instance.

    Args:
        df (pd.DataFrame): Input dataset.
        anomaly (pd.Series): The anomaly to compare subsets against.
        top_n (int): Number of top subsets to return.
        num_samples (int): Used to determine population size and generations.

    Returns:
        list of SubsetContainer: Top-N subsets with lowest similarity (Euclidean distance).
    """
    print(f"▶️ Running Genetic Algorithm at {datetime.now().strftime('%H:%M:%S')}")

    # GA Parameters
    population_size = max(50, num_samples // 20)  # Scale with num_samples
    generations = max(25, num_samples // 40)  # Scale with num_samples
    mutation_rate = 0.15
    elitism_size = max(5, population_size // 10)

    total_rows = df.shape[0]
    total_cols = len(df.columns)

    # Initialize population
    population = []
    seen_combinations = set()

    for _ in range(population_size):
        chromosome = create_random_chromosome(total_rows, total_cols)
        row_comb = tuple(sorted(chromosome.rows))

        # Ensure diversity in initial population
        attempts = 0
        while row_comb in seen_combinations and attempts < 10:
            chromosome = create_random_chromosome(total_rows, total_cols)
            row_comb = tuple(sorted(chromosome.rows))
            attempts += 1

        seen_combinations.add(row_comb)
        evaluate_chromosome(chromosome, df, anomaly)
        population.append(chromosome)

    # Evolution loop
    for generation in range(generations):
        # Sort population by fitness (higher is better)
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Create next generation
        new_population = []

        # Elitism: Keep best individuals
        new_population.extend(population[:elitism_size])

        # Generate offspring
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)

            child1, child2 = crossover(parent1, parent2)

            # Mutation
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            # Evaluate children
            evaluate_chromosome(child1, df, anomaly)
            evaluate_chromosome(child2, df, anomaly)

            new_population.extend([child1, child2])

        # Trim to exact population size
        population = new_population[:population_size]

    # Final evaluation and sorting
    population.sort(key=lambda x: x.fitness, reverse=True)

    # Convert top chromosomes to SubsetContainer format
    top_subsets = []
    for i in range(min(top_n, len(population))):
        chromosome = population[i]
        container = chromosome.to_subset_container(df, anomaly)
        # Re-evaluate to ensure we have the container with distance
        distance = container.get_euclidian_distance()
        top_subsets.append((distance, container))

    # Sort by distance (lower is better)
    top_subsets.sort(key=lambda x: x[0])

    print(f"✅ Genetic Algorithm completed at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Best fitness: {population[0].fitness:.4f}, Best distance: {-population[0].fitness:.4f}")

    return [subset for _, subset in top_subsets]