from Utilities.SubsetContainer import SubsetContainer
from Utilities import Constants as util
import random
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
    r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, total_rows))
    c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, total_cols))
    rows = random.sample(range(total_rows), r)
    return Chromosome(rows, c, total_rows, total_cols)


def create_distance_based_chromosome(total_rows, total_cols, df, anomaly, seed):
    """Create chromosome with points at specific distance ranges from anomaly."""
    r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, total_rows))
    c = random.randint(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, total_cols))

    # Calculate distances to all points
    anomaly_vec = anomaly.values.flatten()
    distances = []
    for idx in range(total_rows):
        row_vec = df.iloc[idx].values
        dist = np.linalg.norm(row_vec - anomaly_vec)
        distances.append((dist, idx))

    distances.sort()

    # Select from different distance ranges based on seed
    ranges = len(distances) // 3
    if seed % 3 == 0:  # Close points
        candidates = [idx for _, idx in distances[:ranges]]
    elif seed % 3 == 1:  # Medium distance points
        candidates = [idx for _, idx in distances[ranges:2 * ranges]]
    else:  # Far points
        candidates = [idx for _, idx in distances[2 * ranges:]]

    rows = random.sample(candidates, min(r, len(candidates)))
    if len(rows) < r:  # Fill remaining with random
        remaining = set(range(total_rows)) - set(rows)
        rows.extend(random.sample(list(remaining), r - len(rows)))

    return Chromosome(rows, c, total_rows, total_cols)


def create_feature_spread_chromosome(total_rows, total_cols, seed):
    """Create chromosome with diverse feature combinations."""
    r = random.randint(util.MIN_ROWS_AMOUNT, min(util.MAX_ROWS_AMOUNT, total_rows))
    # Vary column selection based on seed
    c_options = list(range(util.MIN_COLS_AMOUNT, min(util.MAX_COLS_AMOUNT, total_cols) + 1))
    c = c_options[seed % len(c_options)]
    rows = random.sample(range(total_rows), r)
    return Chromosome(rows, c, total_rows, total_cols)


def create_diverse_population(total_rows, total_cols, population_size, df, anomaly):
    """Create initial population with diverse strategies."""
    population = []
    seen_combinations = set()
    strategies = ['random', 'distance_based', 'feature_spread']

    for i in range(population_size):
        strategy = strategies[i % len(strategies)]

        if strategy == 'distance_based':
            chromosome = create_distance_based_chromosome(total_rows, total_cols, df, anomaly, i)
        elif strategy == 'feature_spread':
            chromosome = create_feature_spread_chromosome(total_rows, total_cols, i)
        else:
            chromosome = create_random_chromosome(total_rows, total_cols)

        # Ensure uniqueness with more attempts
        row_comb = tuple(sorted(chromosome.rows))
        attempts = 0
        while row_comb in seen_combinations and attempts < 50:
            chromosome = create_random_chromosome(total_rows, total_cols)
            row_comb = tuple(sorted(chromosome.rows))
            attempts += 1

        seen_combinations.add(row_comb)
        population.append(chromosome)

    return population


def evaluate_chromosome(chromosome, df, anomaly):
    """FIXED: Evaluate chromosome fitness (higher AFES = higher fitness)."""
    if chromosome.fitness is not None:
        return chromosome.fitness

    container = chromosome.to_subset_container(df, anomaly)

    afes_score = container.get_explanation_score()

    chromosome.distance = afes_score  # Store AFES score
    chromosome.fitness = afes_score  # Higher AFES = higher fitness
    return chromosome.fitness


def diversity_tournament_selection(population, tournament_size=5):
    """Tournament selection with diversity consideration."""
    tournament = random.sample(population, min(tournament_size, len(population)))

    # 70% fitness, 30% diversity consideration
    if random.random() < 0.7:
        return max(tournament, key=lambda x: x.fitness)
    else:
        # Select based on uniqueness of row combination
        unique_combinations = {}
        for chrom in tournament:
            key = tuple(sorted(chrom.rows))
            if key not in unique_combinations:
                unique_combinations[key] = []
            unique_combinations[key].append(chrom)

        # Prefer less common combinations
        rarest_key = min(unique_combinations.keys(), key=lambda k: len(unique_combinations[k]))
        return max(unique_combinations[rarest_key], key=lambda x: x.fitness)


def tournament_selection(population, tournament_size=3):
    """Standard tournament selection (kept for compatibility)."""
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.fitness)


def local_search(chromosome, df, anomaly, max_iterations=3):
    """Simple local search to improve chromosome."""
    current_fitness = evaluate_chromosome(chromosome, df, anomaly)

    for _ in range(max_iterations):
        # Try small modifications
        backup_rows = chromosome.rows.copy()
        backup_cols = chromosome.cols

        # Try swapping one row
        if len(chromosome.rows) > 0:
            old_row = random.choice(list(chromosome.rows))
            available_rows = set(range(chromosome.total_rows)) - chromosome.rows
            if available_rows:
                new_row = random.choice(list(available_rows))
                chromosome.rows.remove(old_row)
                chromosome.rows.add(new_row)

                chromosome.fitness = None
                new_fitness = evaluate_chromosome(chromosome, df, anomaly)

                if new_fitness <= current_fitness:  # Revert if worse
                    chromosome.rows = backup_rows
                    chromosome.cols = backup_cols
                    chromosome.fitness = current_fitness
                else:
                    current_fitness = new_fitness

    return chromosome


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


def enhanced_crossover(parent1, parent2, df, anomaly):
    """Crossover with local optimization."""
    child1, child2 = crossover(parent1, parent2)

    # Apply local search to children
    child1 = local_search(child1, df, anomaly)
    child2 = local_search(child2, df, anomaly)

    return child1, child2


def adaptive_mutate(chromosome, generation, max_generations, base_rate=0.15):
    """Mutation rate that decreases over generations."""
    # Start high, decrease over time
    adaptive_rate = base_rate * (1 - generation / max_generations) + 0.05

    if random.random() < adaptive_rate:
        # Enhanced mutation with multiple changes
        num_mutations = random.randint(1, 2)  # 1-2 mutations per event

        for _ in range(num_mutations):
            if random.random() < 0.7:  # Row mutation
                available_rows = set(range(chromosome.total_rows)) - chromosome.rows

                if len(chromosome.rows) > util.MIN_ROWS_AMOUNT and available_rows:
                    mutation_type = random.choice(['replace', 'add', 'remove'])

                    if mutation_type == 'replace' or (
                            mutation_type == 'add' and len(chromosome.rows) >= util.MAX_ROWS_AMOUNT):
                        old_row = random.choice(list(chromosome.rows))
                        new_row = random.choice(list(available_rows))
                        chromosome.rows.remove(old_row)
                        chromosome.rows.add(new_row)
                    elif mutation_type == 'add' and len(chromosome.rows) < util.MAX_ROWS_AMOUNT:
                        new_row = random.choice(list(available_rows))
                        chromosome.rows.add(new_row)
                    elif mutation_type == 'remove':
                        old_row = random.choice(list(chromosome.rows))
                        chromosome.rows.remove(old_row)
            else:  # Column mutation
                chromosome.cols = max(util.MIN_COLS_AMOUNT,
                                      min(util.MAX_COLS_AMOUNT,
                                          chromosome.cols + random.randint(-1, 1)))

    chromosome.fitness = None
    chromosome.distance = None
    return chromosome


def mutate(chromosome, mutation_rate=0.1):
    """Standard mutate function (kept for compatibility)."""
    return adaptive_mutate(chromosome, 0, 1, mutation_rate)


def get_sub_dfs(df, anomaly, top_n=10, num_samples=1000):
    """
    Genetic Algorithm for identifying top-N subsets with highest AFES scores.
    """

    # Enhanced parameters for better quality
    population_size = max(100, num_samples // 15)
    generations = max(50, num_samples // 20)
    elitism_size = max(10, population_size // 8)

    total_rows = df.shape[0]
    total_cols = len(df.columns)

    # Better initialization with diverse strategies
    population = create_diverse_population(total_rows, total_cols, population_size, df, anomaly)

    # Evaluate initial population
    for chromosome in population:
        evaluate_chromosome(chromosome, df, anomaly)

    best_fitness_history = []
    stagnation_counter = 0

    # Evolution loop
    for generation in range(generations):
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        current_best = population[0].fitness
        best_fitness_history.append(current_best)

        # Check for stagnation
        if len(best_fitness_history) > 10:
            recent_improvement = best_fitness_history[-1] - best_fitness_history[-10]
            if recent_improvement < 0.001:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

        # Apply diversity injection if stagnated
        if stagnation_counter > 5:
            replace_count = population_size // 5
            for i in range(replace_count):
                idx = -(i + 1)
                population[idx] = create_random_chromosome(total_rows, total_cols)
                evaluate_chromosome(population[idx], df, anomaly)
            stagnation_counter = 0

        # Create next generation
        new_population = []

        # Elitism: Keep best individuals (no change needed)
        new_population.extend(population[:elitism_size])

        # Generate offspring
        while len(new_population) < population_size:
            parent1 = diversity_tournament_selection(population)
            parent2 = diversity_tournament_selection(population)

            child1, child2 = enhanced_crossover(parent1, parent2, df, anomaly)

            child1 = adaptive_mutate(child1, generation, generations)
            child2 = adaptive_mutate(child2, generation, generations)

            # Evaluate children
            evaluate_chromosome(child1, df, anomaly)
            evaluate_chromosome(child2, df, anomaly)

            new_population.extend([child1, child2])

        population = new_population[:population_size]

    # Final evaluation and sorting
    population.sort(key=lambda x: x.fitness, reverse=True)

    # Convert top chromosomes to SubsetContainer format
    top_subsets = []
    for i in range(min(top_n, len(population))):
        chromosome = population[i]
        container = chromosome.to_subset_container(df, anomaly)
        afes_score = container.get_explanation_score()
        top_subsets.append((afes_score, container))

    # Sort by AFES score descending (highest first)
    top_subsets.sort(key=lambda x: x[0], reverse=True)

    return [subset for _, subset in top_subsets]