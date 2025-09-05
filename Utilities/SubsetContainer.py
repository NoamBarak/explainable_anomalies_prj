import numpy as np
import math
import Utilities.Constants as constans


class SubsetContainer:
    def __init__(self, subset, anomaly, sim_features_amount):
        self.subset = subset
        self.all_features = constans.COLUMNS
        self.sim_features, self.diff_features = self.sort_features_by_similarity(anomaly, sim_features_amount)
        # Calculate AFES (Anomaly Feature Explanation Score) using Definition 2
        self.explanation_score = self.calc_afes_score(anomaly)
        # Keep distance for backward compatibility, but it's now explanation score
        self.distance = self.explanation_score

    def get_subset(self):
        return self.subset

    def get_explanation_score(self):
        return self.explanation_score

    def set_explanation_score(self, explanation_score):
        self.explanation_score = explanation_score
        self.distance = explanation_score  # Keep distance in sync

    def get_euclidian_distance(self):
        """Deprecated: Use get_explanation_score() instead"""
        return self.explanation_score

    def set_euclidian_distance(self, euclidian_distance):
        """Deprecated: Use set_explanation_score() instead"""
        self.set_explanation_score(euclidian_distance)

    def sort_features_by_similarity(self, anomaly, sim_features_amount):
        feature_differences = {}
        for feature in self.all_features:
            # Calculate the absolute difference between subset and anomaly for each feature
            difference = abs(self.subset[feature].mean() - anomaly[feature])
            feature_differences[feature] = float(difference.iloc[0])

        sorted_features = sorted(feature_differences, key=feature_differences.get)

        sim_features = sorted_features[:sim_features_amount]
        diff_features = sorted_features[sim_features_amount:]

        return sim_features, diff_features

    def calc_euclidian_distance_definition1(self, v, features):
        """
        Definition 1 (Euclidean distance between a vector and a matrix)
        Given a matrix D ∈ R^(n×m) and a vector v ∈ R^m,
        sE(D, v) := 1 / (1 + ||(1/n * Σ(i=1 to n) Di) - v||)
        where Di is the ith row in D and || · || is a norm function.
        """
        if v is None:
            return None

        features = list(features)
        D = self.subset[features].to_numpy()  # Matrix D
        v = np.asarray(v[features]) if hasattr(v, '__getitem__') else np.asarray(v)  # Vector v

        # Calculate mean of rows: (1/n * Σ(i=1 to n) Di)
        n = D.shape[0]
        mean_vector = np.mean(D, axis=0)  # This is (1/n * Σ Di)

        # Calculate Euclidean norm: ||mean_vector - v||
        euclidean_norm = np.linalg.norm(mean_vector - v)

        # Apply the formula: sE(D, v) = 1 / (1 + euclidean_norm)
        sE = 1 / (1 + euclidean_norm)

        return sE

    def calc_euclidian_distance(self, sample, features):
        """
        Original implementation - kept for backward compatibility
        """
        if sample is None:
            return None

        features = list(features)
        subset = self.subset[features].to_numpy()
        sample = np.asarray(sample)

        distances = np.linalg.norm(subset - sample, axis=1)
        return np.mean(distances)

    def calc_afes_score(self, anomaly):
        """
        Definition 2 (anomaly feature explanation score - AFES)
        g(D', s, Fdiff, Fsim) = ω1 · (1/|D'|) * Σ(r∈D') sim(D', r) +
                                ω2 · sim(D', s)|Fsim - ω3 · sim(D', s)|Fdiff
        where sim is the similarity function (using Definition 1)
        """
        # First term: ω1 · (1/|D'|) * Σ(r∈D') sim(D', r)
        sum_similarities = 0
        subset_size = len(self.subset)

        for _, row in self.subset.iterrows():
            # sim(D', r) using Definition 1
            row_similarity = self.calc_euclidian_distance_definition1(row, self.all_features)
            sum_similarities += row_similarity

        term1 = constans.OMEGA_1 * (sum_similarities / subset_size)

        # Second term: ω2 · sim(D', s)|Fsim
        term2 = constans.OMEGA_2 * self.calc_euclidian_distance_definition1(anomaly, self.sim_features)

        # Third term: ω3 · sim(D', s)|Fdiff
        term3 = constans.OMEGA_3 * self.calc_euclidian_distance_definition1(anomaly, self.diff_features)

        # Final AFES score
        afes_score = term1 + term2 - term3

        return afes_score

    def calc_explanation_score(self, anomaly):
        """
        Updated to use AFES (Definition 2) instead of original implementation
        """
        return self.calc_afes_score(anomaly)

    def calc_subset_entropy(self):
        """
        Original implementation - kept unchanged
        """
        subset_entropy = 0
        subset_size = len(self.subset)

        for _, row in self.subset.iterrows():
            row_key = tuple(row.items())
            if row_key in self.entropy_cache:
                subset_entropy += self.entropy_cache[row_key]
            else:
                row_entropy = 0
                for col in self.df:
                    cur_prob = self.calc_prob_of_val(col, row[col])
                    if cur_prob > 0:
                        row_entropy += cur_prob * math.log(cur_prob, 2)
                row_entropy = -row_entropy
                self.entropy_cache[row_key] = row_entropy
                subset_entropy += row_entropy

        return subset_entropy / subset_size if subset_size > 0 else 0