import numpy as np
import math
import Utilities.Constants as constans


class SubsetContainer:
    def __init__(self, subset, anomaly, sim_features_amount):
        self.subset = subset
        self.all_features = constans.COLUMNS
        self.sim_features, self.diff_features = self.sort_features_by_similarity(anomaly, sim_features_amount)
        # self.distance = self.calc_explanation_score(anomaly)
        self.distance = self.calc_euclidian_distance(anomaly, self.all_features)

    def get_subset(self):
        return self.subset

    def get_euclidian_distance(self):
        return self.distance

    def set_euclidian_distance(self, euclidian_distance):
        self.distance = euclidian_distance


    def sort_features_by_similarity(self,anomaly, sim_features_amount):
        feature_differences = {}
        for feature in self.all_features:
            # Calculate the absolute difference between subset and anomaly for each feature
            difference = abs(self.subset[feature].mean() - anomaly[feature])
            feature_differences[feature] = float(difference.iloc[0])

        sorted_features = sorted(feature_differences, key=feature_differences.get)

        sim_features = sorted_features[:sim_features_amount]
        diff_features = sorted_features[sim_features_amount:]

        return sim_features, diff_features


    def calc_euclidian_distance(self, sample, features):
        """
        Definition 1 (Euclidean distance between a vector and a matrix)
        Euclidean distances between D_prime and a sample based on the specified features (features).
        """
        # if sample is None:
        #     return None
        # features = list(features)
        # subset_mean = np.mean(self.subset[features], axis=0)
        # euclidian_distance = 1/(1 + np.linalg.norm(subset_mean - sample))
        # return euclidian_distance

        if sample is None:
            return None  # Or raise ValueError("Sample cannot be None")

        features = list(features)  # Ensure features is a list
        subset = self.subset[features].to_numpy()  # Convert to NumPy array if not already

        # Ensure sample is a NumPy array
        sample = np.asarray(sample)

        # Compute Euclidean distances for all rows
        distances = np.linalg.norm(subset - sample, axis=1)

        return np.mean(distances)

    def calc_explanation_score(self, anomaly):
        """
        Definition 2 (anomaly feature explanation score).
        """
        sum_distances = 0
        for _, row in self.subset.iterrows():
            sum_distances += self.calc_euclidian_distance(row, self.all_features)

        res = (constans.OMEGA_1/len(self.subset)) * sum_distances
        res = res + constans.OMEGA_2 * self.calc_euclidian_distance(anomaly[self.sim_features], self.sim_features)
        res = res - constans.OMEGA_3 * self.calc_euclidian_distance(anomaly[self.diff_features], self.diff_features)

        return res



    def calc_subset_entropy(self):
        subset_entropy = 0
        subset_size = len(self.subset)

        for _, row in self.subset.iterrows():
            # Convert row to a tuple for caching
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

        # Return the average entropy for the subset
        return subset_entropy / subset_size if subset_size > 0 else 0








