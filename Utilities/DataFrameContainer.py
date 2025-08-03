import pandas as pd
import os
from Utilities import Constants as constants

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class DataFrameContainer:
    def __init__(self):
        # file_path = constants.PROJECT_PATH + "\Data\house_prices_train.csv"
        file_path = constants.PROJECT_PATH + "\Data\AmesHousing.csv"
        df = pd.read_csv(file_path)
        df = df.sample(n=300, random_state=42)
        selected_columns = constants.COLUMNS     # Select specific columns from the DataFrame
        self.original_df = pd.DataFrame(df[selected_columns])

        imputer = SimpleImputer(strategy="median")
        imputed_data = imputer.fit_transform(self.original_df)

        self.original_df = pd.DataFrame(imputed_data, columns=self.original_df.columns)


        # Normalize the data
        # self.full_data = (self.original_df - self.original_df.min()) / (self.original_df.max() - self.original_df.min())

        scaler = StandardScaler()
        self.full_data = scaler.fit_transform(self.original_df)
        self.full_data = pd.DataFrame(self.full_data, columns=selected_columns)

        model = IsolationForest(n_estimators=100, max_samples=0.7,
                                contamination=0.05, max_features=1.0, bootstrap=False, n_jobs=-1,
                                verbose=0, random_state=2020)
        model.fit(self.full_data)
        predictions = model.predict(self.full_data)

        # Separate the data into normal and anomalies
        self.anomalies = self.full_data[predictions == -1]
        self.normal_data = self.full_data[predictions == 1]

        # Metadata about the DataFrame
        self.rows_amount = self.normal_data.shape[0]
        self.cols_amount = self.normal_data.shape[1]
        self.cols_names = self.normal_data.columns
        self.entropy_cache = {}  # Initialize a cache for entropy calculations
        self.prob_cache = {}  # Initialize a cache for probability calculations

    def get_df(self):
        return self.df

    def calc_prob_of_val(self, col_name, val):
        """
        Calculate the probability of a specific value occurring in a specified column.
        """
        # Check if the probability for this column and value pair is already cached
        cache_key = (col_name, val)
        if cache_key in self.prob_cache:
            return self.prob_cache[cache_key]

        count_of_val = len(self.normal_data[self.normal_data[col_name] == val])
        prob = count_of_val / self.rows_amount

        self.prob_cache[cache_key] = prob   # Store the calculated probability in the cache

        return prob
