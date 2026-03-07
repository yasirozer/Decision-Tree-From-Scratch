import numpy as np
import pandas as pd
from Node import Node


class Splitter:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def gini(self, y):
        indexA = 0.0
        total_samples = len(y)
        for y_value in np.unique(y):
            p = len(y[y == y_value])
            indexA += (p / total_samples) ** 2
        return 1 - indexA

    def mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def best_split(self, X, y):
        n_samples, n_features = X.shape

        best_feature_index = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        best_gini = float("inf")

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            if len(unique_values) < 2:
                continue

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2

                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                y_left = y[left_indices]
                y_right = y[right_indices]

                left_gini = self.gini(y_left)
                right_gini = self.gini(y_right)

                weighted_gini = (
                    (len(left_indices) / n_samples) * left_gini
                    + (len(right_indices) / n_samples) * right_gini
                )

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature_index = feature_index
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        return best_feature_index, best_threshold, best_left_indices, best_right_indices
    

X = np.array([
    [2.1, 5.0],
    [2.5, 6.0],
    [3.7, 1.0],
    [4.0, 2.0]
])
y = np.array([0, 0, 1, 1])  
if __name__ == "__main__":
    split = Splitter()
    y = np.array([0, 0, 1, 1])
    result = split.gini(y)
    print(result)