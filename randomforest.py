"""
Random Forest implementation using the Random Subspace Method.

Each tree in the forest is trained on ALL training samples, but with a
randomly selected subset of features (columns). This is the key difference
from Bagging, which samples rows instead of columns.

The number of features per tree varies randomly between 1 and n_features-1,
increasing diversity among the trees. Final predictions are made by
majority voting across all trees.
"""

import numpy as np
import pandas as pd
from Tree import Tree


class RandomForest:
    def __init__(self, n_trees=50, max_depth=5, min_samples_split=2,
                 random_state=None):
        """
        Parameters:
            n_trees:           Number of decision trees in the forest (between 15-100).
            max_depth:         Maximum depth allowed for each individual tree.
            min_samples_split: Minimum number of samples required to split a node.
            random_state:      Seed for the random number generator (for reproducibility).
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []             # Stores each trained Tree object
        self.feature_indices = []   # Stores the feature subset indices used by each tree

    def fit(self, X, y):
        """
        Train the random forest using the Random Subspace Method.

        For each of the n_trees trees:
          1. Randomly pick how many features this tree will see: k in [1, n_features-1]
          2. Randomly select k feature columns (without replacement)
          3. Train a decision tree on ALL samples but only those k features

        By varying k per tree, we get a mix of:
          - Specialist trees (k=1): focus on a single feature
          - Moderate trees (k=2): capture pairwise interactions
          - Generalist trees (k=3): see most of the feature space
        We never use k=n_features so every tree sees a true subspace.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape

        # Reproducible random number generator
        rng = np.random.default_rng(self.random_state)

        self.trees = []
        self.feature_indices = []

        for i in range(self.n_trees):
            # Randomly decide how many features this tree gets (1 to n_features-1)
            k = rng.integers(1, n_features)

            # Randomly pick k feature indices without replacement
            feat_idx = np.sort(rng.choice(n_features, size=k, replace=False))

            # Slice the training data to only include the selected features
            X_sub = X[:, feat_idx]

            # Build and train a single decision tree on this feature subspace
            tree = Tree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_sub, y)

            self.trees.append(tree)
            self.feature_indices.append(feat_idx)

        return self

    def predict(self, X):
        """
        Predict class labels using majority voting.

        Each tree makes a prediction using only its own feature subset.
        For each sample, the class that receives the most votes across
        all trees is selected as the final prediction.
        """
        X = np.asarray(X)

        # Gather predictions from every tree (shape: n_trees x n_samples)
        # Each tree only sees the columns it was trained on
        all_preds = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self.feature_indices)
        ])

        # For each sample, pick the class with the most votes
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=all_preds.dtype)
        for i in range(n_samples):
            votes = all_preds[:, i]
            values, counts = np.unique(votes, return_counts=True)
            predictions[i] = values[np.argmax(counts)]

        return predictions


# ---- Standalone execution: train RF and print test metrics ----
if __name__ == "__main__":
    # Load train and test datasets
    train_df = pd.read_excel(r"C:\Users\Asus\Desktop\Projects\CART-From-Scratch\X_train.xlsx")
    test_df = pd.read_excel(r"C:\Users\Asus\Desktop\Projects\CART-From-Scratch\X_test.xlsx")

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    # Train random forest with 50 trees
    rf = RandomForest(n_trees=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    test_preds = rf.predict(X_test)

    # --- Multi-class performance metrics ---
    # Uses one-vs-rest approach: for each class, treat it as "positive"
    # and all other classes as "negative", then macro-average the results.
    classes = np.unique(np.concatenate([y_test, test_preds]))
    total_tp = 0
    total_tn = 0
    recalls = []
    precisions = []
    tn_rates = []

    for cls in classes:
        tp = np.sum((y_test == cls) & (test_preds == cls))   # Correct positive
        tn = np.sum((y_test != cls) & (test_preds != cls))   # Correct negative
        fp = np.sum((y_test != cls) & (test_preds == cls))   # False positive
        fn = np.sum((y_test == cls) & (test_preds != cls))   # False negative
        total_tp += tp
        total_tn += tn
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        tn_rates.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    accuracy = np.sum(y_test == test_preds) / len(y_test)
    recall = np.mean(recalls)           # Macro-averaged TP rate (recall)
    precision = np.mean(precisions)     # Macro-averaged precision
    tn_rate = np.mean(tn_rates)         # Macro-averaged TN rate
    f_score = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

    print("=" * 40)
    print("Random Forest - Test Results:")
    print("=" * 40)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"TP Rate (Recall): {recall:.4f}")
    print(f"TN Rate: {tn_rate:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F-Score: {f_score:.4f}")
    print(f"Total Number of TP: {total_tp}")
    print(f"Total Number of TN: {total_tn}")
    print("=" * 40)
