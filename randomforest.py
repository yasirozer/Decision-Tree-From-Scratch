"""
Random Forest implementation using the Random Subspace Method.
Each tree in the forest is trained on all samples but with a randomly
selected subset of features. Final predictions are made by majority voting.
"""

import numpy as np
import pandas as pd
from Tree import Tree


class RandomForest:
    def __init__(self, n_trees=50, max_depth=5, min_samples_split=2,
                 max_features="sqrt", random_state=None):
        """
        Parameters:
            n_trees:          Number of decision trees in the forest (15-100).
            max_depth:        Maximum depth for each individual tree.
            min_samples_split: Minimum samples required to split a node.
            max_features:     Number of features per tree. Accepts "sqrt", "log2",
                              an int (exact count), or a float (fraction of total).
            random_state:     Seed for reproducibility.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []             # List of trained Tree objects
        self.feature_indices = []   # Feature subset used by each tree

    def _get_max_features(self, n_features):
        """Resolve the max_features parameter into an integer count."""
        if self.max_features == "sqrt":
            return max(2, int(np.sqrt(n_features)))
        if self.max_features == "log2":
            return max(2, int(np.log2(n_features)))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if isinstance(self.max_features, float):
            return max(2, int(self.max_features * n_features))
        return n_features

    def fit(self, X, y):
        """
        Train the random forest using the Random Subspace Method.
        For each tree: randomly select a feature subset, then train the tree
        on all training samples but only with those selected features.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)

        k = self._get_max_features(n_features)
        self.trees = []
        self.feature_indices = []

        for _ in range(self.n_trees):
            # Random Subspace: select k random features (columns) without replacement
            feat_idx = np.sort(rng.choice(n_features, size=k, replace=False))
            X_sub = X[:, feat_idx]

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
        Each tree predicts using its own feature subset, then the most
        frequent prediction across all trees is selected for each sample.
        """
        X = np.asarray(X)

        # Collect predictions from all trees (shape: n_trees x n_samples)
        all_preds = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self.feature_indices)
        ])

        # Majority vote for each sample
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=all_preds.dtype)
        for i in range(n_samples):
            votes = all_preds[:, i]
            values, counts = np.unique(votes, return_counts=True)
            predictions[i] = values[np.argmax(counts)]

        return predictions


if __name__ == "__main__":
    train_df = pd.read_excel(r"C:\Users\Asus\Desktop\Projects\CART-From-Scratch\X_train.xlsx")
    test_df = pd.read_excel(r"C:\Users\Asus\Desktop\Projects\CART-From-Scratch\X_test.xlsx")

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    rf = RandomForest(n_trees=50, max_depth=5, max_features="sqrt", random_state=42)
    rf.fit(X_train, y_train)

    test_preds = rf.predict(X_test)

    # --- Multi-class performance metrics (one-vs-rest per class, then macro average) ---
    classes = np.unique(np.concatenate([y_test, test_preds]))
    total_tp = 0
    total_tn = 0
    recalls = []
    precisions = []
    tn_rates = []

    for cls in classes:
        tp = np.sum((y_test == cls) & (test_preds == cls))
        tn = np.sum((y_test != cls) & (test_preds != cls))
        fp = np.sum((y_test != cls) & (test_preds == cls))
        fn = np.sum((y_test == cls) & (test_preds != cls))
        total_tp += tp
        total_tn += tn
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        tn_rates.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    accuracy = np.sum(y_test == test_preds) / len(y_test)
    recall = np.mean(recalls)           # Macro-averaged recall
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
