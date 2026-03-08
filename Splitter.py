"""
Splitter class responsible for finding the best binary split at each node.
Handles both numeric features (threshold-based midpoint splits) and
categorical/nominal features (subset-based splits using sorted category ordering).
Uses Gini impurity as the splitting criterion.
"""

import numpy as np


class Splitter:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self._numeric_features = None   # Boolean array: True if feature is numeric

    def gini(self, y):
        """
        Calculate Gini impurity for a set of labels.
        Gini = 1 - sum(p_i^2) where p_i is the proportion of class i.
        Returns 0 for pure nodes, max ~0.5 for binary, higher for multi-class.
        """
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / counts.sum()
        return 1.0 - np.sum(proportions ** 2)

    def detect_feature_types(self, X):
        """
        Detect whether each feature column is numeric or categorical.
        Called once during Tree.fit() to avoid repeated type-checking at every node.
        """
        n_features = X.shape[1]
        self._numeric_features = np.zeros(n_features, dtype=bool)
        for i in range(n_features):
            try:
                X[:, i].astype(float)
                self._numeric_features[i] = True
            except (ValueError, TypeError):
                self._numeric_features[i] = False

    def best_split(self, X, y):
        """
        Find the best binary split across all features.
        For numeric features: tries midpoint thresholds between consecutive unique values.
        For categorical features: sorts categories by target class distribution,
            then tries all k-1 split points along that ordering (incremental mask approach).
        Returns: (feature_index, threshold, categories_set, left_indices, right_indices)
            - threshold is set for numeric splits, categories for categorical splits.
        """
        n_samples, n_features = X.shape

        best_gini = float("inf")
        best_feature = None
        best_threshold = None
        best_categories = None
        best_left = None
        best_right = None

        for fi in range(n_features):
            col = X[:, fi]
            unique = np.unique(col)

            # Skip features with only one unique value (no possible split)
            if len(unique) < 2:
                continue

            if self._numeric_features[fi]:
                # --- Numeric feature: try midpoint thresholds ---
                fv = col.astype(float)
                uv = unique.astype(float)
                uv.sort()
                # Midpoints between consecutive sorted unique values
                thresholds = (uv[:-1] + uv[1:]) / 2.0

                for thr in thresholds:
                    left_mask = fv <= thr
                    n_left = np.count_nonzero(left_mask)
                    n_right = n_samples - n_left
                    if n_left == 0 or n_right == 0:
                        continue

                    # Weighted Gini impurity of the split
                    wg = ((n_left / n_samples) * self.gini(y[left_mask])
                          + (n_right / n_samples) * self.gini(y[~left_mask]))

                    if wg < best_gini:
                        best_gini = wg
                        best_feature = fi
                        best_threshold = thr
                        best_categories = None
                        best_left = np.where(left_mask)[0]
                        best_right = np.where(~left_mask)[0]
            else:
                # --- Categorical feature: sorted subset splits ---
                classes = np.unique(y)

                # Precompute boolean mask and a sorting score for each category.
                # Score = weighted average of class indices by their proportions,
                # which creates a meaningful 1D ordering of categories.
                cat_masks = {}
                cat_scores = {}
                for cat in unique:
                    mask = col == cat
                    cat_masks[cat] = mask
                    y_cat = y[mask]
                    n_cat = len(y_cat)
                    score = sum(
                        i * np.count_nonzero(y_cat == cls) / n_cat
                        for i, cls in enumerate(classes)
                    )
                    cat_scores[cat] = score

                # Sort categories by their score
                sorted_cats = sorted(unique, key=lambda c: cat_scores[c])

                # Incrementally build the left mask by adding one category at a time.
                # This avoids rebuilding the mask from scratch for each split point.
                left_mask = np.zeros(n_samples, dtype=bool)
                left_set = set()
                for i in range(len(sorted_cats) - 1):
                    cat = sorted_cats[i]
                    left_mask |= cat_masks[cat]
                    left_set.add(cat)

                    n_left = np.count_nonzero(left_mask)
                    n_right = n_samples - n_left
                    if n_left == 0 or n_right == 0:
                        continue

                    wg = ((n_left / n_samples) * self.gini(y[left_mask])
                          + (n_right / n_samples) * self.gini(y[~left_mask]))

                    if wg < best_gini:
                        best_gini = wg
                        best_feature = fi
                        best_threshold = None
                        best_categories = set(left_set)
                        best_left = np.where(left_mask)[0]
                        best_right = np.where(~left_mask)[0]

        return best_feature, best_threshold, best_categories, best_left, best_right
