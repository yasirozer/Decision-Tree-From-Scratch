"""
CART (Classification and Regression Trees) Decision Tree implementation.
Supports both numeric and categorical features.
Uses Gini impurity for splitting and majority voting for leaf predictions.
Includes matplotlib-based tree visualization.
"""

import numpy as np

from Node import Node
from Splitter import Splitter


class Tree:
    def __init__(self, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None
        self.splitter = Splitter(
            min_samples_split=min_samples_split,
            max_depth=max_depth if max_depth is not None else 100,
        )

    def fit(self, X, y):
        """
        Build the decision tree from training data.
        Feature types (numeric vs categorical) are detected once here
        to avoid repeated checks at every node.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.splitter.detect_feature_types(X)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """
        Recursively build the tree. Returns a leaf node if any stopping
        condition is met (max depth, pure node, min samples, or no
        impurity improvement). Otherwise splits and recurses on children.
        """
        # Stopping conditions: min samples, pure node, or max depth reached
        if (
            len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return Node(value=self._most_common_label(y))

        feature, threshold, categories, left_indices, right_indices = self.splitter.best_split(X, y)

        # No valid split found
        if left_indices is None or right_indices is None:
            return Node(value=self._most_common_label(y))

        # Check if the impurity decrease is worth splitting
        parent_impurity = self.splitter.gini(y)
        left_y = y[left_indices]
        right_y = y[right_indices]

        weighted_child_impurity = (
            (len(left_y) / len(y)) * self.splitter.gini(left_y)
            + (len(right_y) / len(y)) * self.splitter.gini(right_y)
        )
        impurity_decrease = parent_impurity - weighted_child_impurity

        if impurity_decrease < self.min_impurity_decrease:
            return Node(value=self._most_common_label(y))

        # Recurse on left and right subsets
        left_child = self._build_tree(X[left_indices], left_y, depth + 1)
        right_child = self._build_tree(X[right_indices], right_y, depth + 1)

        return Node(
            feature=feature,
            threshold=threshold,
            categories=categories,
            left=left_child,
            right=right_child,
        )

    def predict(self, X):
        """Predict class labels for each sample by traversing the tree."""
        X = np.asarray(X)
        return np.array([self._traverse(self.root, sample) for sample in X])

    def _traverse(self, node, sample):
        """
        Walk a single sample down the tree until a leaf is reached.
        Categorical splits check set membership; numeric splits compare against threshold.
        """
        if node.is_leaf_node():
            return node.value

        if node.is_categorical_split():
            if sample[node.feature] in node.categories:
                return self._traverse(node.left, sample)
            return self._traverse(node.right, sample)
        else:
            if float(sample[node.feature]) <= node.threshold:
                return self._traverse(node.left, sample)
            return self._traverse(node.right, sample)

    def _most_common_label(self, y):
        """Return the most frequent class label (majority vote for leaf nodes)."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    # ---- Plotting methods ----

    def plot(self, feature_names=None, figsize=None):
        """Visualize the decision tree using matplotlib."""
        if self.root is None:
            raise ValueError("Tree has not been fitted yet.")

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for plotting. Install it with 'python -m pip install matplotlib'."
            ) from exc

        total_leaves = max(self._count_leaves(self.root), 1)
        max_depth = max(self._tree_depth(self.root), 1)

        # Scale figure size based on tree complexity
        if figsize is None:
            width = max(14, total_leaves * 3.5)
            height = max(8, max_depth * 3)
            figsize = (width, height)

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")

        # Assign x positions to leaves first, then derive parent positions
        leaf_positions = {}
        self._assign_leaf_x(self.root, leaf_positions, counter=[0], total_leaves=total_leaves)

        self._plot_tree(
            node=self.root,
            ax=ax,
            y=0.95,
            y_step=0.85 / max_depth,
            feature_names=feature_names,
            leaf_positions=leaf_positions,
        )

        plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
        plt.show()

    def _count_leaves(self, node):
        """Count total leaf nodes in the subtree (used for spacing in plot)."""
        if node.is_leaf_node():
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)

    def _tree_depth(self, node):
        """Calculate the maximum depth of the subtree (used for vertical spacing)."""
        if node is None or node.is_leaf_node():
            return 1
        return 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))

    def _node_label(self, node, feature_names=None):
        """
        Generate the display text for a node.
        Leaf: shows predicted class. Split: shows feature name and condition.
        For categorical splits with many categories, the label is truncated.
        """
        if node.is_leaf_node():
            return f"Leaf\nclass = {node.value}"

        if feature_names is not None and node.feature < len(feature_names):
            feature_label = feature_names[node.feature]
        else:
            feature_label = f"X[{node.feature}]"

        if node.is_categorical_split():
            cats = sorted(str(c) for c in node.categories)
            if len(cats) <= 3:
                cats_str = ", ".join(cats)
            else:
                cats_str = ", ".join(cats[:2]) + f"\n... +{len(cats)-2} more"
            return f"{feature_label}\nin {{{cats_str}}}"
        return f"{feature_label} <= {node.threshold:.3f}"

    def _assign_leaf_x(self, node, positions, counter, total_leaves):
        """
        Assign evenly spaced x positions to leaf nodes (left to right).
        Internal nodes will be centered between their children later.
        """
        if node.is_leaf_node():
            margin = 0.05
            spacing = (1.0 - 2 * margin) / max(total_leaves - 1, 1)
            positions[id(node)] = margin + counter[0] * spacing
            counter[0] += 1
            return

        self._assign_leaf_x(node.left, positions, counter, total_leaves)
        self._assign_leaf_x(node.right, positions, counter, total_leaves)
        # Internal node x = midpoint of its children's x range
        positions[id(node)] = (self._subtree_min_x(node, positions)
                               + self._subtree_max_x(node, positions)) / 2

    def _subtree_min_x(self, node, positions):
        """Get the leftmost x position in this subtree."""
        if node.is_leaf_node():
            return positions[id(node)]
        return self._subtree_min_x(node.left, positions)

    def _subtree_max_x(self, node, positions):
        """Get the rightmost x position in this subtree."""
        if node.is_leaf_node():
            return positions[id(node)]
        return self._subtree_max_x(node.right, positions)

    def _plot_tree(self, node, ax, y, y_step, feature_names, leaf_positions):
        """
        Recursively draw nodes and edges using precomputed leaf positions.
        This avoids overlap by spacing leaves evenly and centering parents.
        """
        x = leaf_positions[id(node)]
        label = self._node_label(node, feature_names)

        fontsize = 8
        box_style = {
            "boxstyle": "round,pad=0.4",
            "fc": "#e8f0fe" if not node.is_leaf_node() else "#e6f4ea",
            "ec": "#4a4a4a",
        }
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, bbox=box_style)

        if node.is_leaf_node():
            return

        child_y = y - y_step
        left_x = leaf_positions[id(node.left)]
        right_x = leaf_positions[id(node.right)]

        # Draw edges to children
        ax.annotate("", xy=(left_x, child_y + 0.02), xytext=(x, y - 0.02),
                     arrowprops={"arrowstyle": "-", "lw": 1.2, "color": "#666"})
        ax.annotate("", xy=(right_x, child_y + 0.02), xytext=(x, y - 0.02),
                     arrowprops={"arrowstyle": "-", "lw": 1.2, "color": "#666"})

        # Edge labels
        ax.text((x + left_x) / 2 - 0.01, (y + child_y) / 2, "True",
                ha="center", va="center", fontsize=7, color="#2e7d32")
        ax.text((x + right_x) / 2 + 0.01, (y + child_y) / 2, "False",
                ha="center", va="center", fontsize=7, color="#c62828")

        self._plot_tree(node.left, ax, child_y, y_step, feature_names, leaf_positions)
        self._plot_tree(node.right, ax, child_y, y_step, feature_names, leaf_positions)
