"""
Node class representing a single node in the decision tree.
Each node is either an internal (split) node or a leaf (prediction) node.
Supports both numeric splits (threshold-based) and categorical splits (set-based).
"""


class Node:
    def __init__(self, feature=None, threshold=None, categories=None,
                 left=None, right=None, value=None):
        self.feature = feature          # Index of the feature used for splitting
        self.threshold = threshold      # Numeric split: sample goes left if <= threshold
        self.categories = categories    # Categorical split: sample goes left if value in this set
        self.left = left                # Left child node (True branch)
        self.right = right              # Right child node (False branch)
        self.value = value              # Predicted class label (only set for leaf nodes)

    def is_leaf_node(self):
        """Return True if this node is a leaf (has a predicted value, no children)."""
        return self.value is not None

    def is_categorical_split(self):
        """Return True if this node splits on a categorical feature."""
        return self.categories is not None
