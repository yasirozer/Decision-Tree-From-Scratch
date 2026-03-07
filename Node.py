class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # The index of the feature used for splitting
        self.threshold = threshold  # The threshold value for splitting
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # The predicted value for leaf nodes

    def is_leaf_node(self):
        return self.value is not None