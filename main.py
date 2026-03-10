"""
Main script for running the CART Decision Tree and Random Forest models.
Loads training/test data from Excel, trains both models, prints performance
metrics, and displays the decision tree plot.
"""

import numpy as np
import pandas as pd
from Tree import Tree

# --- Load data ---
train_df = pd.read_excel(r"C:\Users\Asus\Desktop\Projects\CART-From-Scratch\X_train.xlsx")
test_df = pd.read_excel(r"C:\Users\Asus\Desktop\Projects\CART-From-Scratch\X_test.xlsx")

# Features = all columns except last, Target = last column
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

feature_names = train_df.columns[:-1].tolist()

# --- Train decision tree ---
tree = Tree(max_depth=10)
tree.fit(X_train, y_train)

train_preds = tree.predict(X_train)
test_preds = tree.predict(X_test)


def compute_metrics(y_true, y_pred):
    """
    Compute multi-class performance metrics using one-vs-rest approach.
    For each class: calculates TP, TN, FP, FN treating that class as positive.
    Final scores are macro-averaged across all classes.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    total_tp = 0
    total_tn = 0
    recalls = []
    precisions = []
    tn_rates = []

    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))   # Correctly predicted as this class
        tn = np.sum((y_true != cls) & (y_pred != cls))   # Correctly predicted as NOT this class
        fp = np.sum((y_true != cls) & (y_pred == cls))   # Incorrectly predicted as this class
        fn = np.sum((y_true == cls) & (y_pred != cls))   # Missed this class

        total_tp += tp
        total_tn += tn
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        tn_rates.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    accuracy = np.sum(y_true == y_pred) / len(y_true)
    recall = np.mean(recalls)           # Macro-averaged recall (TP Rate)
    precision = np.mean(precisions)     # Macro-averaged precision
    tn_rate = np.mean(tn_rates)         # Macro-averaged TN rate
    f_score = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

    return {
        "Accuracy": accuracy,
        "TP Rate (Recall)": recall,
        "TN Rate": tn_rate,
        "Precision": precision,
        "F-Score": f_score,
        "Total Number of TP": total_tp,
        "Total Number of TN": total_tn,
    }


def print_metrics(title, y_true, y_pred):
    """Print a formatted metrics table with the given title."""
    print("=" * 40)
    print(f"{title}:")
    print("=" * 40)
    metrics = compute_metrics(y_true, y_pred)
    for name, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{name}: {value}")
        else:
            print(f"{name}: {value:.4f}")
    print("=" * 40)


# --- Decision Tree results ---
print("DECISION TREE")
print_metrics("Train Results", y_train, train_preds)
print()
print_metrics("Test Results", y_test, test_preds)

# --- Random Forest results ---
print()
print("RANDOM FOREST")
from randomforest import RandomForest

rf = RandomForest(n_trees=50, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
rf_test_preds = rf.predict(X_test)

print_metrics("Test Results", y_test, rf_test_preds)

# --- Show decision tree plot ---
tree.plot(feature_names=feature_names)
