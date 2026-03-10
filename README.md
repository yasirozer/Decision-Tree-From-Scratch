## CART From Scratch – Decision Tree & Random Forest

This project implements a **CART decision tree classifier** and a **Random Forest** classifier **from scratch in Python**, using only `numpy`, `pandas` and `matplotlib` (plus `openpyxl` for reading Excel files).

It was written for an assignment where:
- Part A: implement CART and evaluate it with detailed performance metrics.
- Part B: build a Random Forest using the **Random Subspace Method** (random feature subsets per tree) and report the same metrics.

The data set is a real-estate classification problem (multi‑class), where the last column is the class label.

---

## Project structure

```text
CART-From-Scratch/
  Node.py          # Node object for the decision tree
  Splitter.py      # Finds the best split (numeric + categorical)
  Tree.py          # CART decision tree implementation + plotting
  randomforest.py  # Random Forest implementation (random subspace)
  main.py          # Entry point: trains tree + forest, prints metrics, plots tree
  X_train.xlsx     # Training data (features + target in last column)
  X_test.xlsx      # Test data (same column structure as train)
```

### Core components

- **`Node` (`Node.py`)**
  - Represents a single node in the tree.
  - Can be either:
    - **Leaf node**: stores only `value` (predicted class).
    - **Split node**:
      - `feature`: index of feature used for splitting.
      - `threshold`: numeric split – sample goes left if `feature <= threshold`.
      - `categories`: categorical split – sample goes left if value is in this set.
      - `left`, `right`: child nodes.

- **`Splitter` (`Splitter.py`)**
  - Responsible for **finding the best split** at a node using **Gini impurity**.
  - Supports:
    - **Numeric features**: tries midpoints between sorted unique values as thresholds.
    - **Categorical features**: orders categories by class distribution and tries all
      split points along that order (Random Subspace can still hide some features).

- **`Tree` (`Tree.py`)**
  - Full **CART classifier**:
    - Recursively builds the tree using `Splitter.best_split`.
    - Stopping criteria: max depth, pure node, not enough samples, or too small impurity decrease.
    - Prediction: traverse from root to leaf using split rules.
  - Includes a **custom matplotlib plot** that:
    - Supports both numeric and categorical splits.
    - Automatically scales figure size by depth / number of leaves.
    - Places nodes to avoid overlap and labels edges as `True` / `False`.

- **`RandomForest` (`randomforest.py`)**
  - Implements a Random Forest using the **Random Subspace Method**:
    - For each tree:
      - Chooses a **random number of features** `k` between `1` and `n_features-1`.
      - Randomly selects `k` feature columns **without replacement**.
      - Trains a `Tree` only on those features but with **all training samples**.
    - This creates:
      - “Specialist” trees (k=1),
      - “Moderate” trees (k=2),
      - “Generalist” trees (k=3, …).
    - Final prediction is made with **majority voting** across all trees.

- **`main.py`**
  - Loads train / test Excel files.
  - Trains a **decision tree** and a **random forest**.
  - Computes detailed **multi‑class metrics**:
    - Accuracy
    - TP Rate (Recall)
    - TN Rate
    - Precision
    - F‑Score
    - Total Number of TP
    - Total Number of TN
  - Prints metrics separately for:
    - Decision tree (Train + Test)
    - Random forest (Test)
  - Displays the decision tree plot with feature names.

---

## Requirements

- Python 3.10+ (project was developed on a MSYS2 Python 3.10 / 3.14 environment)
- Python packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `openpyxl` (for reading `.xlsx` files)

Install dependencies, for example:

```bash
py -3 -m pip install numpy pandas matplotlib openpyxl
```

> Note: on some systems `python` or `python3` is used instead of `py -3`.

---

## Data files

The scripts expect the following files in the project root:

- `X_train.xlsx`
- `X_test.xlsx`

In the current version, the paths in `main.py` and `randomforest.py` are **absolute Windows paths**:

```python
train_df = pd.read_excel(r"C:\Users\Asus\Desktop\Projects\CART-From-Scratch\X_train.xlsx")
test_df = pd.read_excel(r"C:\Users\Asus\Desktop\Projects\CART-From-Scratch\X_test.xlsx")
```

If you move the project, you should either:

- Update these paths to your local path, **or**
- Replace them with relative paths:

```python
train_df = pd.read_excel("X_train.xlsx")
test_df = pd.read_excel("X_test.xlsx")
```

The data format is:

- All columns except the **last**: features.
- Last column: target class label (multi‑class).

---

## Running the project

Open a terminal in the `CART-From-Scratch` directory and run:

```bash
py -3 main.py
```

This will:

1. Read `X_train.xlsx` and `X_test.xlsx`.
2. Train a **CART decision tree** with `max_depth=10`.
3. Train a **Random Forest** with `n_trees=50`, `max_depth=5`, random subspace per tree.
4. Print metrics in this format:

```text
DECISION TREE
========================================
Train Results:
========================================
Accuracy: 0.www
TP Rate (Recall): 0.xxx
TN Rate: 0.yyy
Precision: 0.zzz
F-Score: 0.aaa
Total Number of TP: bbb
Total Number of TN: ccc
========================================

========================================
Test Results:
...
========================================

RANDOM FOREST
========================================
Test Results:
...
========================================
```

5. Open a matplotlib window showing the learned decision tree.

You can also run just the Random Forest script directly:

```bash
py -3 randomforest.py
```

It will train the forest on `X_train.xlsx`, evaluate on `X_test.xlsx`, and print the same test metrics.

---

## How the metrics are computed

For a **multi‑class** problem, simple binary formulas for TP/TN/FP/FN are not enough.  
This project uses a **one‑vs‑rest** strategy:

- For each class `c`:
  - Treat `c` as the **positive** class.
  - Treat all other classes as **negative**.
  - Compute TP, TN, FP, FN for that class.
- Then compute:
  - Recall (TP rate) = TP / (TP + FN)
  - Precision = TP / (TP + FP)
  - TN rate = TN / (TN + FP)
- Finally, **macro‑average** these scores across all classes.

The F‑Score (F1) is then:

\[
F = \frac{2 \cdot \text{precision} \cdot \text{recall}}
         {\text{precision} + \text{recall}}
\]

(if both precision and recall are > 0).

The “Total Number of TP/TN” printed are the **sums** of TP/TN over all one‑vs‑rest views.

---

## Random Subspace Method in this project

- Let \( d \) be the number of features.
- For each tree:
  - Randomly choose an integer \( k \in \{1, \dots, d-1\} \).
  - Randomly choose \( k \) feature indices without replacement.
  - Train a full decision tree on **all rows** but only these \( k \) columns.
- Different trees see different “views” of the feature space:
  - Some trees specialise on a single feature,
  - Some see pairs/triples of features,
  - No tree sees all features at once.

This reduces correlation between trees and generally improves the Random Forest’s
generalisation performance compared to a single tree.

---

## Limitations and possible improvements

- Paths to Excel files are currently hard‑coded and machine‑specific.
- There is no automatic hyper‑parameter search (e.g. tuning `max_depth`, `n_trees`).
- Plotting is designed for small/medium trees; very deep trees may still be crowded.

Despite these limitations, the code is intentionally written to be **readable and educational**, so you can:

- Step through splits and verify Gini calculations.
- Experiment with different tree depths or number of trees.
- Try other ensemble strategies (e.g. bagging on rows in addition to feature subspaces).

