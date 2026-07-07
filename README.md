# Decision Tree & Random Forest From Scratch

A CART style decision tree classifier and a bagging based random forest
classifier, both implemented from scratch in Python using only numpy for
array math. No scikit-learn, no other ML library, anywhere in this repo.

## Why this exists

Most people use `sklearn.tree.DecisionTreeClassifier` and
`RandomForestClassifier` without ever seeing what is inside them. This repo
builds both algorithms from the ground up so the mechanics are visible and
testable:

* how a tree picks the best (feature, threshold) split at every node using
  Gini impurity or entropy
* why an unconstrained tree overfits, and how `max_depth` /
  `min_samples_split` / `min_samples_leaf` control that
* how bagging (bootstrap sampling) plus random feature subsampling turns a
  collection of individually overfit trees into a forest that generalizes
  much better than any single tree
* how out of bag (OOB) scoring gives you a free validation estimate without
  holding out a separate test set
* how per feature impurity reduction gives you an interpretable feature
  importance ranking

## What is in the repo

| File | Purpose |
| --- | --- |
| `decision_tree.py` | `DecisionTreeClassifier`, plus `gini` / `entropy` impurity functions |
| `random_forest.py` | `RandomForestClassifier` (bagging + feature subsampling + majority vote + OOB score) |
| `metrics.py` | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report` |
| `datasets.py` | Synthetic dataset generators: `make_blobs`, `make_moons`, `train_test_split` |
| `benchmark.py` | Runnable script comparing a deep tree, a shallow tree, and a forest on an easy and a hard dataset |
| `tests/` | 29 unit tests covering impurity math, splitting, overfitting behavior, ensembling, OOB scoring, and metrics |

## How to run it

Requires Python 3.9+ and numpy.

```bash
pip install -r requirements.txt

# Run the test suite
python -m unittest discover -s tests -v

# Run the benchmark / demo
python benchmark.py
```

Minimal usage example:

```python
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier
from datasets import make_moons, train_test_split
from metrics import accuracy_score

X, y = make_moons(n_samples=400, noise=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("tree accuracy:", accuracy_score(y_test, tree.predict(X_test)))

forest = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=0)
forest.fit(X_train, y_train)
print("forest accuracy:", accuracy_score(y_test, forest.predict(X_test)))
print("forest OOB estimate:", forest.oob_score(X_train, y_train))
```

Sample `benchmark.py` output (numbers will vary slightly by machine but the
qualitative story will not):

```
=== moons (hard, non-linear boundary) (n=400, features=2, classes=2) ===
DecisionTree (unbounded)     acc=0.942 precision=0.942 recall=0.941 f1=0.942 | depth=7 leaves=16
DecisionTree (max_depth=3)   acc=0.892 precision=0.895 recall=0.890 f1=0.891 | depth=3 leaves=6
RandomForest (50 trees)      acc=0.958 precision=0.958 recall=0.959 f1=0.958 | oob_acc=0.961
```

The forest's OOB accuracy (computed only from predictions on rows each tree
never saw during training) tracks the held out test accuracy closely,
which is exactly what OOB scoring is supposed to demonstrate.

## Design decisions

* **Splitting is exhaustive per feature, not histogram based.** For each
  candidate feature the algorithm sorts by that feature's values and
  evaluates every midpoint between consecutive distinct values as a
  threshold. This is `O(n log n)` per feature per node rather than an
  approximate binning approach, which keeps the results exact and easy to
  unit test at the cost of some speed on very large datasets.
* **Impurity is pluggable.** `gini` and `entropy` are both implemented as
  plain functions and selected via a `criterion` string, so adding a third
  criterion (e.g. misclassification error) only means adding one function
  and one dictionary entry.
* **The forest reuses the tree, not a copy of its logic.** `RandomForestClassifier`
  drives `DecisionTreeClassifier` through its existing `max_features`
  parameter for feature subsampling; there is no duplicated splitting code
  between the two classes.
* **OOB scoring re-derives bagging indices deterministically.** Rather than
  storing an `(n_samples, n_estimators)` boolean membership matrix, `oob_score`
  replays the same `RandomState` sequence used during `fit` to figure out
  which rows were out of bag for each tree. This keeps memory flat with
  respect to the number of trees.
* **Testing strategy.** The 29 tests split into three groups: closed form
  checks against hand computed impurity values (e.g. a 50/50 binary split
  must have gini exactly 0.5 and entropy exactly 1.0 bit), behavioral checks
  on synthetic data (perfectly separable clusters must be classified
  perfectly, `max_depth` must be respected, an unconstrained tree must grow
  deeper than a depth-limited one), and ensemble level checks (a forest must
  not do dramatically worse than a single tree on noisy data, OOB accuracy
  must land in a sane range, `bootstrap=False` must reject `oob_score`).
