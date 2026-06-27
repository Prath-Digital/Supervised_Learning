# Self Exercise - 6.9

---

## Q.1

**Question**: What is the difference between a model parameter and a hyperparameter? Give two examples of each for a RandomForestClassifier.

**Answer**:

**Model Parameters** are learned from the training data during the fitting process.
- Examples for RandomForestClassifier: `n_estimators` (number of trees - actually a hyperparameter), wait, correction:
  - True model parameters: The internal decision rules (splits) in each tree, feature thresholds, leaf values, etc.

**Hyperparameters** are set before training and control the learning process.
- Examples for RandomForestClassifier:
  1. `n_estimators` - number of trees in the forest.
  2. `max_depth` - maximum depth of each tree.
  3. `min_samples_split`, `criterion` (gini/entropy), `max_features`.

---

## Q.2

**Question**: Why can you not use test data to select hyperparameters? What is the correct evaluation procedure?

**Answer**:

Using test data for hyperparameter tuning leads to **data leakage** and overly optimistic performance estimates. The test set should only be used once at the very end for final evaluation.

**Correct procedure**:
1. Split data → Train + Validation (or use cross-validation on train set).
2. Tune hyperparameters using only the training data + validation/cross-validation.
3. Evaluate the final model on the untouched test set.

---

## Q.3

**Question**: Explain k-fold cross-validation. Why is k=5 the standard default, and what are the trade-offs of using k=2 vs k=10?

**Answer**:

**k-fold CV**: Data is split into k equal parts. Model is trained k times, each time using k-1 folds for training and 1 fold for validation. Final score is the average.

- **k=5**: Good balance between bias and variance + computational cost.
- **k=2**: High variance in score estimate, less reliable.
- **k=10**: Lower bias, but higher computational cost (more model fits).

---

## Q.4

**Question**: What is Grid Search CV? Write the formula for the total number of model fits given a grid of size G and k folds.

**Answer**:

**Grid Search CV** exhaustively tries every possible combination of hyperparameters in a defined grid.

**Total model fits** = `G × k`
where G = number of combinations in the hyperparameter grid.

---

## Q.5

**Question**: What is Random Search CV? Why does it outperform Grid Search when only a few hyperparameters strongly affect model performance?

**Answer**:

**Random Search CV** samples random combinations from the hyperparameter space (instead of all combinations).

It often outperforms Grid Search because:
- Better exploration of high-dimensional spaces.
- When a few hyperparameters matter most, it is more likely to find good values faster.
- More efficient when some parameters have little impact.

---

## Q.6

**Question**: Explain the Bergstra & Bengio (2012) argument: why does random sampling cover a low-dimensional effective subspace better than a grid?

**Answer**:

Bergstra & Bengio showed that in high-dimensional hyperparameter spaces, **most dimensions have little effect**. Random search explores more unique values along each dimension, while grid search wastes evaluations on many irrelevant combinations.

---

## Q.7

**Question**: What does cv_results_["std_test_score"] tell you? When would you prefer a model with lower mean score but lower standard deviation?

**Answer**:

`std_test_score` shows the **variability** of the model's performance across CV folds.

Prefer lower mean + lower std when:
- You need stable/reliable performance (e.g., production systems).
- Avoiding models that overfit certain folds.

---

## Q.8

**Question**: What is the risk of evaluating multiple tuned models on the test set and picking the best? What is the correct alternative?

**Answer**:

Risk: **Optimistic bias** / information leakage from test set → overestimation of true performance.

**Correct approach**: Use nested cross-validation or keep a completely separate final test set used only once.

---

## Q.9

**Question**: You run GridSearchCV with n_estimators ∈ {50, 100, 200}, max_depth ∈ {None, 5, 10}, cv=5. How many total model fits does this require? Show your calculation.

**Answer**:

Number of combinations = 3 (n_estimators) × 3 (max_depth) = **9**

Total fits = 9 × 5 (folds) = **45 model fits**.

---

## Q.10

**Question**: What is nested cross-validation? How does it differ from flat cross-validation, and when should you use it?

**Answer**:

**Nested CV** has an inner loop for hyperparameter tuning and an outer loop for performance estimation.

- Flat CV: Hyperparameter tuning + evaluation on same data → biased.
- Nested CV: Unbiased performance estimate of the full pipeline.

Use when you need a reliable estimate of how well your *tuning process* performs.

---

## Q.11

**Question**: What scoring metric would you use in GridSearchCV for an imbalanced binary classification problem (5% positive)? Justify your choice.

**Answer**:

Recommended: **`roc_auc`** or **`average_precision`** (PR-AUC).

Reason: Accuracy is misleading with heavy imbalance. ROC-AUC is robust; PR-AUC is better when positive class is rare.

---

## Q.12

**Question**: How does RandomizedSearchCV sample from a continuous distribution such as scipy.stats.uniform(0.1, 0.9)? Why is this more powerful than a fixed grid of three values?

**Answer**:

It samples continuously (e.g., any float between 0.1 and 0.9).

**Advantage**: Much better coverage of the space. A fixed grid (e.g., [0.1, 0.5, 0.9]) might miss the optimal value entirely.

---

## Q.13

**Question**: A Grid Search finds best params at the boundary of your grid (e.g., max_depth=10, which is the maximum in your grid). What does this tell you, and what should you do next?

**Answer**:

It suggests the optimum lies **outside** your current grid.

**Next step**: Expand the grid (e.g., try `max_depth=15, 20, None`) and search again.

---

## Q.14

**Question**: Write the sklearn code to: (1) define a param distributions dict with randint and uniform distributions, (2) run RandomizedSearchCV with n_iter=100, cv=5, scoring="roc_auc", (3) print best_params_ and best_score_.

**Answer**:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': randint(2, 20),
    'max_features': uniform(0.1, 0.9)
}

search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='roc_auc',
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best score:", search.best_score_)
```

---

