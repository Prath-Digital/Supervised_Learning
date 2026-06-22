# Self Exercise - 4.2

## Q1. What is cross-validation and why is it used instead of a single train/test split?

### Answer
Cross-validation is a resampling technique used to evaluate a machine learning model's performance on unseen data.

Instead of using only one train/test split, the dataset is divided into multiple subsets (folds), and the model is trained and evaluated multiple times.

### Why use Cross-Validation?
- Provides a more reliable estimate of model performance.
- Reduces dependence on a particular train/test split.
- Uses data more efficiently.
- Helps detect overfitting and underfitting.
- Useful when datasets are small.

### Limitation of Single Train/Test Split
A single split may accidentally produce:
- An easy test set (overly optimistic results)
- A difficult test set (overly pessimistic results)

Cross-validation averages performance across multiple splits, giving a more robust estimate.

---

## Q2. Describe the K-Fold Cross-Validation algorithm step by step in your own words.

### Answer

K-Fold Cross-Validation works as follows:

### Step 1
Choose a value of K (commonly 5 or 10).

### Step 2
Split the dataset into K equal-sized folds.

### Step 3
Select one fold as the validation set.

### Step 4
Use the remaining K−1 folds as the training set.

### Step 5
Train the model.

### Step 6
Evaluate it on the validation fold.

### Step 7
Repeat the process until every fold has served once as validation data.

### Step 8
Compute the average of all validation scores.

### Formula

Average Score:

\[
CV = \frac{1}{K}\sum_{i=1}^{K} Score_i
\]

The average score becomes the final performance estimate.

---

## Q4. What is the trade-off between choosing a small K (e.g., K=2) versus a large K (e.g., K=10)?

### Answer

### Small K

Example: K = 2

Advantages:
- Faster computation
- Less training time

Disadvantages:
- Higher bias
- Less reliable performance estimate

---

### Large K

Example: K = 10

Advantages:
- Lower bias
- More accurate estimate
- Better use of data

Disadvantages:
- More computation
- Longer training time

---

### Summary

| K Value | Bias | Variance | Computation |
|----------|----------|----------|----------|
| Small K | High | Low | Low |
| Large K | Low | Moderate | High |

K = 5 and K = 10 are most commonly used.

---

## Q4. What is Leave-One-Out Cross-Validation (LOOCV)? When is it most appropriate to use?

### Answer

LOOCV is a special case of K-Fold Cross-Validation where:

\[
K = n
\]

where n is the number of observations.

For each iteration:
- One sample is used for testing.
- Remaining n−1 samples are used for training.

This process repeats for every observation.

### Advantages
- Maximum use of available data.
- Low bias.

### Appropriate When
- Dataset is very small.
- Every observation is valuable.
- Computational cost is manageable.

---

## Q5. Why does LOOCV have high variance? How does this compare to K-Fold with moderate K?

### Answer

In LOOCV:

- Training sets are nearly identical.
- Test results depend heavily on one observation.

A single unusual observation can significantly affect the score.

Therefore:

- Bias is low.
- Variance is high.

---

### K-Fold (K=5 or K=10)

Each validation fold contains multiple observations.

Benefits:
- More stable estimates.
- Lower variance.
- Better practical performance.

---

### Comparison

| Method | Bias | Variance |
|----------|----------|----------|
| LOOCV | Very Low | High |
| K=5 or K=10 | Slightly Higher | Lower |

---

## Q6. What is Stratified K-Fold? How does it differ from standard K-Fold and when should you use it?

### Answer

Stratified K-Fold preserves class proportions in every fold.

Example:

Dataset:

- 90% Class A
- 10% Class B

Each fold maintains approximately the same ratio.

### Standard K-Fold

Randomly splits data without considering class labels.

This can create imbalanced folds.

### Stratified K-Fold

Maintains class distribution across all folds.

### Use When

- Classification problems
- Imbalanced datasets
- Rare classes exist

It generally produces more reliable evaluation metrics.

---

## Q7. Why is shuffling data forbidden in Time Series Cross-Validation? What is the consequence of shuffling?

### Answer

Time series data contains chronological order.

Future observations must never influence past predictions.

Example:

2022 → 2023 → 2024

Shuffling destroys temporal structure.

### Consequences

- Data leakage
- Unrealistically high performance
- Invalid evaluation

The model indirectly gains information from the future.

Therefore, chronological ordering must always be preserved.

---

## Q8. What is an expanding window in Time Series Split? How does it differ from a sliding window?

### Answer

### Expanding Window

Training data grows over time.

Example:

Train: [1]

Test: [2]

Train: [1,2]

Test: [3]

Train: [1,2,3]

Test: [4]

The training set continually expands.

---

### Sliding Window

Training window remains fixed size.

Example:

Train: [1,2,3]

Test: [4]

Train: [2,3,4]

Test: [5]

Train: [3,4,5]

Test: [6]

Old observations are discarded.

---

### Difference

| Method | Training Size |
|----------|----------|
| Expanding Window | Increases |
| Sliding Window | Fixed |

---

## Q9. What is data leakage in the context of cross-validation? Give one concrete example of how it occurs.

### Answer

Data leakage occurs when information from the validation or test set influences the training process.

This causes overly optimistic performance estimates.

### Example

Suppose feature scaling is performed before cross-validation:

Incorrect:

1. Scale entire dataset.
2. Perform K-Fold.

The scaler has already seen validation data.

Correct:

1. Split fold.
2. Fit scaler only on training fold.
4. Transform training and validation separately.

This prevents leakage.

---

## Q10. What metric would you use to aggregate results across CV folds, and how would you report uncertainty?

### Answer

The metric depends on the problem.

### Regression

Common metrics:

- RMSE
- MAE
- R²

### Classification

Common metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

### Aggregation

Compute mean score:

\[
\bar{x} = \frac{1}{K}\sum_{i=1}^{K}x_i
\]

---

### Uncertainty

Report standard deviation:

\[
s = \sqrt{\frac{\sum(x_i-\bar{x})^2}{K-1}}
\]

Example:

Accuracy = 92.4% ± 1.8%

This indicates variability across folds.

---

## Q11. Can cross-validation be used for hyperparameter tuning? If yes, what is the risk and how do you mitigate it?

### Answer

Yes.

Cross-validation is commonly used to select hyperparameters.

Example:

- λ in Ridge/Lasso
- Learning Rate
- Tree Depth
- Number of Neighbors

### Risk

Repeated tuning can overfit to validation folds.

The selected hyperparameters may perform well only on those folds.

### Mitigation

Use:

- Nested Cross-Validation
- Separate holdout test set

This provides an unbiased estimate of performance.

---

## Q12. What is nested cross-validation? When and why would you use it over standard K-Fold?

### Answer

Nested Cross-Validation uses two loops.

### Inner Loop

Performs hyperparameter tuning.

### Outer Loop

Evaluates final model performance.

Structure:

Outer Fold
    └── Inner Cross-Validation
            └── Hyperparameter Search

### Why Use It?

Standard K-Fold can produce optimistic results when tuning hyperparameters.

Nested Cross-Validation separates:

- Model selection
- Performance evaluation

This produces a less biased estimate.

### Use When

- Comparing models
- Academic research
- Small datasets
- Extensive hyperparameter tuning

### Advantages

- Reduced selection bias
- More trustworthy performance estimates
- Better scientific rigor

