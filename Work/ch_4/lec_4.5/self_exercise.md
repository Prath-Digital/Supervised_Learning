# Self Exercise - 4.5

## Q1. What is Bootstrap Aggregation (Bagging)? Write the formal prediction rule for a bagged ensemble of regression trees.

### Answer

Bootstrap Aggregation (Bagging) is an ensemble learning technique that reduces variance by training multiple models on different bootstrap samples and averaging their predictions.

For a regression problem with B trees:

\[
\hat{f}_{bag}(x)
=
\frac{1}{B}
\sum_{b=1}^{B}
\hat{f}^{(b)}(x)
\]

where:

- \(B\) = number of trees
- \(\hat{f}^{(b)}(x)\) = prediction from the \(b\)-th tree

The final prediction is the average of all tree predictions.

### Benefits

- Reduces variance
- Improves generalization
- Less prone to overfitting than a single tree

---

## Q2. What is a bootstrap sample? How is it drawn, and what fraction of original samples does it typically omit?

### Answer

A bootstrap sample is created by randomly sampling observations **with replacement** from the original dataset.

Suppose the dataset contains:

\[
n
\]

observations.

A bootstrap sample also contains:

\[
n
\]

observations, but some observations may appear multiple times while others may not appear at all.

### Probability of Omission

The probability that a specific observation is not selected in one draw is:

\[
1-\frac{1}{n}
\]

After \(n\) draws:

\[
\left(1-\frac{1}{n}\right)^n
\]

As \(n \to \infty\):

\[
\left(1-\frac{1}{n}\right)^n
\rightarrow e^{-1}
\approx 0.368
\]

### Result

Approximately:

\[
36.8\%
\]

of observations are omitted from a bootstrap sample.

These become Out-of-Bag samples.

---

## Q4. What is the Out-of-Bag (OOB) error? How is it computed and why is it a valid generalisation estimate?

### Answer

Out-of-Bag (OOB) error is an internal validation estimate used in Random Forests.

### Procedure

For each observation:

1. Identify trees that did not use that observation during training.
2. Collect predictions from those trees.
4. Average the predictions.
4. Compare with the true value.

For regression:

\[
OOB\ Error
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_{i,OOB})^2
\]

### Why It Works

The observation was never seen by those trees during training.

Therefore OOB predictions behave similarly to predictions on unseen data.

### Advantages

- No separate validation set required
- Computationally efficient
- Reliable estimate of test performance

---

## Q4. How does Random Forest extend Bagging? What additional randomisation does it introduce at each split?

### Answer

Bagging creates diversity by using different bootstrap samples.

Random Forest adds another source of randomness.

### At Each Split

Instead of evaluating all features:

\[
p
\]

features,

the algorithm randomly selects:

\[
m
\]

candidate features.

Only those features compete for the split.

### Result

- Less correlation between trees
- More diversity
- Lower ensemble variance
- Better generalization

This feature subsampling is the key difference between Bagging and Random Forest.

---

## Q5. Why does averaging the predictions of multiple trees reduce variance? Derive the variance reduction formula for B uncorrelated trees.

### Answer

Assume:

- Each tree has variance:

\[
\sigma^2
\]

- Trees are independent.

The ensemble prediction is:

\[
\bar{f}
=
\frac{1}{B}
\sum_{b=1}^{B}f_b
\]

Variance:

\[
Var(\bar{f})
=
Var
\left(
\frac{1}{B}
\sum_{b=1}^{B}f_b
\right)
\]

For independent trees:

\[
Var(\bar{f})
=
\frac{1}{B^2}
\sum_{b=1}^{B}\sigma^2
\]

\[
=
\frac{B\sigma^2}{B^2}
\]

\[
=
\frac{\sigma^2}{B}
\]

### Conclusion

Variance decreases proportionally to:

\[
\frac{1}{B}
\]

More trees generally reduce variance.

---

## Q6. Why are Random Forest trees intentionally correlated, and what mechanism controls the degree of correlation?

### Answer

Random Forest trees are not completely independent because:

- Bootstrap samples overlap
- Trees are trained on the same dataset

Some correlation naturally exists.

### Why Correlation Matters

The variance reduction formula improves when correlation decreases.

For correlated trees:

\[
Var(\bar{f})
=
\rho\sigma^2
+
\frac{(1-\rho)\sigma^2}{B}
\]

where:

\[
\rho
\]

is the correlation between trees.

### Controlling Correlation

The parameter:

\[
max\_features
\]

controls correlation.

Smaller values:

- More randomness
- Lower correlation

Larger values:

- Less randomness
- Higher correlation

---

## Q7. What is the role of the hyperparameter m (max_features)? What are the typical recommended values for regression, and why?

### Answer

The hyperparameter:

\[
m = max\_features
\]

determines how many randomly selected features are considered at each split.

### Effects

Small m:

- More diverse trees
- Lower correlation
- Higher randomness

Large m:

- Stronger individual trees
- Higher correlation

### Common Recommendations

For regression:

\[
m \approx \sqrt{p}
\]

or

\[
m \approx \frac{p}{3}
\]

where:

\[
p
\]

is the total number of features.

These values often balance tree strength and diversity.

---

## Q8. What is the bias–variance trade-off for Random Forest? Does averaging trees reduce bias, variance, or both?

### Answer

Random Forest primarily reduces variance.

### Single Deep Tree

- Low bias
- High variance

### Random Forest

- Similar bias
- Much lower variance

### Reason

Averaging stabilizes predictions.

### Summary

| Component | Effect |
|------------|---------|
| Bias | Slight change |
| Variance | Significant reduction |

Thus Random Forest improves generalization mainly through variance reduction.

---

## Q9. Why does a single deep decision tree overfit while a forest of deep trees does not? Explain using bias–variance terms.

### Answer

A deep tree:

- Fits training data extremely well
- Captures noise
- Has high variance

Thus:

\[
Training\ Error \approx 0
\]

but test error can be large.

### Random Forest

Each tree still has:

- Low bias
- High variance

However averaging many trees reduces variance:

\[
Variance_{forest}
<
Variance_{single\ tree}
\]

The result is:

- Low bias
- Lower variance
- Better test performance

---

## Q10. What is Feature Importance in a Random Forest? How is it computed across multiple trees?

### Answer

Feature Importance measures the contribution of a feature to reducing prediction error.

For a split:

\[
Gain
=
RSS_{parent}
-
(RSS_{left}+RSS_{right})
\]

### For Each Tree

1. Compute gain for every split.
2. Sum gains for each feature.

### Across the Forest

Average feature importance across all trees:

\[
Importance_j
=
\frac{1}{B}
\sum_{b=1}^{B}
Importance_{j,b}
\]

Values are normalized so they sum to 1.

---

## Q11. How does increasing the number of trees (n_estimators) affect bias, variance, and computational cost?

### Answer

### Bias

Bias remains nearly unchanged.

The individual trees remain the same type of learner.

### Variance

Variance decreases as more trees are averaged.

Initially:

- Large improvement

Later:

- Diminishing returns

### Computational Cost

Increasing:

\[
n\_estimators
\]

causes:

- More memory usage
- Longer training time
- Longer prediction time

### Summary

| Metric | Effect |
|----------|----------|
| Bias | Nearly Constant |
| Variance | Decreases |
| Cost | Increases |

---

## Q12. What are the key differences between Random Forest Regression and Gradient Boosting Regression in terms of how individual trees are built and combined?

### Answer

### Random Forest

#### Tree Construction

Trees are built independently.

#### Training Style

Parallel.

#### Combination

Predictions are averaged:

\[
\hat{y}
=
\frac{1}{B}
\sum_{b=1}^{B}\hat{y}_b
\]

#### Goal

Reduce variance.

---

### Gradient Boosting

#### Tree Construction

Trees are built sequentially.

#### Training Style

Each new tree corrects previous errors.

#### Combination

Predictions are added:

\[
\hat{y}
=
\sum_{b=1}^{B}
\eta f_b(x)
\]

where:

\[
\eta
\]

is the learning rate.

#### Goal

Reduce bias.

---

## Summary Table

| Property | Random Forest | Gradient Boosting |
|-----------|---------------|------------------|
| Training | Parallel | Sequential |
| Main Goal | Reduce Variance | Reduce Bias |
| Tree Independence | Independent | Dependent |
| Overfitting Risk | Lower | Higher |
| Training Speed | Faster | Slower |
| Hyperparameter Sensitivity | Lower | Higher |
| Interpretability | Moderate | Moderate |

### Key Takeaway

Random Forest is a bagging-based ensemble that primarily reduces variance through averaging many deep trees, while Gradient Boosting builds trees sequentially to reduce bias by correcting previous errors.
