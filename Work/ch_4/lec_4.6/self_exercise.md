# Self Exercise - 4.6

## Q1
**What does the oob_score parameter do in sklearn's RandomForestRegressor, and why is it useful?**

**Answer:** It enables Out-of-Bag (OOB) evaluation using samples not included in each tree's bootstrap sample, providing an internal estimate of generalization performance without a separate validation set.

## Q2
**How is the OOB error computed?**

**Answer:** For each sample, predictions are averaged only from trees where that sample was out-of-bag. The resulting prediction is compared to the true target to compute OOB error.

## Q3
**Effect of max_features=1.0?**

**Answer:** All features are considered at every split. This makes Random Forest behave similarly to Bagging and increases tree correlation.

## Q4
**Effect of increasing n_estimators?**

**Answer:** Test error typically decreases and then plateaus. Variance decreases while bias remains nearly unchanged.

## Q5
**max_depth=None vs max_depth=3?**

**Answer:** Deep trees have lower bias and higher variance. Shallow trees have higher bias and lower variance.

## Q6
**Feature importance in Random Forest?**

**Answer:** Importance is computed from the average impurity/RSS reduction contributed by each feature across all trees.

## Q7
**Limitations of impurity-based feature importance?**

**Answer:** Biased toward high-cardinality features and can be misleading when features are correlated.

## Q8
**What is Permutation Importance?**

**Answer:** Measures performance degradation after randomly shuffling a feature. It often provides a more reliable estimate of feature usefulness.

## Q9
**Low train MSE, high test MSE?**

**Answer:** Overfitting. Possible fixes:
- Decrease max_depth
- Increase min_samples_leaf
- Increase min_samples_split

## Q10
**High train MSE, high test MSE?**

**Answer:** Underfitting. Possible fixes:
- Increase max_depth
- Decrease min_samples_leaf
- Decrease min_samples_split

## Q11
**Complexity of Random Forest vs single tree?**

**Answer:** A Random Forest with B trees is roughly B times more expensive than a single tree, though training can be parallelized.

## Q12
**Random Forest vs XGBoost/Gradient Boosting?**

**Answer:** Random Forest is easier to tune and more robust. XGBoost often achieves higher accuracy but requires more tuning and sequential training.
