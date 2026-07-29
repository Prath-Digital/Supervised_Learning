# Self Exercise - 11.1

**Q.1 What is Bagging, and what problem does it solve compared to a single Decision Tree?**

**Answer:**  
Bagging (Bootstrap Aggregating) is an ensemble technique that trains multiple models (usually decision trees) on different bootstrap samples of the training data and then aggregates their predictions (by averaging for regression or majority voting for classification).

Compared to a single Decision Tree, Bagging primarily solves the **high-variance** problem. A single deep Decision Tree tends to overfit the training data and produces unstable predictions that change significantly with small changes in the data. By averaging many such trees trained on different samples, Bagging reduces variance while keeping bias relatively low, leading to more stable and accurate models.

---

**Q.2 What is bootstrap sampling — how does sampling with replacement differ from sampling without replacement?**

**Answer:**  
Bootstrap sampling is a resampling technique in which we draw samples of the same size as the original dataset **with replacement**.

- **Sampling with replacement**: After an observation is selected, it is put back into the pool, so the same observation can appear multiple times in the sample. Consequently, some observations from the original data will not appear at all in a given bootstrap sample (these are the Out-of-Bag samples).
- **Sampling without replacement**: Once an observation is selected, it is removed from the pool, so each observation can appear at most once. The resulting sample is just a random subset of the original data.

Bootstrap sampling (with replacement) is the foundation of Bagging and Random Forests.

---

**Q.3 In a Random Forest, how many features are considered at each split, and why is it not all features?**

**Answer:**  
In a Random Forest, at each split of a tree only a random subset of features is considered:

- Classification: typically \(\sqrt{p}\) features (where \(p\) is the total number of features)
- Regression: typically \(p/3\) features

It is **not** all features because using the full feature set would make the trees highly correlated with each other. By restricting the number of candidate features at each split, Random Forest forces the trees to be more diverse (decorrelated). Averaging diverse trees reduces the variance of the ensemble much more effectively than averaging highly similar trees.

---

**Q.4 What is the Out-Of-Bag (OOB) error, and how is it computed without a separate test set?**

**Answer:**  
The Out-Of-Bag (OOB) error is an internal estimate of the generalization error of a Random Forest (or any bagged ensemble).

For each tree, the observations that were **not** included in its bootstrap sample (approximately 37% of the data on average) are called the OOB samples for that tree. These OOB samples are used as a test set for that particular tree.

To compute the overall OOB error:

1. For every training observation, collect the predictions from all trees for which that observation was OOB.
2. Aggregate those predictions (majority vote or average).
3. Compare the aggregated prediction with the true label and compute the error rate (or MSE).

Because every observation is left out of some trees, the OOB error provides a reliable estimate of test-set performance without needing a separate validation/test set.

---

**Q.5 How does Random Forest make a final prediction for regression? How does it differ for classification?**

**Answer:**

- **Regression**: The Random Forest averages the continuous predictions of all individual trees.  
  \[
  \hat{y} = \frac{1}{T} \sum\_{t=1}^{T} \hat{y}\_t
  \]  
  where \(T\) is the number of trees.

- **Classification**: Each tree outputs a class label (or class probabilities). The final prediction is obtained by **majority voting** (the class that receives the most votes) or by averaging the class probabilities across trees and then choosing the class with the highest average probability.

---

**Q.6 What does 'ensemble' mean, and why do ensembles of weak learners often outperform a single strong learner?**

**Answer:**  
An **ensemble** is a collection of multiple models (base learners) whose individual predictions are combined to produce a final prediction.

Ensembles of weak learners often outperform a single strong learner because:

- Individual weak learners (e.g., shallow trees) have high bias or high variance, but their errors tend to be uncorrelated.
- Combining many diverse weak learners reduces both bias and variance through averaging or voting.
- The diversity among models allows the ensemble to capture different aspects of the data, leading to better generalization than a single complex model that may overfit.

---

**Q.7 How does Random Forest reduce variance compared to a single deep Decision Tree?**

**Answer:**  
A single deep Decision Tree has low bias but very high variance (it overfits noise and is sensitive to small changes in the training data).

Random Forest reduces variance by:

1. Training many trees on different bootstrap samples (Bagging).
2. Introducing additional randomness by considering only a random subset of features at each split.

Because the trees are both trained on different data and forced to be diverse, their errors cancel each other out when averaged. The final ensemble therefore has much lower variance while retaining the low bias of deep trees.

---

**Q.8 How does Random Forest reduce variance compared to a single deep Decision Tree?**

**Answer:**  
(Same as Q.7 – the question is repeated in the original list.)

A single deep Decision Tree has low bias but very high variance. Random Forest reduces this variance by averaging many decorrelated trees trained via bagging and random feature selection at each split. The averaging of diverse models cancels out individual tree errors, producing a more stable predictor.

---

**Q.9 Name at least four hyperparameters that control a Random Forest and describe the effect of each.**

**Answer:**

1. **`n_estimators`** (number of trees)
   - More trees usually improve performance and reduce variance, but with diminishing returns and higher computational cost.

2. **`max_features`** (number of features considered at each split)
   - Smaller values increase tree diversity (lower correlation → lower variance) but may increase bias. Larger values make trees more similar.

3. **`max_depth`**
   - Limits how deep each tree can grow. Shallower trees reduce overfitting (lower variance) but may increase bias. Deep trees capture more complex patterns.

4. **`min_samples_split`** / **`min_samples_leaf`**
   - Control the minimum number of samples required to split a node or to be in a leaf. Higher values prevent the trees from creating very specific rules that overfit noise.

Other important ones: `max_leaf_nodes`, `bootstrap` (whether to use bootstrap sampling), `criterion` (Gini, entropy, MSE, etc.).

---

**Q.10 What is the bias-variance tradeoff, and where does Random Forest sit on that spectrum vs a single tree?**

**Answer:**  
The **bias-variance tradeoff** describes the tension between:

- **Bias**: error due to overly simplistic assumptions (underfitting).
- **Variance**: error due to sensitivity to fluctuations in the training data (overfitting).

A single deep Decision Tree typically has **low bias** and **high variance**.

A Random Forest keeps the **low bias** of deep trees while significantly **reducing variance** through bagging and feature randomness. Consequently, Random Forest sits in a more favorable region of the bias-variance spectrum: low bias + low-to-moderate variance, usually resulting in better generalization than a single tree.

---

**Q.11 In what situations might a single Decision Tree outperform a Random Forest?**

**Answer:**  
A single Decision Tree may outperform (or be preferable to) a Random Forest when:

- The dataset is very small – ensembles need enough data to create diverse bootstrap samples.
- Interpretability is critical – a single tree is fully transparent; a forest is a black box.
- The underlying relationship is extremely simple and can be captured by a shallow tree.
- Computational resources or latency constraints are severe (a single tree is much faster to train and predict).
- There is almost no noise and the tree does not overfit (rare in practice).

In most real-world noisy datasets with moderate-to-large size, Random Forest is superior.

---

**Q.12 What are the key differences between Bagging and Boosting as ensemble strategies?**

**Answer:**

| Aspect               | Bagging                          | Boosting                                          |
| -------------------- | -------------------------------- | ------------------------------------------------- |
| Training style       | Parallel (independent models)    | Sequential (each model depends on previous)       |
| Focus                | Reduce variance                  | Reduce bias (and also variance)                   |
| Sample weighting     | Equal weights, bootstrap samples | Higher weight on previously misclassified samples |
| Typical base learner | Deep trees (high variance)       | Weak learners / shallow trees                     |
| Aggregation          | Simple average or majority vote  | Weighted average / vote                           |
| Overfitting risk     | Lower (averaging helps)          | Higher if not regularized                         |
| Famous algorithms    | Random Forest, Bagged Trees      | AdaBoost, Gradient Boosting, XGBoost, LightGBM    |

Bagging builds many independent models and averages them; Boosting builds models sequentially, each correcting the mistakes of the previous ones.
