# Self Exercise - 6.8

---

## Q.1 How does a Random Forest Classifier aggregate tree predictions? Explain both hard voting and soft voting with formulas.

**Hard Voting (Majority Vote)**  
Each tree predicts a class. The final prediction is the class with the most votes:  
$$ \hat{y} = \text{mode}\{ \hat{y}_1, \hat{y}_2, \dots, \hat{y}_B \} $$  
where $B$ is the number of trees.

**Soft Voting (Probability Averaging)**  
Averages the predicted probabilities from all trees and picks the class with highest average probability:  
$$ P(y=k | x) = \frac{1}{B} \sum_{b=1}^{B} P_b(y=k | x) $$  
$$ \hat{y} = \arg\max_k P(y=k | x) $$  

Soft voting usually performs better as it considers confidence levels.

---

## Q.2 What is the OOB (Out-of-Bag) accuracy in a Random Forest Classifier? Why is it a valid generalisation estimate without a separate validation set?

**OOB Accuracy** is the accuracy computed using the predictions made on samples that were **not** included in the bootstrap sample for each tree (approximately 37% of data per tree).  

It acts as a built-in cross-validation score. Because each tree is evaluated on data it hasn't seen during training, OOB provides an **unbiased estimate** of generalization performance without needing a separate hold-out validation set.

---

## Q.3 Why does Random Forest Classifier almost always outperform a single Decision Tree Classifier? Use the bias-variance decomposition to explain.

A single deep Decision Tree has **low bias** but **very high variance** (overfitting).  

Random Forest reduces variance dramatically through:
- Bagging (Bootstrap Aggregating)
- Random feature selection at each split

The ensemble maintains roughly the same low bias while significantly lowering variance → much better generalization.

---

## Q.4 What is the role of max_features in a Random Forest Classifier? How does it reduce tree correlation and improve the ensemble?

`max_features` controls how many features are considered when looking for the best split at each node.  

- Default for classification: `'sqrt'` (√n_features)

By limiting features, different trees see different subsets of features → **decorrelates** the trees.  
Less correlated trees → lower ensemble variance when averaged.

---

## Q.5 What is the recommended default value of max_features for classification? Why is it 'sqrt' rather than using all features?

**Recommended default: `'sqrt'`** (square root of total features).  

Using all features would make trees highly correlated (similar to plain bagging). Using `sqrt` introduces randomness in feature selection, which is crucial for the "strength + diversity" trade-off that makes Random Forests powerful.

---

## Q.6 What does oob_score=True compute? Under what conditions would OOB accuracy be an unreliable estimate of test accuracy?

`oob_score=True` computes the Out-of-Bag score automatically during training.  

**OOB can be unreliable when:**
- Dataset is very small
- Strong class imbalance (without `class_weight`)
- Data has temporal structure (time series)
- Very few trees (`n_estimators` too low)

---

## Q.7 How is feature importance computed in a Random Forest Classifier? Name two limitations of this impurity-based importance measure.

Computed as the **total decrease in node impurity** (usually Gini) brought by splits on that feature, averaged across all trees.  

**Limitations:**
1. Biased towards high-cardinality features (features with many unique values)
2. Does not handle correlated features well (importance gets split among them)

---

## Q.8 What is Permutation Importance? How does it differ from impurity-based importance, and when should you prefer it?

**Permutation Importance** measures the drop in model performance (accuracy, AUC, etc.) when a feature’s values are randomly shuffled.  

**Advantages over impurity-based:**
- Model-agnostic
- Less biased toward high-cardinality features
- Captures interaction effects better

**Prefer it** for reliable feature ranking and model interpretation.

---

## Q.9 A Random Forest Classifier has train accuracy = 1.00 and OOB accuracy = 0.73. Is this overfitting? What three hyperparameter changes would you make and in which direction?

**Yes, this indicates overfitting.**  

**Recommended changes:**
1. **Increase** `min_samples_leaf` or `min_samples_split` (stronger regularization)
2. **Decrease** `max_depth` (shallower trees)
3. Slightly **increase** `max_features` or tune `n_estimators`

---

## Q.10 Increasing n_estimators from 10 to 1000 in a Random Forest never increases bias. Explain why, using the ensemble prediction formula.

The ensemble prediction is:  
$$ \hat{y}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x) $$  

Increasing the number of trees $B$ only adds more terms to the average. The bias of the ensemble remains approximately equal to the bias of the individual trees. Only the variance decreases (by the law of large numbers).

---

## Q.11 Compare soft voting and hard voting in a Random Forest. In which scenario does soft voting provide a significantly better prediction?

- **Hard Voting**: Majority class vote  
- **Soft Voting**: Average of predicted probabilities  

**Soft voting is significantly better when:**
- Trees produce well-calibrated probabilities
- You need better ranking performance (e.g., AUC)
- Dealing with imbalanced classes

---

## Q.12 You are working with an imbalanced dataset (5% positive). Which hyperparameter of RandomForestClassifier directly addresses class imbalance without external resampling?

**`class_weight='balanced'`** or **`class_weight='balanced_subsample'`**  

This automatically assigns weights inversely proportional to class frequencies.

---

## Q.13 How does Random Forest Classifier handle multiclass classification? Describe the voting mechanism for 3 or more classes.

Random Forest uses **one-vs-rest** internally for multiclass.  

- **Hard voting**: Class with the highest number of votes across all trees wins.  
- **Soft voting**: Probabilities are averaged per class across trees; class with highest average probability wins.

---

## Q.14 Write the sklearn code to: (1) fit a RandomForestClassifier with oob_score, (2) select a threshold from the ROC curve to achieve Recall ≥ 0.90 on the positive class, (3) apply that threshold to predict_proba output to generate final labels.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import numpy as np

# 1. Fit with OOB score
rf = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)

# 2. Get probabilities & find threshold for Recall >= 0.90
probs = rf.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, probs)

# Find the smallest threshold where recall >= 0.90
idx = np.where(tpr >= 0.90)[0][0]
best_threshold = thresholds[idx]

print(f"Best threshold: {best_threshold:.4f}")

# 3. Apply threshold
y_pred = (probs >= best_threshold).astype(int)
```