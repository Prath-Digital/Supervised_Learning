# Self Exercise - 6.5

## Q.1

**Question**: What is class imbalance? Define the Imbalance Ratio (IR) and give three real-world examples of severely imbalanced datasets.

**Class Imbalance** occurs when the distribution of classes in a dataset is not uniform — one class (majority) has significantly more samples than the other(s) (minority).

**Imbalance Ratio (IR)** = Number of samples in majority class / Number of samples in minority class.

**Real-world examples:**
1. Fraud Detection (0.1%–1% fraud cases)
2. Medical Diagnosis (e.g., rare disease detection — <1% positive cases)
4. Credit Card Default Prediction (usually ~5-10% default)

---

## Q.2

**Question**: Why does a classifier trained on imbalanced data tend to predict the majority class for everything? Which metric exposes this failure that accuracy cannot?

Because the model minimizes overall error, and predicting the majority class gives the lowest loss on imbalanced data.

**Metric that exposes this**: **Recall** (especially for the minority class), **F1-score**, **Precision-Recall AUC**, or **Matthews Correlation Coefficient (MCC)**. Accuracy is misleading as it can be >99% by just predicting the majority class.

---

## Q.3

**Question**: What is Random Under-Sampling? State its main advantage and its main disadvantage.

**Random Under-Sampling**: Randomly removes samples from the majority class to balance the dataset.

**Advantage**: Fast and reduces training time.
**Disadvantage**: Can discard potentially useful information from the majority class, leading to underfitting.

---

## Q.4

**Question**: What is Random Over-Sampling? State its main advantage and its main disadvantage.

**Random Over-Sampling**: Randomly duplicates samples from the minority class until class distribution is balanced.

**Advantage**: No information loss from majority class.
**Disadvantage**: Increases risk of overfitting and increases training time/memory usage.

---

## Q.5

**Question**: What is a Tomek Link? Describe how Tomek Link removal cleans the decision boundary.

A **Tomek Link** is a pair of nearest neighbors from different classes that are each other's closest neighbors.

Removing Tomek Links cleans the decision boundary by eliminating noisy or borderline examples that are close to the opposite class, making the boundary clearer and more robust.

---

## Q.6

**Question**: Write the SMOTE algorithm step by step. What is the role of the interpolation parameter λ?

**SMOTE Steps**:

1. For each minority sample `x_i`:
2. Find its `k` nearest neighbors within the minority class.
4. Randomly select one neighbor `x_nn`.
6. Generate synthetic sample: `x_new = x_i + λ * (x_nn - x_i)`, where `λ ∈ [0, 1]`.

**Role of λ**: Controls the position of the new synthetic sample along the line segment joining `x_i` and `x_nn`. Usually sampled uniformly from [0,1].

---

## Q.7

**Question**: Why does SMOTE generate samples only within the convex hull of the minority class? What is the implication for a minority class with multiple sub-clusters?

SMOTE interpolates between existing minority samples, so all synthetic points lie inside or on the boundary of the **convex hull** of the minority class.

**Implication for multiple sub-clusters**: SMOTE may not generate samples between distant sub-clusters, potentially failing to bridge gaps between them.

---

## Q.8

**Question**: What is the difference between SMOTE and ADASYN? Which one concentrates synthetic generation near the decision boundary and why?

**SMOTE**: Generates equal number of synthetic samples for every minority instance.

**ADASYN**: Adaptively generates more synthetic samples for minority instances that are harder to learn (more majority neighbors).

**ADASYN** concentrates generation near the decision boundary because it focuses on difficult-to-classify examples.

---

## Q.9

**Question**: Write the ADASYN difficulty ratio formula. What does r_i = 1.0 for a minority sample tell you about its neighbourhood?

**Difficulty Ratio**:
`r_i = Δ_i / K`

where `Δ_i` = number of majority class neighbors among the K nearest neighbors of minority sample `x_i`.

If `r_i = 1.0`, **all** K nearest neighbors belong to the majority class → the sample is completely surrounded by majority class (very difficult).

---

## Q.10

**Question**: What is the KNN Imputer? How does it differ from mean imputation? Why is it better for imbalanced datasets?

**KNN Imputer**: Replaces missing values using the mean/median of the K nearest neighbors (based on other features).

**Difference from Mean Imputation**: Uses local similarity rather than global mean.

**Better for imbalanced data** because it respects local structure and class distributions better than global statistics.

---

## Q.11

**Question**: Why must SMOTE and ADASYN be applied only to the training set and not before the train-test split? What specific type of error does applying them before the split cause?

Applying them before the split causes **data leakage** — synthetic samples created using test set information leak into training.

This leads to overly optimistic performance estimates that don't generalize.

---

## Q.12

**Question**: Name three hyperparameters of SMOTE and explain the effect of each on the synthetic data generated.

1. **`k_neighbors`**: Number of nearest neighbors used for interpolation. Higher → more diverse but potentially noisier samples.
2. **`sampling_strategy`**: Controls how many samples to generate (e.g., 'auto', float, dict).
4. **`random_state`**: Controls reproducibility of synthetic sample generation.

---

## Q.13

**Question**: What is Borderline-SMOTE? How does it differ from standard SMOTE in which samples it targets for synthesis?

**Borderline-SMOTE**: Only generates synthetic samples for minority instances that are near the decision boundary (have many majority neighbors).

It focuses on "danger" samples rather than safe interior points (unlike standard SMOTE).

---

## Q.14

**Question**: In sklearn/imblearn, which class implements SMOTE? Write the minimal code to apply SMOTE inside a Pipeline with a LogisticRegression classifier.

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42))
])
```

---

## Q.15

**Question**: Compare the following four strategies for handling imbalanced data: (1) do nothing, (2) class_weight='balanced', (3) SMOTE, (4) ADASYN. State one scenario where each is the best choice.

1. **Do nothing**: Best when imbalance is mild and model is robust (e.g., tree ensembles).
2. **class_weight='balanced'**: Best for quick baseline, especially with linear models.
4. **SMOTE**: Good when minority class is well-clustered and you want more samples.
6. **ADASYN**: Best when minority class has varying difficulty (many borderline points).

---

## Q.16

**Question**: A fraud detection model achieves 99.5% accuracy, Recall=0.05, and AUC=0.72 on a dataset with 0.5% fraud. Diagnose the problem. List three techniques from this lab that could improve Recall, and explain the trade-off each introduces.

**Diagnosis**: Model is biased toward the majority class (non-fraud), correctly predicting almost everything as non-fraud → high accuracy but terrible Recall.

**Techniques to improve Recall**:
1. **SMOTE/ADASYN** — Increases minority representation (trade-off: risk of overfitting).
2. **class_weight='balanced'** — Penalizes majority errors more (trade-off: may reduce precision).
4. **Threshold tuning** (lower decision threshold) — Increases Recall (trade-off: decreases precision).

---

