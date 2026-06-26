# Self Exercise - 6.4

## Q.1

**Question**: What are the two axes of the ROC curve? Write their full names and formulas.

**Answer**:

**True Positive Rate (TPR)** on the Y-axis and **False Positive Rate (FPR)** on the X-axis.

- **TPR** (also called Sensitivity or Recall) = TP / (TP + FN)
- **FPR** = FP / (FP + TN)

Where:
- TP = True Positives
- FN = False Negatives
- FP = False Positives
- TN = True Negatives

---

## Q.2

**Question**: Why does the ROC curve always start at (0, 0) and end at (1, 1)? Explain in terms of what happens at threshold=1.0 and threshold=0.0.

**Answer**:

At **threshold = 1.0**, the classifier predicts everything as negative. Therefore:
- TP = 0 → TPR = 0
- FP = 0 → FPR = 0
→ Point (0, 0)

At **threshold = 0.0**, the classifier predicts everything as positive. Therefore:
- TP = all actual positives → TPR = 1
- FP = all actual negatives → FPR = 1
→ Point (1, 1)

---

## Q.3

**Question**: What does the random-chance diagonal on the ROC plot represent? What classifier produces exactly this line?

**Answer**:

The diagonal line from (0,0) to (1,1) represents **random guessing** (a classifier that has no discriminative power).

A classifier that assigns scores randomly (or predicts the positive class with probability equal to the prevalence) will produce this line.

---

## Q.4

**Question**: Write the Trapezoid Rule formula for computing AUC. Describe geometrically what each term represents.

**Answer**:

**AUC using Trapezoidal Rule**:

$$
AUC = \sum_{i=1}^{n-1} \frac{(FPR_{i+1} - FPR_i) \times (TPR_i + TPR_{i+1})}{2}
$$

- $(FPR_{i+1} - FPR_i)$ = width of the trapezoid (horizontal segment)
- $(TPR_i + TPR_{i+1})/2$ = average height of the two vertical sides

This approximates the area under the ROC curve by summing areas of trapezoids formed between consecutive points.

---

## Q.5

**Question**: State the Wilcoxon–Mann–Whitney probabilistic interpretation of AUC. What does AUC = 0.80 mean in plain English?

**Answer**:

**AUC** is the probability that a randomly chosen **positive** instance is ranked higher than a randomly chosen **negative** instance by the classifier.

**AUC = 0.80** means: There is an 80% chance that the model will assign a higher score to a randomly selected positive example than to a randomly selected negative example.

---

## Q.6

**Question**: What does AUC = 0.5 imply about a classifier? What does AUC = 0.4 imply, and how would you fix it with zero code changes?

**Answer**:

- **AUC = 0.5**: The classifier is no better than random guessing.
- **AUC = 0.4**: The classifier is *worse* than random. It is systematically ranking negatives higher than positives.

**Fix**: Invert the prediction scores (multiply by -1 or do `1 - score`). This turns AUC 0.4 into 0.6.

---

## Q.7

**Question**: Why is AUC threshold-invariant? Why is this an advantage when comparing two classifiers?

**Answer**:

AUC measures the **ranking quality** of the model across **all possible thresholds**, not performance at one specific threshold.

**Advantage**: You can fairly compare two models even if their optimal operating thresholds are different. It focuses purely on discriminative ability.

---

## Q.8

**Question**: On the ROC curve, what causes a step upward (TPR increase)? What causes a step rightward (FPR increase)?

**Answer**:

- **Step upward (↑ TPR)**: The threshold is lowered such that one or more **positive** samples are now classified as positive.
- **Step rightward (→ FPR)**: The threshold is lowered such that one or more **negative** samples are now classified as positive.

---

## Q.9

**Question**: What is the Youden Index? Write the formula and explain how to use it to select the optimal decision threshold from the ROC curve.

**Answer**:

**Youden Index (J)** = TPR + TNR - 1 = Sensitivity + Specificity - 1

Or equivalently: **J = TPR - FPR**

**Usage**: Choose the threshold that **maximizes** the Youden Index. This gives the best balance between sensitivity and specificity (maximizing vertical distance from the diagonal).

---

## Q.10

**Question**: Explain the difference between AUC-ROC and AUC-PR (Precision-Recall AUC). When is AUC-PR more informative?

**Answer**:

- **AUC-ROC**: Plots TPR vs FPR. Good for overall ranking, less affected by class imbalance.
- **AUC-PR**: Plots Precision vs Recall. Focuses on the positive class.

**AUC-PR is more informative** when the dataset is **highly imbalanced** (positive class is rare), as ROC can give overly optimistic results.

---

## Q.11

**Question**: A model achieves AUC-ROC = 0.99 on a dataset with 1% positive samples. Why should this immediately raise a suspicion of data leakage? How would you investigate?

**Answer**:

Such a high AUC on a heavily imbalanced dataset is suspicious because it often indicates **data leakage** (e.g., test set information leaking into training features).

**Investigation steps**:
1. Check for ID columns, timestamps, or features that shouldn't be available at prediction time.
2. Look at feature importance — are any suspiciously perfect separators?
4. Retrain with strict temporal split or proper cross-validation.
6. Inspect prediction scores on known negative samples.

---

## Q.12

**Question**: How does class imbalance affect AUC-ROC vs Accuracy? Give a numerical example showing that AUC-ROC is more robust to imbalance.

**Answer**:

**Accuracy** becomes misleading in imbalanced data (e.g., 99% negative → predicting all negative gives 99% accuracy).

**AUC-ROC** is much more robust as it evaluates ranking ability.

**Example**:
- 100 positives, 9900 negatives
- Dummy model predicts all negative: Accuracy = 99%, AUC-ROC = 0.5
- Good model: Accuracy = 95%, AUC-ROC = 0.95

---

## Q.13

**Question**: In sklearn, which function computes AUC-ROC directly from probability scores? Write the exact function call.

**Answer**:

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_true, y_score)  # y_score = probability of positive class
```

---

## Q.14

**Question**: A binary classifier is used for medical diagnosis. The business requires Recall ≥ 0.96. Describe step by step how you would use the ROC curve to select the operating threshold, and state what happens to Precision and FPR at that threshold.

**Answer**:

**Step-by-step**:
1. Generate ROC curve using `roc_curve(y_true, y_scores)`.
2. Find the smallest threshold where **TPR (Recall) ≥ 0.95**.
4. Use that threshold for final predictions.

**Consequences**:
- **Precision** will usually drop (more false positives).
- **FPR** will increase (you accept more false alarms to catch almost all true cases).

---


---
*Generated by Python script*