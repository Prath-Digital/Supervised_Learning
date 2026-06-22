# Self Exercise - 5.3

---

## Q1. What are True Positive (TP), False Positive (FP), True Negative (TN), and False Negative (FN)? Give a one-sentence real-world example of each using a medical test.

**True Positive (TP)**: The model correctly predicts the positive class.  
Example: A medical test correctly identifies that a patient **has** cancer.

**False Positive (FP)**: The model incorrectly predicts the positive class.  
Example: A medical test says a healthy patient **has** cancer (false alarm).

**True Negative (TN)**: The model correctly predicts the negative class.  
Example: A medical test correctly identifies that a patient **does not have** cancer.

**False Negative (FN)**: The model incorrectly predicts the negative class.  
Example: A medical test says a patient with cancer **does not have** it (missed diagnosis).

---

## Q2. Draw a 2×2 confusion matrix and label every cell. Which cells represent correct predictions?

```
                  Actual Positive     Actual Negative
Predicted Positive       TP                  FP
Predicted Negative       FN                  TN
```

**Correct predictions** are the **diagonal cells**: **TP** and **TN**.

---

## Q4. Write the formula for Accuracy. Why can a 99% accurate model be completely useless?

**Accuracy** = $\frac{TP + TN}{TP + TN + FP + FN}$

A model can be 99% accurate but useless in highly **imbalanced datasets**.  
Example: If only 1% of people have a rare disease, a model that always predicts "No disease" will be ~99% accurate but completely fails to detect any actual cases.

---

## Q5. What is Type I Error? Give its name, formula, and one scenario where it is the more dangerous mistake.

**Type I Error** (False Positive) = $\frac{FP}{TP + FP}$

This is a **false alarm**.  
**Dangerous scenario**: Convicting an innocent person in criminal justice.

---

## Q5. What is Type II Error? Give its name, formula, and one scenario where it is the more dangerous mistake.

**Type II Error** (False Negative) = $\frac{FN}{TP + FN}$

This is a **missed detection**.  
**Dangerous scenario**: Failing to detect cancer in a patient, allowing the disease to progress untreated.

---

## Q6. Explain the trade-off between Type I and Type II Errors. What happens to each when you raise the decision threshold from 0.5 to 0.9?

Raising the threshold (e.g., 0.5 → 0.9) makes the model **more conservative** in predicting positive:

- **Type I Error (FP)** **decreases** → fewer false alarms
- **Type II Error (FN)** **increases** → more missed positives

This is the classic **Precision-Recall trade-off**.

---

## Q7. Write the formula for Precision and explain what it measures. When is maximising Precision the right goal?

**Precision** = $\frac{TP}{TP + FP}$

**Measures**: Of all instances the model predicted as positive, how many are **actually** positive?

**Best when**: False positives are very costly (e.g., spam detection, fraud alerts, recommending expensive treatments).

---

## Q8. Write the formula for Recall (Sensitivity / TPR) and explain what it measures. When is maximising Recall the right goal?

**Recall (Sensitivity / TPR)** = $\frac{TP}{TP + FN}$

**Measures**: Of all **actual** positive cases, how many did the model correctly identify?

**Best when**: Missing a positive case is dangerous (e.g., cancer screening, disease outbreak detection, security threats).

---

## Q9. Derive the F1 Score as the harmonic mean of Precision and Recall. Why is the harmonic mean more appropriate than the arithmetic mean here?

**F1 Score** = $2 \times \frac{Precision \times Recall}{Precision + Recall}$

The **harmonic mean** penalizes extreme imbalance between Precision and Recall more heavily. If one metric is very low, F1 drops significantly — unlike the arithmetic mean which can mask poor performance.

---

## Q10. What is the F_β score? What does β control? Give a real-world example where F2 is preferred over F1.

**F_β Score** = $(1 + \beta^2) \times \frac{Precision \times Recall}{\beta^2 \times Precision + Recall}$

- $\beta = 1$ → F1 (equal weight)
- $\beta > 1$ → weights Recall more
- $\beta < 1$ → weights Precision more

**Example**: In **cancer screening**, F2 ($\beta=2$) is preferred because missing a cancer case (low Recall) is far worse than a false positive.

---

## Q11. Define TPR and FPR. Write the formula for each and explain in plain English what each measures.

**TPR (True Positive Rate / Recall)** = $\frac{TP}{TP + FN}$  
→ How good is the model at **finding** actual positives?

**FPR (False Positive Rate)** = $\frac{FP}{FP + TN}$  
→ How often does the model raise **false alarms** on actual negatives?

---

## Q12. What is the ROC curve? What are its axes? What does it show that a single Precision and Recall value cannot?

The **ROC Curve** (Receiver Operating Characteristic) plots:

- **X-axis**: False Positive Rate (FPR)
- **Y-axis**: True Positive Rate (TPR)

It shows the **trade-off** between sensitivity and specificity **across all possible thresholds**, giving a complete picture of model performance.

---

## Q14. What does AUC (Area Under the ROC Curve) measure? State its range and give the probabilistic interpretation.

**AUC** measures the model's ability to distinguish between positive and negative classes.

- Range: **0.5** (no better than random) to **1.0** (perfect classifier)

**Probabilistic interpretation**: The probability that the model ranks a randomly chosen **positive** instance higher than a randomly chosen **negative** instance.

---

### Key Formulas Summary

- **Accuracy**: $\frac{TP + TN}{Total}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1**: $2 \times \frac{P \times R}{P + R}$
- **FPR**: $\frac{FP}{FP + TN}$