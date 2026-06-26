# Self Exercise - 6.2

## Q.1

**Question:** Write the L2-regularised logistic regression objective. What does the λ term add to the gradient update?

**L2-Regularised Logistic Regression Objective:**

For binary classification, the regularised loss is:

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2 $$

Where $h_\theta(x) = \sigma(\theta^T x)$ is the sigmoid function.

**Effect of λ on gradient update:**

The gradient becomes:

$$ \frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j $$

The λ term adds **weight decay** (shrinkage) to the gradient, pulling weights towards zero during gradient descent.

---

## Q.2

**Question:** What is the difference between L1 and L2 regularisation in logistic regression? Which one produces sparse weights and why?

**L1 Regularisation (Lasso):** Adds $\frac{\lambda}{m} \sum |\theta_j|$ to the loss.

**L2 Regularisation (Ridge):** Adds $\frac{\lambda}{2m} \sum \theta_j^2$ to the loss.

**Key Differences:**
- L1 produces **sparse weights** (many exactly zero).
- L2 produces small but non-zero weights.

**Why L1 creates sparsity:** The L1 penalty has a sharp corner at zero in the constraint region, so the optimum often lies on the axes (making some weights exactly zero). L2 has a smooth circular constraint.

---

## Q.3

**Question:** In sklearn LogisticRegression, the parameter is C, not λ. What is the relationship? What does C=0.001 imply about regularisation strength?

In scikit-learn, `C` is the **inverse** of the regularisation strength:

$$ C = \frac{1}{\lambda} $$

- Higher `C` → weaker regularisation (larger λ would be small).
- Lower `C` → stronger regularisation.

**C=0.001** means **very strong regularisation** (λ = 1000), which heavily penalises large weights.

---

## Q.4

**Question:** Write the Softmax function for K classes. Verify that for K=2 it reduces to the binary sigmoid.

**Softmax Function:**

$$ \sigma(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} \quad \text{for } k = 1, \dots, K $$

**For K=2:**

Let $z_1 = z$, $z_2 = 0$ (common in binary case):

$$ \sigma(z)_1 = \frac{e^z}{e^z + 1} = \frac{1}{1 + e^{-z}} = \text{sigmoid}(z) $$

It reduces exactly to the binary sigmoid.

---

## Q.5

**Question:** What is the multiclass cross-entropy loss? Write it using the indicator function and explain each term.

**Multiclass Cross-Entropy Loss:**

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \mathbb{1}\{y^{(i)} = k\} \log(\hat{p}_k^{(i)}) $$

Where $\hat{p}_k^{(i)}$ is the predicted probability for class $k$ (from softmax).

**Explanation:**
- $\mathbb{1}\{y^{(i)} = k\}$: Indicator function (1 if true class is $k$, else 0).
- $\log(\hat{p}_k^{(i)})$: Log probability of the correct class.
- The loss penalises the model for assigning low probability to the correct class.

---

## Q.6

**Question:** Explain One-vs-Rest (OvR) multiclass logistic regression. What is a limitation of OvR probabilities?

**One-vs-Rest (OvR):** Train $K$ separate binary classifiers. For class $k$, treat it as positive and all others as negative.

At inference, choose the class with highest probability.

**Limitation:** Probabilities from different binary models are **not calibrated** with each other. They don't sum to 1 and can be inconsistent across classes.

---

## Q.7

**Question:** Define Precision and Recall. In what scenario is high Recall more important than high Precision? Give a real-world example.

- **Precision** = $\frac{TP}{TP + FP}$ — Of all predicted positives, how many are actually positive?
- **Recall** = $\frac{TP}{TP + FN}$ — Of all actual positives, how many did we catch?

**High Recall is more important** when the cost of missing a positive (False Negative) is very high.

**Example:** Cancer screening — You want to catch **all** possible cancer cases even if it means many false alarms (which can be ruled out later).

---

## Q.8

**Question:** What is the F1-score? Why is it the harmonic mean of Precision and Recall rather than the arithmetic mean?

**F1-Score:**

$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + 	ext{Recall}} $$

It is the **harmonic mean**.

**Why harmonic mean?** It gives a balanced measure that penalises extreme imbalances between Precision and Recall more severely than the arithmetic mean. If either Precision or Recall is zero, F1 becomes zero.

---

## Q.9

**Question:** What is the ROC curve? What do the axes represent? What does AUC measure and what is its range?

**ROC Curve (Receiver Operating Characteristic):** Plots **True Positive Rate (Recall)** vs **False Positive Rate** at various threshold values.

- **X-axis:** False Positive Rate (FPR = FP / (FP + TN))
- **Y-axis:** True Positive Rate (TPR = Recall)

**AUC (Area Under Curve):** Measures how well the model separates classes. Range: **0.5 (random)** to **1.0 (perfect)**.

---

## Q.10

**Question:** What is the Precision-Recall curve and when should you use it instead of the ROC curve?

**Precision-Recall Curve:** Plots Precision (Y) vs Recall (X) at different thresholds.

**Use PR curve instead of ROC when:**
- Classes are **highly imbalanced** (e.g., 95% negative).
- You care more about the positive class performance.
- ROC can be overly optimistic in imbalanced settings.

---

## Q.11

**Question:** A logistic regression model is trained on data with 95% negative and 5% positive samples. The model predicts 'negative' for every sample and achieves 95% accuracy. What metric should you use instead and what would it report?

**Use Precision, Recall, or F1-score** instead of accuracy.

In this case:
- Accuracy = 95%
- Recall = **0%** (catches no positive cases)
- Precision is undefined or 0 (no positive predictions)

This highlights why accuracy is misleading in imbalanced datasets.

---

## Q.12

**Question:** Compare logistic regression to a neural network with one sigmoid output neuron and no hidden layers. Are they the same model? Justify your answer.

**Yes, they are mathematically identical.**

A neural network with:
- No hidden layers
- One output neuron with sigmoid activation

is **exactly** logistic regression. The weights and bias correspond directly to the parameters $\theta$ in logistic regression.

---

