# Self Exercise - 6.1

## Q.1

**Question:** Write the L2-regularised logistic regression objective. What does the λ term add to the gradient update?

**Answer:**

**L2-regularised Logistic Regression Objective:**

For binary logistic regression with L2 regularization:

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2 $$

where $h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$ is the sigmoid function.

**Effect of λ on gradient update:**

The gradient becomes:
$$ \frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j $$

The λ term adds **weight decay**: it shrinks the weights towards zero during gradient descent, preventing overfitting.

---

## Q.2

**Question:** What is the difference between L1 and L2 regularisation in logistic regression? Which one produces sparse weights and why?

**Answer:**

**L1 Regularization (Lasso):**
- Penalty term: $\frac{\lambda}{m} \sum |\theta_j|$
- Produces **sparse weights** (many θⱼ become exactly zero).
- Creates feature selection.

**L2 Regularization (Ridge):**
- Penalty term: $\frac{\lambda}{2m} \sum \theta_j^2$
- Shrinks weights but rarely makes them exactly zero.
- Better for handling multicollinearity.

**Why L1 produces sparsity:** The diamond-shaped constraint region of L1 has corners on the axes, so the optimum often lies on an axis (making some coefficients zero).

---

## Q.3

**Question:** In sklearn LogisticRegression, the parameter is C, not λ. What is the relationship? What does C=0.001 imply about regularisation strength?

**Answer:**

In scikit-learn, `C` is the **inverse** of the regularization strength:

$$ C = \frac{1}{\lambda} $$

- Higher `C` → weaker regularization (larger λ is stronger).
- `C=0.001` means **very strong regularization** (λ = 1000), which heavily penalizes large weights.

---

## Q.4

**Question:** Write the Softmax function for K classes. Verify that for K=2 it reduces to the binary sigmoid.

**Answer:**

**Softmax Function:**

$$ \sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}} \quad \text{for } i = 1, \dots, K $$

**For K=2:**

Let $z_1 = z$, $z_2 = 0$ (common in binary case):

$$ \sigma(z)_1 = \frac{e^z}{e^z + e^0} = \frac{e^z}{1 + e^z} = \frac{1}{1 + e^{-z}} $$

Which is exactly the **sigmoid** function. ✅

---

## Q.5

**Question:** What is the multiclass cross-entropy loss? Write it using the indicator function and explain each term.

**Answer:**

**Multiclass Cross-Entropy Loss:**

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \mathbb{1}\{y^{(i)} = k\} \log(\hat{p}_k^{(i)}) $$

where $\mathbb{1}\{y^{(i)} = k\}$ is the indicator function (1 if true class is k, else 0), and $\hat{p}_k^{(i)}$ is the predicted probability from softmax.

- Encourages the model to assign high probability to the correct class.
- Penalizes confident wrong predictions heavily.

---

## Q.6

**Question:** Explain One-vs-Rest (OvR) multiclass logistic regression. What is a limitation of OvR probabilities?

**Answer:**

**One-vs-Rest (OvR):** Train K separate binary classifiers. For class k, treat it as positive and all others as negative.

**Prediction:** Choose the class with highest probability.

**Limitation:** The probabilities from different classifiers are **not calibrated** with each other (they don't necessarily sum to 1). This can lead to inconsistent confidence scores across classes.

---

## Q.7

**Question:** Define Precision and Recall. In what scenario is high Recall more important than high Precision? Give a real-world example.

**Answer:**

**Precision** = TP / (TP + FP) — "Of all positive predictions, how many were correct?"

**Recall** = TP / (TP + FN) — "Of all actual positives, how many did we catch?"

**High Recall preferred when** missing a positive case is very costly.

**Example:** Cancer detection — You prefer to flag many healthy people for further tests (low precision) rather than miss even one cancer patient (high recall).

---

## Q.8

**Question:** What is the F1-score? Why is it the harmonic mean of Precision and Recall rather than the arithmetic mean?

**Answer:**

**F1-Score:**

$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

**Why harmonic mean?** It is more sensitive to low values. If either precision or recall is very low, F1 drops significantly — unlike arithmetic mean which can be misleading when one metric is high and the other is low.

---

## Q.9

**Question:** What is the ROC curve? What do the axes represent? What does AUC measure and what is its range?

**Answer:**

**ROC Curve (Receiver Operating Characteristic):**

- **X-axis:** False Positive Rate (FPR = FP / (FP + TN))
- **Y-axis:** True Positive Rate (TPR = Recall)

**AUC (Area Under Curve):** Measures the model's ability to distinguish classes.
- Range: **0.5 to 1.0** (0.5 = random guessing, 1.0 = perfect classifier).

---

## Q.10

**Question:** What is the Precision-Recall curve and when should you use it instead of the ROC curve?

**Answer:**

**Precision-Recall Curve:**

- X-axis: Recall
- Y-axis: Precision

**Use PR curve instead of ROC when:**
- Classes are highly imbalanced (especially when positives are rare).
- You care more about the positive class performance.

---

## Q.11

**Question:** A logistic regression model is trained on data with 95% negative and 5% positive samples. The model predicts 'negative' for every sample and achieves 95% accuracy. What metric should you use instead and what would it report?

**Answer:**

**Accuracy is misleading** here (always predicting majority class gives 95% accuracy).

**Better metrics:**
- **Recall** = 0 (catastrophic for positive class)
- **Precision** = undefined or 0 (no positive predictions)
- **F1-score** = 0
- **AUC** would be around 0.5

**Recommended:** Use **F1-score**, **PR-AUC**, or **balanced accuracy**.

---

## Q.12

**Question:** Compare logistic regression to a neural network with one sigmoid output neuron and no hidden layers. Are they the same model? Justify your answer.

**Answer:**

**Yes, they are mathematically equivalent.**

A neural network with:
- One dense layer with sigmoid activation
- No hidden layers

is **exactly** logistic regression.

The only differences are:
- Implementation details (optimization, regularization defaults)
- API and training procedures
- Feature engineering expectations

---

