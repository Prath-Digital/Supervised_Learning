# Self Exercise - 11.3

---

## Q.1 What does 'Adaptive' mean in AdaBoost — what adapts after each round?

**Answer:**  
"Adaptive" refers to the fact that the **sample weights** are adaptively updated after every boosting round.

After each weak learner is trained, the algorithm increases the weights of the samples that were misclassified and decreases the weights of the samples that were correctly classified. Consequently, the next weak learner is forced to pay more attention to the previously hard (misclassified) examples. This sequential re-weighting of the training distribution is the adaptive mechanism that gives AdaBoost its name.

---

## Q.2 How are initial sample weights set in AdaBoost, and why that value?

**Answer:**  
All sample weights are initialized uniformly:

$$
w_i^{(1)} = \frac{1}{N} \quad \text{for } i = 1, 2, \dots, N
$$

where $N$ is the number of training samples.

This choice ensures that every training example starts with equal importance. No sample is privileged or ignored at the beginning; the algorithm discovers which examples are difficult purely through the subsequent adaptive updates.

---

## Q.3 What is a weak learner — what accuracy threshold qualifies a model as 'weak'?

**Answer:**  
A **weak learner** (or weak classifier) is any model whose performance is only slightly better than random guessing.

For binary classification the formal requirement is that its weighted error rate $\varepsilon$ satisfies

$$
\varepsilon < 0.5
$$

(i.e., accuracy strictly greater than 50 %). In practice the most common weak learners are decision stumps (depth-1 trees) or very shallow trees.

---

## Q.4 How is the learner weight $\alpha$ computed, and what does it measure?

**Answer:**  
Given the weighted error $\varepsilon_t$ of the $t$-th weak learner $h_t$, its coefficient is

$$
\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \varepsilon_t}{\varepsilon_t}\right)
$$

$\alpha_t$ measures the **importance** (or confidence) of that weak learner in the final ensemble:

- A learner with very low error ($\varepsilon_t \to 0$) receives a large positive $\alpha_t$.
- A learner with error close to 0.5 receives a small $\alpha_t$.
- An error $\ge 0.5$ would produce a non-positive $\alpha_t$, which is why such learners are rejected.

---

## Q.5 What happens to sample weights after a misclassification — and after a correct prediction?

**Answer:**  
After a weak learner $h_t$ with weight $\alpha_t$ has been obtained, every sample weight is updated by

$$
w_i^{(t+1)} = \frac{w_i^{(t)}}{Z_t} \exp\bigl(-\alpha_t\, y_i\, h_t(x_i)\bigr)
$$

where $Z_t$ is a normalisation constant that makes the weights sum to 1.

- **Misclassification** ($y_i h_t(x_i) = -1$): the exponent becomes $+\alpha_t$, so the weight is **multiplied by $e^{\alpha_t}$** (increased).
- **Correct classification** ($y_i h_t(x_i) = +1$): the exponent becomes $-\alpha_t$, so the weight is **multiplied by $e^{-\alpha_t}$** (decreased).

Thus hard examples receive higher relative weight for the next round.

---

## Q.6 Why must the weighted error $\varepsilon$ be less than 0.5 for AdaBoost to work?

**Answer:**  
If $\varepsilon_t \ge 0.5$, then

$$
\alpha_t = \frac12\ln\Bigl(\frac{1-\varepsilon_t}{\varepsilon_t}\Bigr) \le 0.
$$

A non-positive $\alpha_t$ means the learner is no better (or is worse) than random guessing and would either contribute nothing or actively harm the ensemble. The theoretical guarantees of AdaBoost (that the training error decreases exponentially) rely on every weak learner having an edge $\gamma_t = \frac12 - \varepsilon_t > 0$. Therefore the algorithm aborts or discards any learner whose weighted error is not strictly less than ½.

---

## Q.7 How does AdaBoost combine multiple weak learners into a final prediction (classification)?

**Answer:**  
The final strong classifier is a **weighted majority vote**:

$$
H(x) = \operatorname{sign}\Biggl(\sum\_{t=1}^{T} \alpha_t\, h_t(x)\Biggr)
$$

Each weak learner $h_t$ casts a vote whose magnitude is proportional to its reliability $\alpha_t$. The sign of the weighted sum decides the predicted class.

---

## Q.8 How does AdaBoost differ for regression vs classification — what changes?

**Answer:**  
Classic AdaBoost was designed for binary classification. For regression several variants exist (AdaBoost.R, AdaBoost.R2, etc.). The principal changes are:

| Aspect               | Classification                | Regression                                      |
| -------------------- | ----------------------------- | ----------------------------------------------- |
| Loss / error         | 0-1 (misclassification)       | Absolute or squared residual                    |
| Sample-weight update | Exponential in $\pm\alpha$    | Proportional to the magnitude of the residual   |
| Combination rule     | Weighted majority vote (sign) | Weighted average (or median) of the predictions |
| Weak learner         | Classifier (usually a stump)  | Regressor (usually a shallow tree)              |

In short, the adaptive re-weighting still occurs, but it is driven by continuous residuals rather than discrete misclassification indicators, and the final aggregation changes from voting to averaging.

---

## Q.9 What is the exponential loss function in AdaBoost, and why exponential?

**Answer:**  
AdaBoost can be derived as stage-wise additive modelling that minimises the **exponential loss**

$$
L(y, f(x)) = \exp\bigl(-y\, f(x)\bigr), \qquad y\in\{+1,-1\}.
$$

**Why exponential?**

- It is a smooth, strictly convex upper bound on the 0-1 loss.
- Minimising it with a greedy coordinate-descent procedure yields exactly the classic AdaBoost weight-update and $\alpha$-formula.
- The exponential growth for large negative margins places heavy emphasis on hard examples, which is precisely the adaptive behaviour desired by boosting.

(The same population minimiser is obtained by logistic loss, but the exponential form leads to the particularly simple closed-form updates of AdaBoost.)

---

## Q.10 What are the main hyperparameters of AdaBoost, and how does each affect performance?

**Answer:**  
The most important hyperparameters (as exposed by scikit-learn and similar libraries) are:

| Hyperparameter     | Effect                                                                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `n_estimators` (T) | Number of weak learners. Larger → lower bias, higher risk of overfitting and longer training.                                            |
| `learning_rate`    | Multiplies every $\alpha_t$. Smaller values require more estimators but often improve generalisation (shrinkage).                        |
| `base_estimator`   | The weak learner class (almost always a decision stump or shallow tree). Deeper trees increase capacity but reduce the “weak” character. |
| `algorithm`        | SAMME (discrete) vs SAMME.R (real-valued probabilities). SAMME.R usually converges faster.                                               |

Secondary controls include the maximum depth of the base trees and any regularisation parameters of the base learner.

---

## Q.11 How does AdaBoost handle outliers compared to other ensemble methods?

**Answer:**  
AdaBoost is **sensitive to outliers and noisy labels**.

Because misclassified points receive exponentially increasing weights, a few persistent outliers can dominate the sample distribution. Subsequent weak learners are forced to fit those outliers, which can degrade overall performance.

In contrast:

- Bagging / Random Forests treat all samples equally (or with bootstrap sampling) and are far more robust.
- Gradient Boosting can use more robust loss functions (Huber, absolute loss, etc.) and therefore tolerates outliers better.
- Modern robust boosting variants replace the exponential loss by bounded or non-convex losses to mitigate this weakness.

---

## Q.12 What are the key differences between AdaBoost and Gradient Boosting?

**Answer:**

| Aspect                             | AdaBoost                                                   | Gradient Boosting                                            |
| ---------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| Loss function                      | Fixed exponential loss                                     | Arbitrary differentiable loss (squared, logistic, Huber, …)  |
| How “hard” examples are emphasised | Explicit re-weighting of the sample distribution           | Fitting the negative gradient (pseudo-residuals) of the loss |
| Combination of learners            | Weighted majority vote / weighted sum of $\alpha_t h_t$    | Simple additive expansion $F*m = F*{m-1} + \nu\cdot h_m$     |
| Typical base learner               | Decision stumps                                            | Shallow regression trees                                     |
| Robustness to outliers             | Sensitive (exponential loss)                               | Can be made robust by choice of loss                         |
| Flexibility                        | Primarily classification (binary / multi-class extensions) | Naturally handles regression and classification              |
| Shrinkage / regularisation         | Learning-rate multiplier on $\alpha$                       | Explicit learning-rate $\nu$ and tree regularisation         |

In essence, AdaBoost is a special case of functional gradient descent that happens to use the exponential loss and an explicit re-weighting view, while Gradient Boosting is the more general framework.

---

_End of document_  
_Generated as a study reference for AdaBoost fundamentals._
