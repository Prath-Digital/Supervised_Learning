# Lab Work - 11.4

---

## Q1. What does Gradient Boosting optimise, and how does it differ from AdaBoost’s optimisation strategy?

**Answer:**  
Gradient Boosting optimises an arbitrary differentiable **loss function** by performing gradient descent in **function space**. At each iteration it fits a weak learner to the **negative gradient** (pseudo-residuals) of the loss with respect to the current ensemble prediction, then adds a scaled version of that learner to the ensemble.

**Difference from AdaBoost:**

- **AdaBoost** re-weights the training samples: it increases the weight of misclassified examples so that subsequent weak learners focus on them. It is equivalent to gradient descent on the **exponential loss**.
- **Gradient Boosting** does **not** re-weight samples. Instead it directly fits each new tree to the residual errors (negative gradients) of the current model. This allows the use of any differentiable loss (squared error, absolute error, logistic loss, Huber, etc.).

---

## Q2. What are pseudo-residuals, and why are they called ‘pseudo’?

**Answer:**  
Pseudo-residuals are the **negative gradients** of the chosen loss function with respect to the current model prediction:

$$
r*{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]*{F=F\_{m-1}}
$$

They are called **pseudo**-residuals because only for the squared-error loss do they coincide exactly with the ordinary residuals $y*i - F*{m-1}(x_i)$. For every other loss (absolute error, logistic loss, Huber, etc.) the negative gradient is a different quantity that merely _plays the role_ of a residual; hence the qualifier “pseudo”.

---

## Q3. Why does Gradient Boosting fit new trees on residuals and not on the original target?

**Answer:**  
Fitting successive trees on the original target would simply re-learn the same patterns and quickly overfit. By fitting each new tree to the **residuals** (or more generally the negative gradients) of the current ensemble, every tree is forced to focus exclusively on the **remaining errors**. This sequential residual-fitting is exactly gradient descent in function space: each added tree moves the overall prediction a small step in the direction that most reduces the loss. The result is a progressive reduction of bias while the individual trees remain simple (low variance).

---

## Q4. What is the role of the learning rate (shrinkage) in Gradient Boosting?

**Answer:**  
The learning rate $\nu$ (also called shrinkage) multiplies the contribution of each newly added tree:

$$
F*m(x) = F*{m-1}(x) + \nu\, h_m(x), \qquad 0 < \nu \le 1
$$

Its main roles are:

1. **Regularisation** – a small $\nu$ prevents any single tree from making a large correction, reducing the risk of overfitting.
2. **Finer optimisation** – smaller steps allow the algorithm to approach the minimum of the loss more carefully, often yielding better generalisation.
3. **Trade-off with the number of trees** – a lower learning rate usually requires more boosting rounds to reach the same training loss.

Typical practical values lie in the range $0.01$–$0.1$.

---

## Q5. What happens if you use too many boosting rounds without regularisation?

**Answer:**  
Without regularisation (no shrinkage, no early stopping, deep trees, etc.) each additional tree continues to fit the remaining training residuals, including noise. Consequently:

- Training loss keeps decreasing (often to near zero).
- The model starts to **overfit** the training data.
- Validation / test error eventually rises after an initial decrease.

The classic remedy is a combination of a small learning rate, early stopping on a validation set, and constraints on tree complexity.

---

## Q6. What is the difference between the loss function for regression GBM and classification GBM?

**Answer:**

| Task               | Common loss functions              | Pseudo-residual (negative gradient) |
| ------------------ | ---------------------------------- | ----------------------------------- |
| **Regression**     | Squared error $\frac12(y-F)^2$     | $y - F$                             |
|                    | Absolute error $\lvert y-F\rvert$  | $\operatorname{sign}(y-F)$          |
|                    | Huber loss                         | piecewise linear / quadratic        |
| **Classification** | Binomial deviance / logistic loss  | $y - p$ (where $p=\sigma(F)$)       |
|                    | Multinomial deviance (multi-class) | $y_k - p_k$ for each class          |
|                    | Exponential loss (AdaBoost-style)  | $-y\,e^{-yF}$                       |

The key conceptual difference is that regression losses measure the discrepancy between a continuous prediction and a continuous target, while classification losses measure the discrepancy between predicted probabilities (or log-odds) and discrete class labels.

---

## Q7. How is the initial prediction $F_0$ computed in Gradient Boosting for regression?

**Answer:**  
$F_0$ is the constant value that minimises the chosen loss over the whole training set:

$$
F*0(x) = \arg\min*{\gamma}\sum\_{i=1}^{n} L(y_i,\gamma)
$$

For the **squared-error loss** this constant is simply the **mean** of the target values:

$$
F*0(x) = \bar{y} = \frac1n\sum*{i=1}^{n} y_i
$$

(For absolute-error loss it is the median; for other losses the corresponding optimal constant is used.)

---

## Q8. What is the relationship between Gradient Boosting and gradient descent?

**Answer:**  
Gradient Boosting **is** gradient descent performed in **function space** rather than in parameter space.

- Ordinary gradient descent updates a finite-dimensional parameter vector $\theta$ by stepping in the direction of $-\nabla\_\theta L$.
- Gradient Boosting updates an additive function $F$ by successively adding weak learners that approximate the negative functional gradient $-\frac{\partial L}{\partial F}$.

Each new tree is therefore a step of functional gradient descent. The learning-rate parameter $\nu$ plays exactly the same role as the step-size in classical gradient descent.

---

## Q9. What are the key hyperparameters in Gradient Boosting — name four and explain each?

**Answer:**

1. **Number of boosting rounds / trees (`n_estimators`)**  
   Controls how many sequential trees are added. More trees improve training fit but increase the risk of overfitting; usually combined with early stopping.

2. **Learning rate / shrinkage (`learning_rate`, $\nu$)**  
   Scales the contribution of each tree. Smaller values regularise the model and generally improve generalisation, at the cost of needing more trees.

3. **Maximum tree depth (`max_depth`)**  
   Limits the complexity of each individual tree. Shallow trees (depth 1–5) keep variance low; deeper trees can capture higher-order interactions but raise the chance of overfitting.

4. **Subsample ratio (`subsample`)**  
   Fraction of training rows used to grow each tree (stochastic gradient boosting). Values < 1 introduce randomness that reduces variance and often improves generalisation, analogous to bagging.

(Other important ones: `min_samples_leaf`, `min_samples_split`, column subsampling, regularisation penalties.)

---

## Q10. How does tree depth affect the bias-variance tradeoff in Gradient Boosting?

**Answer:**

- **Shallow trees** (depth 1–3) have **high bias** and **low variance**. They capture only simple, low-order interactions.
- **Deep trees** have **low bias** and **high variance**; they can model complex interactions but easily overfit noise.

In Gradient Boosting the sequential addition of many trees already reduces bias. Therefore the preferred strategy is to keep individual trees **shallow** (low variance) and let the boosting process drive the bias down. Using deep trees tends to inflate variance without a commensurate gain in bias reduction, often leading to poorer generalisation.

---

## Q11. Why are shallow trees (stumps or depth-2 trees) typically used as base learners in GBM?

**Answer:**

1. **They are weak learners** – the classic boosting theory requires weak base learners; a stump is the weakest useful tree.
2. **Bias reduction is already provided by the sequential process** – many shallow trees successively correct residuals, so each tree does not need to be powerful.
3. **Variance control** – shallow trees have low variance; adding them does not rapidly inflate the ensemble’s variance.
4. **Computational efficiency** – shallow trees are fast to grow and evaluate.
5. **Interaction control** – a tree of depth $d$ can model interactions of order up to $d$; keeping $d$ small prevents unnecessary high-order interactions that often overfit.

Empirically, depths of 1–5 (most commonly 2–4) give excellent results on the majority of tabular problems.

---

## Q12. What is the key difference between Gradient Boosting and XGBoost?

**Answer:**  
Classic Gradient Boosting (e.g. Friedman’s original algorithm or scikit-learn’s `GradientBoosting`) performs a **first-order** gradient descent: each tree is fit to the negative gradient of the loss.

**XGBoost** performs a **second-order (Newton) approximation**. It uses both the first derivative (gradient) **and** the second derivative (Hessian) of the loss to construct a more accurate quadratic approximation of the loss at each step. In addition XGBoost introduces:

- explicit **L1 / L2 regularisation** on leaf weights,
- advanced split-finding algorithms (approximate, histogram-based),
- native handling of missing values,
- parallel and distributed tree construction,
- sparsity-aware algorithms,

all of which make it substantially faster, more regularised and more scalable than the original Gradient Boosting formulation.
