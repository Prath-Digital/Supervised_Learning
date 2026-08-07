# Self Exercise - 11.6

A comprehensive section-wise guide covering fundamental concepts, mathematics, hyperparameters, and comparisons with other gradient boosting libraries.

---

## Section 1: Fundamentals of XGBoost

### Q1. What does XGBoost stand for, and what does ‘extreme’ refer to in the name?

**Answer:**

**XGBoost** stands for **eXtreme Gradient Boosting**.

The term **“extreme”** refers to the engineering optimizations and algorithmic enhancements that push the limits of computational performance and scalability beyond traditional gradient boosting implementations. These include:

- Parallelized tree construction
- Cache-aware memory access patterns
- Out-of-core computation for datasets larger than memory
- Sparsity-aware algorithms
- Regularization techniques built into the objective function
- Efficient handling of missing values

XGBoost was developed by Tianqi Chen and Carlos Guestrin and introduced in the 2016 paper _“XGBoost: A Scalable Tree Boosting System”_.

---

## Section 2: Core Mathematics & Tree Construction

### Q2. What is the Similarity Score in XGBoost, and how is it calculated?

**Answer:**

The **Similarity Score** measures how similar the residuals (or gradients) are within a leaf node. It quantifies the quality of a potential leaf and is used to evaluate splits.

**Formula:**

$$
\text{Similarity Score} = \frac{\left( \sum_{i \in I} g_i \right)^2}{\sum_{i \in I} h_i + \lambda}
$$

Where:

- $g_i$ = first derivative (gradient) of the loss function for instance $i$
- $h_i$ = second derivative (Hessian) of the loss function for instance $i$
- $I$ = set of instances in the leaf
- $\lambda$ = L2 regularization parameter (`reg_lambda`)

**Special cases:**

- **Regression** (squared error loss): $g_i = -(y_i - \hat{y}_i)$ (negative residual), $h_i = 1$
  $$
  \text{Similarity Score} = \frac{\left( \sum \text{residuals} \right)^2}{\text{number of residuals} + \lambda}
  $$
- **Classification** (logistic loss): $h_i = p_i(1 - p_i)$, where $p_i$ is the previously predicted probability.

A higher similarity score indicates that the residuals in the leaf are more homogeneous (similar in direction and magnitude).

---

### Q3. What is the Split Gain in XGBoost — how is it used to decide whether to make a split?

**Answer:**

**Split Gain** measures the improvement in the objective function obtained by splitting a node into two child nodes.

**Formula:**

$$
\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
$$

Where:

- $G_L, G_R$ = sum of gradients in left and right child
- $H_L, H_R$ = sum of Hessians in left and right child
- $\lambda$ = L2 regularization on leaf weights
- $\gamma$ = complexity penalty for adding a new leaf (`gamma` / `min_split_loss`)

**Decision rule:**

- Compute Gain for all candidate splits.
- Choose the split with the **highest Gain**.
- If the best Gain is **≤ 0** (or less than a threshold), **do not split** — the node becomes a leaf.

The Gain formula can be interpreted as:

> (Left Similarity + Right Similarity − Parent Similarity) − γ

---

### Q4. What is the role of the regularisation parameter λ in XGBoost’s objective function?

**Answer:**

**λ (`reg_lambda`)** is the **L2 regularization** parameter applied to the leaf weights.

**Role in the objective function:**

$$
\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2
$$

Where $T$ is the number of leaves and $w_j$ are the leaf weights.

**Effects:**

1. Appears in the **denominator** of both the Similarity Score and the optimal leaf weight:
   $$
   w_j^* = -\frac{G_j}{H_j + \lambda}
   $$
2. **Shrinks leaf weights** toward zero → reduces model complexity.
3. **Discourages extreme predictions** from leaves that contain few samples or high-variance gradients.
4. Higher λ → more conservative model → helps prevent overfitting.

Default value in XGBoost is **1**.

---

### Q5. What is γ (gamma) in XGBoost — how does it implement tree pruning?

**Answer:**

**γ (`gamma` / `min_split_loss`)** is the **minimum loss reduction** required to make a further partition on a leaf node.

**How pruning works:**

In the Gain formula, γ is subtracted:

$$
\text{Gain} = \underbrace{\text{(Left + Right Similarity − Parent Similarity)}}_{\text{improvement}} - \gamma
$$

- If Gain > 0 → split is beneficial → keep the split.
- If Gain ≤ 0 → split does not improve the objective enough to justify the added complexity → **prune** the split (node remains a leaf).

**Key points:**

- Larger γ → more conservative algorithm → shallower trees.
- γ acts as a **pre-pruning** threshold during tree growth.
- It is one of the most effective parameters for controlling overfitting.
- Default value is **0** (no minimum gain required).

---

### Q6. What is the difference between the gradient (gᵢ) and Hessian (hᵢ) in XGBoost?

**Answer:**

| Aspect               | Gradient ($g_i$)                                     | Hessian ($h_i$)                                       |
| -------------------- | ---------------------------------------------------- | ----------------------------------------------------- |
| **Definition**       | First derivative of the loss w.r.t. prediction       | Second derivative of the loss w.r.t. prediction       |
| **Mathematical**     | $g_i = \partial_{\hat{y}_i} l(y_i, \hat{y}_i)$       | $h_i = \partial_{\hat{y}_i}^2 l(y_i, \hat{y}_i)$      |
| **Interpretation**   | Direction and magnitude of the error (residual)      | Curvature / local confidence / weight of the instance |
| **Role**             | Tells _how much_ and _in which direction_ to correct | Tells _how reliable_ that correction is               |
| **Regression (MSE)** | $g_i = \hat{y}_i - y_i$ (or negative residual)       | $h_i = 1$ (constant)                                  |
| **Binary Logistic**  | $g_i = p_i - y_i$                                    | $h_i = p_i(1 - p_i)$                                  |

**Why both are needed:**
XGBoost uses a **second-order (Newton) approximation**. The gradient provides the direction of steepest descent, while the Hessian scales the step size according to the local curvature, leading to faster and more accurate convergence than pure first-order gradient boosting.

---

### Q7. How does XGBoost compute leaf output weights — write the formula and explain each term?

**Answer:**

**Optimal leaf weight formula:**

$$
w_j^* = -\frac{G_j}{H_j + \lambda}
$$

Where:

- $G_j = \sum_{i \in I_j} g_i$ — sum of gradients of all instances falling into leaf $j$
- $H_j = \sum_{i \in I_j} h_i$ — sum of Hessians of all instances falling into leaf $j$
- $\lambda$ — L2 regularization parameter
- $I_j$ — set of training instances assigned to leaf $j$

**Explanation of terms:**

- **Numerator ($-G_j$)**: Points in the opposite direction of the average gradient → corrects the prediction error.
- **Denominator ($H_j + \lambda$)**:
  - $H_j$ acts as a weighting factor (instances with higher curvature have more influence).
  - $\lambda$ prevents the weight from becoming too large when $H_j$ is small (few samples or low-confidence instances).

This closed-form solution comes from setting the derivative of the second-order approximated objective with respect to $w_j$ to zero.

---

### Q8. Why does XGBoost use a second-order Taylor expansion of the loss — what advantage does it give?

**Answer:**

**Reason:**
Traditional gradient boosting uses only the **first-order** gradient (like gradient descent). XGBoost approximates the loss function with a **second-order Taylor expansion**:

$$
l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)
$$

**Advantages:**

1. **Newton’s method** instead of pure gradient descent → faster convergence and better step-size selection.
2. **Closed-form optimal leaf weights** can be derived analytically.
3. The objective becomes a simple quadratic function of the leaf weights, enabling efficient optimization.
4. The same framework works for **any twice-differentiable loss function** (squared error, logistic, Poisson, etc.) without needing custom derivations for each.
5. Incorporates curvature information (Hessian), leading to more accurate updates, especially for complex loss surfaces.

This is one of the key algorithmic innovations that distinguish XGBoost from classic Gradient Boosting Machines (GBM).

---

## Section 3: Hyperparameters & Practical Usage

### Q9. What are the key hyperparameters in XGBoost — name five and explain the effect of each?

**Answer:**

Here are five of the most important hyperparameters:

| Hyperparameter            | Default | Effect                                                                                                                                                           |
| ------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`learning_rate` (eta)** | 0.3     | Step-size shrinkage. Lower values make the model more robust but require more trees. Controls how much each tree contributes to the final prediction.            |
| **`max_depth`**           | 6       | Maximum depth of each tree. Higher values increase model complexity and risk of overfitting; lower values make the model more conservative.                      |
| **`min_child_weight`**    | 1       | Minimum sum of Hessian (instance weight) needed in a child. Larger values prevent the model from creating leaves with too few samples → stronger regularization. |
| **`subsample`**           | 1.0     | Fraction of training instances sampled for each tree. Values < 1 introduce randomness (stochastic gradient boosting) and help prevent overfitting.               |
| **`colsample_bytree`**    | 1.0     | Fraction of features sampled when constructing each tree. Reduces correlation between trees and acts as a form of regularization / feature selection.            |

**Other important ones:**

- `gamma` / `min_split_loss` — minimum gain to split
- `reg_lambda` / `reg_alpha` — L2 / L1 regularization
- `n_estimators` — number of boosting rounds

---

### Q10. How does XGBoost handle missing values during training and prediction?

**Answer:**

XGBoost uses a **sparsity-aware split finding** algorithm:

**During Training:**

1. For every candidate split on a feature, XGBoost evaluates **two** possibilities:
   - Send all missing values to the **left** child
   - Send all missing values to the **right** child
2. It chooses the direction (and threshold) that yields the **highest Gain**.
3. The optimal default direction is stored in the tree node.

**During Prediction:**

- If a feature value is missing, the instance follows the **learned default direction** for that node.

**Key advantages:**

- No need for explicit imputation.
- Works natively with sparse matrices (only non-missing values need to be stored and processed).
- Computational complexity is linear in the number of **non-missing** entries.
- Handles missing values, zero entries, and one-hot encoding artifacts in a unified way.

This is one of XGBoost’s most practical and powerful features.

---

### Q11. What is the difference between XGBoost’s `tree_method` ‘exact’ and ‘hist’ — when would you use each?

**Answer:**

| Aspect            | `tree_method='exact'`                                       | `tree_method='hist'`                                     |
| ----------------- | ----------------------------------------------------------- | -------------------------------------------------------- |
| **Algorithm**     | Exact greedy algorithm                                      | Histogram-based approximate algorithm                    |
| **Split finding** | Enumerates **all** possible split points                    | Bins continuous features into discrete bins (histograms) |
| **Accuracy**      | Highest (considers every candidate)                         | Slightly approximate but usually very close              |
| **Speed**         | Slow on large datasets                                      | Much faster                                              |
| **Memory**        | Higher                                                      | Lower (operates on bin indices)                          |
| **Best for**      | Small-to-medium datasets where maximum accuracy is critical | Large datasets, production systems, GPU training         |

**When to use:**

- **`exact`**: Small datasets (< ~10k–50k rows), when you need the absolute best possible splits, or for research/reproducibility.
- **`hist`**: Default recommendation for most real-world problems. Significantly faster, supports additional features (e.g., `max_leaves`, categorical features in newer versions), and is the basis for GPU acceleration.

Modern XGBoost defaults to `hist` in many interfaces because the accuracy-speed trade-off strongly favors it on larger data.

---

## Section 4: Comparison with Other Libraries

### Q12. How do XGBoost, LightGBM, and CatBoost differ in their core tree-building strategies?

**Answer:**

| Aspect                   | **XGBoost**                                              | **LightGBM**                                                 | **CatBoost**                                                 |
| ------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Tree growth**          | Level-wise (depth-wise)                                  | Leaf-wise (best-first)                                       | Symmetric / Oblivious trees                                  |
| **Description**          | Grows all nodes at the current depth before going deeper | Always splits the leaf that gives the largest loss reduction | Same split condition used across all nodes at the same depth |
| **Tree shape**           | More balanced                                            | Can become deep and unbalanced                               | Perfectly balanced                                           |
| **Speed**                | Fast (especially with `hist`)                            | Often the fastest on large data                              | Competitive; very fast inference                             |
| **Overfitting tendency** | Moderate (controlled by max_depth)                       | Higher if `num_leaves` is not tuned                          | Lower due to symmetric structure                             |
| **Categorical features** | Native support (recent versions) or needs encoding       | Native (efficient)                                           | Best-in-class (Ordered Target Statistics)                    |
| **Missing values**       | Native (sparsity-aware)                                  | Native                                                       | Native                                                       |
| **Key strength**         | Mature ecosystem, strong regularization, highly tunable  | Extreme speed + memory efficiency                            | Excellent categorical handling + robust defaults             |

**Summary of strategies:**

- **XGBoost** prioritizes controlled, level-wise growth with strong regularization.
- **LightGBM** prioritizes speed via leaf-wise growth and histogram techniques (GOSS + EFB).
- **CatBoost** prioritizes robustness and categorical feature handling through ordered boosting and symmetric trees.
