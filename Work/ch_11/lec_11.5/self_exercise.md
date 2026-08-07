# Self Exercise - 11.5

## Q1. What does 'Light' refer to in LightGBM — what makes it lighter than standard GBM?

**Answer:**  
The "Light" in LightGBM (Light Gradient Boosting Machine) refers to its significantly higher training speed and lower memory usage compared to traditional Gradient Boosting Machines (GBM) and even XGBoost.

It achieves this primarily through two novel techniques introduced in the original paper:

- **GOSS (Gradient-based One-Side Sampling)** — reduces the number of data instances used for finding splits.
- **EFB (Exclusive Feature Bundling)** — reduces the number of features by bundling mutually exclusive ones.

Combined with histogram-based algorithms and leaf-wise tree growth, these make LightGBM much more efficient (often 10–20× faster) while maintaining or improving accuracy.

---

## Q2. What is histogram-based feature binning, and what are its two main advantages?

**Answer:**  
Histogram-based feature binning discretizes continuous feature values into a fixed number of discrete bins (default usually 255). Instead of evaluating every possible split point on sorted feature values, the algorithm builds histograms of gradients and sample counts per bin and finds the best split by scanning these bins.

**Two main advantages:**

1. **Reduced computational cost** — After building the histogram (O(#data)), finding the best split costs only O(#bins), which is much smaller than O(#data).
2. **Lower memory usage** — Continuous features are stored as integer bin indices rather than full floating-point values, and histogram subtraction further speeds up computation for sibling leaves.

---

## Q3. What is leaf-wise tree growth — how does it differ from level-wise growth?

**Answer:**  
**Leaf-wise (best-first) growth** selects the leaf that yields the largest loss reduction (information gain) and splits only that leaf, regardless of its depth.

**Level-wise (depth-wise) growth** (used by most traditional GBMs and XGBoost by default) expands all leaves at the current depth before moving to the next level, producing more balanced/symmetric trees.

**Key difference:** For the same number of leaves, leaf-wise growth typically achieves lower loss (better accuracy) because it prioritizes the most beneficial splits. However, it can lead to deeper, more unbalanced trees and a higher risk of overfitting, especially on small datasets. LightGBM therefore provides `max_depth` and `num_leaves` to control this.

---

## Q4. What hyperparameter limits tree complexity in LightGBM’s leaf-wise strategy?

**Answer:**  
The primary hyperparameter is **`num_leaves`** (default 31). It directly controls the maximum number of leaves in a tree and is the main lever for model complexity under leaf-wise growth.

Secondary controls include:

- `max_depth` — hard limit on tree depth (default −1 = unlimited).
- `min_data_in_leaf` / `min_child_samples` — minimum number of samples required in a leaf.

Note: Setting `num_leaves = 2^max_depth` is generally not recommended, as leaf-wise trees grow deeper than level-wise ones for the same number of leaves.

---

## Q5. What is GOSS (Gradient-based One-Side Sampling) — how does it reduce training data without losing information?

**Answer:**  
GOSS is a smart sampling method that keeps **all** instances with large gradients (the ones the model is currently struggling with the most) and randomly samples only a small fraction of instances with small gradients.

To keep the information gain estimate unbiased, the gradients of the sampled small-gradient instances are amplified by a constant factor `(1 − a)/b`, where `a` is the fraction of large-gradient instances kept (`top_rate`) and `b` is the sampling rate of the rest (`other_rate`).

Because large-gradient instances contribute most to the split gain, GOSS can discard a large portion of the data while preserving almost the same accuracy, substantially reducing computation.

---

## Q6. What is EFB (Exclusive Feature Bundling) — what type of features does it target?

**Answer:**  
EFB bundles **mutually exclusive features** (features that rarely take non-zero values at the same time) into a single feature.

It primarily targets **sparse features**, especially those arising from one-hot encoding of high-cardinality categorical variables or other sparse indicator features. By reducing the effective number of features, EFB lowers the cost of histogram construction from O(#data × #features) to O(#data × #bundles), where #bundles ≪ #features.

---

## Q7. How does LightGBM handle categorical features differently from standard GBM?

**Answer:**  
LightGBM has **native support for categorical features**. Instead of requiring one-hot encoding (which creates many sparse binary columns), it can treat categorical columns directly.

It finds the optimal split by sorting categories according to the training objective (e.g., mean gradient) and then evaluating contiguous partitions of the sorted categories — similar to treating them as ordered. This is both faster and often more accurate than one-hot encoding, and it avoids the sparsity problems that one-hot encoding creates for other algorithms.

---

## Q8. What is the effect of increasing `num_leaves` on model complexity and overfitting risk?

**Answer:**  
Increasing `num_leaves` **increases model complexity** because each tree can become more flexible and capture more intricate patterns.

Because LightGBM grows trees leaf-wise, a larger `num_leaves` also tends to produce deeper trees, which raises the **risk of overfitting**, especially on small or noisy datasets.

Best practice: keep `num_leaves` smaller than `2^max_depth` and pair it with regularization parameters such as `min_data_in_leaf`, `lambda_l1`/`lambda_l2`, or early stopping.

---

## Q9. How does `learning_rate` interact with `num_iterations` in LightGBM — what is the tradeoff?

**Answer:**  
`learning_rate` (shrinkage) controls the contribution of each tree. A smaller learning rate requires a larger number of trees (`num_iterations` / `n_estimators`) to reach the same training loss.

**Tradeoff:**

- Smaller learning rate + more iterations → usually better generalization and lower risk of overfitting, but longer training time.
- Larger learning rate + fewer iterations → faster training but higher chance of under- or over-fitting and less stable results.

In practice, people often use a moderate-to-small learning rate (e.g., 0.01–0.1) together with early stopping on a validation set.

---

## Q10. What is the difference between LightGBM’s dart boosting mode and standard gbdt mode?

**Answer:**

- **gbdt** (default): Standard Gradient Boosting Decision Tree. Each new tree is added to correct the residual errors of the previous ensemble.
- **dart** (Dropouts meet Multiple Additive Regression Trees): At each iteration, a random subset of previously built trees is dropped (with probability controlled by `drop_rate` and `skip_drop`). The new tree is trained on the residual after dropout, and the dropped trees are later scaled.

DART helps reduce overfitting (similar to dropout in neural networks) and can improve accuracy on noisy data, but it is slower and less stable than pure gbdt because of the randomness introduced by dropping trees.

---

## Q11. When would you prefer LightGBM over XGBoost — what dataset characteristics favour each?

**Answer:**  
**Prefer LightGBM when:**

- Dataset is very large (many rows and/or many features).
- Features are high-dimensional and sparse (benefits strongly from EFB and GOSS).
- Training speed and memory efficiency are critical.
- You have categorical features that can be handled natively.

**Prefer XGBoost when:**

- Dataset is smaller or medium-sized (leaf-wise growth can overfit more easily).
- You need more stable / reproducible results or a more mature ecosystem.
- You rely on features such as monotonic constraints, feature interaction constraints, or certain advanced regularizations that XGBoost historically handled more robustly.
- You want level-wise growth by default (more balanced trees).

In modern versions the gap has narrowed (especially with XGBoost’s histogram mode), but LightGBM still tends to win on large-scale sparse data.

---

## Q12. What regularisation parameters does LightGBM offer, and what does each penalise?

**Answer:**  
LightGBM provides several regularization parameters:

| Parameter                                | What it penalises / controls                                          |
| ---------------------------------------- | --------------------------------------------------------------------- |
| `lambda_l1` (reg_alpha)                  | L1 regularization on leaf weights (sparsity)                          |
| `lambda_l2` (reg_lambda)                 | L2 regularization on leaf weights (smoothness)                        |
| `min_split_gain`                         | Minimum gain required to make a split                                 |
| `min_data_in_leaf` / `min_child_samples` | Minimum number of samples in a leaf (prevents overly specific leaves) |
| `min_sum_hessian_in_leaf`                | Minimum sum of Hessian (second-order gradient) in a leaf              |
| `max_depth`                              | Maximum tree depth                                                    |
| `num_leaves`                             | Maximum number of leaves (primary complexity control)                 |
| `feature_fraction` / `bagging_fraction`  | Column and row subsampling (stochastic regularization)                |
| `path_smooth`                            | Further smooths leaf values                                           |

These can be combined with early stopping for effective control of overfitting under the leaf-wise growth strategy.
