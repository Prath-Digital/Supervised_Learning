# Self Exercise - 11.2

---

## Q1. What does 'Bootstrap Aggregating' mean — what do each of the two words refer to?

**Answer:**  
**Bootstrap Aggregating** (commonly shortened to **Bagging**) is an ensemble learning technique.

- **Bootstrap** refers to the statistical method of creating multiple new training datasets by randomly sampling **with replacement** from the original training data. Each of these resampled datasets is called a _bootstrap sample_.
- **Aggregating** refers to combining the predictions of the multiple models (one trained on each bootstrap sample) into a single final prediction — usually by averaging (regression) or majority voting (classification).

Together, the name describes the process of training many models on bootstrap samples and then aggregating their outputs.

---

## Q2. How is a bootstrap sample constructed — what is sampling with replacement?

**Answer:**  
A **bootstrap sample** is constructed by drawing _n_ observations from the original training set of size _n_, where each draw is performed **independently and with replacement**.

**Sampling with replacement** means that after an observation is selected, it is put back into the pool and can be selected again. Consequently:

- Some original observations appear multiple times in the bootstrap sample.
- Some original observations do not appear at all.

On average, a bootstrap sample of size _n_ contains approximately **63.2 %** of the unique original observations (and the remaining ~36.8 % are left out).

---

## Q3. What is the Out-Of-Bag (OOB) sample, and what is it used for?

**Answer:**  
The **Out-Of-Bag (OOB) sample** for a particular bootstrap sample consists of all the original training observations that were **not** selected in that bootstrap sample (approximately 36.8 % of the data).

**Uses:**

- It provides a built-in validation set for the model trained on that bootstrap sample.
- By aggregating predictions on OOB observations across all models, we obtain the **OOB error estimate** — a convenient, approximately unbiased estimate of the generalization error without needing a separate validation set or cross-validation.

---

## Q4. Why does Bagging reduce variance — explain using the law of large numbers in plain terms.

**Answer:**  
Each individual model is trained on a slightly different bootstrap sample, so their predictions on a given point fluctuate (high variance).

When we average (or vote with) a large number of such models, the random fluctuations tend to cancel each other out. By the **law of large numbers**, the average of many roughly independent random variables converges to their expected value.

In plain terms: the “noise” that makes one model overfit or underfit in a particular way is averaged away, producing a more stable final prediction. The variance of the ensemble prediction decreases roughly as $1/M$ (where $M$ is the number of base models), provided the models are not perfectly correlated.

---

## Q5. Does Bagging reduce bias? Why or why not?

**Answer:**  
**No**, Bagging does **not** meaningfully reduce bias.

Bias is the systematic error that remains even if we had infinite data. Because every base learner is trained on a bootstrap sample drawn from the _same_ original dataset, each learner inherits essentially the same bias as a single model trained on the full data. Averaging many similarly biased models still yields a biased ensemble.

Bagging primarily attacks the **variance** component of the error, leaving bias largely unchanged.

---

## Q6. What type of base learner benefits most from Bagging, and why?

**Answer:**  
**High-variance, low-bias (unstable) learners** benefit most — classic examples are deep, unpruned decision trees.

**Why?**

- Unstable learners change their predictions substantially when the training data changes slightly. Bagging exploits this instability by creating diverse models whose errors cancel out when aggregated.
- Stable, high-bias learners (e.g., linear regression, shallow trees, k-NN with large _k_) produce very similar models on different bootstrap samples; averaging them yields little variance reduction.

---

## Q7. How are predictions combined in Bagging for regression? How does it differ for classification?

**Answer:**

- **Regression:** Predictions of the individual models are **averaged** (arithmetic mean).
- **Classification:** Predictions are combined by **majority vote** (the class that receives the most votes wins). Soft voting (averaging class probabilities) is also sometimes used.

The key difference is the aggregation rule: continuous averaging versus discrete voting.

---

## Q8. What is the relationship between the number of bootstrap samples and model stability?

**Answer:**  
As the number of bootstrap samples (i.e., the number of base learners) increases:

- The ensemble prediction becomes more stable (lower variance).
- The benefit follows a diminishing-returns pattern: after a few hundred models the additional reduction in variance is usually small.
- Computational cost grows linearly with the number of models, so there is a practical trade-off between stability and runtime/memory.

In the limit of infinitely many models the ensemble converges to the expected prediction under the bootstrap distribution.

---

## Q9. How does Bagging differ from a single decision tree trained on all the data?

**Answer:**

| Aspect               | Single Decision Tree                   | Bagging Ensemble                      |
| -------------------- | -------------------------------------- | ------------------------------------- |
| Training data        | Full original dataset                  | Many bootstrap samples                |
| Variance             | High (sensitive to small data changes) | Substantially reduced                 |
| Bias                 | Low (if tree is deep)                  | Essentially the same                  |
| Overfitting tendency | High                                   | Much lower                            |
| Prediction           | Single tree path                       | Average / majority vote of many trees |
| Interpretability     | High                                   | Lower (black-box ensemble)            |

Bagging produces a more robust, less overfitted model at the cost of interpretability and extra computation.

---

## Q10. What is the OOB error estimate, and why is it considered an approximately unbiased estimate?

**Answer:**  
The **OOB error estimate** is obtained by predicting each training observation using **only** the models for which that observation was out-of-bag (i.e., never seen during that model’s training). The resulting errors are then averaged.

**Why approximately unbiased?**  
Because each observation is evaluated only by models that never trained on it, the estimate behaves like an independent test-set evaluation. For a sufficiently large number of bootstrap samples it closely approximates leave-one-out cross-validation error and therefore provides a nearly unbiased estimate of the true generalization error — without the need for a separate hold-out set.

---

## Q11. What are the computational trade-offs of increasing the number of base learners in Bagging?

**Answer:**  
**Benefits of more base learners:**

- Greater variance reduction → more stable and usually more accurate predictions.
- More reliable OOB error estimate.

**Costs:**

- Training time scales linearly with the number of models.
- Memory / storage requirements increase (each model must be kept).
- Prediction latency also increases (all models must be evaluated).

In practice one chooses a number large enough that further increases yield negligible gains (commonly a few hundred trees for random forests).

---

## Q12. How does Bagging relate to Random Forest — what does Random Forest add on top of Bagging?

**Answer:**  
**Random Forest = Bagging + extra randomization of features.**

- **Bagging** already trains many decision trees on different bootstrap samples and aggregates them.
- **Random Forest** adds a second source of randomness: at each split of each tree, only a **random subset of features** is considered (instead of all features).

This extra feature randomization further decorrelates the trees, which leads to even greater variance reduction and usually better performance than plain bagging of trees.
