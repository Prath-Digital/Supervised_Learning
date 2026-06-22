# Self Exercise - 4.1

## Q1. What is the objective function of Ridge Regression? Write it in full mathematical notation.

### Answer
Ridge Regression minimizes the following objective function:

\[
\min_{\beta_0,\beta}
\left(
\sum_{i=1}^{n}
(y_i - \beta_0 - \sum_{j=1}^{p} x_{ij}\beta_j)^2
+
\lambda \sum_{j=1}^{p} \beta_j^2
\right)
\]

where:

- \(y_i\) = actual target value
- \(x_{ij}\) = j-th feature of i-th observation
- \(\beta_j\) = coefficient of feature j
- \(\beta_0\) = intercept
- \(\lambda\) = regularization parameter

The second term is the L2 penalty.

---

## Q2. What is the objective function of Lasso Regression? How does it differ from Ridge?

### Answer

Lasso Regression minimizes:

\[
\min_{\beta_0,\beta}
\left(
\sum_{i=1}^{n}
(y_i - \beta_0 - \sum_{j=1}^{p} x_{ij}\beta_j)^2
+
\lambda \sum_{j=1}^{p} |\beta_j|
\right)
\]

Difference from Ridge:

| Ridge | Lasso |
|---------|---------|
| Uses L2 penalty | Uses L1 penalty |
| Shrinks coefficients | Shrinks and removes coefficients |
| Rarely produces exact zeros | Produces sparse models |
| Good for multicollinearity | Good for feature selection |

---

## Q4. What does the regularization parameter λ control — what happens when λ = 0 and when λ is very large?

### Answer

The parameter \(\lambda\) controls the strength of regularization.

#### When λ = 0

No regularization:

\[
\text{Ridge/Lasso} = \text{Ordinary Least Squares}
\]

The model fully fits the training data.

#### When λ is very large

The penalty dominates:

- Ridge coefficients approach zero.
- Lasso coefficients become exactly zero.

This may cause underfitting.

---

## Q4. Why does the L2 penalty produce a closed-form solution while L1 does not?

### Answer

The L2 penalty:

\[
\sum \beta_j^2
\]

is smooth and differentiable everywhere.

Therefore derivatives can be set equal to zero and solved analytically.

The L1 penalty:

\[
\sum |\beta_j|
\]

is not differentiable at:

\[
\beta_j = 0
\]

Therefore no general closed-form solution exists and iterative optimization methods are required.

---

## Q5. What is the geometric interpretation of the Ridge constraint — describe the shape of the feasible region.

### Answer

Ridge imposes:

\[
\sum_{j=1}^{p} \beta_j^2 \le t
\]

This creates:

- Circle in 2D
- Sphere in 3D
- Hypersphere in higher dimensions

The least-squares contours intersect the smooth boundary of the sphere.

This usually shrinks coefficients but rarely makes them exactly zero.

---

## Q6. What is the geometric interpretation of the Lasso constraint — why does it produce corner solutions (sparse coefficients)?

### Answer

Lasso imposes:

\[
\sum_{j=1}^{p} |\beta_j| \le t
\]

The feasible region is:

- Diamond in 2D
- Octahedron in 3D

The corners lie on coordinate axes.

Least-squares contours frequently touch these corners first.

At corners:

\[
\beta_j = 0
\]

Thus Lasso naturally creates sparse solutions.

---

## Q7. What is multicollinearity, and why does Ridge handle it better than OLS?

### Answer

Multicollinearity occurs when predictors are highly correlated.

Problems in OLS:

- Unstable coefficients
- Large variance
- Sensitive to small data changes

Ridge adds:

\[
\lambda \sum \beta_j^2
\]

which stabilizes matrix inversion and reduces coefficient variance.

As a result:

- More stable estimates
- Better generalization
- Less sensitivity to correlated features

---

## Q8. Define the bias–variance trade-off in the context of regularization — how does increasing λ affect each?

### Answer

Prediction error consists of:

\[
\text{Error}
=
\text{Bias}^2
+
\text{Variance}
+
\text{Noise}
\]

As λ increases:

### Bias
Increases

The model becomes simpler.

### Variance
Decreases

The model becomes more stable.

Thus regularization trades variance reduction for increased bias.

The goal is to find λ that minimizes total error.

---

## Q9. What is Elastic Net? How does it combine Ridge and Lasso, and when would you use it instead?

### Answer

Elastic Net combines L1 and L2 penalties:

\[
\min
\left(
RSS
+
\lambda
\left(
\alpha \sum |\beta_j|
+
(1-\alpha)
\sum \beta_j^2
\right)
\right)
\]

where:

- α = 1 → Lasso
- α = 0 → Ridge

Advantages:

- Performs feature selection
- Handles correlated features
- More stable than pure Lasso

Useful when many predictors are correlated.

---

## Q10. How do you select the optimal value of λ in practice? Name and explain at least one technique.

### Answer

The most common technique is Cross-Validation.

### k-Fold Cross Validation

1. Split data into k folds.
2. Train model on k−1 folds.
4. Validate on remaining fold.
4. Repeat for all folds.
5. Compute average validation error.
6. Select λ with lowest error.

Common choice:

\[
k = 5
\]

or

\[
k = 10
\]

This balances bias and variance effectively.

---

## Q11. Does regularization affect the intercept term β₀? Should it? Explain your reasoning.

### Answer

Typically the intercept is NOT regularized.

Objective functions become:

#### Ridge

\[
RSS + \lambda \sum_{j=1}^{p} \beta_j^2
\]

#### Lasso

\[
RSS + \lambda \sum_{j=1}^{p} |\beta_j|
\]

Notice β₀ is excluded.

Reason:

The intercept only shifts predictions.

Penalizing it does not reduce model complexity and can introduce unnecessary bias.

---

## Q12. What assumptions of OLS are violated when regularization becomes necessary, and how does each penalty address them?

### Answer

Regularization becomes useful when OLS struggles due to:

### Multicollinearity

Highly correlated predictors.

**Ridge:**
Reduces coefficient variance.

**Lasso:**
Selects among correlated predictors.

---

### High Dimensionality

When:

\[
p \ge n
\]

OLS becomes unstable or impossible.

**Ridge:** Works effectively.

**Lasso:** Performs variable selection.

---

### Overfitting

Model captures noise instead of signal.

**Ridge:** Shrinks coefficients.

**Lasso:** Shrinks and removes coefficients.

---

### Summary

| Problem | Ridge | Lasso |
|----------|----------|----------|
| Multicollinearity | Excellent | Good |
| Feature Selection | No | Yes |
| Overfitting | Yes | Yes |
| Sparse Models | No | Yes |
| High-Dimensional Data | Yes | Yes |

