# Self Exercise - 9.4

---

## Q1

**Why does SVM require feature scaling even more than logistic regression? What happens to the margin if features are on vastly different scales?**

**Answer:**  
SVM optimizes the **margin** (distance between the decision hyperplane and the nearest points of each class). The geometric margin is measured in Euclidean space, so every feature contributes equally to the distance calculation.

If one feature has a much larger scale than others (e.g., income in thousands vs. age in tens), that feature will dominate the distance metric. The optimizer will therefore focus almost exclusively on the large-scale feature, producing a distorted (and usually suboptimal) margin and decision boundary.

Logistic regression is less sensitive because its loss is based on a linear combination of features followed by a sigmoid; an intercept and coefficient scaling can partially compensate. SVM has no such “free” compensation for the geometric margin, which is why **feature scaling (StandardScaler or MinMaxScaler) is almost mandatory** for SVM (except when using a linear kernel on already well-scaled or sparse text data).

---

## Q2

**What does `model.n_support_` return in sklearn SVC? What does `model.support_vectors_` contain, and how would you use it to verify the margin?**

**Answer:**

- `model.n_support_` → array of shape `(n_classes,)`. It contains the **number of support vectors belonging to each class**.
- `model.support_vectors_` → array of shape `(n_SV, n_features)`. It contains the **actual feature vectors of the support vectors** (the training points that lie on or inside the margin).

To verify the margin with a linear kernel:

```python
# For binary classification, linear kernel
w = model.coef_[0]
b = model.intercept_[0]
# Functional margin of a support vector should be ≈ 1 (soft-margin allows ≤ 1)
functional_margins = y_sv * (model.support_vectors_ @ w + b)
# Geometric margin = 2 / ||w||
geometric_margin = 2 / np.linalg.norm(w)
```

Support vectors should have functional margins close to 1 (or less if they are slack variables).

---

## Q3

**Write sklearn code to fit `SVC(kernel="linear", C=10)` on scaled data and extract the coefficient vector `w`. What does each element of `w` represent?**

**Answer:**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Assume X_train, y_train already exist
pipe = make_pipeline(StandardScaler(), SVC(kernel="linear", C=10))
pipe.fit(X_train, y_train)

# Extract w from the SVC step
w = pipe.named_steps["svc"].coef_[0]   # shape (n_features,)
b = pipe.named_steps["svc"].intercept_[0]
```

Each element $w_j$ is the **weight (importance) of feature $j$** in the linear decision function

$$
f(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b.
$$

Larger $|w_j|$ means feature $j$ has a stronger influence on the side of the hyperplane a point falls on. The sign of $w_j$ indicates the direction of the effect.

---

## Q4

**Compare SVC and LinearSVC: what optimisation solver does each use, and when should you prefer LinearSVC over SVC?**

**Answer:**

| Aspect             | SVC (kernel="linear")       | LinearSVC                                    |
| ------------------ | --------------------------- | -------------------------------------------- |
| Library            | libsvm                      | liblinear                                    |
| Optimisation       | Quadratic programming (SMO) | Coordinate descent / dual coordinate descent |
| Multiclass default | One-vs-One                  | One-vs-Rest                                  |
| Loss (default)     | Hinge                       | Squared hinge                                |
| Scalability        | $O(n^2)$–$O(n^3)$           | Near-linear in $n$                           |

**Prefer LinearSVC when:**

- Number of samples is large ($n \gtrsim 10^4$–$10^5$)
- Data is high-dimensional and (approximately) linearly separable
- You need faster training and lower memory usage

Use `SVC(kernel="linear")` only for small-to-medium data or when you need the exact same solver behaviour as the non-linear kernels.

---

## Q5

**Explain the `make_moons` dataset. Why does a linear SVM fail on it, and which kernel typically succeeds?**

**Answer:**  
`sklearn.datasets.make_moons` generates two interleaving half-moon (crescent) shapes in 2-D. The two classes are **not linearly separable**.

A linear SVM can only draw a straight line (hyperplane). Any straight line will leave a substantial number of points on the wrong side, resulting in high training and test error.

The **RBF (Gaussian) kernel** typically succeeds because it maps the data into a higher-dimensional space where the two moons become linearly separable, producing a smooth curved decision boundary that follows the shape of the moons.

---

## Q6

**What does `gamma` control in the RBF kernel? Describe the decision boundary shape when `gamma=0.01` vs `gamma=100`.**

**Answer:**  
The RBF kernel is

$$
K(\mathbf{x}\_i,\mathbf{x}\_j) = \exp(-\gamma\|\mathbf{x}\_i-\mathbf{x}\_j\|^2).
$$

`gamma` controls the **width of the Gaussian** (influence radius of each support vector).

- **Small gamma (0.01)** → wide Gaussians → smooth, almost linear decision boundary; high bias, low variance.
- **Large gamma (100)** → very narrow Gaussians → highly wiggly, complex boundary that can tightly fit every training point; low bias, high variance (risk of overfitting).

---

## Q7

**You fit `SVC(kernel="rbf", C=1, gamma=10)` and get train accuracy = 1.00, test accuracy = 0.65. What is happening, and what two hyperparameter changes would you make?**

**Answer:**  
The model is **severely overfitting**.

- High `gamma=10` creates extremely local decision regions.
- Combined with moderate `C=1`, the model memorizes the training set (perfect train accuracy) but generalizes poorly.

**Recommended changes:**

1. **Decrease gamma** (try `gamma=0.1` or `'scale'`).
2. **Decrease C** (try `C=0.1` or `0.01`) to increase regularization, or keep C and rely mainly on a smaller gamma.

A proper GridSearchCV / RandomizedSearchCV over both parameters is the systematic solution.

---

## Q8

**What is the Gram matrix in SVM? How is it computed, and what does its rank tell you about the kernel feature space?**

**Answer:**  
The **Gram matrix** (kernel matrix) $K$ is the $n\times n$ matrix whose entries are

$$
K\_{ij} = K(\mathbf{x}\_i,\mathbf{x}\_j) = \langle\phi(\mathbf{x}\_i),\phi(\mathbf{x}\_j)\rangle,
$$

where $\phi$ is the (possibly implicit) feature map of the kernel.

It is computed by evaluating the kernel function on every pair of training samples.

The **rank of $K$** equals the dimension of the subspace spanned by the mapped training points $\{\phi(\mathbf{x}\_i)\}$.

- Full rank $= n$ → the points are linearly independent in feature space.
- Rank $r < n$ → the effective dimensionality of the feature space (for these points) is only $r$.

---

## Q9

**How does GridSearchCV with SVC prevent data leakage? What happens if you scale the entire dataset before running GridSearchCV?**

**Answer:**  
`GridSearchCV` (with a Pipeline that contains the scaler) fits the scaler **only on the training fold** of each cross-validation split and then transforms both the train and validation folds with that scaler. This keeps the validation fold completely unseen.

If you scale the **entire** dataset first and then run GridSearchCV, information from the validation/test folds leaks into the scaling parameters (mean and variance). The model sees a slightly “contaminated” validation set, producing optimistically biased scores and a potentially poorer final model.

**Correct pattern:**

```python
pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC())])
grid = GridSearchCV(pipe, param_grid, cv=5)
```

---

## Q10

**What does `class_weight="balanced"` do in SVC? Write the formula for how it modifies the C parameter per class.**

**Answer:**  
`class_weight="balanced"` automatically adjusts the penalty parameter $C$ for each class inversely proportional to class frequencies:

$$
w*c = \frac{n*{\text{samples}}}{n\_{\text{classes}} \times n_c}
$$

where $n_c$ is the number of samples of class $c$.

Then the effective regularization for class $c$ becomes

$$
C_c = C \times w_c.
$$

Minority classes receive a larger $C_c$ (stronger penalty for misclassification), helping the SVM pay more attention to them.

---

## Q11

**Explain the One-vs-One strategy for multi-class SVM. How many binary classifiers are trained for $K$ classes? What is the prediction rule?**

**Answer:**  
In **One-vs-One (OvO)** a binary SVM is trained for **every pair of classes**.  
Number of classifiers = $\dfrac{K(K-1)}{2}$.

**Prediction:** each binary classifier votes for one of the two classes it was trained on. The class that receives the **most votes** is chosen (majority voting). Ties can be broken by the decision-function values or randomly.

(sklearn SVC uses OvO by default; LinearSVC uses One-vs-Rest.)

---

## Q12

**A Breast Cancer SVM model has very high recall but low precision on the malignant class. What threshold or `class_weight` adjustment would you make, and why?**

**Answer:**  
High recall + low precision on the malignant class means the model is predicting “malignant” too often (many false positives).

**Actions:**

1. **Raise the decision threshold** for the malignant class (e.g., from 0.0 to a positive value) so that only stronger evidence is accepted as malignant.
2. **Decrease the class weight of the malignant class** (or increase the weight of the benign class) so the optimizer is less aggressive about capturing every malignant sample.

Both reduce false positives and therefore improve precision, at the possible cost of a modest drop in recall.

---

## Q13

**Compare SVM (RBF kernel) and Random Forest on three dimensions:**  
**(a)** sensitivity to irrelevant features,  
**(b)** training time for $n=100{,}000$,  
**(c)** decision boundary shape.

**Answer:**

| Dimension                       | RBF-SVM                                                                                    | Random Forest                                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| (a) Irrelevant features         | Sensitive - all features affect the kernel distance; scaling & feature selection important | Relatively robust - trees can ignore irrelevant features via split selection |
| (b) Training time $n=100{,}000$ | Very slow / often impractical ($O(n^2)$–$O(n^3)$)                                          | Fast and scalable (parallelizable, near-linear)                              |
| (c) Decision boundary           | Smooth, continuous, curved (global)                                                        | Piece-wise constant, axis-aligned, can be jagged                             |

---

## Q14

**Write a complete sklearn Pipeline: `StandardScaler` → `SVC(kernel="rbf")`. Run `GridSearchCV` over `svc__C ∈ {0.1, 1, 10}` and `svc__gamma ∈ {0.01, 0.1, "scale"}` with `cv=5`. Print `best_params_` and test AUC-ROC.**

**Answer:**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", probability=True))  # probability=True needed for AUC
])

param_grid = {
    "svc__C": [0.1, 1, 10],
    "svc__gamma": [0.01, 0.1, "scale"]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

y_proba = grid.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)
print("Test AUC-ROC:", round(auc, 4))
```
