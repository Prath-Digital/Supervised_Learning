# Self Exercise - 9.2

---

## Q.1 Write the Minkowski distance formula. Show that p=1 gives Manhattan and p=2 gives Euclidean distance.

**Answer:**

**Minkowski Distance Formula:**

$$
d(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{n} |x_i - y_i|^p \right)^{1/p}
$$

- When **p = 1** → **Manhattan Distance** (L1 norm)
- When **p = 2** → **Euclidean Distance** (L2 norm)

---

## Q.2 What is Cosine distance? When is it used instead of Euclidean in KNN? Give one real-world domain where it is the standard choice.

**Answer:**

**Cosine Distance** = $1 - \frac{\mathbf{A} \cdot \mathbf{B}}{||\mathbf{A}|| \ ||\mathbf{B}||}$

**Used instead of Euclidean when:**
- Direction matters more than magnitude
- High-dimensional sparse data (e.g., text)
- Vectors have varying lengths

**Standard in:** Text classification, Recommendation systems, Document similarity (TF-IDF).

---

## Q.3 What is Hamming distance? Why can you not apply Euclidean distance directly to categorical features?

**Answer:**

**Hamming Distance**: Number of positions where two strings/vectors differ.

**Why not Euclidean?**
- Categorical features have no natural numerical ordering
- Arbitrary encoding creates misleading distances
- Violates metric assumptions of continuous space

---

## Q.4 What is weighted KNN (weights="distance")? Write the weight formula and explain when it outperforms uniform voting.

**Answer:**

**Weight formula:** Usually $w_i = \frac{1}{d(x, x_i)}$ (inverse distance)

**Outperforms uniform voting when:**
- Neighbor density varies significantly
- Presence of noise/outliers
- Local patterns differ across feature space

---

## Q.5 What is the curse of dimensionality? How does it affect the quality of nearest neighbors as the number of features grows?

**Answer:**

As dimensions increase:
- Volume of space grows exponentially
- Data becomes sparse
- All points become roughly equidistant
- Nearest neighbor loses meaning

**Impact on KNN:** Performance degrades sharply beyond ~20-50 dimensions.

---

## Q.6 Compare KD-Tree and Ball-Tree nearest neighbor algorithms. When does each perform better?

**Answer:**

| Aspect           | KD-Tree              | Ball-Tree                  |
|------------------|----------------------|----------------------------|
| Space Partition  | Axis-aligned        | Hyperspheres               |
| Best for         | Low dimensions (< 20) | Higher dimensions          |
| Curse of dim.    | Suffers badly        | More robust                |

---

## Q.7 You apply KNN without scaling and get 70% accuracy. After StandardScaler, accuracy rises to 88%. Explain exactly why scaling had such a large effect.

**Answer:**

Features with larger ranges dominate distance calculations. Without scaling, features like "salary" (0-100000) completely overshadow "age" (0-100).

---

## Q.8 What is PCA dimensionality reduction, and why can it improve KNN performance on high-dimensional datasets?

**Answer:**

PCA projects data onto principal components of maximum variance, reducing noise and redundancy while preserving most information.

**Benefits for KNN:** Reduces curse of dimensionality, improves distance metrics.

---

## Q.9 Explain the algorithm="auto" parameter in sklearn's KNeighborsClassifier. How does sklearn decide which algorithm to use internally?

**Answer:**

`algorithm="auto"` lets sklearn choose:
- **KD_Tree** for low-dimensional data
- **Ball_Tree** for medium dimensions
- **Brute** for very high dimensions or small datasets

---

## Q.10 A GridSearchCV over n_neighbors ∈ {3,5,7,9,11}, weights ∈ {"uniform", "distance"}, metric ∈ {"euclidean", "manhattan"} with cv=5 — how many total model fits does this require? Show your calculation.

**Answer:**

5 values of `n_neighbors` × 2 weights × 2 metrics × 5 CV folds = **100 model fits**

---

## Q.11 Your k-sweep shows CV accuracy peaks at k=7 and then declines. What does the declining curve for large k tell you about the bias of the model?

**Answer:**

Increasing k → higher **bias**, lower **variance**. The model becomes smoother and more biased toward the overall class distribution.

---

## Q.12 KNN achieves 96% test accuracy on training data distribution. In production, accuracy drops to 71%. What are two likely causes of this distribution shift, and how would you detect it?

**Answer:**

**Likely causes:**
1. **Covariate shift** (feature distribution changed)
2. **Concept drift** (relationship between features and target changed)

**Detection:** Kolmogorov-Smirnov test, PSI, monitoring feature distributions.

---

## Q.13 Compare KNN and SVM (RBF kernel) on four dimensions: (a) training time, (b) prediction time, (c) sensitivity to irrelevant features, (d) interpretability.

**Answer:**

| Dimension              | KNN                     | SVM (RBF)                  |
|------------------------|-------------------------|----------------------------|
| Training Time          | Very fast               | Slow (quadratic)           |
| Prediction Time        | Slow (O(n))             | Fast                       |
| Irrelevant Features    | Very sensitive          | More robust                |
| Interpretability       | High (instance-based)   | Low (black box)            |

---

## Q.14 Write sklearn code to: (1) run GridSearchCV on KNeighborsClassifier... (2) print best params & best CV F1-weighted, (3) evaluate the best estimator on the test set and print AUC-ROC.

**Answer:**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, roc_auc_score

param_grid = {
    'n_neighbors': [3,5,7,9,11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1_weighted')
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV F1:", grid.best_score_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("Test AUC-ROC:", roc_auc_score(y_test, y_prob))
