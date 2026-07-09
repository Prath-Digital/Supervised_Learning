# Self Exercise - 9.1

## Q1

### Question
What is the K-Nearest Neighbors algorithm? Explain how it makes a prediction for a new point in classification.

### Answer
K-Nearest Neighbors (KNN) is a supervised, non-parametric, instance-based learning algorithm used for classification and regression. During prediction, it computes the distance from the new sample to every training sample, selects the k closest neighbors, and assigns the class that appears most frequently among those neighbors (majority voting).

---

## Q2

### Question
Why is KNN called a 'lazy learner'? What happens during the training phase, and when is all the computation done?

### Answer
KNN is called a lazy learner because it does not build an explicit model during training. The training phase simply stores the training data. Most computation happens during prediction when distances to all training points are calculated.

---

## Q3

### Question
Write the formula for Euclidean distance between two points in d dimensions. Compute the distance between (2,3) and (5,7).

### Answer
Formula: d(x,y)=√Σ(xi-yi)^2. For (2,3) and (5,7): √((5-2)^2+(7-3)^2)=√(9+16)=√25=5.

---

## Q4

### Question
What is Manhattan distance? Write its formula and compute it for the same two points: (2,3) and (5,7).

### Answer
Formula: d(x,y)=Σ|xi-yi|. For (2,3) and (5,7): |5-2|+|7-3|=3+4=7.

---

## Q5

### Question
What is the effect of increasing k from 1 to n on the bias and variance of a KNN classifier?

### Answer
As k increases, variance decreases and bias increases. Small k (e.g.,1) may overfit, while very large k may underfit because predictions become overly smooth.

---

## Q6

### Question
Why is feature scaling essential for KNN? Give a numerical example where unscaled features produce the wrong nearest neighbor.

### Answer
KNN relies on distance. Features with larger scales dominate. Example: Age (20-60) and Salary (20,000-200,000). Salary differences overwhelm age unless scaling (StandardScaler/MinMaxScaler) is applied.

---

## Q7

### Question
What is the time complexity of KNN at prediction time for n training points and d features? Why is this a problem for large datasets?

### Answer
Prediction complexity is O(nd) because distances to all n samples are computed across d features. This becomes slow for large datasets and high-dimensional data.

---

## Q8

### Question
What is a KD-Tree? How does it speed up nearest neighbor search, and under what conditions does it lose its advantage?

### Answer
A KD-Tree partitions space into hierarchical regions, reducing search time compared to brute force. It performs well in low dimensions but loses efficiency in high-dimensional spaces due to the curse of dimensionality.

---

## Q9

### Question
A KNN classifier with k=1 achieves 100% training accuracy. Explain why this is expected and does not indicate a good model.

### Answer
Each training sample is its own nearest neighbor, so k=1 memorizes the dataset and achieves perfect training accuracy. This often indicates overfitting rather than good generalization.

---

## Q10

### Question
How does the KNN decision boundary change as k increases? Describe the boundary shape for k=1 vs a very large k.

### Answer
For k=1, decision boundaries are highly irregular and sensitive to noise. As k increases, boundaries become smoother. For very large k, the classifier tends toward predicting the majority class.

---

## Q11

### Question
What distance metric is default in sklearn's KNeighborsClassifier? What is the Minkowski distance, and how does it generalise Euclidean and Manhattan?

### Answer
The default metric is Minkowski (p=2), which is equivalent to Euclidean distance. Minkowski generalizes Manhattan (p=1) and Euclidean (p=2) using (Σ|xi-yi|^p)^(1/p).

---

## Q12

### Question
What is the curse of dimensionality in the context of KNN? How does it affect the quality of nearest neighbors in high-dimensional space?

### Answer
As dimensions increase, distances between points become similar, making nearest neighbors less meaningful. KNN accuracy often decreases unless dimensionality reduction or feature selection is used.

---

## Q13

### Question
Compare KNN to Logistic Regression: which is better for linearly separable data? Which is better for complex nonlinear boundaries? Justify.

### Answer
Logistic Regression is usually better for linearly separable problems because it learns a linear decision boundary efficiently. KNN is better for complex nonlinear boundaries because it makes local decisions based on neighboring samples.

---

## Q14

### Question
Write sklearn code to: (1) scale features with StandardScaler, (2) fit KNeighborsClassifier with k=7 and metric="manhattan", (3) print test accuracy and F1-weighted score.

### Answer
```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=7, metric="manhattan")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-weighted:", f1_score(y_test, y_pred, average="weighted"))
```

---

