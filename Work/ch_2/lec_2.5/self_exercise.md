# Self Exercise - 2.5

---

## Q.1. What is the objective (cost) function used in Multiple Linear Regression?

---

**Answer**

The objective (cost) function used in Multiple Linear Regression is the Mean Squared Error (MSE), which measures the average of the squares of the errors between the predicted values and the actual values. The MSE is defined as:
$$ J(\beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$
where \( y_i \) is the actual value, \( \hat{y}_i \) is the predicted value, and \( n \) is the number of data points.

---

## Q.2. Write the mathematical expression for the Mean Squared Error (MSE) cost function in matrix form.

---

**Answer**

$$ J(\beta) = \frac{1}{n} (Y - X\beta)^T (Y - X\beta) $$
where \( Y \) is the vector of actual values, \( X \) is the matrix of input features, and \( \beta \) is the vector of regression coefficients.

---

## Q.3. Why do we square the errors in the cost function instead of using absolute values?

---

**Answer**

Squaring the errors gives more weight to larger errors, making the model more sensitive to outliers. This can lead to a better fit for the majority of the data points. Additionally, the squared error function is differentiable everywhere, which makes it easier to optimize using calculus-based methods like gradient descent.

---

## Q.4. What is the Normal Equation in Multiple Linear Regression?

---

**Answer**

The Normal Equation is a closed-form solution to find the optimal regression coefficients in Multiple Linear Regression without the need for iterative optimization methods like Gradient Descent. It is given by the formula:
$$ \beta = (X^T X)^{-1} X^T Y $$
where \( X \) is the matrix of input features, \( Y \) is the vector of actual values, and \( \beta \) is the vector of regression coefficients.

---

## Q.5. Write the formula to calculate regression coefficients using the Normal Equation.

---

**Answer**

$$ \beta = (X^T X)^{-1} X^T Y $$
where \( X \) is the matrix of input features, \( Y \) is the vector of actual values, and \( \beta \) is the vector of regression coefficients.

---

## Q.6. What is the role of the transpose of matrix X (XT) in the Normal Equation?

---

**Answer**

The transpose of matrix X (XT) is used to ensure that the dimensions of the matrices are compatible for multiplication. Specifically, it allows us to compute the product X^T Y, which is necessary for finding the optimal regression coefficients. Additionally, the term X^T X is used to compute the covariance matrix of the input features, which is essential for determining the relationships between the features and the target variable.

---

## Q.7. What condition must be satisfied for (XTX)-1 to exist?

---

**Answer**

For (XTX)-1 to exist, the matrix (XTX) must be invertible, which means it must be a square matrix with a non-zero determinant. This requires that the input features in X are linearly independent. If there are linearly dependent features (multicollinearity), the matrix (XTX) will be singular, and its inverse will not exist.

---

## Q.8. What is meant by matrix invertibility, and why is it important in regression?

---

**Answer**

Matrix invertibility refers to the property of a square matrix that allows it to have an inverse. A matrix A is invertible if there exists another matrix A^-1 such that AA^-1 = A^-1A = I, where I is the identity matrix. In regression, invertibility is important because it ensures that we can compute the regression coefficients using the Normal Equation. If the matrix (XTX) is not invertible, we cannot find a unique solution for the coefficients, which can lead to issues in model fitting and interpretation.

---

## Q.9. How does multicollinearity affect the matrix (XTX)?

---

**Answer**

Multicollinearity occurs when two or more input features in the matrix X are highly correlated. This leads to the matrix (XTX) being close to singular or singular, making it non-invertible. As a result, the Normal Equation cannot be used to find a unique solution for the regression coefficients, leading to unstable and unreliable estimates.

---

## Q.10. What are the limitations of using the Normal Equation for large datasets?

---

**Answer**

The Normal Equation has several limitations for large datasets:

1. **Computational Complexity**: The Normal Equation requires computing the inverse of the matrix (X^T X), which has a time complexity of O(n^3), where n is the number of features. For large datasets with many features, this can be computationally expensive and time-consuming.

2. **Memory Requirements**: Computing and storing the matrix (X^T X) and its inverse can require significant memory, especially for high-dimensional datasets.

3. **Numerical Stability**: For large datasets, the matrix (X^T X) can be ill-conditioned, leading to numerical instability in the computation of its inverse.

These limitations make Gradient Descent a more practical choice for large-scale regression problems.

---

## Q.11. How does Gradient Descent differ from the Normal Equation approach?

---

**Answer**

Gradient Descent is an iterative optimization algorithm that minimizes the cost function by updating the regression coefficients in the direction of the steepest descent. It does not require the computation of the inverse of any matrix, making it suitable for large datasets. In contrast, the Normal Equation provides a closed-form solution by directly computing the regression coefficients using matrix operations, which can be computationally expensive for large datasets.

---

## Q.12. Why is Gradient Descent preferred over the Normal Equation in some cases?

---

**Answer**

Gradient Descent is preferred over the Normal Equation in cases where the dataset is large or has a high number of features, as it avoids the computational complexity and memory requirements associated with matrix inversion. Additionally, Gradient Descent can be used for non-linear regression problems and can be easily adapted to include regularization techniques, whereas the Normal Equation is limited to linear regression without regularization.

---
