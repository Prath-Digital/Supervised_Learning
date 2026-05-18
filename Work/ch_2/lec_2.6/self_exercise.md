# Self Exercise - 2.6

---

## Q1. What is Polynomial Regression, and how does it differ from Linear Regression?

---

**Answer:**

Polynomial Regression is a regression technique used to model nonlinear relationships between variables by introducing polynomial terms of input features.

Difference:
- Linear Regression fits a straight line.
- Polynomial Regression fits a curve.
- Linear: y = b0 + b1x
- Polynomial: y = b0 + b1x + b2x² + b3x³ + ...


---

## Q2. Why is Polynomial Regression still considered a linear model in parameters?

---

**Answer:**

Polynomial Regression is considered linear because the model remains linear with respect to its coefficients (parameters), even though input variables are transformed into polynomial terms.

---

## Q3. Write the mathematical equation of a polynomial regression model of degree n.

---

**Answer:**

A polynomial regression equation of degree n is:

y = b0 + b1x + b2x² + b3x³ + ... + bnxⁿ + ε

Where:
- b0,b1,...bn = coefficients
- x = input variable
- ε = error term


---

## Q4. What is the role of polynomial features in transforming the input data?

---

**Answer:**

Polynomial features create additional transformed features such as x², x³, etc., allowing linear models to capture nonlinear relationships.

---

## Q5. How does increasing the degree of the polynomial affect model flexibility?

---

**Answer:**

Increasing polynomial degree increases model flexibility and allows fitting more complex patterns. However, too high a degree may lead to overfitting.

---

## Q6. What is the intuition behind fitting a curved relationship using polynomial regression?

---

**Answer:**

Real-world data often follows nonlinear patterns. Polynomial terms help bend the fitted line into curves that better represent the relationship.

---

## Q7. What problem can arise when using a very high-degree polynomial model?

---

**Answer:**

Very high-degree polynomial models can cause:
- Overfitting
- High variance
- Poor generalization
- Numerical instability


---

## Q8. Explain the concept of overfitting in the context of polynomial regression.

---

**Answer:**

Overfitting happens when a model learns training data too closely, including noise and outliers, leading to poor performance on unseen data.

---

## Q9. How can polynomial regression be implemented using feature transformation before applying linear 
---regr
ession?

**Answer:**

Steps:
1. Convert x into polynomial features (x², x³, etc.)
2. Apply Linear Regression on transformed data
3. Predict using trained model

Example:
from sklearn.preprocessing import PolynomialFeatures


---

## Q10. Why is feature scaling important when working with higher-degree polynomial features?

---

**Answer:**

Higher powers can create very large feature values, causing unstable optimization and slow convergence. Scaling keeps values balanced.

---

## Q11. What is the role of regularization (like Ridge/Lasso) in polynomial regression?

---

**Answer:**

Regularization reduces overfitting by penalizing large coefficients:
- Ridge → shrinks coefficients
- Lasso → shrinks and removes unnecessary features


---

## Q12. In what type of real-world scenarios is polynomial regression more suitable than simple linear regression?
---

**Answer:**

Polynomial regression is useful in:
- Stock market trends
- Population growth
- Weather forecasting
- Biological growth patterns
- Sales forecasting
- Medical data analysis


---

