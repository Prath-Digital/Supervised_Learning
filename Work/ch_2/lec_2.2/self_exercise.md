# Self Exercise - 2.2

---

## Q1. What is the objective of the Least Squares Method in linear regression?

---

**Answer:**

The objective of the Least Squares Method is to find the line that minimizes the sum of squared differences between the observed values and the predicted values from the regression model.

---

## Q2. Why do we minimize the Sum of Squared Errors (SSE) instead of absolute errors?

---

**Answer:**

We minimize SSE because it penalizes larger errors more than smaller ones, leading to a more stable and unique solution. Additionally, SSE is differentiable, which allows for easier optimization using calculus.

---

## Q3. What is the mathematical expression for the Sum of Squared Errors (SSE) in linear regression?

---

**Answer:**

The SSE expression is:
$$
SSE = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_i))^2
$$

---

## Q4. How do partial derivatives help in finding the optimal values of $\beta_0$ and $\beta_1$?

---

**Answer:**

Partial derivatives with respect to $\beta_0$ and $\beta_1$ allow us to determine how the SSE changes as we adjust these coefficients. Setting these derivatives to zero helps us find the points where the SSE is minimized, leading to the optimal values of $\beta_0$ and $\beta_1$.

---

## Q5. What does setting the partial derivatives of SSE to zero achieve?

---

**Answer:**

Setting the partial derivatives of SSE to zero identifies the critical points where the SSE is at a minimum, which corresponds to the best-fitting line for the data.

---

## Q6. What is the closed-form formula for estimating the slope coefficient $\beta_1$?

---

**Answer:**

The formula for estimating $\beta_1$ is:
$$
\beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
$$
---

## Q7. What does the term $\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$ represent in the $\beta_1$ formula?

---

**Answer:**

The term $\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$ represents the covariance between x and y, which measures how x and y vary together.

---

## Q8. What is the closed-form formula for estimating the intercept coefficient $\beta_0$?

---

**Answer:**

The formula is:

$$
\beta_0 = \bar{y} - \beta_1 \bar{x}
$$

---

## Q9. Why is the mean of $x$ and mean of $y$ used in the intercept calculation?

---

**Answer:**

The means of x and y are used in the intercept calculation to ensure that the regression line passes through the point of averages, which helps in accurately positioning the line relative to the data.

---

## Q10. How does the variability of $x$ affect the slope $\beta_1$?

---

**Answer:**

Greater variability in x increases the denominator of the $\beta_1$ formula, which affects the sensitivity and stability of the slope estimate.

---

## Q11. What happens to $\beta_1$ if x and y are uncorrelated?

---

**Answer:**

If x and y are uncorrelated, their covariance becomes close to zero, making $\beta_1$ approximately zero.

---

## Q12. Why is the least squares estimation method considered unbiased under standard regression assumptions?

---

**Answer:**

Under standard assumptions such as linearity, independence, homoscedasticity, and zero-mean errors, the expected values of the estimated coefficients equal the true coefficients, making the estimators unbiased.

---

