# Self Exercise - 2.4

---

## Q.1. What is Multiple Linear Regression, and how does it differ from Simple Linear Regression?

---

**Answer:**

Multiple Linear Regression is a statistical technique that models the relationship between a dependent variable and two or more independent variables. It differs from Simple Linear Regression, which only considers one independent variable.

---

## Q.2. Write the general mathematical equation of Multiple Linear Regression.

---

**Answer:**

$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon$

---

## Q.3. What do the terms dependent variable ($Y$) and independent variables ($X_1$, $X_2$, ... , $X_n$) represent?

---

**Answer:**

- $Y$ represents the dependent variable (the outcome we want to predict).
- $X_1, X_2, \ldots, X_n$ represent the independent variables (the predictors or features used to make the prediction).

---

## Q.4. What is the meaning of coefficients ($\beta_0$, $\beta_1$, $\beta_2$, ... ) in the regression equation?

---

**Answer:**

The coefficients ($\beta_0$, $\beta_1$, $\beta_2$, ... ) represent the estimated parameters of the regression model. $\beta_0$ is the intercept, and $\beta_1$, $\beta_2$, ..., $\beta_n$ are the slopes for each independent variable.

---

## Q.5. What is the role of the intercept ($\beta_0$) in the model?

---

**Answer:**

The intercept ($\beta_0$) represents the expected value of the dependent variable when all independent variables are zero.

---

## Q.6. What is the error term ($\epsilon$) in Multiple Linear Regression, and why is it important?

---

**Answer:**

The error term ($\epsilon$) represents the difference between the observed values of the dependent variable and the values predicted by the regression model. It accounts for the variability in the dependent variable that cannot be explained by the independent variables. The error term is important because it captures the noise or unexplained variation in the data, which is essential for understanding the limitations of the model and making accurate predictions.

---

## Q.7. How can Multiple Linear Regression be represented in matrix form?

---

**Answer:**

Multiple Linear Regression can be represented in matrix form as:
$$Y = X\beta + \epsilon$$
Where:
- $Y$ is the vector of observed dependent variable values.
- $X$ is the design matrix containing the independent variable values (including a column of ones for the intercept).
- $\beta$ is the vector of coefficients (parameters) to be estimated.
- $\epsilon$ is the vector of error terms.

---

## Q.8. What is the significance of representing regression using matrices?

---

**Answer:**

Representing regression using matrices allows for a more compact and computationally efficient way to express and solve the regression problem, especially when dealing with multiple independent variables.

---

## Q.9. Explain the concept of design matrix (X) in regression.

---

**Answer:**

The design matrix (X) is a matrix that contains the values of the independent variables for all observations in the dataset. Each row corresponds to an observation, and each column corresponds to an independent variable. The first column typically consists of ones to account for the intercept term in the regression model.

---

## Q.10. What does the vector $\beta$ (beta) represent in matrix formulation?

---

**Answer:**

The vector $\beta$ represents the coefficients (parameters) of the regression model that we want to estimate. Each element of $\beta$ corresponds to the effect of a specific independent variable on the dependent variable.

---

## Q.11. What is the meaning of the equation $Y = X\beta + \epsilon$?

---

**Answer:**

The equation $Y = X\beta + \epsilon$ represents the Multiple Linear Regression model in matrix form. It states that the observed values of the dependent variable ($Y$) can be expressed as a linear combination of the independent variables (through the design matrix $X$ and coefficients $\beta$) plus an error term ($\epsilon$) that accounts for the variability not explained by the model.

---

## Q.12. Why is matrix formulation useful for solving regression problems computationally?

---

**Answer:**

Matrix formulation allows for the use of linear algebra techniques to efficiently compute the coefficients of the regression model, especially when dealing with large datasets and multiple independent variables. It enables the application of methods such as least squares estimation to find the best-fitting line or hyperplane in a computationally efficient manner.

---