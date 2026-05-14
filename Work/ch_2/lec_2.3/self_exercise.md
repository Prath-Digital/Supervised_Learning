# Self Exercise - 2.3

---

Q.1. What is the purpose of evaluation metrics in simple linear regression?

---

**Answer:**

The purpose of evaluation metrics in simple linear regression is to assess the performance of the regression model. They help in quantifying how well the model fits the data and how accurately it predicts the target variable. Evaluation metrics provide insights into the model's accuracy, precision, and overall effectiveness in capturing the underlying relationship between the independent and dependent variables.

---

Q.2. Define Mean Squared Error (MSE) and explain how it is calculated.

---

**Answer:**

Mean Squared Error (MSE) is a commonly used evaluation metric in regression analysis. It measures the average squared difference between the predicted and actual values. MSE is calculated by taking the sum of squared errors and dividing it by the number of observations.

---

Q.3. What are the advantages and disadvantages of using MSE?

---

**Answer:**

Advantages of MSE:
- It penalizes larger errors more heavily due to the squaring, which can be useful when large errors are particularly undesirable.
- It is differentiable, making it suitable for optimization algorithms.

Disadvantages of MSE:
- It is sensitive to outliers because of the squaring operation.
- The units of MSE are the square of the units of the target variable, which can make interpretation difficult.

---

Q.4. Define Mean Absolute Error (MAE) and explain its significance.

---

**Answer:**

Mean Absolute Error (MAE) is a evaluation metric that measures the average absolute difference between the predicted and actual values. It is less sensitive to outliers compared to MSE and provides a more intuitive measure of error in the same units as the target variable.


---

Q.5. How is MAE different from MSE in terms of sensitivity to outliers?

---

**Answer:**

MAE is less sensitive to outliers compared to MSE because it uses the absolute difference rather than the squared difference. This means that extreme values have a lesser impact on the overall error measurement in MAE than they do in MSE.


---

Q.6. What is Root Mean Squared Error (RMSE) and why is it commonly used?

---

**Answer:**

Root Mean Squared Error (RMSE) is the square root of the Mean Squared Error (MSE). It provides a measure of the average magnitude of the errors in the same units as the target variable, making it more interpretable than MSE. RMSE is commonly used because it penalizes larger errors more heavily and is differentiable, which makes it suitable for optimization algorithms.


---

Q.7. How is RMSE related to MSE?

---

**Answer:**

RMSE is the square root of MSE. This means that RMSE is always a positive value and is in the same units as the target variable, making it more interpretable than MSE.

---

Q.8. What is the R2 Score (Coefficient of Determination)?

---

**Answer:**

The R2 Score, also known as the Coefficient of Determination, is a statistical measure that represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, where a value of 1 indicates that the model perfectly explains the variance in the target variable, while a value of 0 indicates that the model does not explain any of the variance. Negative values can occur when the model performs worse than a horizontal line (mean of the target variable).

---

Q.9. What does an R2 value of 0, 1, and negative indicate?

---

**Answer:**

- An R2 value of 0 indicates that the model does not explain any of the variance in the target variable.
- An R2 value of 1 indicates that the model perfectly explains the variance in the target variable.
- A negative R2 value indicates that the model performs worse than a horizontal line (mean of the target variable).

---

Q.10. What is the limitation of using only R2 score for model evaluation?

---

**Answer:**

Using only the R2 score for model evaluation has several limitations:
- It does not provide information about the magnitude of the errors.
- It can be misleading when comparing models with different numbers of predictors.
- It does not indicate whether the model is overfitting or underfitting.
- It assumes a linear relationship between the predictors and the target variable.

---

Q.11. Define Adjusted R2 Score and explain how it differs from R2.

---

**Answer:**

Adjusted R2 Score is a modified version of the R2 score that adjusts for the number of predictors in the model. It penalizes the addition of irrelevant predictors, making it a more reliable metric for comparing models with different numbers of predictors.

---

Q.12. Why is Adjusted R2 preferred over R2 when dealing with multiple predictors?

---

**Answer:**

Adjusted R2 is preferred over R2 when dealing with multiple predictors because it accounts for the number of predictors in the model. While R2 can increase with the addition of more predictors, even if they are not relevant, Adjusted R2 will only increase if the new predictor improves the model's performance. This makes Adjusted R2 a more reliable metric for evaluating models with multiple predictors.

---
