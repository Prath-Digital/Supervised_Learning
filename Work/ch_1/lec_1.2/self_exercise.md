# Self Exercise - 1.2

---

## Q.1. What are supervised learning algorithms, and how do they learn from labeled data?

---

### Answer:

Supervised learning algorithms are a type of machine learning algorithm that learn from labeled data. In supervised learning, the algorithm is provided with a dataset that includes input features and corresponding output labels. The goal of the algorithm is to learn a mapping from the input features to the output labels, allowing it to make predictions on new, unseen data.

---

## Q.2. What are some common supervised learning algorithms used in regression tasks?

---

### Answer:

Some common supervised learning algorithms used in regression tasks include:
    - Linear Regression
    - Polynomial Regression
    - Support Vector Regression
    - Random Forest Regression
    - Gradient Boosting Regression (e.g., XGBoost)

---

## Q.3. What are some common supervised learning algorithms used in classification tasks?

---

### Answer:

Some common supervised learning algorithms used in classification tasks include:
    - Logistic Regression
    - Decision Trees
    - Random Forest
    - Support Vector Machines
    - K-Nearest Neighbors

---

## Q.4. Why is supervised learning widely used in real-world applications?

---

### Answer:

Supervised learning is widely used in real-world applications because it can effectively learn from labeled data, which is often available in many domains. It allows for accurate predictions and classifications, making it suitable for tasks such as image recognition, natural language processing, and medical diagnosis. Additionally, supervised learning algorithms can be easily evaluated and fine-tuned using metrics like accuracy, precision, and recall, making them practical for various applications.

---

## Q.5. What is the fundamental difference between regression and classification algorithms?

---

### Answer:

| Feature                 | Regression                                                                                                    | Classification                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Output Type**         | Continuous numerical value (e.g., price, salary, temperature)                                                 | Discrete category or label (e.g., Yes/No, Spam/Ham)                                                             |
| **Question Type**       | “How much?” or “What is the value?”                                                                           | “Which category?” or “Yes or No?”                                                                               |
| **Common Algorithms**   | Linear Regression, Polynomial Regression, Ridge, Lasso, SVR, Decision Tree Regressor, Random Forest Regressor | Logistic Regression, KNN, SVM, Decision Tree Classifier, Random Forest Classifier, Naive Bayes, Neural Networks |
| **Evaluation Metrics**  | MSE, RMSE, MAE, R² Score                                                                                      | Accuracy, Precision, Recall, F1-Score, ROC-AUC                                                                  |
| **Example**             | Predicting house price ($250,000)                                                                             | Predicting whether an email is Spam or Not Spam                                                                 |
| **Scikit-learn Import** | `LinearRegression()`                                                                                          | `LogisticRegression()`                                                                                          |

---

## Q.6. In what type of problems is a regression algorithm preferred?

---

### Answer:

A regression algorithm is preferred in problems where the goal is to predict a continuous numerical value. Examples include predicting house prices, stock prices, or any scenario where the output is a quantity that can take on a wide range of values. Regression algorithms are suitable for tasks that involve forecasting, trend analysis, and any situation where understanding the relationship between input features and a continuous target variable is important.

---

## Q.7. In what type of problems is a classification algorithm preferred?

---

### Answer:

A classification algorithm is preferred in problems where the goal is to predict a discrete category or label. Examples include email spam detection, image recognition (e.g., identifying objects in images), and medical diagnosis (e.g., classifying diseases). Classification algorithms are suitable for tasks that involve categorizing data into distinct classes, making decisions based on input features, and any scenario where the output is a finite set of categories.

---

## Q.8. Why is predicting continuous values considered regression while predicting categories is classification?

---

### Answer:

Predicting continuous values is considered regression because it involves estimating a numerical output that can take on any value within a range. The focus is on modeling the relationship between input features and a continuous target variable. In contrast, predicting categories is classification because it involves assigning input data to discrete classes or labels. The focus is on determining which category an input belongs to based on its features, rather than estimating a numerical value.

---

## Q.9. What is Simple Linear Regression, and what problem does it solve?

---

### Answer:

Simple Linear Regression is a type of regression algorithm that models the relationship between a single independent variable (input feature) and a dependent variable (target) by fitting a linear equation to the observed data. The equation takes the form $\hat{y} = \beta_0 + \beta_1X$ Where $\hat{y}$ is the predicted value, $\beta_0$ is the intercept, $\beta_1$ is the slope of the line, and $X$ is the independent variable. Simple Linear Regression solves problems where there is a linear relationship between the input feature and the target variable, allowing for predictions of the target based on new input values.

---

## Q.10. What are the assumptions of linear regression related to the relationship between variables?

---

### Answer:

![Linear Regression Assumptions](./assets/s_q10.png)
*Note: This image is taken from a part of notes on lecture 1.2, see full notes for details*

---

## Q.11. What does the assumption of "normality of errors" mean in linear regression?

---

### Answer:

The assumption of "normality of errors" in linear regression means that the residuals (the differences between the observed values and the predicted values) are normally distributed. This assumption is important for making valid inferences about the coefficients and for conducting hypothesis tests. If the errors are not normally distributed, it can affect the reliability of confidence intervals and p-values, leading to incorrect conclusions about the significance of predictors.

---

## Q.12. How does the assumption of "no multicollinearity" impact regression model accuracy?

---

### Answer:

The assumption of "no multicollinearity" means that the independent variables (features) in the regression model should not be highly correlated with each other. If multicollinearity is present, it can lead to unstable estimates of the regression coefficients, making it difficult to determine the individual effect of each predictor on the target variable. This can reduce the accuracy of the model and make it less interpretable, as it becomes challenging to identify which features are truly influencing the target variable.

---

*End of the file*
