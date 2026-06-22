# Self Exercise - 4.8

## Q1

### Question

What is the ε-insensitive loss function? Write it mathematically and explain why points inside the tube incur zero loss.

### Answer

_Write your answer here._

---

## Q2

### Question

Write the SVR primal optimisation problem in full. Identify every symbol and explain its role.

### Answer

_Write your answer here._

---

## Q3

### Question

What do the slack variables ξᵢ and ξᵢ* represent geometrically? When are they non-zero?

### Answer

_Write your answer here._

---

## Q4

### Question

What does the hyperparameter C control in SVR? What happens to the model as C → 0 and as C → ∞?

### Answer

_Write your answer here._

---

## Q5

### Question

What does the hyperparameter ε control? What is the effect of a very large ε on the number of support vectors?

### Answer

_Write your answer here._

---

## Q6

### Question

Write the SVR dual objective. Why is the dual formulation important? What does it enable that the primal alone does not?

### Answer

_Write your answer here._

---

## Q7

### Question

What is a support vector in SVR? How does it differ from the definition in Support Vector Classification?

### Answer

_Write your answer here._

---

## Q8

### Question

State the KKT complementary slackness conditions for SVR and explain what each condition implies about a data point's position relative to the tube.

### Answer

_Write your answer here._

---

## Q9

### Question

Explain the kernel trick: how does replacing xᵢᵀxⱼ with K(xᵢ, xⱼ) in the dual enable non-linear regression without explicit feature maps?

### Answer

_Write your answer here._

---

## Q10

### Question

Compare the RBF kernel and the polynomial kernel: what does each assume about the data, and when would you prefer one over the other?

### Answer

_Write your answer here._

---

## Q11

### Question

How does SVR differ from Ridge Regression? Compare their loss functions, optimisation problems, and sensitivity to outliers.

### Answer

_Write your answer here._

---

## Q12

### Question

What is the computational complexity of SVR prediction at test time in terms of n (training samples)? Why is it often smaller in practice than the worst case?

### Answer

_Write your answer here._

---

## Q13

### Question

What is the computational complexity of SVR prediction at test time in terms of n (training samples)? Why is it often smaller in practice than the worst case?

### Answer

_Write your answer here._

---

## Q14

### Question

What are model.support_vectors_ and model.dual_coef_ in sklearn SVR? How do they relate to the dual formulation from Part 1?

### Answer

_Write your answer here._

---

## Q15

### Question

How does the SVR prediction formula work at test time? Write it out using support vectors and dual coefficients.

### Answer

_Write your answer here._

---

## Q16

### Question

What is the effect of increasing C from 0.01 to 1000 on (a) the number of support vectors, (b) training MSE, and (c) test MSE?

### Answer

_Write your answer here._

---

## Q17

### Question

What is the effect of increasing ε from 0.01 to 0.2 on (a) the tube width, (b) the number of support vectors, and (c) bias–variance trade-off?

### Answer

_Write your answer here._

---

## Q18

### Question

For the RBF kernel, what does gamma hyperparameter control? What is the effect of very high gamma vs very low gamma on the fitted curve?

### Answer

_Write your answer here._

---

## Q19

### Question

You have a dataset with 10,000 samples and 50 features. Should you use SVR or Random Forest Regression? Justify your choice.

### Answer

_Write your answer here._

---

## Q20

### Question

How does SVR handle outliers compared to ordinary least squares regression? Why is this a consequence of the ε-insensitive loss?

### Answer

_Write your answer here._

---

## Q21

### Question

What is LinearSVR in sklearn and how does it differ from SVR(kernel='linear') in terms of computational complexity?

### Answer

_Write your answer here._

---

## Q22

### Question

A train–test error gap of 40 units is observed with SVR(C=100, epsilon=0.01, kernel='rbf', gamma=10). List three hyperparameter changes and the direction of each to close the gap.

### Answer

_Write your answer here._

---

## Q23

### Question

How would you interpret a residual plot where most residuals are exactly 0 for SVR? What does that tell you about ε and the data?

### Answer

_Write your answer here._

---

## Q24

### Question

Compare SVR to Gradient Boosting Regression: what are the key differences in how they handle non-linearity, outliers, and computational scaling with dataset size?

### Answer

_Write your answer here._

---

