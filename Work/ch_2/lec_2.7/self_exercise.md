# Self Exercise - 2.7

## Q1. What is Gradient Descent, and why is it used in machine learning?

Gradient Descent is an optimization algorithm used to minimize a model's cost function by iteratively updating parameters in the direction of the negative gradient. It is used in machine learning to help models learn the best parameter values and improve prediction accuracy.

---

## Q2. What is the main objective of Gradient Descent in optimization problems?

The main objective of Gradient Descent is to minimize the cost (loss) function and find optimal parameter values that produce the lowest prediction error.

---

## Q3. Explain the concept of a cost function in Gradient Descent.

A cost function measures how far a model's predictions are from actual values. Gradient Descent attempts to minimize this function during training.

---

## Q4. What does the gradient represent in the context of optimization?

The gradient represents the direction and rate of the steepest increase of a function. In optimization, we move in the opposite direction to reduce the error.

---

## Q5. How does Gradient Descent update model parameters?

Gradient Descent updates parameters using:

θ = θ − α × ∇J(θ)

where:
- θ = parameters
- α = learning rate
- ∇J(θ) = gradient of cost function

---

## Q6. What is the role of the learning rate (α) in Gradient Descent?

The learning rate controls the size of parameter updates. It determines how quickly or slowly Gradient Descent moves toward the minimum.

---

## Q7. What happens if the learning rate is too high or too low?

If too high:
- Model may overshoot the minimum
- Training can become unstable

If too low:
- Training becomes very slow
- May take many iterations to converge

---

## Q8. What is Batch Gradient Descent?

Batch Gradient Descent computes gradients using the entire training dataset before updating model parameters.

---

## Q9. How does Batch Gradient Descent compute the gradient?

It calculates the gradient by averaging the errors across all training samples before updating parameters.

---

## Q10. What are the advantages of Batch Gradient Descent?

Advantages:
- Stable convergence
- Accurate gradient estimation
- Smooth optimization path
- Deterministic results

---

## Q11. What are the limitations of Batch Gradient Descent when working with large datasets?

Limitations:
- Computationally expensive
- Requires high memory
- Slow for large datasets
- Updates happen less frequently

---

## Q12. How does Batch Gradient Descent differ from Stochastic and Mini-Batch Gradient Descent?

Batch Gradient Descent:
Uses all data for one update.

Stochastic Gradient Descent (SGD):
Uses one sample at a time.

Mini-Batch Gradient Descent:
Uses small groups of samples.

Mini-Batch provides a balance between speed and stability.

---

