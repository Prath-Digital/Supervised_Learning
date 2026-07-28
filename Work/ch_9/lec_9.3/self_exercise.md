# Self Exercise - 9.3

---

## Detailed Answers

### Q.1 What is a hyperplane in d-dimensional space? Write the equation and explain the role of w and b.

A hyperplane in d-dimensional space is a (d-1)-dimensional subspace that divides the space into two half-spaces. 

**Equation:**  
\[ \mathbf{w} \cdot \mathbf{x} + b = 0 \]  
where:
- \(\mathbf{w}\) is the **normal vector** (weight vector) to the hyperplane, which determines its orientation.
- \(b\) is the **bias** term, which shifts the hyperplane away from the origin.
- \(\mathbf{x}\) is a point in the d-dimensional feature space.

The sign of \(\mathbf{w} \cdot \mathbf{x} + b\) determines which side of the hyperplane a point lies on.

### Q.2 Define the margin in an SVM classifier. Write the formula for margin width and explain why maximising it is the objective.

The **margin** is the distance between the decision boundary (hyperplane) and the closest data points (support vectors) from each class.

**Margin width formula:**  
The functional margin is \( y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \).  
The geometric margin for the hyperplane is:  
\[ \frac{2}{\|\mathbf{w}\|} \]

**Why maximise it?**  
Maximising the margin leads to better generalization and robustness to unseen data. A wider margin reduces overfitting and provides a larger separation between classes, improving classification confidence and performance on new examples.

### Q.3 What are support vectors? Why do only support vectors determine the decision boundary?

**Support vectors** are the data points that lie closest to the decision boundary (on the margin boundaries, i.e., where \( y_i (\mathbf{w} \cdot \mathbf{x}_i + b) = 1 \)).

Only support vectors determine the decision boundary because:
- The optimization problem is constrained by these points.
- Moving or removing non-support vectors does not affect the hyperplane (as long as they remain correctly classified with sufficient margin).
- The solution depends solely on the Lagrange multipliers corresponding to support vectors (α_i > 0 in dual form).

### Q.4 Write the hard-margin SVM primal optimisation problem. What condition must the data satisfy for a solution to exist?

**Primal optimization problem (Hard-margin SVM):**  
Minimize:  
\[ \frac{1}{2} \|\mathbf{w}\|^2 \]  

Subject to:  
\[ y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \forall i = 1, \dots, n \]

**Condition for solution to exist:** The data must be **linearly separable** — there must exist a hyperplane that perfectly separates the two classes with no points in the wrong half-space.

### Q.5 What is the geometric margin of a point x_i with label y_i? Write the formula and compute it for w=(2,0), b=-4, x=(3,1), y=+1.

The **geometric margin** of a point \(\mathbf{x}_i\) is the perpendicular distance from the point to the decision hyperplane.  

**Formula:**  
\[ \gamma_i = \frac{y_i (\mathbf{w} \cdot \mathbf{x}_i + b)}{\|\mathbf{w}\|} \]

**Computation:**  
\(\mathbf{w} = (2, 0)\), \(b = -4\), \(\mathbf{x} = (3, 1)\), \(y = +1\)  

\(\mathbf{w} \cdot \mathbf{x} + b = 2\cdot3 + 0\cdot1 - 4 = 6 - 4 = 2\)  
\(\|\mathbf{w}\| = \sqrt{2^2 + 0^2} = 2\)  
Geometric margin: \(\gamma = \frac{1 \cdot 2}{2} = 1\)

### Q.6 What is a slack variable ξ_i in soft-margin SVM? What does ξ=0, 0<ξ≤1, and ξ>1 each mean geometrically?

Slack variables \(\xi_i \geq 0\) allow for some misclassifications or points within the margin to handle non-separable data.

**Geometric interpretations:**
- \(\xi_i = 0\): Point is correctly classified and lies **outside or on** the margin boundary.
- \(0 < \xi_i \leq 1\): Point is correctly classified but lies **inside** the margin (between the hyperplane and the margin boundary).
- \(\xi_i > 1\): Point is **misclassified** (on the wrong side of the hyperplane).

### Q.7 What is the role of C in soft-margin SVM? Describe the behaviour of the decision boundary as C→0 and C→∞.

**Role of C:** C is the **regularization parameter** that controls the trade-off between maximizing the margin and minimizing classification errors (sum of slack variables).

- As \(C \to \infty\): Approaches hard-margin SVM. Prioritizes zero violations → narrower margin if needed, but tries to classify all points correctly. Can lead to overfitting.
- As \(C \to 0\): Allows many violations. Focuses on large margin even if many points are misclassified → smoother, more general decision boundary (underfitting risk).

### Q.8 Write the hinge loss formula. Show that it is zero for correctly classified points with margin ≥1 and grows linearly for violations.

**Hinge loss for a single point:**  
\[ L(y, f(\mathbf{x})) = \max(0, 1 - y \cdot f(\mathbf{x})) \]  
where \( f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b \).

**Properties:**
- If \( y \cdot f(\mathbf{x}) \geq 1 \) (correctly classified with margin ≥1): Loss = 0.
- If \( y \cdot f(\mathbf{x}) < 1 \): Loss = \(1 - y \cdot f(\mathbf{x})\), which grows **linearly** as the violation increases.

### Q.9 What is the kernel trick? Why does the SVM dual formulation allow kernels to be substituted without computing φ(x) explicitly?

The **kernel trick** allows SVMs to operate in high-dimensional (or infinite-dimensional) feature spaces without explicitly computing the mapping φ(x).

In the **dual formulation**, the optimization depends only on inner products \(\mathbf{x}_i \cdot \mathbf{x}_j\). Replacing this with a kernel function \( K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j) \) implicitly works in the feature space defined by φ without ever computing φ(x).

### Q.10 Write the RBF kernel formula. What does γ control? Describe the decision boundary when γ is very large vs very small.

**RBF (Radial Basis Function) Kernel:**  
\[ K(\mathbf{x}, \mathbf{x}') = \exp\left( -\gamma \|\mathbf{x} - \mathbf{x}'\|^2 \right) \]

**γ controls** the influence/reach of a single training example (inverse of variance in Gaussian).

- **Very large γ**: Decision boundary becomes very wiggly/tight around training points (low bias, high variance → overfitting).
- **Very small γ**: Decision boundary becomes smoother, almost linear-like (high bias, low variance → underfitting).

### Q.11 What is Mercer's theorem? Why does it matter when choosing a kernel function for SVM?

**Mercer's theorem** states that a continuous, symmetric kernel function \(K(\mathbf{x}, \mathbf{x}')\) is a valid (positive semi-definite) kernel if and only if the Gram matrix is positive semi-definite for any set of points. This guarantees the existence of a feature map φ such that \(K(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle\).

It matters because only Mercer kernels ensure the SVM optimization problem remains convex and has a global optimum.

### Q.12 Compare Linear, Polynomial, and RBF kernels: when would you choose each? Give one real-world use case per kernel.

- **Linear Kernel** \( K(\mathbf{x}, \mathbf{x}') = \mathbf{x} \cdot \mathbf{x}' \): Fast, works well in high dimensions. Choose when data is linearly separable or approximately so.  
  *Use case:* Text classification (e.g., spam detection with TF-IDF features).

- **Polynomial Kernel** \( K(\mathbf{x}, \mathbf{x}') = (\mathbf{x} \cdot \mathbf{x}' + c)^d \): Captures interactions up to degree d. Choose for data with moderate polynomial relationships.  
  *Use case:* Image recognition with hand-crafted polynomial features.

- **RBF Kernel**: Most flexible, maps to infinite dimensions. Choose for complex, non-linear decision boundaries when data size is moderate.  
  *Use case:* Handwriting recognition or bioinformatics (protein classification).

### Q.13 A hard-margin SVM fails on a dataset. Name two possible reasons and state how soft-margin SVM addresses each.

1. **Data is not linearly separable** (overlapping classes or noise): Soft-margin introduces slack variables to allow some misclassifications.
2. **Outliers** that force the margin to be very small: Soft-margin down-weights the influence of outliers via the C parameter, allowing a larger overall margin.

### Q.14 Show that soft-margin SVM is equivalent to L2-regularised hinge-loss minimisation. Write both forms and identify the correspondence between C and the regularisation strength λ.

**Soft-margin SVM Primal:**  
\[ \min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \]  
s.t. \( y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i \), \(\xi_i \geq 0\)

**Equivalent L2-regularised Hinge Loss:**  
\[ \min_{\mathbf{w}, b} \frac{\lambda}{2} \|\mathbf{w}\|^2 + \frac{1}{n} \sum_{i=1}^n \max(0, 1 - y_i (\mathbf{w} \cdot \mathbf{x}_i + b)) \]

**Correspondence:** \( C = \frac{1}{n\lambda} \) (or equivalently \(\lambda = \frac{1}{nC}\)). C controls the penalty for errors (inverse to regularization strength λ).