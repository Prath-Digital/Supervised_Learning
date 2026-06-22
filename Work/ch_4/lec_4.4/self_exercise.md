# Self Exercise - 4.4

## Q1. What is pruning in a decision tree? Distinguish between pre-pruning and post-pruning.

### Answer

Pruning is the process of reducing the size and complexity of a decision tree to improve generalization and prevent overfitting.

The idea is to remove branches that provide little predictive value on unseen data.

### Pre-Pruning (Early Stopping)

The tree growth is stopped before a fully grown tree is created.

Common stopping criteria:

- max_depth
- min_samples_split
- min_samples_leaf
- max_leaf_nodes
- minimum impurity decrease

#### Advantages

- Faster training
- Smaller trees
- Lower computational cost

#### Disadvantages

- May stop too early
- Can miss useful patterns

---

### Post-Pruning

A large tree is grown first and then unnecessary branches are removed.

#### Advantages

- Often achieves better performance
- Makes pruning decisions using more information

#### Disadvantages

- More computationally expensive
- Requires growing the full tree first

---

### Summary

| Aspect | Pre-Pruning | Post-Pruning |
|----------|------------|------------|
| When Applied | During Growth | After Growth |
| Speed | Faster | Slower |
| Risk | Underfitting | Lower |
| Accuracy | Sometimes Lower | Often Better |

---

## Q2. What is Cost-Complexity Pruning (also called Weakest-Link Pruning)? Describe the algorithm step by step.

### Answer

Cost-Complexity Pruning is a post-pruning method used in CART decision trees.

It balances:

- Training error
- Tree complexity

The objective function is:

\[
R_{\alpha}(T)=R(T)+\alpha|T|
\]

where:

- \(R(T)\) = training error
- \(|T|\) = number of leaf nodes
- \(\alpha\) = complexity penalty

---

### Algorithm

#### Step 1

Grow a large tree.

#### Step 2

Compute the effective alpha for every internal node.

#### Step 3

Identify the weakest link.

The weakest link is the branch whose removal causes the smallest increase in training error.

#### Step 4

Prune that branch.

#### Step 5

Repeat until only the root remains.

#### Step 6

Generate a sequence of nested trees.

#### Step 7

Use cross-validation to choose the best tree.

---

## Q4. What does the hyperparameter ccp_alpha represent, and how does increasing it affect the final tree?

### Answer

ccp_alpha is the complexity penalty parameter used in Cost-Complexity Pruning.

The objective function is:

\[
R_{\alpha}(T)=R(T)+\alpha|T|
\]

where:

\[
\alpha = ccp\_alpha
\]

---

### Small ccp_alpha

- Little pruning
- Larger tree
- Lower bias
- Higher variance

---

### Large ccp_alpha

- More aggressive pruning
- Smaller tree
- Higher bias
- Lower variance

---

### Summary

| ccp_alpha | Tree Size | Bias | Variance |
|------------|-----------|--------|-----------|
| Small | Large | Low | High |
| Large | Small | High | Low |

---

## Q4. How does cross-validation help select the optimal value of ccp_alpha or max_depth?

### Answer

Cross-validation estimates how well each hyperparameter value generalizes to unseen data.

### Procedure

1. Select candidate values.
2. Perform K-Fold Cross-Validation.
4. Train trees using each value.
4. Evaluate validation performance.
5. Average scores across folds.
6. Select the value with the best average score.

Example:

\[
ccp\_alpha \in \{0,0.001,0.01,0.1\}
\]

The alpha producing the lowest validation error is selected.

This prevents choosing a tree that merely memorizes training data.

---

## Q5. What is the hypothesis space of a regression tree? What class of functions can it represent?

### Answer

The hypothesis space is the collection of all functions a model can represent.

For regression trees:

\[
f(x)=c_m
\]

for observations belonging to region:

\[
R_m
\]

The tree partitions feature space into rectangular regions.

Within each region:

\[
f(x)
=
\text{constant}
\]

Thus regression trees represent:

### Piecewise Constant Functions

\[
f(x)=
\begin{cases}
c_1 & x\in R_1\\
c_2 & x\in R_2\\
\vdots\\
c_m & x\in R_m
\end{cases}
\]

---

## Q6. Why do decision trees produce step-function (piecewise constant) predictions rather than smooth curves?

### Answer

Each leaf stores a single constant prediction.

Example:

\[
\hat y = 50
\]

for all observations inside a region.

When the input crosses a split boundary:

\[
x < t
\]

to

\[
x \ge t
\]

the prediction suddenly changes.

This creates discontinuities.

Therefore decision trees generate:

- Piecewise constant outputs
- Step functions
- Non-smooth prediction surfaces

---

## Q7. What is feature importance in a Decision Tree? How is it computed from the tree structure?

### Answer

Feature importance measures how useful a feature is for reducing prediction error.

For regression trees:

\[
Importance_j
=
\sum
\text{Reduction in RSS}
\]

over all splits using feature \(j\).

---

### Computation

For every split:

\[
Gain
=
RSS_{parent}
-
(RSS_{left}+RSS_{right})
\]

The gains are summed across the tree.

Finally:

\[
Importance_j
=
\frac{\text{Total Gain of Feature }j}
{\text{Total Gain Across All Features}}
\]

The values are normalized so they sum to 1.

---

## Q8. Name two limitations of feature importance as computed by a Decision Tree. Why can it be misleading?

### Answer

### Limitation 1: Bias Toward High-Cardinality Features

Features with many possible split points often appear artificially important.

Example:

- Customer ID
- ZIP Code

Such features may receive high importance despite weak predictive power.

---

### Limitation 2: Correlated Features

When features are highly correlated:

- One feature may receive most importance
- Another equally useful feature may appear unimportant

This can create misleading interpretations.

---

### Better Alternatives

- Permutation Importance
- SHAP Values

---

## Q9. What is the max_depth hyperparameter? How does it affect bias and variance independently?

### Answer

max_depth controls the maximum number of splits from root to leaf.

---

### Small max_depth

Example:

\[
max\_depth = 2
\]

Effects:

- Simpler tree
- Higher bias
- Lower variance

May underfit.

---

### Large max_depth

Example:

\[
max\_depth = 20
\]

Effects:

- More complex tree
- Lower bias
- Higher variance

May overfit.

---

### Summary

| max_depth | Bias | Variance |
|------------|--------|-----------|
| Small | High | Low |
| Large | Low | High |

---

## Q10. What is min_samples_split? What is min_samples_leaf? How do they differ in effect on the tree?

### Answer

### min_samples_split

Minimum number of observations required before a node can be split.

Example:

\[
min\_samples\_split=10
\]

Nodes with fewer than 10 samples cannot split.

---

### min_samples_leaf

Minimum number of observations allowed in each leaf.

Example:

\[
min\_samples\_leaf=5
\]

Every leaf must contain at least 5 samples.

---

### Difference

| Parameter | Controls |
|------------|-----------|
| min_samples_split | Whether splitting is allowed |
| min_samples_leaf | Minimum leaf size |

Increasing either parameter generally:

- Simplifies the tree
- Reduces overfitting
- Increases bias

---

## Q11. A regression tree has training MSE = 0 and test MSE = 80. What is the diagnosis and what are three hyperparameter-level fixes?

### Answer

### Diagnosis

The model is severely overfitting.

Training error:

\[
MSE_{train}=0
\]

indicates perfect memorization.

Test error:

\[
MSE_{test}=80
\]

indicates poor generalization.

---

### Fix 1

Reduce:

\[
max\_depth
\]

This limits complexity.

---

### Fix 2

Increase:

\[
min\_samples\_leaf
\]

Leaves become larger and more stable.

---

### Fix 3

Increase:

\[
ccp\_alpha
\]

More aggressive pruning removes unnecessary branches.

---

Additional fixes:

- Increase min_samples_split
- Use cross-validation
- Use ensemble methods

---

## Q12. What are the main advantages and disadvantages of Decision Tree Regressors compared to Linear Regression and Random Forests?

### Answer

### Compared to Linear Regression

#### Advantages

- Captures nonlinear relationships
- Handles interactions automatically
- No feature scaling required

#### Disadvantages

- Higher variance
- Less stable
- Stepwise predictions

---

### Compared to Random Forests

#### Advantages

- Easier to interpret
- Faster to explain
- Visualizable

#### Disadvantages

- Lower predictive accuracy
- More prone to overfitting
- Less robust

---

## Summary Table

| Model | Interpretability | Accuracy | Variance |
|---------|-----------------|----------|----------|
| Linear Regression | High | Moderate | Low |
| Decision Tree | High | Good | High |
| Random Forest | Moderate | Excellent | Low |

### Key Takeaway

Decision Trees are excellent for interpretability and capturing nonlinear relationships, but they often require pruning and careful tuning to avoid overfitting. Random Forests usually achieve better predictive performance, while Linear Regression remains preferable when relationships are approximately linear.

