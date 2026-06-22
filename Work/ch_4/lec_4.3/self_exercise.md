# Self Exercise - 4.3

## Q1. What is the objective function minimized at each node of a Decision Tree Regressor? Write it formally.

### Answer

At each node, a Decision Tree Regressor seeks the split that minimizes the Residual Sum of Squares (RSS).

For a split dividing the data into left and right regions:

\[
RSS_{split} = \sum_{i \in R_{left}} (y_i - \hat{y}_{left})^2
+ \sum_{i \in R_{right}} (y_i - \hat{y}_{right})^2
\]

where:

- \(R_{left}\) = left child region
- \(R_{right}\) = right child region
- \(\hat{y}_{left}\) = mean target value in left region
- \(\hat{y}_{right}\) = mean target value in right region

The algorithm chooses the split with the smallest RSS.

---

## Q2. Why is the mean of the target values in a leaf region the optimal prediction under MSE loss? Derive it.

### Answer

Suppose a leaf contains observations:

\[
y_1, y_2, ..., y_n
\]

We wish to find a constant prediction \(c\) minimizing:

\[
L(c)=\sum_{i=1}^{n}(y_i-c)^2
\]

Differentiate:

\[
\frac{dL}{dc}
=
-2\sum_{i=1}^{n}(y_i-c)
\]

Set equal to zero:

\[
\sum_{i=1}^{n}(y_i-c)=0
\]

\[
\sum y_i - nc =0
\]

\[
c=\frac{1}{n}\sum_{i=1}^{n}y_i
\]

Therefore, the mean minimizes MSE loss.

---

## Q4. What is Recursive Binary Splitting? Describe the algorithm step by step in your own words.

### Answer

Recursive Binary Splitting is the process used to build regression trees.

### Steps

1. Start with the entire dataset.
2. Evaluate all possible features.
4. Evaluate candidate split points for each feature.
4. Compute RSS for every candidate split.
5. Select the split producing minimum RSS.
6. Divide the dataset into two child nodes.
7. Repeat the process recursively on each child node.
8. Stop when a stopping criterion is reached.

Examples of stopping criteria:

- Maximum depth reached
- Minimum samples per leaf reached
- No significant RSS reduction

The process is called "recursive" because each child node is split again.

---

## Q4. What are candidate split points? How are they generated from a continuous feature?

### Answer

A candidate split point is a potential threshold used to divide observations.

Suppose a feature contains:

\[
[2, 5, 8, 12]
\]

Candidate splits are generated at midpoints:

\[
\frac{2+5}{2}=4.5
\]

\[
\frac{5+8}{2}=6.5
\]

\[
\frac{8+12}{2}=10
\]

Candidate thresholds become:

\[
4.5,\;6.5,\;10
\]

Each threshold is evaluated using RSS.

The threshold with the smallest RSS is selected.

---

## Q5. What is the Residual Sum of Squares (RSS)? How is it used to evaluate a candidate split?

### Answer

RSS measures prediction error:

\[
RSS=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

Smaller RSS indicates better predictions.

For a candidate split:

\[
RSS_{split}
=
RSS_{left}
+
RSS_{right}
\]

The tree evaluates all candidate splits and selects the split that minimizes total RSS.

---

## Q6. What is a greedy algorithm in the context of decision tree learning? What does it sacrifice?

### Answer

A greedy algorithm makes the best decision at the current step without considering future consequences.

In decision trees:

- The algorithm chooses the split that immediately minimizes RSS.
- It does not look ahead multiple levels.

### Advantages

- Fast
- Computationally efficient
- Easy to implement

### Sacrifice

The algorithm may miss the globally optimal tree because a locally optimal split may not lead to the best overall structure.

---

## Q7. Why is finding the globally optimal regression tree NP-hard?

### Answer

Finding the globally optimal tree requires:

- Evaluating all possible features
- Evaluating all possible split points
- Evaluating all possible tree structures

The number of possible trees grows exponentially with:

- Number of observations
- Number of features
- Tree depth

Because exhaustive search becomes computationally infeasible, the problem is NP-hard.

Therefore practical algorithms use greedy approximations.

---

## Q8. What is a leaf node in a regression tree, and how is its predicted value determined?

### Answer

A leaf node is a terminal node that is not split further.

When a new observation reaches a leaf, the tree outputs a constant value.

The prediction is:

\[
\hat{y}_{leaf}
=
\frac{1}{n}
\sum_{i=1}^{n} y_i
\]

where \(n\) is the number of observations in that leaf.

Thus the prediction equals the mean target value of the observations contained in that leaf.

---

## Q9. What is the depth of a tree, and how does increasing depth affect bias and variance?

### Answer

Tree depth is the maximum number of splits from the root node to a leaf node.

### Shallow Trees

- High bias
- Low variance
- May underfit

### Deep Trees

- Low bias
- High variance
- May overfit

### Summary

| Tree Depth | Bias | Variance |
|------------|-------|----------|
| Small | High | Low |
| Large | Low | High |

Increasing depth typically reduces training error but increases overfitting risk.

---

## Q10. What is overfitting in a decision tree? What happens to training RSS and test RSS as the tree grows deeper?

### Answer

Overfitting occurs when a tree learns noise instead of the true underlying pattern.

As depth increases:

### Training RSS

Generally decreases continuously.

\[
RSS_{train}\downarrow
\]

because the tree becomes more flexible.

### Test RSS

Initially decreases.

After a certain point:

\[
RSS_{test}\uparrow
\]

because the model begins fitting noise.

This produces the classic bias-variance trade-off.

---

## Q11. What are the min_samples_split and min_samples_leaf hyperparameters? How do they control tree complexity?

### Answer

### min_samples_split

Minimum number of samples required to split a node.

Example:

\[
min\_samples\_split = 10
\]

Nodes with fewer than 10 observations cannot be split.

---

### min_samples_leaf

Minimum number of samples required in a leaf node.

Example:

\[
min\_samples\_leaf = 5
\]

Every leaf must contain at least 5 observations.

---

### Effect

Increasing either parameter:

- Produces simpler trees
- Reduces variance
- Reduces overfitting

Decreasing them:

- Produces deeper trees
- Increases variance
- Increases overfitting risk

---

## Q12. How does a regression tree differ from a classification tree in terms of the splitting criterion and the leaf output?

### Answer

### Regression Tree

#### Splitting Criterion

Uses:

- RSS
- MSE reduction
- Variance reduction

#### Leaf Output

Predicts:

\[
\hat{y}
=
\text{Mean of target values}
\]

Output is continuous.

Examples:

- House price
- Temperature
- Sales

---

### Classification Tree

#### Splitting Criterion

Uses:

- Gini Impurity
- Entropy
- Information Gain

#### Leaf Output

Predicts:

- Most common class
- Class probabilities

Output is categorical.

Examples:

- Spam/Not Spam
- Fraud/Not Fraud
- Disease/No Disease

---

## Summary Table

| Feature | Regression Tree | Classification Tree |
|----------|----------------|---------------------|
| Target Variable | Continuous | Categorical |
| Split Criterion | RSS / MSE | Gini / Entropy |
| Leaf Prediction | Mean Value | Majority Class |
| Output Type | Numeric | Category |

