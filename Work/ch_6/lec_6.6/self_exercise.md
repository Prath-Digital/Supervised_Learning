# Self Exercise - 6.6

---

## Q1
**Question**: Write the Gini Impurity formula. What is the maximum possible Gini value for a binary classification problem and when does it occur?

**Answer**:

**Gini Impurity Formula**:
$$
Gini = 1 - \sum_{i=1}^{c} p_i^2
$$
where $p_i$ is the proportion of samples belonging to class $i$.

**Maximum Gini for binary classification**: **0.5**

It occurs when classes are perfectly balanced (50% each class).

---

## Q2
**Question**: Write the Shannon Entropy formula. What is the maximum possible entropy for a binary classification problem and when does it occur?

**Answer**:

**Shannon Entropy Formula**:
$$
Entropy = -\sum_{i=1}^{c} p_i \log_2(p_i)
$$

**Maximum Entropy for binary classification**: **1**

It occurs at 50-50 class distribution.

---

## Q3
**Question**: What is Information Gain? Write the formula. Why is Information Gain biased toward features with many categories?

**Answer**:

**Information Gain (IG)** is the reduction in impurity after splitting.

**Formula**:
$$
IG = Impurity(parent) - \sum_{i=1}^{k} \frac{n_i}{n} \times Impurity(child_i)
$$

**Bias toward many categories**: Features with high cardinality can create many small pure nodes, inflating IG even if the split isn't meaningful for generalization.

---

## Q4
**Question**: What is the CART algorithm? How does it use Gini Impurity to choose splits?

**Answer**:

**CART** (Classification And Regression Tree) is a binary recursive partitioning algorithm.

It selects the split that maximizes **Gini Gain** (reduction in weighted Gini impurity) at each node.

---

## Q5
**Question**: What is the difference between Gini Impurity and Entropy as splitting criteria? Which does sklearn's DecisionTreeClassifier use by default?

**Answer**:

**Differences**:
- Gini is computationally faster (no logs).
- Entropy is slightly more sensitive.
- Both yield very similar trees in practice.

**sklearn default**: `criterion='gini'`

---

## Q6
**Question**: What does a pure node mean in a Decision Tree? What is its Gini Impurity? What is its Entropy?

**Answer**:

A **pure node** contains samples from only **one class**.

- Gini Impurity = **0**
- Entropy = **0**

---

## Q7
**Question**: What shape are the decision regions of a Decision Tree Classifier? Why can't a Decision Tree represent a diagonal boundary without many splits?

**Answer**:

Decision Trees produce **axis-aligned rectangular** decision boundaries.

They require many splits to approximate diagonal boundaries (staircase pattern) because each split is parallel to one axis.

---

## Q8
**Question**: Explain the greedy nature of Decision Tree learning. Why does the CART algorithm not guarantee a globally optimal tree?

**Answer**:

Decision Trees are **greedy** — they choose the best split at the current node without looking ahead.

Finding the globally optimal tree is NP-hard, so greedy heuristics are used, which may lead to suboptimal trees.

---

## Q9
**Question**: What is the stopping criterion for growing a Decision Tree? Name four conditions that stop further splitting.

**Answer**:

Stopping criteria include:

1. Node reaches **maximum depth** (`max_depth`)
2. Node has fewer than **min_samples_split** samples
3. Child nodes would have fewer than **min_samples_leaf** samples
4. Node becomes **pure** (all samples same class)

---

## Q10
**Question**: What is overfitting in a Decision Tree Classifier? Which hyperparameter is the most direct control and in which direction should you adjust it to reduce overfitting?

**Answer**:

**Overfitting**: Tree becomes too complex and memorizes noise in training data.

**Best hyperparameter**: `max_depth`

**To reduce overfitting**: **Decrease** `max_depth`

---

## Q11
**Question**: A node contains 8 samples: 6 of class A, 2 of class B. Compute its Gini Impurity and its Entropy. Show all arithmetic.

**Answer**:

$p_A = 0.75$, $p_B = 0.25$

**Gini**:
$$
1 - (0.75^2 + 0.25^2) = 1 - (0.5625 + 0.0625) = 0.375
$$

**Entropy**:
$$
-(0.75 \log_2 0.75 + 0.25 \log_2 0.25) \approx 0.811
$$

---

## Q12
**Question**: A split produces two child nodes: left child has 4A, 0B; right child has 2A, 4B. Compute the weighted Gini after the split. How much Gini Gain does this split produce given the parent from the previous question?

**Answer**:

Parent Gini = **0.375**

Left (pure): Gini = **0**

Right (2A,4B): 
$$
Gini = 1 - (1/3)^2 - (2/3)^2 = 4/9 \approx 0.444
$$

Weighted Gini:
$$
(4/8)\times0 + (6/8)\times(4/9) = 1/3 \approx 0.333
$$

**Gini Gain**: $0.375 - 0.333 = 0.042$

---
**End of Document**