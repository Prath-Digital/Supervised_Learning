# Self Exercise - 6.7

## Q1

### Question

Why must you use stratify=y when splitting classification data? What
problem does it prevent?

### Answer

Using `stratify=y` preserves class proportions in the train and test
sets, preventing class imbalance after splitting and ensuring reliable
model evaluation.

------------------------------------------------------------------------

## Q2

### Question

What does export_text() output for a Decision Tree? What information
does each line contain?

### Answer

`export_text()` prints a readable text representation of the tree. Each
line shows the split feature, threshold, indentation (depth), and
predicted class for leaf nodes.

------------------------------------------------------------------------

## Q3

### Question

What does clf.feature_importances\_ measure in a Decision Tree
Classifier? How is it computed from the tree structure?

### Answer

It measures each feature's contribution based on the weighted reduction
in impurity (Gini/Entropy) across all splits using that feature. The
values sum to 1.

------------------------------------------------------------------------

## Q4

### Question

What is the difference between max_depth and min_samples_split? How does
each prevent overfitting?

### Answer

`max_depth` limits tree depth, while `min_samples_split` specifies the
minimum number of samples required to split a node. Both reduce
excessive tree growth.

------------------------------------------------------------------------

## Q5

### Question

What does min_samples_leaf do? Give a scenario where setting
min_samples_leaf=5 is better than max_depth=3.

### Answer

It enforces a minimum number of samples in every leaf. On noisy
datasets, it allows deeper but more reliable trees than simply
restricting depth.

------------------------------------------------------------------------

## Q6

### Question

What does the filled=True option in plot_tree show? What does a very
pale node indicate about class purity?

### Answer

`filled=True` colors nodes by class and purity. A pale node indicates
mixed classes and therefore low purity.

------------------------------------------------------------------------

## Q7

### Question

What is the shape of the decision boundary of a Decision Tree
Classifier? Why is this a limitation for some datasets?

### Answer

Decision trees create axis-aligned rectangular boundaries, making them
less suitable for diagonal or curved decision regions.

------------------------------------------------------------------------

## Q8

### Question

A Decision Tree Classifier has train accuracy=1.00 and test
accuracy=0.60. List three hyperparameter changes with direction to fix
this.

### Answer

Decrease `max_depth`, increase `min_samples_split`, increase
`min_samples_leaf` (or increase `ccp_alpha`).

------------------------------------------------------------------------

## Q9

### Question

A Decision Tree Classifier has train accuracy≈0.62 and test
accuracy≈0.61. List three changes to improve performance without
overfitting.

### Answer

Increase `max_depth`, decrease `min_samples_split`, decrease
`min_samples_leaf`, and engineer better features.

------------------------------------------------------------------------

## Q10

### Question

How does class_weight='balanced' affect a Decision Tree Classifier? When
should you use it?

### Answer

It assigns higher weights to minority classes, reducing bias toward
majority classes. Use it for imbalanced datasets.

------------------------------------------------------------------------

## Q11

### Question

What is cost-complexity pruning (ccp_alpha) in sklearn? Describe the
procedure to select the optimal ccp_alpha using cross-validation.

### Answer

Generate candidate alphas with `cost_complexity_pruning_path()`,
evaluate them using cross-validation, choose the best alpha, and retrain
the model.

------------------------------------------------------------------------

## Q12

### Question

Compare Decision Tree Classifier vs Random Forest Classifier: which has
lower bias? Which has lower variance? Which is more interpretable? Which
is generally more accurate on tabular data?

### Answer

Random Forest generally has lower variance, lower bias, and higher
accuracy, while a Decision Tree is much easier to interpret.

------------------------------------------------------------------------
