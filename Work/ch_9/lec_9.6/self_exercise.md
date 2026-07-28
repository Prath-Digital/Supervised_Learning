# Self Exercise - 9.6

---

## Q.1  
**What does GaussianNB store during training? Name the two arrays and describe what each contains.**

**Answer:**  
GaussianNB stores two main arrays learned from the training data:

- **`theta_`**: shape `(n_classes, n_features)` – the **mean** of each feature for each class.  
- **`var_`** (formerly `sigma_`): shape `(n_classes, n_features)` – the **variance** of each feature for each class.

These are the parameters of the Gaussian probability density function used for each class-feature pair.

---

## Q.2  
**What is `var_smoothing` in GaussianNB? What problem does it solve, and how does it modify the variance estimate?**

**Answer:**  
`var_smoothing` is a hyperparameter (default `1e-9`) that adds a small amount of variance for numerical stability.

**Problem it solves:**  
When a feature has zero (or near-zero) variance within a class, the Gaussian density involves division by a very small number, which can cause numerical instability or overflow/underflow.

**How it modifies the variance:**  
It computes  
$$
\epsilon = \texttt{var\_smoothing} \times \max(\text{variance of all features})
$$
and then adds this $\epsilon$ to every estimated variance:  
$$
\texttt{var\_}[c, i] \leftarrow \texttt{var\_}[c, i] + \epsilon
$$

---

## Q.3  
**Why is GaussianNB less sensitive to feature scaling than SVM or KNN? Explain from the perspective of how each model uses feature values.**

**Answer:**  
- **GaussianNB** models each feature independently with its own mean and variance. Because the likelihood is based on standardized distance $(x - \mu)/\sigma$, absolute scale differences between features are automatically normalized by the per-feature variance.  
- **SVM** (especially with RBF or linear kernels) and **KNN** rely on Euclidean (or other) distances that treat all dimensions equally. Features with larger numeric ranges dominate the distance calculation, so scaling is critical.

Thus GaussianNB is inherently more robust to unscaled features.

---

## Q.4  
**What does CountVectorizer produce? How does it differ from TfidfVectorizer, and when would you prefer each for Naive Bayes?**

**Answer:**  
- **CountVectorizer** produces a matrix of raw term frequencies (integer counts of each word/token).  
- **TfidfVectorizer** produces a matrix of TF-IDF weights (term frequency × inverse document frequency), which down-weights common words.

**Preference for Naive Bayes:**  
- Prefer **CountVectorizer** with MultinomialNB / ComplementNB when the absolute occurrence counts matter (classic bag-of-words Multinomial model).  
- Prefer **TfidfVectorizer** when you want to reduce the influence of very frequent words; it often improves performance, especially with ComplementNB.

---

## Q.5  
**Write the Laplace-smoothed probability formula for MultinomialNB. What happens when $\alpha=0$ and a word appears in test but not training data?**

**Answer:**  
Laplace-smoothed probability for word $w$ given class $c$:

$$
\hat{P}(w \mid c) = \frac{N_{wc} + \alpha}{N_c + \alpha |V|}
$$

where  
- $N_{wc}$ = count of word $w$ in documents of class $c$,  
- $N_c$ = total word count in class $c$,  
- $|V|$ = vocabulary size,  
- $\alpha$ = smoothing parameter (usually 1 for Laplace).

**When $\alpha=0$:**  
If a word appears in the test set but was never seen in training for that class, $N_{wc}=0$, so $\hat{P}(w \mid c)=0$. The entire posterior for that class becomes zero, and the model cannot make a proper prediction (or assigns zero probability to the class).

---

## Q.6  
**How does BernoulliNB differ from MultinomialNB in what it models? Give one scenario where BernoulliNB outperforms MultinomialNB.**

**Answer:**  
- **MultinomialNB** models the **frequency** (count) of each feature/word.  
- **BernoulliNB** models only the **presence/absence** (binary occurrence) of each feature.

**Scenario where BernoulliNB is better:**  
Short documents or binary feature spaces (e.g., “does this email contain the word ‘viagra’?”) where the mere presence of a term is more informative than how many times it appears. BernoulliNB also works well when features are truly binary indicators.

---

## Q.7  
**What is the Brier score? Write the formula and explain what a Brier score of 0 vs 0.25 means in practice.**

**Answer:**  
The **Brier score** is a proper scoring rule that measures the accuracy of probabilistic predictions:

$$
\text{BS} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2
$$

where $p_i$ is the predicted probability of the positive class and $y_i \in \{0,1\}$ is the true label.

- **Brier score = 0**: Perfect predictions (always predicts probability 1 when the event occurs and 0 when it does not).  
- **Brier score = 0.25**: Equivalent to always predicting 0.5 (a completely uninformative “coin-flip” forecast).

Lower is better.

---

## Q.8  
**What is a calibration curve (reliability diagram)? What does it mean when the curve lies above the diagonal vs below it?**

**Answer:**  
A **calibration curve** (reliability diagram) plots the predicted probability (x-axis) against the observed frequency of the positive class (y-axis), usually in bins.

- Ideal calibration → the curve lies on the diagonal $y=x$.  
- **Curve above the diagonal**: The model is **under-confident** (observed frequency is higher than predicted probability).  
- **Curve below the diagonal**: The model is **over-confident** (observed frequency is lower than predicted probability).

---

## Q.9  
**How do you extract the most informative words per class from a fitted MultinomialNB? What array do you sort and why?**

**Answer:**  
After fitting a MultinomialNB, the log probabilities of features given each class are stored in **`feature_log_prob_`** (shape `(n_classes, n_features)`).

To find the most informative words for a class:

```python
# For class i
top_indices = np.argsort(clf.feature_log_prob_[i])[::-1][:n]
top_words = vectorizer.get_feature_names_out()[top_indices]
```

We sort **`feature_log_prob_`** because higher log-probability means the word is more characteristic of that class.

---

## Q.10  
**What is ComplementNB? When does it outperform MultinomialNB, and what does it model instead of $P(\text{word}|\text{class})$?**

**Answer:**  
**ComplementNB** is a variant of Multinomial Naive Bayes designed for imbalanced datasets (especially text classification).

It outperforms MultinomialNB particularly on **imbalanced class distributions**.

Instead of estimating $P(\text{word} \mid \text{class})$, it estimates the probability of a word given the **complement of the class** (i.e., all other classes). Classification is then based on which class has the poorest complement match.

---

## Q.11  
**GaussianNB achieves 99% training accuracy but 68% test accuracy on a dataset with 50 features and 200 samples. What is the most likely cause, and what two steps would you take to investigate?**

**Answer:**  
**Most likely cause:** Overfitting due to the high feature-to-sample ratio (50 features, only 200 samples). GaussianNB estimates a mean and variance per class per feature; with limited data many of these estimates are unreliable.

**Two investigation steps:**
1. Examine the estimated variances (`var_`) – look for near-zero or extremely small values that indicate unstable estimates.  
2. Evaluate with cross-validation (or a proper train/validation split) and/or apply feature selection / dimensionality reduction (e.g., SelectKBest, PCA) and re-measure the train–test gap.

---

## Q.12  
**Explain `partial_fit()` in GaussianNB. What makes it unique among sklearn classifiers, and when would you use it in production?**

**Answer:**  
`partial_fit(X, y, classes=None)` allows **incremental / online learning**. The model updates its mean and variance estimates from successive batches of data without needing to see the entire dataset at once.

**What makes it unique:**  
GaussianNB is one of the few sklearn classifiers that natively supports true online updates of its sufficient statistics (mean & variance) via a numerically stable algorithm.

**Production use cases:**  
- Streaming data / continuous data arrival  
- Very large datasets that do not fit in memory  
- Situations where the model must be updated frequently with new batches

---

## Q.13  
**A 3-class MultinomialNB model always predicts class 0. What is the most likely cause? How do you fix it using the `class_prior` parameter?**

**Answer:**  
**Most likely cause:** Severe class imbalance – class 0 has a much higher prior probability, so the model almost always selects it.

**Fix with `class_prior`:**  
Force equal (or more balanced) priors:

```python
clf = MultinomialNB(class_prior=[1/3, 1/3, 1/3])
# or any other desired prior distribution
```

This overrides the empirical class frequencies learned from the training data.

---

## Q.14  
**Compare Naive Bayes and Logistic Regression on the generative–discriminative axis. In which data regime (small $n$ or large $n$) does each tend to perform better, and why?**

**Answer:**  
- **Naive Bayes** is a **generative** model: it models the joint distribution $P(X,y) = P(y)P(X|y)$ and then applies Bayes’ rule.  
- **Logistic Regression** is a **discriminative** model: it models the conditional $P(y|X)$ directly.

**Data regimes:**  
- **Small $n$ (few samples):** Naive Bayes often performs better because it has fewer parameters to estimate (thanks to the independence assumption) and can reach its asymptotic error faster.  
- **Large $n$ (many samples):** Logistic Regression tends to outperform because the independence assumption of NB becomes a limiting bias, while LR can learn more flexible decision boundaries given enough data.