# Self Exercise - 9.5

---

### Q-1

**Write the definition of conditional probability $P(A|B)$. Why is $P(A|B)$ generally different from $P(B|A)$?**

**Answer:**  
The conditional probability of event $A$ given that event $B$ has occurred is defined as

$$
P(A|B) = \frac{P(A \cap B)}{P(B)} \quad \text{(provided $P(B) > 0$)}
$$

$P(A|B)$ is generally different from $P(B|A)$ because the two quantities answer different questions:

- $P(A|B)$ is the probability of $A$ occurring _given that $B$ has already occurred_.
- $P(B|A)$ is the probability of $B$ occurring _given that $A$ has already occurred_.

These are equal only in special cases (e.g., when $P(A) = P(B)$). In general they differ because the sample spaces being conditioned upon are different.

---

### Q-2

**State the chain rule of probability for two events. Extend it to three events $A$, $B$, $C$.**

**Answer:**  
**For two events:**

$$
P(A \cap B) = P(A) \cdot P(B|A) = P(B) \cdot P(A|B)
$$

**For three events:**

$$
P(A \cap B \cap C) = P(A) \cdot P(B|A) \cdot P(C|A \cap B)
$$

(The order of conditioning can be rearranged in any convenient way.)

---

### Q-3

**What does it mean for two events to be statistically independent? Write the formal condition and give one real-world example of independent events.**

**Answer:**  
Two events $A$ and $B$ are statistically independent if the occurrence of one does not change the probability of the other.

**Formal condition:**

$$
P(A \cap B) = P(A) \cdot P(B)
$$

(equivalently $P(A|B) = P(A)$ or $P(B|A) = P(B)$).

**Real-world example:**  
The outcome of a fair coin toss (heads/tails) and the outcome of rolling a fair die (1-6) are independent events.

---

### Q-4

**State the Law of Total Probability. When is it needed, and how is it used as the denominator in Bayes’ theorem?**

**Answer:**  
**Law of Total Probability:**  
If $\{B_1, B_2, \dots, B_n\}$ is a partition of the sample space (mutually exclusive and exhaustive), then for any event $A$

$$
P(A) = \sum_{i=1}^{n} P(A|B_i) P(B_i)
$$

**When needed:**  
Whenever we know the conditional probabilities $P(A|B_i)$ and the prior probabilities $P(B_i)$, but we need the unconditional (marginal) probability $P(A)$.

**Use in Bayes’ theorem:**  
It supplies the denominator (the marginal likelihood / evidence):

$$
P(B_j|A) = \frac{P(A|B_j)P(B_j)}{\sum_{i} P(A|B_i)P(B_i)}
$$

---

### Q-5

**Write Bayes' theorem and name every term: prior, likelihood, marginal likelihood, and posterior.**

**Answer:**

$$
P(H|D) = \frac{P(D|H) \, P(H)}{P(D)}
$$

- **Prior** $P(H)$: probability of the hypothesis _before_ seeing the data.
- **Likelihood** $P(D|H)$: probability of the data given the hypothesis.
- **Marginal likelihood (evidence)** $P(D)$: total probability of the data (obtained via the Law of Total Probability).
- **Posterior** $P(H|D)$: probability of the hypothesis _after_ seeing the data.

---

### Q-6

**A disease has prevalence 1%. A test has sensitivity 95% and specificity 90%. Compute $P(\text{Disease}|\text{Positive Test})$ using Bayes’ theorem. Show all steps.**

**Answer:**  
Let $D$ = has disease, $+$ = positive test.

Given:

- $P(D) = 0.01$ => $P(\neg D) = 0.99$
- Sensitivity $P(+|D) = 0.95$
- Specificity $P(-|\neg D) = 0.90$ => False-positive rate $P(+|\neg D) = 0.10$

By Bayes’ theorem:

$$
P(D|+) = \frac{P(+|D)\,P(D)}{P(+)}
$$

Marginal probability of a positive test (Law of Total Probability):

$$
\begin{align*}
P(+) &= P(+|D)P(D) + P(+|\neg D)P(\neg D)\\
&= (0.95)(0.01) + (0.10)(0.99)\\
&= 0.0095 + 0.099 = 0.1085
\end{align*}
$$

Therefore

$$
P(D|+) = \frac{0.0095}{0.1085} \approx 0.0876 \quad (8.76\%)
$$

---

### Q-7

**What is the false-positive paradox? Why does a high-sensitivity test still produce many false positives for a rare disease?**

**Answer:**  
The **false-positive paradox** (also called the base-rate fallacy in medical testing) is the counter-intuitive result that even a highly accurate test can yield a large number of false positives when the disease is rare, so that most positive results are actually false positives.

**Why it happens:**  
When prevalence is very low, the absolute number of healthy people is huge. Even a small false-positive rate applied to that large population produces many more false positives than the true positives coming from the small diseased population.

In the numerical example of Q-6, out of 10 000 people only 100 have the disease (95 true positives) while 990 false positives occur among the healthy, so the majority of positive tests are false.

---

### Q-8

**What is the naive independence assumption in Naive Bayes? Write the full classification rule as an argmax expression.**

**Answer:**  
**Naive independence assumption:**  
Given the class label $y$, the features $x_1, x_2, \dots, x_n$ are conditionally independent:

$$
P(x_1,\dots,x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)
$$

**Full classification rule:**

$$
\hat{y} = \arg\max_{y} \; P(y) \prod_{i=1}^{n} P(x_i \mid y)
$$

(or equivalently in log-space for numerical stability).

---

### Q-9

**Why does Naive Bayes work well in practice despite the violated independence assumption?**

**Answer:**  
Although the independence assumption is almost always false, Naive Bayes still performs well for two main reasons:

1. **Classification only needs the correct ranking of class posteriors**, not accurate probability estimates. Even if the absolute probabilities are wrong, the class with the highest posterior is often still the correct one.
2. **Bias-variance trade-off**: the strong independence assumption greatly reduces variance (fewer parameters to estimate). When training data are limited this reduction in variance outweighs the increase in bias, leading to better generalization.

---

### Q-10

**What is Gaussian Naive Bayes? Write the likelihood formula $P(x_i \mid y)$ for a continuous feature.**

**Answer:**  
Gaussian Naive Bayes assumes that each continuous feature, given the class, follows a normal (Gaussian) distribution.

**Likelihood formula:**

$$
P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_{y,i}^{2}}} \exp\left( -\frac{(x_i - \mu_{y,i})^{2}}{2\sigma_{y,i}^{2}} \right)
$$

where $\mu_{y,i}$ and $\sigma_{y,i}^{2}$ are the mean and variance of feature $i$ estimated from the training examples of class $y$.

---

### Q-11

**What is Multinomial Naive Bayes? Write the Laplace-smoothed probability formula for a word given a class.**

**Answer:**  
Multinomial Naive Bayes models the data as counts (e.g., word frequencies in a document) drawn from a multinomial distribution. It is the classic model used for text classification.

**Laplace-smoothed probability of word $w$ given class $c$:**

$$
P(w \mid c) = \frac{\text{count}(w,c) + \alpha}{\sum_{w'} \text{count}(w',c) + \alpha |V|}
$$

where

- $\text{count}(w,c)$ = number of times word $w$ appears in documents of class $c$,
- $|V|$ = vocabulary size,
- $\alpha$ = smoothing parameter (usually $\alpha = 1$ for classic Laplace smoothing).

---

### Q-12

**What is the zero-frequency problem in Naive Bayes? How does Laplace smoothing (add-$\alpha$ smoothing) fix it?**

**Answer:**  
**Zero-frequency problem:**  
If a feature value (e.g., a word) never appears in the training data for a particular class, its estimated probability is zero. Because Naive Bayes multiplies probabilities, a single zero makes the entire class posterior zero, so that class can never be chosen—even if all other evidence strongly supports it.

**Laplace (add-$\alpha$) smoothing** fixes the problem by adding a small positive constant $\alpha$ to every count:

$$
P(x_i \mid y) = \frac{\text{count}(x_i,y) + \alpha}{N_y + \alpha \cdot |V|}
$$

This guarantees that every probability is strictly positive.

---

### Q-13

**Why must Naive Bayes compute in log-space for long documents? Show the underflow problem with a numerical example.**

**Answer:**  
For a long document the product $\prod P(x_i \mid y)$ can contain hundreds or thousands of factors, each smaller than 1. The product quickly becomes smaller than the smallest representable floating-point number and underflows to zero.

**Numerical example:**  
Suppose every conditional probability is $0.1$ and we have a document of 400 tokens:

$$
0.1^{400} = 10^{-400}
$$

Double-precision floating point can only represent numbers down to roughly $10^{-308}$. Consequently $0.1^{400}$ underflows to 0.0, making the posterior zero for every class.

Working in log-space

$$
\log P(y \mid \mathbf{x}) = \log P(y) + \sum_{i} \log P(x_i \mid y)
$$

converts the product into a sum and avoids underflow.

---

### Q-14

**Compare Naive Bayes and Logistic Regression: which is generative and which is discriminative? What are the practical implications of this distinction for training data size?**

**Answer:**

- **Naive Bayes** is a **generative** classifier: it models the joint distribution $P(\mathbf{x},y) = P(y)P(\mathbf{x}|y)$ and then applies Bayes’ rule.
- **Logistic Regression** is a **discriminative** classifier: it models the posterior $P(y|\mathbf{x})$ directly.

**Practical implications for training-data size:**

- Generative models (Naive Bayes) typically reach their asymptotic performance with **less data** because they make stronger assumptions (the independence assumption) and therefore have fewer parameters to estimate.
- Discriminative models (Logistic Regression) usually need **more data** to estimate the decision boundary accurately, but once enough data are available they often achieve higher asymptotic accuracy because they do not waste capacity modelling the input distribution $P(\mathbf{x})$.
