## Problem formulation

We are given a dataset $(X_1, y_1), \dots, (X_n, y_n)$ where $X_i \in \mathbb{R}^d$ and $y_i \in \{0, 1\}$.
The task is to privately estimate the logistic regression coefficient vector

$$
\hat{\theta} \in \operatorname*{arg\,min}_{\theta \in \mathbb{R}^d}
\frac{1}{n}\sum_{i=1}^n
\left[\log(1+\exp(X_i^\top\theta))-y_iX_i^\top \theta\right].
$$
