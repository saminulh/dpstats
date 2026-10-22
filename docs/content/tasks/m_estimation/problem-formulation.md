## Problem formulation

We are given a loss function $\ell$ and dataset $X_1, \dots, X_n$ where $X_i \in \mathbb{R}^d$ and the task is to privately estimate the empirical loss minimizer

$$
\hat{\theta} \in \operatorname*{arg\,min}_{\theta \in K}
\frac{1}{n}\sum_{i=1}^n \ell(X_i, \theta),
$$

where $K$ is the public parameter domain.

M-estimation algorithms vary greatly by assumptions on $\ell$ (e.g., convexity and twice-differentiability).
To see how to encode this information, see the [configured loss contract](https://saminulh.github.io/dpstats/m_estimation/definition/MEstimationLoss/).
