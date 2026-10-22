## Problem formulation

We are given a dataset $(X_1, y_1), \dots, (X_n, y_n)$ where $X_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$ and we are also given a $j \in \{0, \dots, d-1\}$.
Letting $\hat{\beta}$ be the ordinary least squares regression coefficient vector, namely

$$\hat{\beta} = (X^TX)^{-1}X^Ty,$$

the task is to privately estimate $\hat{\beta}_j$.

We measure the benchmark performance of a private estimator by the absolute distance from $\hat{\beta}_j$. The baseline is the standard error of $\hat{\beta}_j$ that the original study reported (on a simulated dataset, the OLS standard error). This is the estimated nonprivate sampling error. Their ratio has the form of a $|t|$ statistic. See the [metric definition](https://saminulh.github.io/dpstats/single_coefficient_estimation/metrics/#absolute-error-to-ols) for more details.
