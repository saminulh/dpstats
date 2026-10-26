## Problem formulation

We are given a dataset $(X_1, y_1), \dots, (X_n, y_n)$ where $X_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$ and we are also given a $j \in \{0, \dots, d-1\}$.
Letting $\hat{\beta}_{\rm OLS}$ be the ordinary least squares regression coefficient vector, namely

$$\hat{\beta}_{\rm OLS} = (X^\top X)^{-1}X^\top y,$$

the task is to privately estimate $\hat{\beta}_{{\rm OLS}, j}$.

In the benchmark, we measure the error of a private estimate $\hat{\beta}_j$ by its [absolute distance](https://saminulh.github.io/dpstats/single_coefficient_estimation/metrics/#absolute-error-to-ols) to the OLS coefficient: $|\hat{\beta}_j - \hat{\beta}_{{\rm OLS}, j}|$. We take the baseline to be the standard error of $\hat{\beta}_{{\rm OLS}, j}$ as published by the original study. See the [Methodology](https://saminulh.github.io/dpstats/methodology/) page for more details on our rationale.
