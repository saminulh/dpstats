## Problem formulation

We are given a dataset $(X_1, y_1), \dots, (X_n, y_n)$ where $X_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$.
The task is to privately estimate the ordinary least squares regression coefficient vector

$$
\hat{\beta}_{\rm OLS} = (X^\top X)^{-1}X^\top y.
$$

We measure the benchmark performance of a private estimate $\hat{\beta}$ by the root-mean-square difference between its predictions and those of OLS over the sample:

$$
\tfrac{1}{\sqrt n}\lVert X(\hat{\beta} - \hat{\beta}_{\rm OLS})\rVert_2.
$$

The baseline is $\hat{\sigma}\sqrt{d/n}$, where $\hat{\sigma}^2 = \lVert y - X\hat{\beta}_{\rm OLS}\rVert_2^2/(n-d)$: the root-mean-square standard error of the OLS fitted values. Their ratio is the Mahalanobis distance of $\hat{\beta}$ from $\hat{\beta}_{\rm OLS}$ under the estimated OLS covariance $\hat{\sigma}^2(X^\top X)^{-1}$, divided by $\sqrt d$; its square has the form of the F statistic for the hypothesis that the coefficients equal $\hat{\beta}$. See the [metric definition](https://saminulh.github.io/dpstats/linear_regression/metrics/#prediction-rmse-to-ols) for more details.
