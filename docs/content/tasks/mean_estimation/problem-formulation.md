## Problem formulation

Given a dataset $X_1, \dots, X_n \in \mathbb{R}^d$, privately estimate its mean

$$\hat{\mu} = \frac{1}{n} \sum_{i=1}^n X_i.$$

We measure benchmark performance by the sample-covariance Mahalanobis error to the sample mean, with baseline $\sqrt{\operatorname{rank}(\hat{\Sigma})/n}$, where $\hat{\Sigma}$ is the centered sample covariance with divisor $n-1$ and $\hat{\Sigma}^{+}$ is its pseudoinverse. For a private estimate $\tilde{\mu}$, the error is

$$
\begin{cases}
\sqrt{(\tilde{\mu}-\hat{\mu})^\top\hat{\Sigma}^{+}(\tilde{\mu}-\hat{\mu})},
& \tilde{\mu}-\hat{\mu}\in\operatorname{range}(\hat{\Sigma}),\\
+\infty, & \text{otherwise}.
\end{cases}
$$

An estimate outside the sample's affine span has infinite error. The baseline is the plug-in RMS sampling error in this geometry. See the [metric definition](https://saminulh.github.io/dpstats/mean_estimation/metrics/#mahalanobis-error-to-sample-mean) for more details.
