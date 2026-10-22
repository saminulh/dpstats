## Problem formulation

Given row vectors $x_1,\ldots,x_n\in\mathbb{R}^d$, estimate their centered covariance. The population target is $\Sigma=\mathbb{E}[(X-\mu)(X-\mu)^\top]$, where $\mu=\mathbb{E}[X]$.

The sample reference is $S_n=\frac{1}{n}\sum_i(x_i-\bar{x})(x_i-\bar{x})^\top$. It uses divisor $n$ and is zero for one record. The raw second moment $X^\top X/n$ is a different quantity.

An estimate is a finite symmetric $d\times d$ matrix; it need not be positive semidefinite. Frobenius and spectral errors are reported separately against population and sample covariance. Reference values are evaluation data, not public algorithm metadata.
