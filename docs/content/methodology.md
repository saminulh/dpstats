# Benchmark methodology

Here we explain how we built this benchmark: which tasks we
compare, on which data, at which privacy levels, and how we rank algorithms.
It ships with every release.

## 1. Tasks

We benchmark three tasks:

- single coefficient estimation (SCE), which estimates one coefficient of
    interest in a regression;
- linear regression, which estimates the full coefficient vector;
- mean estimation, which estimates the mean of a set of samples.

Every other task in the library is documented and runnable, but has no
associated corpus or leaderboard.

## 2. Data: the public corpus and the withheld corpus

Each problem instance is one concrete problem for a task: a dataset together with the
public facts about it, such as its sample size. Our instances come from
published studies, each a paper together with its replication package. One study may supply many instances, but we limit each study to at most 3 instances per task.

We
split the instances into one of two corpora:

- The **public corpus** is published. You can download its data and every
    algorithm's results on it.
- The **withheld corpus** stays private. We use it only for the leaderboard rankings.

All instances from one study fall in the same corpus, so a study is either
wholly public or wholly withheld.

Each release lists its public-corpus instances, so the current size is always in the download itself.

## 3. Privacy budget

We run every algorithm under approximate differential privacy at four
privacy levels: $\varepsilon$ = 0.3, 1, 3 and 10, with $\delta = 10^{-6}$ for
every instance.

## 4. The algorithms roster

Each benchmarked task has a roster of configured algorithms that are ranked on the leaderboard.
To be eligible for the roster, the algorithm must have a full argument for differential privacy (its privacy interpretation is *standard*; each task's "Native algorithms" page explains the classes).
We also run some algorithms without a full privacy argument on the public corpus for comparison.

The roster consists largely of relevant algorithms found in the literature along with variants that incorporate pre-processing transformations to the data. We typically chose the hyperparameters for each algorithm based on the defaults suggested either by the paper or the code from the algorithm's authors. Each algorithm's docs page explains the choice.

### Tuning

We also tuned the algorithms on a subset of the public corpus, separately at each privacy level. We add the tuned variants to the roster if it beats the default on a held out portion of the public corpus.

## 5. How we measure error and success

For
each task, every problem instance also has a nonprivate estimate. We measure error by how far the private algorithm's estimate falls from the nonprivate estimate under a given metric for the task; for example, in single coefficient estimation (SCE) the error is simply the absolute difference between the algorithm's estimated coefficient and the corresponding OLS coefficient. The chosen metrics all have an analogous sampling error for the nonprivate estimator. For SCE, that quantity is the standard error for the OLS coefficient (either the classical standard error for an OLS coefficient, or a more robust standard error procedure a study may choose to use). We call this sampling error analogue the baseline error for the instance. Below we list the metrics and baselines for each benchmarked task:

- **Single coefficient estimation:** the metric is the absolute difference between the
    private estimate and the ordinary least squares (OLS) estimate of the
    coefficient of interest. The sampling error baseline is the OLS coefficient's standard error reported by the study; this is usually the classical OLS standard error, but the study may choose some other procedure, like bootstrapping.

- **Linear regression:** the metric is the Mahalanobis norm between the private and OLS coefficients:

    $$
    \frac{1}{\sqrt{n}} \bigl\lVert X(\hat\beta - \hat\beta_{\rm OLS}) \bigr\rVert_2 .
    $$

    The sampling error baseline is

    $$
    \sqrt{\frac{s^2 d}{n}}, \qquad s^2 = \frac{\lVert y - X\hat\beta_{\rm OLS} \rVert_2^2}{n - d},
    $$

    which, under typical homoskedastic linear-model assumptions, estimates the OLS sampling error $\frac{1}{\sqrt{n}} \lVert X(\hat\beta_{\rm OLS} - \beta^\star) \rVert_2$. This baseline is used in classical joint F-tests for simple hypothesis tests in linear regression.

- **Mean estimation:** the Mahalanobis error between the private estimate
    and the sample mean, using the sample covariance $S$ with divisor $n-1$:

    $$
    \begin{cases}
    \sqrt{(\hat\mu-\bar x)^\top S^+(\hat\mu-\bar x)},
      & \hat\mu-\bar x\in\operatorname{range}(S),\\
    +\infty, & \text{otherwise}.
    \end{cases}
    $$

    Here $S^+$ is the pseudoinverse. An estimate outside the sample's affine
    span has infinite error. The sampling baseline is
    $\sqrt{\operatorname{rank}(S)/n}$, the plug-in RMS sampling error in this
    geometry. When $S$ has rank zero, the error is zero for an exact match to
    the sample mean and infinite otherwise.

We then say the output of an algorithm is successful at a given threshold $\tau$ if its error is at most $\tau$ times the sampling error baseline. Because private algorithms are random, the success rate for a given threshold and problem instance is a probability that lies in [0, 1] which we estimate by running multiple trials for an algorithm.

## 6. Running algorithms for the leaderboard

In
the previous section we discussed how we measure the success rate of an algorithm on a given problem instance. We run 10 trials on the public corpus (seeds 0-9) and 30 trials on the withheld corpus to estimate the success rate on each instance. Every algorithm is given a time budget of 60 seconds and memory budget of 4GB; a trial that exceeds either resource counts as a failure and is tracked accordingly. To obtain an overall success rate over the corpus at a threshold $\tau$, we average over the success rates over all instances in the corpus. Because multiple instances in the corpus may come from the same study, each instance is weighted such that every study contributes equal weight.

By computing overall success rates for every $\tau$, we form the performance profile curves ([Dolan and Moré, 2002](https://doi.org/10.1007/s101070100263)) shown on the Leaderboard and Explore pages. The official leaderboard ranks algorithms by τ@50, the median error ratio; lower is better. If an algorithm’s success rate never reaches 50%, because it fails or declines to release on more than half of the corpus, it has τ@50 = ∞ and is shown as “∞ (reaches 34%)” with the share in which it achieves finite error. These algorithms rank below every algorithm with a finite τ@50, ordered by the highest success rate they reach and then by the threshold at which they reach it. Rankings are computed separately at each privacy budget.

Besides the ranking over the whole corpus, the leaderboard reports the same ranking within slices of problem instances: instances with dimension $d \le 10$ and $d > 10$, and instances with $n/d \le 100$ and $n/d > 100$ samples per dimension. Mean estimation is sliced by $n/d$ only. Within a slice, every study that contributes instances to it carries equal weight, and the leaderboard shows how many instances and studies each slice contains.
