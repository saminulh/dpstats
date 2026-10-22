# M-Estimation

## Configured Loss Contract

`MEstimationMetadata.loss(data, theta, reduction="none")` returns finite,
nonnegative per-sample costs. Data is always a two-dimensional batch, including
a batch of one. Losses, their `gradient` methods, and optional `hessian` methods
share keyword-only `reduction="none" | "sum" | "mean"`: `none` returns shapes
`(n,)`, `(n,d)`, and `(n,d,d)`; `sum` and `mean` reduce over records to a scalar,
a parameter vector, and a d-by-d matrix, respectively.
Its `gradient(data, theta, reduction="none")` method returns the corresponding
selected gradient or subgradient, and its
`properties(data_domain, parameter_domain)` method returns the mathematical
facts on those declared public domains: convexity, whether the loss is
generalized-linear (it reaches the parameter only through a linear predictor
`f(x^T theta; y)`), whether it is continuously twice differentiable, the L2 loss
Lipschitz constant over the declared parameter domain, and the ambient L2
gradient Lipschitz constant on the whole parameter space. Objective perturbation
admits a loss on convexity, the generalized-linear flag, and finite values for
both constants, from public facts alone and before reading a record; a finite
gradient Lipschitz constant is what rules out a kink in the linear predictor
itself. The twice-differentiability flag describes the loss, but does not add
an admission condition for `ObjectivePerturbationRedberg` or
`AMP`: their RKW profile is C1 and admits Huber loss.
Every built-in loss lives in the transformation module of the task that presents its
problem as M-estimation: the location losses with mean and median estimation,
the residual losses with linear regression. External packages define
an importable module-level `MEstimationLoss` subclass with ordinary configured
fields, so its behavior and nested configuration round-trip together.
Algorithms and metrics assume metadata is correct and do not inspect loss
outputs to certify it at runtime.

## Parameter Domain

`MEstimationMetadata.parameter_domain` is the public constraint set for $\theta$, and its dimension is the parameter dimension. An unconstrained problem declares `EuclideanSpace`. A bounded problem declares a domain with a finite diameter and an exact Euclidean projection, which is what a projected-gradient analysis needs and what makes a squared loss's Lipschitz constant finite at all.

The median-estimation `ToMEstimation` declares the data domain itself: the absolute-error loss has a minimizer at a sample value, which lies in the declared domain, so that domain constrains nothing and is the parameter domain. The mean-estimation `ToMEstimation` declares the data domain's convex enclosure (`Domain.convex_enclosure`: the domain itself when convex, else its bounding box). Projecting a point onto a convex set containing the records brings it no farther from any record, so the squared and Huber losses have a minimizer inside that set and the estimand is unchanged; a bounded data domain thereby gives the squared loss a finite Lipschitz constant, and since mean-estimation domains are bounded, `ERM`, which accepts only `EuclideanSpace`, no longer composes with this transformation. The linear-regression `ToMEstimation` reads the domain off its configured `parameter_radius`, which is part of the configured identity under either spelling. Its default `None` is a value naming the unbounded domain, not an omitted radius: the transformation declares `EuclideanSpace` of the parameter dimension and the estimand is the unconstrained minimizer $\arg\min_{\theta} \frac{1}{n} \sum_i \ell(z_i, \theta)$. A finite radius declares an origin-centered `LpBallDomain` with `p=2`, and **that ball is then part of the estimand**, which becomes the constrained minimizer over $K$ — not in general the Euclidean projection of the unconstrained one, since projection is what the *algorithms* apply to each iterate — so a finite radius is declared on the problem's own scale.

`GD`, `SGD`, and `DPSGD` project every iterate onto the declared domain, so they accept either spelling — projection onto `EuclideanSpace` is the identity. `ERM` implements unconstrained optimization only and refuses any domain other than `EuclideanSpace` with `InvalidRegime`, so it runs under the default while a composition over a declared ball uses the projecting `GD` for a non-private baseline. In the other direction `DPSGD` with `C_grad_clip="loss_lipschitz"` calibrates from the loss Lipschitz constant over the parameter domain, which `SquaredResidualLoss` and `SquaredErrorLoss` leave infinite when that domain is unbounded, so under those losses it refuses `EuclideanSpace` with `InvalidRegime` and needs a finite radius; the Huber and absolute losses take their constants from the threshold and the data domain instead, so they stay admissible under either.

`HuberResidualLoss` takes an absolute `threshold` on the residual. `HuberNormLoss` takes `threshold_per_sqrt_dim`, so its threshold is that multiple of the square root of the dimension: after `Standardize`, which gives every coordinate unit scale, it sits at the same point relative to the data in every dimension.
