# Simple Linear Regression

## Relationship to Linear Regression

`simple_linear_regression` has native scalar-regression algorithms and an
`ToLinearRegression` transformation for composing algorithms from the
[Linear Regression](../../linear_regression/) task. The transformation promotes the
scalar covariate $x$ to a two-column design matrix $[1,x]$, runs the configured
linear-regression algorithm, and reads the coefficients back as
$(\text{slope},\text{intercept})$. Benchmark rosters select such compositions
explicitly; linear-regression algorithms are not registered here automatically.

Because the reference fit here is OLS **with an intercept**, a dataset converted
to `simple_linear_regression` and the same dataset converted to
`linear_regression` (a through-origin $d=1$ fit) are *different benchmark
instances*, not two scorings of the same fit.
