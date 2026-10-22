# Single Coefficient Estimation

## Relationship to Linear Regression

`single_coefficient_estimation` is a **child of** the
[Linear Regression](../../linear_regression/) task. Any linear-regression algorithm
runs here through the `ToLinearRegression` transformation: the composed algorithm preserves
the same $X$, $y$, dimensions, domains, privacy budget, and seed, runs its
configured algorithm to estimate the full $\beta$, and then **projects** the
released result down to coordinate $j$.

Its metadata and input directly extend the linear-regression contracts with
only the public coefficient-of-interest index added to metadata. Task-native
`Standardize` and `DataBounds` transformations therefore provide the same public
affine preconditioning and diagnostic empirical-bounds treatments through
concrete adapters over linear-regression preparation. For an intercept target,
`Standardize` scales without centering: restoring an intercept after centering
also needs the slopes, which the scalar child output does not contain.
