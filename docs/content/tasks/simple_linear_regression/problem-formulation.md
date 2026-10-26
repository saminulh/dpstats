## Problem formulation

We are given a dataset $(x_1, y_1), \dots, (x_n, y_n)$ where $x_i, y_i \in \mathbb{R}$.
The task is to privately estimate the ordinary least squares slope and intercept $(\hat{\alpha}_{\rm OLS}, \hat{\beta}_{\rm OLS})$, the minimizers of

$$\sum_{i=1}^n (y_i - \alpha x_i - \beta)^2.$$

This is a specialization of [LinearRegression](../linear_regression/) to the case where $d=2$ with an intercept in the first column.
