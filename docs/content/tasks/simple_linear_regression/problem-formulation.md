## Problem formulation

We are given a dataset $(x_1, y_1), \dots, (x_n, y_n)$ where $x_i, y_i \in \mathbb{R}$.
The goal is to privately estimate $\hat{\alpha}, \hat{\beta}$, where $\hat{\alpha}, \hat{\beta}$ minimize

$$\sum_{i=1}^n (y_i - \alpha x_i - \beta)^2.$$

This is a specialization of [LinearRegression](../linear_regression/) to the case where $d=2$ with the first dimension being an intercept.
