import numpy as np


def noisy_regression(
    X: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    x_bound: float,
    y_bound: float,
    ridge: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Compute a noisy version of (X'X)^{-1}X'y by adding enough Laplace noise to X'X and X'y for epsilon pure-DP."""
    generator = np.random.default_rng(seed)

    norms = np.linalg.norm(X, axis=1)
    X_clipped = X * np.minimum(1.0, x_bound / np.maximum(norms, 1e-300))[:, None]
    y_clipped = np.clip(y, -y_bound, y_bound)

    d = X.shape[1]
    sensitivity = (d + 1) / 2 * x_bound**2 + np.sqrt(d) * x_bound * y_bound
    scale = sensitivity / epsilon

    upper = np.triu(generator.laplace(scale=scale, size=(d, d)))
    covariance_noise = upper + np.triu(upper, 1).T
    cross_noise = generator.laplace(scale=scale, size=d)

    noisy_XtX = X_clipped.T @ X_clipped + covariance_noise
    noisy_Xty = X_clipped.T @ y_clipped + cross_noise
    return np.linalg.solve(
        noisy_XtX + ridge * np.eye(d),
        noisy_Xty,
    )
