"""The instance properties Explore offers for each task.

Each property is a plain function of one Instance and the collection holding
it. The build computes them through ``collection.instance_features``, and each
Explore notebook carries the source of its task's functions, so the page and
the notebook read one list.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from dpstats.workspace import TaskCollection


def n(instance, collection):
    return instance.metadata.n_samples


def d(instance, collection):
    return instance.metadata.dim


def n_over_d(instance, collection):
    return instance.metadata.n_samples / instance.metadata.dim


def design_condition_number(instance, collection):
    return np.linalg.cond(collection.load_input(instance).X)


def vif(instance, collection):
    """Compute VIF at the coefficient of interest using the model's intercept convention."""
    X = collection.load_input(instance).X
    j = instance.metadata.coefficient_of_interest_index
    target = X[:, j]
    reference = target - target.mean() if instance.metadata.has_intercept else target
    total_ss = reference @ reference
    if total_ss == 0:
        return np.nan  # VIF is undefined when its reference variance is zero.

    others = np.delete(X, j, axis=1)
    beta = np.linalg.lstsq(others, target, rcond=None)[0]
    residual = target - others @ beta
    return float(total_ss / (residual @ residual))


_DESIGN = (n, d, n_over_d, design_condition_number)

PROPERTIES: dict[str, tuple[Callable[[Any, Any], float], ...]] = {
    "covariance_estimation": _DESIGN,
    "linear_regression": _DESIGN,
    "logistic_regression": _DESIGN,
    "m_estimation": (n,),
    "mean_estimation": _DESIGN,
    "median_estimation": (n,),
    "simple_linear_regression": (n,),
    "single_coefficient_estimation": (*_DESIGN, vif),
}


def instance_properties(
    collection: TaskCollection, registries: Sequence[str] | None = None
) -> pd.DataFrame:
    """One row per Instance and one column per property its task lists."""
    return collection.instance_features(
        {
            prop.__name__: partial(prop, collection=collection)
            for prop in PROPERTIES[collection.task.id]
        },
        registries=registries,
    )
