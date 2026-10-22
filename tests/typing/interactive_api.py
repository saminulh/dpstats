"""Run explicitly with pyright to verify concrete interactive types."""

# These negative calls must produce a type diagnostic, or their ignores fail.
# pyright: reportUnnecessaryTypeIgnoreComment=true

from pathlib import Path
from typing_extensions import assert_type

import pandas as pd

import dpstats as dp
from dpstats import analysis
from dpstats.instance import Instance
from dpstats.tasks import linear_regression as lr
from dpstats.tasks import mean_estimation as me
from dpstats.tasks.linear_regression.algorithms import AdaSSP
from dpstats.workspace.collection import TaskCollection


def check_collection_types(workspace: Path) -> None:
    collection = dp.open_collection(workspace, lr.LinearRegression)
    assert_type(
        collection,
        TaskCollection[
            lr.LinearRegressionMetadata,
            lr.LinearRegressionInput,
            lr.LinearRegressionOutput,
        ],
    )
    instances = collection.load_instances()
    assert_type(instances, dict[str, Instance[lr.LinearRegressionMetadata]])
    instance = next(iter(instances.values()))
    assert_type(instance.metadata, lr.LinearRegressionMetadata)
    assert_type(collection.load_input(instance), lr.LinearRegressionInput)
    assert_type(lr.algorithms.AdaSSP(), AdaSSP)
    assert_type(collection.task.Output, type[lr.LinearRegressionOutput])

    assert_type(
        analysis.get_metric_availability(
            instances.values(), lr.metrics.prediction_rmse_to_ols
        ),
        pd.DataFrame,
    )


def check_collection_task_boundary(
    collection: TaskCollection[
        lr.LinearRegressionMetadata, lr.LinearRegressionInput, lr.LinearRegressionOutput
    ],
    instance: Instance[me.MeanEstimationMetadata],
    metadata: lr.LinearRegressionMetadata,
    input: lr.LinearRegressionInput,
    other_metadata: me.MeanEstimationMetadata,
    other_input: me.MeanEstimationInput,
) -> None:
    assert_type(
        collection.add_instance(metadata, input, {}),
        Instance[lr.LinearRegressionMetadata],
    )
    collection.load_input(instance)  # pyright: ignore[reportArgumentType]
    collection.add_instance(
        other_metadata,  # pyright: ignore[reportArgumentType]
        input,
        {},
    )
    collection.add_instance(
        metadata,
        other_input,  # pyright: ignore[reportArgumentType]
        {},
    )
