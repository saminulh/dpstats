"""Execute the public code examples that first-time readers can copy."""

from __future__ import annotations

from pathlib import Path
import re
import runpy

import numpy as np
import pytest

from api_reference import reference_pages
from dpstats.budgets import ApproxDPBudget
from dpstats.tasks import linear_regression as lr
from templates import library_source
from templates.model import Link


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name", ["SSP", "Standardize"])
def test_generated_task_examples_execute_with_their_declared_inputs(name: str) -> None:
    is_transformation = name == "Standardize"
    component = (
        lr.transformations.Standardize if is_transformation else lr.algorithms.SSP
    )
    task_link = Link("Linear Regression", "../index.md")
    page = library_source.component_page(
        lr.LinearRegression,
        name,
        component,
        task_link=task_link,
        inner_link=task_link if is_transformation else None,
        semantics=Link("Privacy guarantees", "../../library/reference/privacy/"),
    )
    scope = {"budget": ApproxDPBudget(eps=3, delta=1e-5), "inner": lr.algorithms.SSP()}
    exec(page.usage, scope)
    assert isinstance(scope["output"], lr.LinearRegressionOutput)
    assert np.isfinite(scope["output"].beta).all()
    assert scope["lr"] is lr
    if is_transformation:
        assert isinstance(scope["algorithm"].transformations, tuple)


def test_generated_metric_access_names_the_existing_metric_objects() -> None:
    access = library_source.metrics(lr.LinearRegression).access
    scope = {}
    exec(access.import_line, scope)
    objects = [eval(line, scope) for line in access.accessor.splitlines()]
    assert lr.metrics.prediction_rmse_to_ols in objects
    assert all(metric.task is lr.LinearRegression for metric in objects)


def test_the_collection_reference_supplies_its_working_public_import() -> None:
    pages, routes = reference_pages(
        {"collections": ROOT / "docs/reference/collections.md"}
    )
    page = pages[routes["dpstats.open_collection"]]
    code = re.search(r"```python\n(.*?)\n```", page, flags=re.DOTALL)
    assert code is not None
    scope = {}
    exec(code[1], scope)
    import dpstats as dp

    assert scope["open_collection"] is dp.open_collection


def test_the_workflow_retains_results_and_reuses_them(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    example = ROOT / "docs/reference/interactive_workflow.py"
    run = runpy.run_path(str(example))
    collection = run["collection"]
    first_results = collection.load_results(results_source="local")
    assert len(first_results) == 18
    assert set(first_results["status"]) == {"success"}
    assert set(run["measured"]["alg_seed"]) == {0, 1, 2}
    assert len(run["curves"]) == 10
    assert np.isfinite(run["measured"]["prediction_rmse_to_ols"]).all()
    assert (tmp_path / "interactive-example/measurements.csv").is_file()
    assert (tmp_path / "interactive-example/profiles.html").is_file()
    stored = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*.parquet")
    }
    rerun = runpy.run_path(str(example))
    assert len(rerun["results"]) == 18
    assert stored == {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*.parquet")
    }
