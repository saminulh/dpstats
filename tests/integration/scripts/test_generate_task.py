"""Smoke test for the task scaffolder (``scripts/generate_task.py``).

A freshly scaffolded task must import and run against the current
``BaseAlgorithm`` API: the template algorithm implements ``run`` (not the
public ``run`` template method), so its class is concrete and instantiable.
This guards against API drift leaving generated tasks broken.
"""

import importlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

from dpstats.budgets import NoPrivacy, PureDPBudget
from dpstats.transport.canonical import canonical_json_bytes
from dpstats.domains import BoxDomain
from dpstats.instance import Instance, load_instance_input, persist_instance
from dpstats.instance import (
    build_instance_registry,
    read_instance_registry,
)
from dpstats.scripts.generate_task import generate_task
from tests.fixtures.accounting_spy import AccountingSpy


def test_scaffolded_task_imports_and_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scaffold a task, import it, and run an algorithm on a generated instance."""
    generate_task("smoke_task", tmp_path)

    target = tmp_path / "smoke_task"
    assert (target / "__init__.py").exists()
    # The scaffolder must not copy stale bytecode from the template tree.
    assert not list(target.rglob("*.pyc"))

    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        pkg = importlib.import_module("smoke_task")

        task = pkg.SmokeTask
        algorithms = importlib.import_module("smoke_task.algorithms")
        assert algorithms.LaplaceMean.TASK is task
        algs = [algorithms.LaplaceMean()]
        run_params = inspect.signature(type(algs[0]).run).parameters
        assert list(run_params) == ["self", "metadata", "input", "budget", "seed"]
        assert all(
            parameter.default is inspect.Parameter.empty
            for parameter in list(run_params.values())[1:]
        )
        with pytest.raises(ValueError, match="greater than or equal"):
            type(algs[0])(noise_multiplier=0.5)

        simulations = importlib.import_module("smoke_task.simulations")
        metadata, input_data, metric_arguments = simulations.simulate_uniform(seed=0)
        budget = PureDPBudget(eps=1.0)
        output = algs[0].run(
            metadata=metadata,
            input=input_data,
            budget=budget,
            seed=0,
        )
        metrics = importlib.import_module("smoke_task.metrics")
        assert metrics.absolute_error.task is task
        score = metrics.absolute_error.calculate(
            output, ground_truth=metric_arguments["ground_truth"]
        )
        assert score >= 0.0
        assert not (target / "tuning_objectives.py").exists()
        assert not (target / "selection_costs.py").exists()
    finally:
        for name in list(sys.modules):
            if name == "smoke_task" or name.startswith("smoke_task."):
                del sys.modules[name]


def test_a_scaffolded_task_is_instance_ready_with_no_extra_declarations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new task needs its value models and ``validate_input``, and nothing else.

    The scaffold declares no custom record-reconstruction method or field-name
    set, so this fails the moment reconstruction stops being generic — which
    is the tax a public extension point must not carry.
    """
    generate_task("record_task", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    task = importlib.import_module("record_task").RecordTask

    data = np.array([0.0, 1.0, 2.0, 3.0])
    store = tmp_path / "store"
    metadata = task.Metadata(
        n_samples=4, domain=BoxDomain(bounds=np.array([[0.0, 3.0]]))
    )
    input = task.Input(data=data)
    instance = persist_instance(metadata, input, {}, store)
    document = build_instance_registry(task.id, [instance])
    restored = read_instance_registry(
        canonical_json_bytes(document.model_dump()), task.id
    )

    materialized = load_instance_input(restored[instance.uuid], store)
    assert canonical_json_bytes(
        materialized.model_dump(mode="json")
    ) == canonical_json_bytes(input.model_dump(mode="json"))


def test_scaffolded_algorithm_calibrates_the_add_remove_sensitivity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accounting_spy: AccountingSpy
) -> None:
    """Pin the scaffold's Laplace scale on a hand-computed toy case.

    The template releases the centered public-``n`` mean
    ``c + sum(x - c) / n_samples``. Adding or removing one admitted record
    moves it by one centered contribution, bounded by the enclosing-ball
    radius of the box ``[100, 101]``, which is ``(101 - 100) / 2 = 0.5``. With
    ``n_samples = 100`` the add/remove-one sensitivity is therefore
    ``0.5 / 100 = 0.005``, and a pure-DP Laplace release at ``eps = 1`` takes
    scale ``sensitivity / eps = 0.005``.
    """
    generate_task("sensitivity_task", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    task = importlib.import_module("sensitivity_task").SensitivityTask
    algorithms = importlib.import_module("sensitivity_task.algorithms")

    metadata = task.Metadata(
        n_samples=100, domain=BoxDomain(bounds=np.array([[100.0, 101.0]]))
    )
    input = task.Input(data=np.full(100, 100.5))
    budget = PureDPBudget(eps=1.0)
    algorithm = algorithms.LaplaceMean()
    releases = np.array(
        [
            algorithm.run(
                metadata=metadata, input=input, budget=budget, seed=seed
            ).mean_est
            for seed in range(400)
        ]
    )

    calibration = accounting_spy.calls[0]
    assert calibration.entry_point == "calibrate_laplace_scale"
    assert calibration.budget == budget
    assert calibration.sensitivity == pytest.approx(0.005)
    # ``E|Laplace(0, b)| = b`` reads the realized scale back off the releases.
    assert np.mean(np.abs(releases - 100.5)) == pytest.approx(0.005, rel=0.2)


def test_scaffolded_algorithm_refuses_a_singleton_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-radius domain gives a zero sensitivity, which the calibration refuses.

    The box ``[3, 3]`` admits only the value ``3``, so its enclosing-ball
    radius and the mean's sensitivity are zero under any offered budget.
    """
    generate_task("singleton_task", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    task = importlib.import_module("singleton_task").SingletonTask
    algorithms = importlib.import_module("singleton_task.algorithms")

    metadata = task.Metadata(
        n_samples=10, domain=BoxDomain(bounds=np.array([[3.0, 3.0]]))
    )
    input = task.Input(data=np.full(10, 3.0))
    algorithm = algorithms.LaplaceMean()

    for budget in (NoPrivacy(), PureDPBudget(eps=1.0)):
        with pytest.raises(ValueError, match="sensitivity must be finite and positive"):
            algorithm.run(metadata=metadata, input=input, budget=budget, seed=0)


@pytest.mark.parametrize(
    "bad_name", ["SmokeTask", "smoke-task", "smoke__task", "1smoke"]
)
def test_scaffolder_rejects_a_name_that_is_not_snake_case(
    tmp_path: Path, bad_name: str
) -> None:
    """The Task id is the package name, so a malformed one is refused up front."""
    with pytest.raises(SystemExit):
        generate_task(bad_name, tmp_path)

    assert not list(tmp_path.iterdir())


def test_scaffolded_names_follow_the_convention(tmp_path: Path) -> None:
    """Task id and display name come from the name; the example class names the method."""
    generate_task("smoke_task", tmp_path)

    task_source = (tmp_path / "smoke_task" / "__init__.py").read_text()
    assert 'id="smoke_task"' in task_source
    assert 'name="Smoke Task"' in task_source
    assert (tmp_path / "smoke_task" / "algorithms" / "laplace_mean.py").exists()
