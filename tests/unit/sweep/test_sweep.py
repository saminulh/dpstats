"""Tests for the authored Sweep reader and writer and its per-Instance budgets.

Sweep refusals use ``validate_sweep``; declared budget groups are
``Sweep.budget_groups_for``, exercised through ``TaskCollection.load_results``
in ``tests/unit/test_collection.py``. Structural round trips remain internal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dpstats.budgets import ApproxDPBudget, PureDPBudget
from dpstats.instance import Instance
from dpstats.sweep import SubinverseSampleSizeDelta
from dpstats.sweep.coordinates import validate_sweep
from dpstats.sweep._sweep import (
    Sweep,
    read_sweep,
    read_sweep_task,
    write_sweep,
)
from tests.registry_components import (
    SIMPLE_TASK,
    InverseSquareDelta,
    OversizedDelta,
)

PURE_BUDGET_PATH = "dpstats.budgets.families:PureDPBudget"
APPROX_BUDGET_PATH = "dpstats.budgets.families:ApproxDPBudget"
INVERSE_SQUARE_DELTA_PATH = "tests.registry_components:InverseSquareDelta"
UNINSTALLED_POLICY_PATH = "no_such_distribution.policies:CustomDelta"


def _write_document(directory: Path, document: Any) -> Path:
    """Author one Sweep document verbatim, without a production writer."""
    path = directory / "sweep.json"
    path.write_text(json.dumps(document))
    return path


def _instance(*, n_samples: int) -> Instance:
    return Instance(
        metadata=SIMPLE_TASK.Metadata(n_samples=n_samples, dim=1),
        input_digest="0" * 64,
        metric_arguments={},
    )


def test_a_complete_sweep_round_trips_through_its_document(tmp_path: Path) -> None:
    """Every declared field survives reading and dumping unchanged."""
    document = {
        "task": "simple_test_task",
        "budgets": [
            {"path": PURE_BUDGET_PATH, "config": {"eps": 1.5}},
            {"path": INVERSE_SQUARE_DELTA_PATH, "config": {"eps": 2.0}},
        ],
        "seeds": [0, 7],
        "time_limit": 12.5,
        "memory_limit_gb": 2.0,
    }

    sweep = read_sweep(_write_document(tmp_path, document))

    assert sweep.task == "simple_test_task"
    assert sweep.budgets == (PureDPBudget(eps=1.5), InverseSquareDelta(eps=2.0))
    assert sweep.seeds == (0, 7)
    assert sweep.time_limit == 12.5
    assert sweep.memory_limit_gb == 2.0
    assert sweep.dump_config().config == {
        **document,
        "algorithms": None,
        "instances": None,
    }


def test_an_omitted_field_reads_the_documented_default(tmp_path: Path) -> None:
    """A Sweep declaring only its Task carries the reader's stated defaults."""
    sweep = read_sweep(_write_document(tmp_path, {"task": "simple_test_task"}))

    assert sweep.budgets is None
    assert sweep.seeds is None
    assert sweep.time_limit is None
    assert sweep.memory_limit_gb is None


def test_reading_the_task_alone_loads_no_declared_component(tmp_path: Path) -> None:
    """Settling an abandoned store must not depend on code the Sweep names."""
    path = _write_document(
        tmp_path,
        {
            "task": "simple_test_task",
            "budgets": [{"path": UNINSTALLED_POLICY_PATH, "config": {"eps": 1.0}}],
        },
    )

    assert read_sweep_task(path) == "simple_test_task"


def test_budgets_resolve_per_instance_and_keep_one_of_each(tmp_path: Path) -> None:
    """A policy offers one budget per Instance, and equal offers collapse."""
    # At n_samples = 4 the policy's rule 1 / n_samples ** 2 gives delta = 0.0625,
    # which the document also declares outright as an approximate-DP budget.
    document = {
        "task": "simple_test_task",
        "budgets": [
            {"path": PURE_BUDGET_PATH, "config": {"eps": 1.0}},
            {"path": INVERSE_SQUARE_DELTA_PATH, "config": {"eps": 2.0}},
            {"path": PURE_BUDGET_PATH, "config": {"eps": 1.0}},
            {"path": APPROX_BUDGET_PATH, "config": {"eps": 2.0, "delta": 0.0625}},
        ],
    }
    sweep = read_sweep(_write_document(tmp_path, document))

    resolved = sweep.budgets_for(_instance(n_samples=4))

    assert resolved == (PureDPBudget(eps=1.0), ApproxDPBudget(eps=2.0, delta=0.0625))


def test_the_subinverse_policy_offers_one_delta_wherever_it_resolves() -> None:
    """The offered delta is the rule's value, not the host's rounding of it."""
    # 317 ** -1.1 is 0.0017735147777247594305931243..., whose first twelve
    # significant digits are 0.00177351477772.
    offered = SubinverseSampleSizeDelta(eps=1.0).resolve(
        SIMPLE_TASK.Metadata(n_samples=317, dim=1)
    )

    assert offered == ApproxDPBudget(eps=1.0, delta=0.00177351477772)
    assert offered.delta == 0.00177351477772
    assert offered == ApproxDPBudget(eps=1.0, delta=0.0017735147777247586)


def test_the_subinverse_policy_canonicalizes_the_decimal_not_a_rounded_float() -> None:
    # 195233 ** -1.1 sits close enough to a half-even rounding boundary at
    # the 12th significant digit that canonicalizing the exact Decimal and
    # canonicalizing float(that Decimal) disagree in the last digit
    # (1.51492610517e-06 vs 1.51492610518e-06). The policy must canonicalize
    # the Decimal directly, matching origin/dev's resolved value.
    offered = SubinverseSampleSizeDelta(eps=1.0).resolve(
        SIMPLE_TASK.Metadata(n_samples=195233, dim=1)
    )

    assert offered.delta == 1.51492610517e-06


def test_the_subinverse_policy_ceiling_bounds_the_smallest_instances() -> None:
    offered = SubinverseSampleSizeDelta(eps=1.0).resolve(
        SIMPLE_TASK.Metadata(n_samples=2, dim=1)
    )

    assert offered == ApproxDPBudget(eps=1.0, delta=0.01)


def test_policies_apart_above_the_identity_precision_offer_distinct_budgets() -> None:
    metadata = SIMPLE_TASK.Metadata(n_samples=317, dim=1)

    assert SubinverseSampleSizeDelta(eps=1.0).resolve(metadata) != (
        SubinverseSampleSizeDelta(eps=1.0 + 1e-6).resolve(metadata)
    )


def test_the_subinverse_policy_separates_neighbouring_sample_sizes() -> None:
    policy = SubinverseSampleSizeDelta(eps=1.0)

    assert policy.resolve(SIMPLE_TASK.Metadata(n_samples=317, dim=1)) != policy.resolve(
        SIMPLE_TASK.Metadata(n_samples=318, dim=1)
    )


@pytest.mark.parametrize("task", ["", ".", "..", "../elsewhere"])
def test_a_task_outside_one_name_is_refused(tmp_path: Path, task: str) -> None:
    """The declared Task is one directory name, never a traversable path."""
    path = _write_document(tmp_path, {"task": task})

    with pytest.raises(ValueError, match="Sweep .* is invalid") as refusal:
        validate_sweep(path)

    assert "must name exactly one Task" in str(refusal.value.__cause__)


def test_an_empty_seed_selection_is_a_valid_no_work_sweep(tmp_path: Path) -> None:
    path = _write_document(tmp_path, {"task": "simple_test_task", "seeds": []})

    assert read_sweep(path).seeds == ()


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_a_seed_outside_the_algorithm_seed_domain_is_refused(
    tmp_path: Path, seed: int
) -> None:
    path = _write_document(tmp_path, {"task": "simple_test_task", "seeds": [0, seed]})

    with pytest.raises(ValueError, match="Sweep .* is invalid") as refusal:
        validate_sweep(path)

    assert "non-negative integer below" in str(refusal.value.__cause__)


@pytest.mark.parametrize("field", ["time_limit", "memory_limit_gb"])
@pytest.mark.parametrize("declared", [0.0, -1.0, "+inf_json", "nan_json"])
def test_a_resource_declaration_that_is_not_positive_and_finite_is_refused(
    tmp_path: Path, field: str, declared: float | str
) -> None:
    """Standard JSON spells a nonfinite limit only as a canonical float token."""
    path = _write_document(tmp_path, {"task": "simple_test_task", field: declared})

    with pytest.raises(ValueError, match="Sweep .* is invalid") as refusal:
        validate_sweep(path)

    assert field in str(refusal.value.__cause__)


def test_an_undeclared_sweep_field_is_refused(tmp_path: Path) -> None:
    path = _write_document(tmp_path, {"task": "simple_test_task", "undeclared": []})

    with pytest.raises(ValueError, match="Sweep .* is invalid") as refusal:
        validate_sweep(path)

    assert "undeclared" in str(refusal.value.__cause__)


def test_a_component_this_installation_cannot_import_is_refused(
    tmp_path: Path,
) -> None:
    """A Sweep naming an absent class fails at read, not at resolution."""
    path = _write_document(
        tmp_path,
        {
            "task": "simple_test_task",
            "budgets": [{"path": UNINSTALLED_POLICY_PATH, "config": {"eps": 1.0}}],
        },
    )

    with pytest.raises(ValueError, match="Sweep .* is invalid") as refusal:
        validate_sweep(path)

    assert isinstance(refusal.value.__cause__, ModuleNotFoundError)


def test_a_policy_offering_no_valid_budget_names_itself_and_the_instance() -> None:
    sweep = Sweep(task="simple_test_task", budgets=(OversizedDelta(eps=1.0),))
    instance = _instance(n_samples=4)

    with pytest.raises(ValueError, match="OversizedDelta offered no valid budget"):
        sweep.budgets_for(instance)


@pytest.mark.parametrize("document", [{}, {"task": None}, {"task": "../elsewhere"}])
def test_a_document_declaring_no_readable_task_is_refused(
    tmp_path: Path, document: Any
) -> None:
    path = _write_document(tmp_path, document)

    with pytest.raises(ValueError, match="declares no readable Task"):
        read_sweep_task(path)


def test_validation_accepts_only_the_expected_task(tmp_path: Path) -> None:
    path = _write_document(tmp_path, {"task": "simple_test_task"})

    assert validate_sweep(path, expected_task="simple_test_task") is None

    with pytest.raises(
        ValueError, match="declares Task 'simple_test_task', not 'mean_estimation'"
    ):
        validate_sweep(path, expected_task="mean_estimation")


def test_writing_a_sweep_refuses_to_replace_unless_asked(tmp_path: Path) -> None:
    sweep = Sweep(task="simple_test_task", seeds=(0, 1))
    path = tmp_path / "sweeps" / "demo.json"
    path.parent.mkdir()

    write_sweep(path, sweep, overwrite=False)
    assert read_sweep(path) == sweep
    with pytest.raises(FileExistsError, match="overwrite=True"):
        write_sweep(path, sweep, overwrite=False)
    changed = Sweep(task="simple_test_task", seeds=(2,))
    write_sweep(path, changed, overwrite=True)
    assert read_sweep(path) == changed


def test_settings_project_the_sweep_as_editable_lists() -> None:
    sweep = Sweep(
        task="simple_test_task",
        algorithms=("a",),
        budgets=(PureDPBudget(eps=1.0),),
        seeds=(0, 7),
        time_limit=3.0,
    )

    assert sweep.settings() == {
        "task": "simple_test_task",
        "algorithms": ["a"],
        "instances": None,
        "budgets": [PureDPBudget(eps=1.0)],
        "seeds": [0, 7],
        "time_limit": 3.0,
        "memory_limit_gb": None,
    }
    assert Sweep(task="simple_test_task").settings()["seeds"] is None
