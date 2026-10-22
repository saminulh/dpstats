"""Public run-sweep display over reused typed Results."""

from pathlib import Path

import pytest

from dpstats.budgets import PureDPBudget
from dpstats.workspace import TaskCollection
from dpstats.scripts.run_sweep import main
from dpstats.results import ResultStatus
from tests.attempts import current_result, write_attempt_partition
from tests.registry_components import SimpleAlgorithm, simple_producer
from tests.sweep_helpers import (
    cli_args,
    local_task_root,
    sweep_coordinates,
    write_sweep,
)


@pytest.mark.parametrize("include_output", [False, True])
def test_run_displays_readable_values_without_changing_typed_results(
    tmp_path: Path,
    collection: TaskCollection,
    capsys: pytest.CaptureFixture[str],
    include_output: bool,
) -> None:
    """CLI text identifies trials without printing complete live objects."""
    recipe = write_sweep(
        collection,
        [SimpleAlgorithm(label="my-method")],
        [simple_producer()],
        [PureDPBudget(eps=1.0)],
    )
    (coordinate,) = sweep_coordinates(collection, recipe)
    write_attempt_partition(
        local_task_root(collection),
        [
            current_result(
                instance_uuid=coordinate.instance_uuid,
                alg_uuid=coordinate.alg_uuid,
                budget=coordinate.budget.dump_config().model_dump(),
            )
        ],
    )

    main(
        [
            *cli_args(recipe, tmp_path),
            *(["--include-output"] if include_output else []),
        ]
    )

    display = capsys.readouterr().out
    assert "success" in display
    assert "ResultStatus." not in display
    assert "my-method" in display
    assert coordinate.instance_uuid in display
    assert "SimpleMetadata(" not in display
    assert "scale=1.0" not in display
    assert ("payload" in display) is include_output
    result = collection.load_results().iloc[0]
    assert result["status"] is ResultStatus.SUCCESS
    assert result["algorithm"].label == "my-method"
    assert result["instance"].uuid == coordinate.instance_uuid
    assert result["budget"] == PureDPBudget(eps=1.0)
