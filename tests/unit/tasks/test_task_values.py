"""Task value bases keep Pydantic and Domain persistence details private."""

from __future__ import annotations

import numpy as np
from pydantic import model_validator

from dpstats.transport.canonical import (
    canonical_json_bytes,
    decode_json_document,
)
from dpstats.domains import BoxDomain, Domain, ProductDomain
from dpstats.task import TaskInput, TaskMetadata, TaskOutput
from dpstats.task import Array1D


class ExampleMetadata(TaskMetadata):
    n_samples: int
    domain: Domain

    @model_validator(mode="after")
    def _validate_metadata(self) -> ExampleMetadata:
        if self.n_samples < 1:
            raise ValueError("n_samples must be positive")
        return self

    def validate_input(self, input: TaskInput) -> None:
        ExampleInput.model_validate(input)


class ExampleInput(TaskInput):
    data: Array1D


class ExampleOutput(TaskOutput):
    estimate: float


def test_task_values_have_plain_contributor_facing_fields():
    metadata = ExampleMetadata(
        n_samples=2,
        domain=BoxDomain(bounds=[[0, 1]]),
    )
    input = ExampleInput(data=[0, 1])
    output = ExampleOutput(estimate=0.5)

    assert isinstance(metadata.domain, Domain)
    assert input.data.dtype == np.dtype(float)
    assert output.estimate == 0.5


def test_domain_typed_metadata_round_trips_through_shared_configuration():
    metadata = ExampleMetadata(
        n_samples=2,
        domain=ProductDomain(
            parts=(
                BoxDomain(bounds=[[0, 1]]),
                BoxDomain(bounds=[[-np.inf, np.inf]]),
            )
        ),
    )

    payload = decode_json_document(
        canonical_json_bytes(metadata.dump_config().model_dump())
    )

    assert payload == {
        "path": f"{__name__}:ExampleMetadata",
        "config": {
            "n_samples": 2,
            "domain": {
                "path": "dpstats.domains:ProductDomain",
                "config": {
                    "parts": [
                        {
                            "path": "dpstats.domains:BoxDomain",
                            "config": {"bounds": [[0.0, 1.0]]},
                        },
                        {
                            "path": "dpstats.domains:BoxDomain",
                            "config": {
                                "bounds": [[-np.inf, np.inf]],
                            },
                        },
                    ]
                },
            },
        },
    }

    restored = ExampleMetadata.load_record(payload)
    assert isinstance(restored.domain, ProductDomain)
    assert all(isinstance(part, BoxDomain) for part in restored.domain.parts)
    assert np.isneginf(restored.domain.parts[1].bounds[0, 0])
    assert np.isposinf(restored.domain.parts[1].bounds[0, 1])
