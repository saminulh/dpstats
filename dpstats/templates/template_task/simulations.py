"""Simulated-instance producer examples for TemplateTask."""

from __future__ import annotations

import numpy as np
from pydantic import JsonValue

from dpstats.domains import BoxDomain

from .types import TemplateTaskInput, TemplateTaskMetadata


def simulate_uniform(
    *, seed: int, domain_min: float = 0.0, domain_max: float = 1.0
) -> tuple[TemplateTaskMetadata, TemplateTaskInput, dict[str, JsonValue]]:
    """Generate 100 scalar uniform observations in a declared Box domain."""
    if domain_min >= domain_max:
        raise ValueError("domain_min must be less than domain_max.")
    data = np.random.default_rng(seed).uniform(domain_min, domain_max, size=100)
    metadata = TemplateTaskMetadata(
        n_samples=100,
        domain=BoxDomain(bounds=np.array([[domain_min, domain_max]])),
    )
    return (
        metadata,
        TemplateTaskInput(data=data),
        {"ground_truth": (domain_min + domain_max) / 2},
    )
