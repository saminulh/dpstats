from __future__ import annotations

import numpy as np


def test_domains_is_the_public_configured_domain_home() -> None:
    from dpstats.domains import BoxDomain, Domain

    domain = BoxDomain(bounds=np.array([[-1.0, 1.0], [0.0, 2.0]]))

    assert domain.dump_config().model_dump() == {
        "path": "dpstats.domains:BoxDomain",
        "config": {"bounds": [[-1.0, 1.0], [0.0, 2.0]]},
    }
    restored = Domain.load_record(domain.dump_config())
    assert isinstance(restored, BoxDomain)
    assert np.array_equal(restored.bounds, domain.bounds)


def test_box_domain_reference_exposes_its_constructor_field() -> None:
    """The generated reference reads field descriptions off the model.

    ``bounds`` is BoxDomain's only constructor field; a docs-only reader
    constructing it needs the reference page to name it and its shape, not
    just discover it from a runtime validation error.
    """
    from dpstats.domains import BoxDomain

    field = BoxDomain.model_fields["bounds"]
    assert field.description
    assert "dim" in field.description or "dimension" in field.description
    assert "2" in field.description
