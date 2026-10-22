"""Build the T24 instance page record.

It reads one instance the Corpus area already holds and, where Explore
measured it, the instance's design condition number. Nothing here re-reads the
release: :mod:`assemble` supplies each of these already loaded, and every
number shown is one the release itself distributes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

from dpstats.domains import (
    BoxDomain,
    Domain,
    EuclideanSpace,
    InterceptDomain,
    OneHotDomain,
    ProductDomain,
)

from site_copy import copy_span, copy_text
from templates.blocks import anchor
from templates.corpus_source import (
    Corpus,
    CorpusInstance,
    CorpusSource,
    label_without_task,
    instance_route,
    licence_html,
)
from templates.model import InstanceDomain, InstancePage

# An instance page sits at "<task>/instances/<uuid>/", three levels below the
# site root whatever the task's slug.
INSTANCE_DEPTH = 3
INSTANCE_CLIMB = "../" * INSTANCE_DEPTH

# The file holding one task's instance cards, beside its other shared data.
INSTANCE_CARDS_NAME = "instances.json"

# The slot naming what each Metadata domain field holds, for the reader.
DOMAIN_NAMES = {
    "design_domain": "domain_covariates",
    "response_domain": "domain_response",
    "domain": "domain_data",
}

_SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_SIGNIFICANT = 4


def number(value: float) -> str:
    """One value to four significant figures, as the site sets numbers.

    Plain decimals between 10⁻⁴ and 10⁶, a power of ten outside them, and a
    true minus sign throughout.
    """
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    if -4 <= exponent < 6:
        digits = _SIGNIFICANT - 1 - exponent
        text = f"{round(value, digits):,.{max(0, digits)}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
    else:
        mantissa = f"{value / 10**exponent:.{_SIGNIFICANT - 1}f}"
        if mantissa.lstrip("-").startswith("10"):
            exponent += 1
            mantissa = f"{value / 10**exponent:.{_SIGNIFICANT - 1}f}"
        mantissa = mantissa.rstrip("0").rstrip(".")
        text = f"{mantissa} × 10{str(exponent).translate(_SUPERSCRIPT)}"
    return text.replace("-", "−")


def _interval(low: float, high: float) -> str:
    """One coordinate's bounds, open at an infinite end."""
    opening = "(" if math.isinf(low) else "["
    closing = ")" if math.isinf(high) else "]"
    return f"{opening}{number(low)}, {number(high)}{closing}"


def _parts(domain: Domain) -> list[Domain]:
    """A domain's factors, in column order, with nested products flattened."""
    if isinstance(domain, ProductDomain):
        return [part for factor in domain.parts for part in _parts(factor)]
    return [domain]


def _span(first: int, width: int) -> str:
    """The one-based column positions a factor covers."""
    return str(first) if width == 1 else f"{first}–{first + width - 1}"


def _counted(key: str, count: int) -> str:
    """*count* through the slot pair ``key_one`` and ``key_many``."""
    number = "one" if count == 1 else "many"
    return copy_span("corpus", f"{key}_{number}", fields={"count": str(count)})


def instance_domain(
    name: str, domain: Domain, coefficient: int | None = None
) -> InstanceDomain:
    """One declared domain as its summary line and its column table, as HTML.

    Args:
        name: The domain's name as the reader reads it, as HTML.
        domain: The declared domain.
        coefficient: The zero-based column the task estimates, if any; its row
            is marked.

    Returns:
        The domain's record. A domain of one column is summarised by its own
        values.
    """
    # Each row keeps its kind as plain text too, for the slot that marks it.
    rows: list[tuple[str, str, str, str]] = []
    counts = {"intercept": 0, "groups": 0, "indicators": 0, "ranges": 0}
    column = 1
    for part in _parts(domain):
        if isinstance(part, InterceptDomain):
            rows.append((_span(column, 1), *_kind("column_intercept"), "1"))
            counts["intercept"] += 1
            column += 1
        elif isinstance(part, BoxDomain):
            for low, high in part.bounds:
                rows.append(
                    (
                        _span(column, 1),
                        *_kind("column_range"),
                        _interval(float(low), float(high)),
                    )
                )
                counts["ranges"] += 1
                column += 1
        elif isinstance(part, EuclideanSpace):
            for _ in range(part.dim):
                rows.append(
                    (
                        _span(column, 1),
                        *_kind("column_range"),
                        _interval(-math.inf, math.inf),
                    )
                )
                counts["ranges"] += 1
                column += 1
        elif isinstance(part, OneHotDomain):
            if part.dim == 1:
                values = (
                    copy_span("corpus", "indicator_zero_or_one")
                    if part.allow_zero
                    else "1"
                )
            else:
                values = copy_span(
                    "corpus",
                    (
                        "indicator_at_most_one"
                        if part.allow_zero
                        else "indicator_exactly_one"
                    ),
                )
            rows.append(
                (
                    _span(column, part.dim),
                    *_kind("column_indicator_group"),
                    values,
                )
            )
            counts["groups"] += 1
            counts["indicators"] += part.dim
            column += part.dim
        else:
            named = type(part).__name__
            rows.append((_span(column, part.dim), escape(named), named, ""))
            column += part.dim
    shown = [
        (
            (
                span,
                copy_span("corpus", "coefficient_mark", fields={"kind": text}),
                values,
            )
            if coefficient is not None and _covers(span, coefficient + 1)
            else (span, kind, values)
        )
        for span, kind, text, values in rows
    ]
    if len(shown) == 1:
        return InstanceDomain(name, shown[0][2] or shown[0][1], tuple(shown))
    summary = [
        part
        for part in (
            copy_span("corpus", "summary_intercept") if counts["intercept"] else "",
            (
                f"{_counted('summary_groups', counts['groups'])} "
                f"({_counted('summary_columns', counts['indicators'])})"
                if counts["groups"]
                else ""
            ),
            _counted("summary_ranges", counts["ranges"]) if counts["ranges"] else "",
        )
        if part
    ]
    return InstanceDomain(name, ", ".join(summary), tuple(shown))


def _kind(key: str) -> tuple[str, str]:
    """A column kind's slot as HTML and as the plain text another slot quotes."""
    return copy_span("corpus", key), copy_text("corpus", key)


def _covers(span: str, position: int) -> bool:
    first, _, last = span.partition("–")
    return int(first) <= position <= int(last or first)


def instance_page(
    corpus: Corpus,
    source: CorpusSource,
    instance: CorpusInstance,
    *,
    condition: float | None,
) -> InstancePage:
    """The T24 record for one instance this release distributes.

    Args:
        corpus: The release this build reads, which the page names.
        source: The source the instance is drawn from.
        instance: The instance.
        condition: The design condition number Explore measured, if any.

    Returns:
        The page record; a fact the release does not carry is left out.
    """
    metadata = instance.metadata
    coefficient = getattr(metadata, "coefficient_of_interest_index", None)
    domains = (
        tuple(
            instance_domain(
                (
                    copy_span("corpus", DOMAIN_NAMES[name])
                    if name in DOMAIN_NAMES
                    else escape(name.replace("_", " ").capitalize())
                ),
                value,
                coefficient if name == "design_domain" else None,
            )
            for name in type(metadata).model_fields
            if isinstance(value := getattr(metadata, name), Domain)
        )
        if metadata is not None
        else ()
    )
    dimension = getattr(metadata, "dim", None)
    facts = [
        (
            ("corpus", "fact_study"),
            anchor(source.title, f"{INSTANCE_CLIMB}corpus/{source.slug}/"),
        ),
        (("site", "fact_field"), escape(instance.field)),
        (("corpus", "fact_data_licence"), licence_html(instance, INSTANCE_CLIMB)),
        (("corpus", "fact_sample_unit"), escape(instance.sample_unit)),
        (("corpus", "fact_samples"), f"{instance.n:,}"),
        (("corpus", "fact_dimension"), instance.dimension),
        (
            ("corpus", "fact_samples_per_dimension"),
            number(instance.n / dimension) if isinstance(dimension, int) else "",
        ),
        (
            ("site", "fact_condition_number"),
            number(condition) if condition is not None else "",
        ),
        (
            ("corpus", "fact_coefficient"),
            (
                copy_span(
                    "corpus",
                    "coefficient_column",
                    fields={"column": str(coefficient + 1)},
                )
                if isinstance(coefficient, int)
                else ""
            ),
        ),
        (
            ("corpus", "fact_weight"),
            f"{instance.weight * 100:.3g}%" if instance.weight is not None else "",
        ),
    ]
    source_facts = instance.source_facts
    return InstancePage(
        title=label_without_task(instance),
        facts=tuple((copy_span(*slot), value) for slot, value in facts if value),
        domains=domains,
        variables=tuple(str(name) for name in source_facts.get("variables", ())),
        coefficient=str(source_facts.get("coefficient") or ""),
        claim=_shown_claim(source_facts.get("claims", ())),
    )


# How prominently the paper states a claim, most prominent first.
_PROMINENCE = ("headline", "primary", "secondary")


def _shown_claim(
    claims: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str, str] | None:
    """The one claim an instance page quotes, as ``(quote, page, table)``.

    The most prominent claim the paper states in prose, the first listed on a
    tie; only where no claim has a quote, the first claim, without one.
    """
    if not claims:
        return None
    quoted = [claim for claim in claims if claim.get("quote")]
    chosen = (
        min(
            quoted,
            key=lambda claim: (
                _PROMINENCE.index(claim["prominence"])
                if claim.get("prominence") in _PROMINENCE
                else len(_PROMINENCE)
            ),
        )
        if quoted
        else claims[0]
    )
    return (
        str(chosen["quote"]) if chosen.get("quote") else None,
        str(chosen.get("page") or ""),
        str(chosen.get("table") or ""),
    )


def instance_card(
    source: CorpusSource,
    instance: CorpusInstance,
    *,
    condition: float | None,
    headline: str,
) -> dict[str, Any]:
    """The reduced card for one instance, as its task's cards file holds it.

    The route is from the site root; the headline baseline's value is given
    only where the instance stores it as a number.
    """
    dimension = getattr(instance.metadata, "dim", None)
    held = dict(instance.references).get(headline)
    numeric = isinstance(held, int | float) and not isinstance(held, bool)
    return {
        "label": label_without_task(instance),
        "route": instance_route(instance),
        "study": source.title,
        "field": instance.field,
        "n": instance.n,
        "d": dimension if isinstance(dimension, int) else None,
        "kappa": condition,
        "baseline": [headline, float(held)] if numeric else None,
    }
