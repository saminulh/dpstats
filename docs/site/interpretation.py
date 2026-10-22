"""How a results surface reads one algorithm's privacy-semantics class.

Only the class the protocol's eligibility names enters official standings.
Explore can explicitly reveal other classes as unranked comparisons. Missing
metadata never establishes eligibility.
"""

from __future__ import annotations


def shown_by_default(privacy_interpretation: str | None, eligibility: str) -> bool:
    """Whether a row carrying this class opens visible and ranked.

    Args:
        privacy_interpretation: The class its release registers for it, or
            ``None`` where the release registers nothing about it.
        eligibility: The privacy interpretation the protocol ranks.
    """
    return privacy_interpretation is not None and privacy_interpretation == eligibility
