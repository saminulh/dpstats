"""Shared hypothesis runtime profile for the property suites.

One home decides how many generated examples the wire-contract properties
draw, so the default pytest gate stays fast while a deeper profile remains
available. Set ``DP_HYPOTHESIS_PROFILE=thorough`` to opt into the deeper
profile; importing this module applies the selected profile.
"""

import os

from hypothesis import settings

settings.register_profile(
    "dp_default",
    max_examples=30,
    deadline=None,
    database=None,
)
settings.register_profile(
    "dp_thorough",
    max_examples=1000,
    deadline=None,
    database=None,
)
settings.load_profile(
    "dp_thorough"
    if os.environ.get("DP_HYPOTHESIS_PROFILE") == "thorough"
    else "dp_default"
)
