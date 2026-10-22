"""The accounted bottom outcome carries the message a collection records."""

import pytest

from dpstats.exceptions import InvalidRegime, NoRelease


def test_no_release_message_is_its_text():
    released = NoRelease("PTR check failed")

    assert str(released) == "PTR check failed"


def test_no_release_with_no_argument_keeps_its_default_message():
    assert str(NoRelease()) == "No result released"


def test_invalid_regime_requires_an_authored_message():
    with pytest.raises(TypeError):
        InvalidRegime()  # type: ignore[call-arg]


def test_invalid_regime_message_is_its_text():
    assert str(InvalidRegime("x")) == "x"
