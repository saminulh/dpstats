"""Shared per-seed fake for ``numpy.random.default_rng``.

``default_rng`` is a process-global name, so replacing it replaces the
generator of every component a test drives, not one component's randomness.
The fake therefore keeps the seed axis visible: it builds one
:class:`FakeGenerator` per ``default_rng`` call and records that seed, so a
test can assert that composed children received the distinct child seeds
``split_seed`` promises. Each generator method records its arguments and its
result, and an unscripted method delegates to a real generator seeded
identically, so a stream nobody scripted stays a genuine pseudorandom stream
and test data built inside the patched window is unaffected.

Because the replaced name is global, every caller reached while it is replaced
is served by this object, not only the component under test. The stand-in
therefore covers each generator method the library calls plus the
``standard_normal`` that test data builders use.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pytest


GeneratorMethod = Literal[
    "normal",
    "standard_normal",
    "laplace",
    "uniform",
    "gumbel",
    "gamma",
    "standard_t",
    "poisson",
    "geometric",
    "logseries",
    "binomial",
    "negative_binomial",
    "random",
    "integers",
    "permutation",
    "choice",
    "multivariate_normal",
]
DrawParameters = Mapping[str, Any]
DrawScript = Callable[[DrawParameters], Any]


@dataclass(frozen=True)
class RandomDraw:
    """One observed generator call.

    Attributes:
        seed: The seed the serving generator was constructed with.
        method: The ``numpy.random.Generator`` method the call used.
        parameters: The call's arguments, under the method's parameter names.
        value: The value the call returned.
    """

    seed: int
    method: GeneratorMethod
    parameters: DrawParameters
    value: Any


def zero_draw(parameters: DrawParameters) -> Any:
    """Serve ``0.0``, or zeros in the requested shape.

    Zero noise makes a mechanism's released value the noise-free quantity, so
    a test can compare that quantity against an independent oracle while the
    recorded draws still expose the noise scales the mechanism asked for.
    Applies to the distribution methods that take a ``size``.
    """
    size = parameters["size"]
    return 0.0 if size is None else np.zeros(size)


def draws_in_order(values: Iterable[Any]) -> DrawScript:
    """Serve prepared values, one per call, in order.

    Args:
        values: The values to serve, consumed as the draws are requested.

    Returns:
        A script that serves the next prepared value and raises
        ``StopIteration`` once they are exhausted.
    """
    remaining = iter(values)

    def serve(_parameters: DrawParameters) -> Any:
        return next(remaining)

    return serve


class FakeGenerator:
    """A ``numpy.random.Generator`` stand-in bound to exactly one seed."""

    def __init__(self, seed: int, randomness: "FakeRandomness") -> None:
        self.seed = seed
        self._randomness = randomness
        self._source = randomness.real_default_rng(seed)

    def _draw(self, method: GeneratorMethod, parameters: DrawParameters) -> Any:
        return self._randomness.serve(self.seed, method, parameters, self._source)

    def normal(self, loc: Any = 0.0, scale: Any = 1.0, size: Any = None) -> Any:
        return self._draw("normal", {"loc": loc, "scale": scale, "size": size})

    def standard_normal(self, size: Any = None) -> Any:
        return self._draw("standard_normal", {"size": size})

    def laplace(self, loc: Any = 0.0, scale: Any = 1.0, size: Any = None) -> Any:
        return self._draw("laplace", {"loc": loc, "scale": scale, "size": size})

    def uniform(self, low: Any = 0.0, high: Any = 1.0, size: Any = None) -> Any:
        return self._draw("uniform", {"low": low, "high": high, "size": size})

    def gumbel(self, loc: Any = 0.0, scale: Any = 1.0, size: Any = None) -> Any:
        return self._draw("gumbel", {"loc": loc, "scale": scale, "size": size})

    def gamma(self, shape: Any, scale: Any = 1.0, size: Any = None) -> Any:
        return self._draw("gamma", {"shape": shape, "scale": scale, "size": size})

    def standard_t(self, df: Any, size: Any = None) -> Any:
        return self._draw("standard_t", {"df": df, "size": size})

    def poisson(self, lam: Any = 1.0, size: Any = None) -> Any:
        return self._draw("poisson", {"lam": lam, "size": size})

    def geometric(self, p: Any, size: Any = None) -> Any:
        return self._draw("geometric", {"p": p, "size": size})

    def logseries(self, p: Any, size: Any = None) -> Any:
        return self._draw("logseries", {"p": p, "size": size})

    def binomial(self, n: Any, p: Any, size: Any = None) -> Any:
        return self._draw("binomial", {"n": n, "p": p, "size": size})

    def negative_binomial(self, n: Any, p: Any, size: Any = None) -> Any:
        return self._draw("negative_binomial", {"n": n, "p": p, "size": size})

    def random(self, size: Any = None) -> Any:
        return self._draw("random", {"size": size})

    def integers(
        self, low: Any, high: Any = None, size: Any = None, endpoint: bool = False
    ) -> Any:
        return self._draw(
            "integers",
            {"low": low, "high": high, "size": size, "endpoint": endpoint},
        )

    def permutation(self, x: Any) -> Any:
        return self._draw("permutation", {"x": x})

    def choice(
        self, a: Any, size: Any = None, replace: bool = True, p: Any = None
    ) -> Any:
        return self._draw("choice", {"a": a, "size": size, "replace": replace, "p": p})

    def multivariate_normal(
        self,
        mean: Any,
        cov: Any,
        size: Any = None,
        check_valid: str = "warn",
        tol: float = 1e-8,
    ) -> Any:
        return self._draw(
            "multivariate_normal",
            {
                "mean": mean,
                "cov": cov,
                "size": size,
                "check_valid": check_valid,
                "tol": tol,
            },
        )


class FakeRandomness:
    """Record of the seeds and draws a test's randomness passed through."""

    def __init__(self, real_default_rng: Callable[[int], np.random.Generator]) -> None:
        self.real_default_rng = real_default_rng
        self._seeds: list[int] = []
        self._draws: list[RandomDraw] = []
        self._scripts: dict[GeneratorMethod, DrawScript] = {}

    @property
    def seeds(self) -> tuple[int, ...]:
        """Return the seeds passed to ``default_rng``, in call order."""
        return tuple(self._seeds)

    def draws_of(self, method: GeneratorMethod) -> tuple[RandomDraw, ...]:
        """Return the recorded draws served by one generator method."""
        return tuple(draw for draw in self._draws if draw.method == method)

    def parameters_of(self, method: GeneratorMethod, parameter: str) -> list[Any]:
        """Return one argument of every recorded call to ``method``, in order."""
        return [draw.parameters[parameter] for draw in self.draws_of(method)]

    def values_of(self, method: GeneratorMethod) -> list[Any]:
        """Return the value every recorded call to ``method`` returned."""
        return [draw.value for draw in self.draws_of(method)]

    def script(self, method: GeneratorMethod, draw: DrawScript) -> None:
        """Serve ``method`` from ``draw`` instead of a real generator."""
        self._scripts[method] = draw

    def serve(
        self,
        seed: int,
        method: GeneratorMethod,
        parameters: DrawParameters,
        source: np.random.Generator,
    ) -> Any:
        """Produce one draw for a fake generator and record it.

        An unscripted method is served by ``source``, the real generator for
        that seed.
        """
        script = self._scripts.get(method)
        if script is None:
            value = getattr(source, method)(**parameters)
        else:
            value = script(parameters)
        self._draws.append(RandomDraw(seed, method, parameters, value))
        return value

    def forget(self) -> None:
        """Discard the seeds and draws recorded so far.

        Test data built inside the patched window passes through the fake like
        any other caller. Forgetting immediately before the subject runs keeps
        the recorded seeds and draws the subject's own.
        """
        self._seeds.clear()
        self._draws.clear()

    def generator(self, seed: int) -> FakeGenerator:
        """Serve one fake generator and record the seed it was built with."""
        self._seeds.append(seed)
        return FakeGenerator(seed, self)


@pytest.fixture
def fake_randomness(monkeypatch: pytest.MonkeyPatch) -> FakeRandomness:
    """Replace ``numpy.random.default_rng`` with a recording per-seed fake."""
    randomness = FakeRandomness(np.random.default_rng)
    monkeypatch.setattr(np.random, "default_rng", randomness.generator)
    return randomness
