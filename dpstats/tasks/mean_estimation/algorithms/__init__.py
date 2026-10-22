"""Built-in mean estimation algorithms, registered in ``ALGORITHMS``."""

from dpstats.algorithm import BaseAlgorithm
from dpstats.tasks.mean_estimation.algorithms.sample_mean import SampleMean
from dpstats.tasks.mean_estimation.algorithms.gaussian_mechanism import (
    GaussianMechanism,
)
from dpstats.tasks.mean_estimation.algorithms.dp_histogram import (
    DPHistogram,
)
from dpstats.tasks.mean_estimation.algorithms.histogram_gaussian import (
    HistogramGaussian,
)
from dpstats.tasks.mean_estimation.algorithms.coinpress import (
    CoinPress,
)
from dpstats.tasks.mean_estimation.algorithms.boxem import BoxEM
from dpstats.tasks.mean_estimation.algorithms.plan import (
    PLANBinarySearchGaussian,
)
from dpstats.tasks.mean_estimation.algorithms.hourglass import (
    Hourglass,
)

__all__ = [
    "SampleMean",
    "GaussianMechanism",
    "DPHistogram",
    "HistogramGaussian",
    "CoinPress",
    "BoxEM",
    "PLANBinarySearchGaussian",
    "Hourglass",
]

ALGORITHMS: tuple[type[BaseAlgorithm], ...] = (
    SampleMean,
    GaussianMechanism,
    DPHistogram,
    HistogramGaussian,
    CoinPress,
    BoxEM,
    PLANBinarySearchGaussian,
    Hourglass,
)
