"""Built-in linear regression algorithms, registered in ``ALGORITHMS``."""

from dpstats.algorithm import BaseAlgorithm
from dpstats.tasks.linear_regression.algorithms.adaops import AdaOPS
from dpstats.tasks.linear_regression.algorithms.adaptive_ssp import AdaSSP
from dpstats.tasks.linear_regression.algorithms.local_output_perturbation import (
    LocalOutputPerturbation,
)
from dpstats.tasks.linear_regression.algorithms.one_pass_dpgd import (
    OnePassDPGD,
)
from dpstats.tasks.linear_regression.algorithms.boosted_adassp import (
    BoostedAdaSSP,
)
from dpstats.tasks.linear_regression.algorithms.dp_ambssgd import DPAMBSSGD
from dpstats.tasks.linear_regression.algorithms.efficient_ptr import (
    EfficientPTR,
)
from dpstats.tasks.linear_regression.algorithms.grid_exponential_mechanism import (
    GridExponentialMechanism,
)
from dpstats.tasks.linear_regression.algorithms.iterative_hessian_mixing import (
    IterativeHessianMixing,
)
from dpstats.tasks.linear_regression.algorithms.ops_ptr import OPSPTR
from dpstats.tasks.linear_regression.algorithms.ols import OLS
from dpstats.tasks.linear_regression.algorithms.projected_ols import (
    ProjectedOLS,
)
from dpstats.tasks.linear_regression.algorithms.ridge import Ridge
from dpstats.tasks.linear_regression.algorithms.tail_robust_dpgd import (
    TailRobustDPGD,
    TailRobustSparseHuberIHT,
)
from dpstats.tasks.linear_regression.algorithms.tukey_em import TukeyEM
from dpstats.tasks.linear_regression.algorithms.ssp import (
    SSP,
)

__all__ = [
    "AdaOPS",
    "AdaSSP",
    "LocalOutputPerturbation",
    "BoostedAdaSSP",
    "DPAMBSSGD",
    "EfficientPTR",
    "GridExponentialMechanism",
    "IterativeHessianMixing",
    "OnePassDPGD",
    "OPSPTR",
    "OLS",
    "ProjectedOLS",
    "Ridge",
    "TailRobustDPGD",
    "TailRobustSparseHuberIHT",
    "TukeyEM",
    "SSP",
]

ALGORITHMS: tuple[type[BaseAlgorithm], ...] = (
    AdaOPS,
    AdaSSP,
    LocalOutputPerturbation,
    BoostedAdaSSP,
    DPAMBSSGD,
    EfficientPTR,
    GridExponentialMechanism,
    IterativeHessianMixing,
    OnePassDPGD,
    OPSPTR,
    OLS,
    ProjectedOLS,
    Ridge,
    TailRobustDPGD,
    TailRobustSparseHuberIHT,
    TukeyEM,
    SSP,
)
