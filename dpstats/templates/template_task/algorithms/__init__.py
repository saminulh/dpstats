"""Algorithms of the template_task task, registered in ``ALGORITHMS``."""

from dpstats.algorithm import BaseAlgorithm

from .laplace_mean import LaplaceMean

ALGORITHMS: tuple[type[BaseAlgorithm], ...] = (LaplaceMean,)
