"""Post-run Metrics for template_task."""

from dpstats.metric import Metric, MetricArgument

from . import TemplateTask
from .types import TemplateTaskOutput

ground_truth = MetricArgument(
    name="ground_truth", description="The population value output.mean_est estimates."
)


def _absolute_error(output: TemplateTaskOutput, *, ground_truth: float) -> float:
    """Absolute difference between output.mean_est and a reference value."""
    return float(abs(output.mean_est - ground_truth))


absolute_error = Metric(
    task=TemplateTask,
    name="absolute_error",
    calculation=_absolute_error,
    requires=(ground_truth,),
    label="Absolute error",
)

METRICS: tuple[Metric, ...] = (absolute_error,)
