"""Run the zero-work resume and report final coordinate and status counts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dpstats import open_collection
from dpstats.analysis import measure_results
from dpstats.tasks import list_built_in_metrics
from dpstats.tasks import MeanEstimation


STORAGE_ROOT = Path("benchmark-storage")
ALGORITHM_REGISTRIES = ("simulated",)
INSTANCE_REGISTRIES = ("simulated",)


def main() -> None:
    collection = open_collection(STORAGE_ROOT, MeanEstimation)
    results = collection.run_sweep(
        collection.list_sweeps()[0],
        algorithm_registries=ALGORITHM_REGISTRIES,
        instance_registries=INSTANCE_REGISTRIES,
    )
    status = collection.get_sweep_status(
        collection.list_sweeps()[0],
        algorithm_registries=ALGORITHM_REGISTRIES,
        instance_registries=INSTANCE_REGISTRIES,
    )
    measurements = measure_results(
        results,
        list_built_in_metrics(MeanEstimation)["l2_error_to_ground_truth"],
    )
    report = {
        "coordinates": len(results),
        "measurements": len(measurements),
        "coverage": status["cells"],
        "statuses": dict(sorted(Counter(results["status"]).items())),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
