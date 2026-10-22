from pathlib import Path

import dpstats as dp
from dpstats import analysis
from dpstats.budgets import ApproxDPBudget
from dpstats.simulations.linear_regression import simulate_hypercube
from dpstats.tasks import linear_regression as lr

workspace = dp.init_workspace(Path("interactive-example"))
collection = dp.open_collection(workspace, lr.LinearRegression)
for n in (60, 120, 240):
    metadata, input_data, references = simulate_hypercube(seed=n, n_samples=n, dim=2)
    collection.add_instance(metadata, input_data, references, label=f"n={n}")

instances = collection.load_instances()
print(
    [
        (item.metadata.n_samples, item.metadata.design_domain.dim)
        for item in instances.values()
    ]
)

algorithms = [
    lr.algorithms.SSP(label="SSP"),
    dp.CompositeAlgorithm(
        transformations=(lr.transformations.Standardize(),),
        algorithm=lr.algorithms.SSP(),
        label="Standardized SSP",
    ),
]
for algorithm in algorithms:
    collection.add_algorithm(algorithm)
metric = lr.metrics.prediction_rmse_to_ols
print(analysis.get_metric_availability(instances.values(), metric))

collection.run_sweep(
    algorithms=algorithms,
    instances=list(instances.values()),
    budgets=[ApproxDPBudget(eps=3, delta=1e-5)],
    seeds=[0, 1, 2],
    mode="missing",
)
results = collection.load_results(results_source="local")
measured = analysis.measure_results(results, metric, collection=collection)
columns = ["algorithm_label", "instance_label", "alg_seed", metric.name]
print(measured[columns])

curves = analysis.compute_profile_curves(
    measured,
    [0, 1, 2, 5, 10],
    metric=metric,
    baseline="ols_prediction_se",
    instances=list(instances.values()),
)
figure = analysis.plot_profiles(
    curves,
    xlabel="Prediction RMSE / OLS prediction standard error",
    ylabel="Fraction within threshold, averaged over Instances",
    xscale="linear",
    hover="Within {tau}: {share}",
    guide="Threshold {tau}",
)
measured[columns].to_csv(workspace / "measurements.csv", index=False)
figure.write_html(workspace / "profiles.html")
