"""The disclosure reads configured structure despite arbitrary labels."""

import csv

from dpstats.algorithm import (
    CompositeAlgorithm,
    Transformation,
    register_algorithm,
)
from dpstats.tasks.linear_regression.transformations.m_estimation import (
    ToMEstimation,
    RobHytResidualLoss,
)
from dpstats.tasks.m_estimation.algorithms.objective_perturbation_kifer import (
    ObjectivePerturbationKifer,
)
from dpstats.tasks.mean_estimation.algorithms import (
    GaussianMechanism,
)
from dpstats.tasks.mean_estimation.transformations import (
    ApproxBounds,
    DataBounds,
    Standardize,
)
from dpstats.workspace.layout import (
    algorithm_declarations_path,
    task_collection_directory,
)


from algorithm_provenance import provenance_payload, link_provenance
from tests.unit.docs.conftest import TunedMean


class UndocumentedMean(GaussianMechanism):
    pass


def test_prepared_description_links_without_reloading_its_configuration(tmp_path):
    algorithm = GaussianMechanism()
    register_algorithm(tmp_path, algorithm)
    cls = type(algorithm)
    routes = {f"{cls.__module__}.{cls.__qualname__}": "library/gaussian.md"}
    expected = provenance_payload(tmp_path, "mean_estimation", routes)
    prepared = provenance_payload(tmp_path, "mean_estimation", {})
    link_provenance(prepared, routes)
    assert prepared == expected


def test_provenance_uses_configuration_not_a_misleading_label(tmp_path):
    algorithm = CompositeAlgorithm(
        label="No transformations + AnotherAlgorithm </script>",
        transformations=(DataBounds(), Standardize()),
        algorithm=GaussianMechanism(),
    )
    uuid = register_algorithm(tmp_path, algorithm)
    cls = GaussianMechanism
    route = "library/tasks/mean_estimation/methods/GaussianMechanism.md"
    payload = provenance_payload(
        tmp_path,
        "mean_estimation",
        {
            f"{cls.__module__}.{cls.__qualname__}": route,
            "dpstats.tasks.mean_estimation.transformations.DataBounds": "library/data-bounds.md",
        },
    )

    assert list(payload) == [uuid]
    component = payload[uuid]["component"]
    assert component["kind"] == "Composition"
    assert component["children"][0]["component"]["docs"] == "library/data-bounds/"
    assert [
        (child["field"], child["component"]["name"]) for child in component["children"]
    ] == [
        ("transformations[0]", "DataBounds"),
        ("transformations[1]", "Standardize"),
        ("algorithm", "GaussianMechanism"),
    ]
    assert (
        component["children"][-1]["component"]["docs"]
        == route.removesuffix(".md") + "/"
    )
    assert payload[uuid]["configuration"]["config"]["label"] == algorithm.label


def test_missing_registry_has_no_invented_provenance(tmp_path):
    assert provenance_payload(tmp_path, "mean_estimation", {}) == {}


def test_recipe_description_is_joined_by_exact_configuration_identity(tmp_path):
    algorithm = CompositeAlgorithm(
        transformations=(ToMEstimation(loss=RobHytResidualLoss(k=1.0)),),
        algorithm=ObjectivePerturbationKifer(),
    )
    uuid = register_algorithm(tmp_path, algorithm)
    description = "PME: RobHyt with KST; <not HTML>"
    provenance_ref = "provenance/algorithms/pme.md"
    ref = tmp_path / provenance_ref
    ref.parent.mkdir(parents=True)
    ref.write_text("# PME\nhttps://arxiv.org/abs/2108.08266v1\n")
    declarations_path = algorithm_declarations_path(
        task_collection_directory(tmp_path, "linear_regression")
    )
    declarations_path.parent.mkdir(parents=True, exist_ok=True)
    with declarations_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["uuid", "description", "provenance_ref"])
        writer.writerow([uuid, description, provenance_ref])
    row = provenance_payload(tmp_path, "linear_regression", {})[uuid]
    assert row["declaration"] == {
        "description": description,
        "provenance_ref": provenance_ref,
    }
    assert row["configuration"] == algorithm.dump_config().model_dump(mode="json")


def test_hyperparameter_rows_come_from_field_introspection_not_raw_json(tmp_path):
    """Rows are each component's own configured fields, not structural ones.

    A composition's own fields (its wrapped transformations and algorithm, and the
    inherited ``label``) are structural and never appear as hyperparameters of
    any component.
    """
    algorithm = CompositeAlgorithm(
        label="Bounded tuned mean",
        transformations=(ApproxBounds(),),
        algorithm=TunedMean(clip_norm=3.0, threshold=2.0),
    )
    uuid = register_algorithm(tmp_path, algorithm)
    component = provenance_payload(tmp_path, "mean_estimation", {})[uuid]["component"]

    assert component["kind"] == "Composition"
    assert component["hyperparameters"] == []

    transformation = component["children"][0]["component"]
    assert transformation["name"] == "ApproxBounds"
    assert {row["name"] for row in transformation["hyperparameters"]} == (
        set(ApproxBounds.model_fields) - set(Transformation.model_fields)
    )

    core = component["children"][-1]["component"]
    assert core["name"] == "TunedMean"
    assert core["hyperparameters"] == [
        {
            "name": "clip_norm",
            "value": "3.0",
            "default": "1.0",
            "search": "0.1 – 10.0 (log scale)",
            "description": "Per-example clipping norm.",
        },
        {
            "name": "noise",
            "value": "'gaussian'",
            "default": "'gaussian'",
            "search": "'gaussian', 'laplace'",
            "description": "Which noise distribution to add.",
        },
        {
            "name": "threshold",
            "value": "2.0",
            "default": "required",
            "search": "",
            "description": "Required cutoff.",
        },
    ]
    assert not any(
        row["name"] == "label"
        for node in (component, transformation, core)
        for row in node["hyperparameters"]
    )


def test_a_configured_object_field_nests_instead_of_serializing_as_a_mapping(tmp_path):
    """A configured field's own class, docs and values -- never a raw mapping."""
    algorithm = CompositeAlgorithm(
        transformations=(ToMEstimation(loss=RobHytResidualLoss(k=1.0)),),
        algorithm=ObjectivePerturbationKifer(),
    )
    uuid = register_algorithm(tmp_path, algorithm)
    component = provenance_payload(tmp_path, "linear_regression", {})[uuid]["component"]

    transformation = component["children"][0]["component"]
    assert transformation["name"] == "ToMEstimation"
    assert not any(row["name"] == "loss" for row in transformation["hyperparameters"])
    assert {row["name"] for row in transformation["hyperparameters"]} == {
        "parameter_radius"
    }

    loss_child = next(
        child for child in transformation["children"] if child["field"] == "loss"
    )["component"]
    assert loss_child["name"] == "RobHytResidualLoss"
    assert loss_child["hyperparameters"] == [
        {
            "name": "k",
            "value": "1.0",
            "default": "required",
            "search": "",
            "description": "Positive tuning constant in response units.",
        }
    ]


def test_each_step_summarises_itself_from_its_own_docstring(tmp_path):
    """The summary is the class's opening paragraph on one line, never inherited."""
    algorithm = CompositeAlgorithm(
        transformations=(Standardize(),), algorithm=GaussianMechanism()
    )
    uuid = register_algorithm(tmp_path, algorithm)
    children = provenance_payload(tmp_path, "mean_estimation", {})[uuid]["component"][
        "children"
    ]
    standardize = children[0]["component"]["summary"]
    assert standardize and "\n" not in standardize
    assert standardize.split()[:3] == Standardize.__doc__.split()[:3]
    assert "summary" in children[1]["component"]

    other = tmp_path / "undocumented"
    uuid = register_algorithm(other, UndocumentedMean())
    undocumented = provenance_payload(other, "mean_estimation", {})[uuid]["component"]
    assert undocumented["summary"] == ""
