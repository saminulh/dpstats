"""The notebook a built Explore page exports reads exactly the view it was given.

The build supplies the panel data and the notebook template; the check writes a
chosen selection into the template the way the page's download does, runs it
against the local release the template names, and counts the curves it plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from recreate_notebook import PARAMETERS_KEY, PARAMETERS_TAG, parameter_literal

TASK = "mean_estimation"


def _parameter_cells(document: dict) -> list[dict]:
    return [
        cell
        for cell in document["cells"]
        if PARAMETERS_TAG in cell["metadata"].get("tags", [])
    ]


def _rewrite(document: dict, chosen: dict) -> dict:
    """Write ``chosen`` into the parameter cells, as the page's download does."""
    values: dict = {}
    for cell in _parameter_cells(document):
        declared = cell["metadata"][PARAMETERS_KEY]
        declared.update({name: chosen[name] for name in declared if name in chosen})
        cell["source"] = [
            cell["source"][0],
            *[
                f"{name} = {parameter_literal(value)}\n"
                for name, value in declared.items()
            ],
            *cell["source"][1 + len(declared) :],
        ]
        values.update(declared)
    return values


PICKED = 3

# Printed by a cell appended after the exported ones.
_REPORT = (
    "import json\n"
    "print('CURVES=' + json.dumps(sorted({v.uuid for v in curves['algorithm']})))\n"
    "print('LINES=' + str(len(figure.data)))\n"
)


def _report(outputs: list[dict]) -> dict[str, str]:
    text = "".join(
        "".join(output.get("text", ""))
        for output in outputs
        if output["output_type"] == "stream"
    )
    return dict(
        line.split("=", 1)
        for line in text.splitlines()
        if line.startswith(("CURVES=", "LINES="))
    )


def test_an_exported_three_curve_view_plots_those_three_curves(
    built_site: Path, tmp_path: Path
) -> None:
    """The plotted algorithms and the one budget are the ones exported."""
    results = built_site / TASK / "public-results"
    index = json.loads((results / "index.json").read_text(encoding="utf-8"))
    document = json.loads((results / "recreate.ipynb").read_text(encoding="utf-8"))
    values = _rewrite(document, {})
    if not Path(values["LOCAL_RELEASE_ROOT"]).is_dir():
        pytest.skip("The local release this build read is not on this machine.")

    picked = [entry["uuid"] for entry in index["algorithms"]][:PICKED]
    panel = next(
        panel for panel in index["panels"] if panel["metric"] == values["METRIC"]
    )
    _rewrite(document, {"ALGORITHMS": picked, "BUDGET": panel["budget"]})
    document["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _REPORT,
        }
    )

    notebook = nbformat.reads(json.dumps(document), as_version=4)
    NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    report = _report(notebook.cells[-1].outputs)

    assert json.loads(report["CURVES"]) == sorted(picked)
    assert int(report["LINES"]) == PICKED


# Printed by a cell appended after the error ratio notebook's own.
_FIGURE_REPORT = (
    "import json\n"
    "print('FIGURE=' + type(figure).__module__ + '.' + type(figure).__name__)\n"
    "print('MEDIANS=' + json.dumps(sorted(t.name for t in figure.data if t.name and 'hovertemplate' in t and t.mode == 'lines')))\n"
    "print('POINTS=' + str(len(points)))\n"
    "print('KEPT=' + str(len(kept)) + '/' + str(len(properties)))\n"
)


def test_an_exported_error_ratio_view_draws_its_plotly_figure(
    built_site: Path, tmp_path: Path
) -> None:
    """The error ratio notebook runs against the release and draws one
    running median per exported algorithm on a Plotly figure."""
    results = built_site / TASK / "public-results"
    index = json.loads((results / "index.json").read_text(encoding="utf-8"))
    document = json.loads(
        (results / "recreate-feature.ipynb").read_text(encoding="utf-8")
    )
    values = _rewrite(document, {})
    if not Path(values["LOCAL_RELEASE_ROOT"]).is_dir():
        pytest.skip("The local release this build read is not on this machine.")

    picked = [entry["uuid"] for entry in index["algorithms"]][:PICKED]
    # A filter set: instances of at least the median sample size.
    sizes = sorted(entry["n"] for entry in index["instances"])
    _rewrite(
        document,
        {
            "ALGORITHMS": picked,
            "INSTANCE_FILTERS": {"n": [sizes[len(sizes) // 2], None]},
        },
    )
    document["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _FIGURE_REPORT,
        }
    )

    notebook = nbformat.reads(json.dumps(document), as_version=4)
    NotebookClient(
        notebook,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    ).execute()
    text = "".join(
        "".join(output.get("text", ""))
        for output in notebook.cells[-1].outputs
        if output["output_type"] == "stream"
    )
    report = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)

    assert report["FIGURE"] == "plotly.graph_objs._figure.Figure"
    assert len(json.loads(report["MEDIANS"])) == PICKED
    assert int(report["POINTS"]) > 0
    kept, total = map(int, report["KEPT"].split("/"))
    assert 0 < kept < total
