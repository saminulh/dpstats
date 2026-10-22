# dpstats

dpstats is a Python library for implementing DP algorithms for common statistical tasks
and interfacing with the dpstatsbench benchmark.

## Quick Start

### Installation

Install the library from the tagged public repository into a Python 3.13 environment:

```bash
pip install "dpstats @ git+https://github.com/saminulh/dpstats.git@v0.1.0"
```

[Install and fetch the benchmark](https://saminulh.github.io/dpstats/library/tutorials/install_and_fetch/)
walks through the environment, opening the notebook in JupyterLab or VS Code,
and fetching the public corpus; no account or token is needed.

## Tutorials

The tutorials are Jupyter notebooks in [`tutorials/`](tutorials/). Each link
below opens the rendered page on the documentation site, with every output;
the notebook beside it is the file to download and run.

Benchmarking an algorithm:

- [Install and fetch the benchmark](https://saminulh.github.io/dpstats/library/tutorials/install_and_fetch/) ([notebook](tutorials/install_and_fetch.ipynb)): install the library, fetch the benchmark, and run one algorithm on one problem instance.
- [Run your own script](https://saminulh.github.io/dpstats/library/tutorials/run_your_own_script/) ([notebook](tutorials/run_your_own_script.ipynb)): wrap a script you already have for a DP algorithm so the library can run it.
- [Compare on the benchmark](https://saminulh.github.io/dpstats/library/tutorials/compare_on_the_benchmark/) ([notebook](tutorials/compare_on_the_benchmark.ipynb)): run that script on single-coefficient estimation instances and compare its errors with benchmarked algorithms.

Using the library:

- [Introduction](https://saminulh.github.io/dpstats/library/tutorials/tasks/) ([notebook](tutorials/tasks.ipynb)): what a task defines: its input, metadata and output types.
- [Algorithms](https://saminulh.github.io/dpstats/library/tutorials/algorithms/) ([notebook](tutorials/algorithms.ipynb)): implement and run an algorithm for a task.
- [Domains](https://saminulh.github.io/dpstats/library/tutorials/domains/) ([notebook](tutorials/domains.ipynb)): state the public bounds available to an algorithm.
- [Privacy budgets and accounting](https://saminulh.github.io/dpstats/library/tutorials/privacy_budgets_and_accounting/) ([notebook](tutorials/privacy_budgets_and_accounting.ipynb)): the typed budget families and how a mechanism spends them.
- [Metrics and baselines](https://saminulh.github.io/dpstats/library/tutorials/metrics_and_baselines/) ([notebook](tutorials/metrics_and_baselines.ipynb)): measure an algorithm's error against statistical baselines.
- [Transformations and composition](https://saminulh.github.io/dpstats/library/tutorials/extend_with_transformations/) ([notebook](tutorials/extend_with_transformations.ipynb)): prepare a problem for an algorithm and convert its output back.

Analyses:

- [How much does choosing a configuration help?](https://saminulh.github.io/dpstats/library/tutorials/tuning_headroom/) ([notebook](tutorials/tuning_headroom.ipynb)): the gain from choosing among private configurations instead of one default.
- [Which instances are hard for SSP?](https://saminulh.github.io/dpstats/library/tutorials/ssp_hard_instances/) ([notebook](tutorials/ssp_hard_instances.ipynb)): relate one configuration's cost to public properties of each instance.

## Command line

```bash
dpstats --help
dpstats <command> --help
```

The generated help is the current command index, and the documentation site
renders those same parsers as its CLI reference.

### Local sweeps and benchmark collection

Install configured algorithms and direct Instances, author a portable one-task
Sweep recipe, preview its gaps, then fill the missing cells:

```bash
dpstats get-sweep-status \
  --sweep <workspace>/local/mean_estimation/sweeps/local.json \
  --workspace <workspace>
dpstats run-sweep \
  --sweep <workspace>/local/mean_estimation/sweeps/local.json \
  --workspace <workspace>
dpstats measure-sweep \
  --sweep <workspace>/local/mean_estimation/sweeps/local.json \
  --workspace <workspace> \
  --metric l2_error_to_ground_truth
```

Official collection executes exact cells from the current transient job
description. Distribution Release fetch is explicit:

```bash
dpstats fetch --workspace <workspace> [--revision <tag-or-commit>]
```

Release projection and publication remain operator workflows.

### Explore Tasks and Algorithms

Built-in discovery answers which Tasks and Algorithms are available; use
`dpstats.tasks.list_built_in_algorithms`/`BUILT_IN_TASKS` in Python to
find the `path` values a roster names.

## Documentation

The [documentation site](https://saminulh.github.io/dpstats/) carries the
tutorials above, the API reference, and the benchmark's tasks and results.
In this repository:

- [DP Tuning](docs/reference/dp_tuning.md): privacy-accounted configuration selection mechanisms.
- [Submission](docs/submission.md): preparing a benchmark submission.

### Build Documentation Site

The site renders each tutorial from the copy the tutorial harness executes, so
run the harness first, and again after editing a notebook:

```bash
uv venv --seed && uv sync --all-extras
pytest tests/integration/test_tutorials.py -m slow -n 0
python docs/site/live_preview.py --build
```

The initial strict build refreshes generated pages from the checkout containing
the preview command, even when it is a Git worktree. The preview then serves
that snapshot and syncs CSS and JavaScript directly from source, reloading the
browser in about a second without regenerating result pages. Rebuild only after
changing templates, Python renderers, documentation, or result inputs by
restarting with `--build`; omit it when reopening an existing snapshot.

To render results, point `DOCS_LOCAL_RELEASE_ROOT` at one Distribution
Release tree:

```bash
DOCS_LOCAL_RELEASE_ROOT=/path/to/distribution-release \
    mkdocs build --strict -f docs/site/mkdocs.yml
```

The release declares the tasks it released, ships the views each page renders,
and states the headline Metric, the threshold, the grid and the Metrics it
carries in its own Benchmark protocol; the site chooses none of them. The first
build measures that release's distribution lane once into a cached snapshot
beside `docs/site`, and later builds render the snapshot without reading
Results again. A local tree is labelled a preview and never presented as a
published release. The public build leaves this variable unset and renders the
release `docs/benchmark.lock` pins.

## Contributing

Development environments are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv venv --seed && uv sync --all-extras
```

That creates the checkout's own `.venv` with the docs toolchain plus the test
and typecheck tooling. Check a change with `pytest tests/`. Before committing
a notebook, strip its outputs and editor metadata:

```bash
nbstripout --keep-id --extra-keys "metadata.kernelspec metadata.language_info" tutorials/*.ipynb
```

## Requirements

- Python 3.10+

The install (`pip install -e .`) covers the whole library: every built-in
algorithm, fetching published releases, and the analysis plots. One extra
exists, for building the documentation site:

```bash
pip install -e ".[docs]"      # mkdocs toolchain — build the docs site

uv sync --extra dev           # [docs] + test/typecheck tooling (development gate environment)
```

## License

MIT
