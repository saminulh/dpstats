# Submit a method

Enter a configured method into the benchmark.

## How to enter

1. [Wrap your algorithm](tutorials/run_your_own_script.ipynb) against an existing task
   contract, in your own code.
2. Package a method request — one configured algorithm, by UUID, as below.
3. Send the bundle to the benchmark operator.

Send questions and bug reports to the addresses on the [contact page](https://saminulh.github.io/dpstats/contact/).

Implementing or registering an algorithm does not include it in a benchmark.

## Package a method request

A method submission is a small, verifiable request for one configured algorithm.
It is not a task collection or collected Attempt records. The benchmark operator chooses
the evaluation corpus, budgets, seeds, and resource caps.

The candidate is one entry of a task-local algorithm registry — the same
installed group a sweep or a tuning run selects, described in
[Sweeps and Results](tutorials/run_your_own_script.ipynb). Install the configured algorithm once:

<!-- no-run: illustrative fragment, not a standalone script -->
```python
from pathlib import Path

from dpstats import open_collection
from dpstats.tasks import LinearRegression

collection = open_collection(Path("benchmark-storage"), LinearRegression)
algorithm_uuid = collection.add_algorithm(
    MyMechanism(label="my-method"), registry="mine"
)
print(algorithm_uuid)
```

Then package exactly one installed entry by UUID. Replace `<algorithm-uuid>`
below with the printed value, not its display label:

```bash
dpstats prepare-submission \
  --workspace benchmark-storage \
  --task linear_regression \
  --algorithm-registry mine \
  --algorithm <algorithm-uuid> \
  --out submission-bundle \
  --target-series v1-experimental \
  --source-url https://github.com/author/method \
  --source-commit <40-character-git-commit>
```

The task is the one the configured algorithm declares, and omitting
`--algorithm-registry` selects from every group installed for it. The selector
must match an installed UUID. That entry must also carry a display label;
the request records it as presentation metadata, never as identity.

A configuration chosen by your own search takes the same path. Build the
chosen configured algorithm with its label, then install and submit it:

<!-- no-run: illustrative fragment, not a standalone script -->
```python
algorithm = MyMechanism(**chosen_settings, label="my-method")
algorithm_uuid = collection.add_algorithm(algorithm, registry="mine")
print(algorithm_uuid)
```

The bundle contains `submission.json`, a one-record
`candidate-algorithm-registry.json`, and `manifest.json`. The candidate record
enters the bundle exactly as it is installed. The request records only the
target series, built-in task, selected label, fully constructed algorithm
configuration, candidate identity, source reference, and an optional
resource note. Verification derives the configured Algorithm object's declared
`privacy_interpretation` for its report: the reading of the release that the
author's privacy argument supports, which review confirms or reclassifies. The
registry does not duplicate that derived property. The bundle never copies an
evaluation configuration or names data, budgets, seeds, or caps.
Author-to-operator context belongs in the resource note.

Before accepting a bundle, an operator verifies it from a normal installation:

```bash
dpstats verify-submission submission-bundle
```

Verification allowlists the bundle schema, checks its SHA-256 manifest,
reconstructs the candidate from the installed source, and prints a
machine-readable report containing the normalized algorithm configuration and
source reference. Verification proves that those installed objects reproduce
the request; it does not prove that the installation came from the stated Git
commit. The operator first checks out and installs the reviewed source commit,
then runs verification. Each of the three entries must be a non-symlink regular
JSON file no larger than 1 MiB. These commands prepare and verify method
requests; they do not publish benchmark results. Citation and version details
for a Distribution Release live in that release's Hugging Face dataset card.

## Eligibility

The release's [benchmark protocol](https://saminulh.github.io/dpstats/methodology/) states its
[eligibility](reference/algorithm/dpstats.algorithm.PrivacyInterpretation.md)
and comparison rules. Preparing a method request does not admit it to the benchmark.

## What happens next

The operator evaluates the request and selects entries into a release. The
published benchmark protocol records the selections
and settings used for that release.

## Extend the library instead

Adding data to your own installation is library work, not a submission:
[add data-backed Instances](https://saminulh.github.io/dpstats/library/reference/instance/).
