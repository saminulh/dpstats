<div class="dp-task-page" hidden></div>

# FillerAlgorithm

What the Algorithm does, in one line.
{ .dp-summary }

What the Algorithm does, in the terms of this task's estimand. It says
what the method is, not why a reader should choose it.

<section class="dp-glance" aria-label="⟨component.glance_label⟩" markdown="1">
<dl markdown="1">
<dt><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.glance_native_task">⟨component.glance_native_task⟩</span></dt>
<dd markdown="span">[Task Name](../t03-task-overview/)</dd>
<dt><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.glance_interpretation">⟨component.glance_interpretation⟩</span></dt>
<dd markdown="span"><span class="dp-badge dp-badge--theoretical badge" title="a theoretical version of the method satisfies the guarantee but the implementation does not.">theoretical</span> · [What this means](../../library/reference/algorithm/dpstats.algorithm.PrivacyInterpretation/)</dd>
<dt><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.glance_requires">⟨component.glance_requires⟩</span></dt>
<dd><ul>
<li>A finite bound the class reads from the metadata.</li>
<li>A budget family the class accepts.</li>
<li>A dimension or sample-size condition.</li>
</ul></dd>
<dt><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.glance_source">⟨component.glance_source⟩</span></dt>
<dd>Algorithm 2 from Surname et al. (2021), with Theorem 3.1 from Surname (2019)</dd>
</dl>
</section>

## Parameters

<dl class="dp-parameters">
<dt id="param-clip_norm"><span class="dp-parameter-name">clip_norm</span> <span class="dp-parameter-type">: float &gt; 0, <span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.parameter_default">⟨component.parameter_default⟩</span></span></dt>
<dd><p>The description declared on the parameter field.</p><p class="dp-parameter-note">Corresponds to a symbol in the cited work; why its default was chosen.</p></dd>
<dt id="param-step_count"><span class="dp-parameter-name">step_count</span> <span class="dp-parameter-type">: int ≥ 1, <span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.parameter_default">⟨component.parameter_default⟩</span></span></dt>
<dd><p>The description declared on the parameter field.</p></dd>
<dt id="param-split_share"><span class="dp-parameter-name">split_share</span> <span class="dp-parameter-type">: float in (0, 1), <span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.parameter_default">⟨component.parameter_default⟩</span></span></dt>
<dd><p>The description declared on the parameter field.</p><p class="dp-parameter-note">Corresponds to a symbol in the cited work; why its default was chosen.</p></dd>
<dt id="param-shrinkage"><span class="dp-parameter-name">shrinkage</span> <span class="dp-parameter-type">: float in [0, 1] or None, <span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.parameter_default">⟨component.parameter_default⟩</span></span></dt>
<dd><p>The description declared on the parameter field.</p></dd>
<dt id="param-tail_rule"><span class="dp-parameter-name">tail_rule</span> <span class="dp-parameter-type">: &#x27;trim&#x27; or &#x27;winsorize&#x27;, <span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.parameter_default">⟨component.parameter_default⟩</span></span></dt>
<dd><p>The description declared on the parameter field.</p></dd>
</dl>

<details class="dp-fold" markdown="1">
<summary><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.implementation_notes">⟨component.implementation_notes⟩</span></summary>

Anything a reader of the code needs and would not guess: a numerical
guard, an ordering that matters, a representation choice.

Deviations from the cited paper:

- One modification, stated as what the implementation does instead.
- A second modification, one bullet each.


</details>

## References

Surname, A. B., Othersurname, C., Thirdsurname, D., Fourthsurname, E., &amp; Fifthsurname, F. (2021). <span data-verbatim>Title of a paper with five authors</span>. Venue Name. <a href="https://arxiv.org/abs/0000.00001">https://arxiv.org/abs/0000.00001</a>
{ .dp-reference }

Surname, A. B. (2019). <span data-verbatim>Title of the cited paper in sentence case</span>. Venue Name. <a href="https://doi.org/10.0000/000000.0000000">https://doi.org/10.0000/000000.0000000</a>
{ .dp-reference }

## Example

```python
from dp_filler.tasks.task_name import FillerAlgorithm

algorithm = FillerAlgorithm(clip_norm=1.0, step_count=64)
output = algorithm.run(instance.input, instance.metadata, budget)
```

## API

```python
class FillerAlgorithm(
    clip_norm: float = 1.0,
    step_count: int = 64,
    split_share: float = 0.5,
    shrinkage: float | None = None,
    tail_rule: str = "trim",
)
```

### `run`

```python
run(input: TaskNameInput, metadata: TaskNameMetadata, budget: Budget) -> TaskNameOutput
```

The method's docstring first paragraph.

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col"><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.api_parameter_column">⟨component.api_parameter_column⟩</span></th><th scope="col"><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.api_type_column">⟨component.api_type_column⟩</span></th><th scope="col">Description</th></tr></thead>
<tbody>
<tr><td><code>input</code></td><td><code>TaskNameInput</code></td><td>The parameter description.</td></tr>
<tr><td><code>metadata</code></td><td><code>TaskNameMetadata</code></td><td>The parameter description.</td></tr>
<tr><td><code>budget</code></td><td><code>Budget</code></td><td>The parameter description.</td></tr>
</tbody>
</table>
</div>

### `applies_to`

```python
applies_to(metadata: TaskNameMetadata, budget: Budget) -> None
```

The method's docstring first paragraph, including what it raises.

<p class="dp-roster-link"><a href="../t22-roster-landing/?core=FillerAlgorithm"><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.roster_algorithm_many">⟨component.roster_algorithm_many⟩</span> →</a>; <span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.roster_also_in">⟨component.roster_also_in⟩</span> <a href="../t22-roster-landing/?core=FillerAlgorithm">Second Task Name (3)</a></p>
