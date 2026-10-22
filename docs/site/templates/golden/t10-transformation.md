<div class="dp-task-page" hidden></div>

# FillerTransformation

What the Transformation does, in one line.
{ .dp-summary }

What the Transformation transports, and what the inner task then solves. It
says what the reduction is, not why a reader should choose it.

<section class="dp-glance" aria-label="⟨component.glance_label⟩" markdown="1">
<dl markdown="1">
<dt><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.glance_task">⟨component.glance_task⟩</span></dt>
<dd markdown="span">[Task Name](../t03-task-overview/) → [Inner Task Name](../t03-task-overview/)</dd>
<dt><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.glance_interpretation">⟨component.glance_interpretation⟩</span></dt>
<dd markdown="span"><span class="dp-badge dp-badge--theoretical badge" title="a theoretical version of the method satisfies the guarantee but the implementation does not.">theoretical</span> · [What this means](../../library/reference/algorithm/dpstats.algorithm.PrivacyInterpretation/)</dd>
<dt><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.glance_requires">⟨component.glance_requires⟩</span></dt>
<dd><ul>
<li>A fact the outer task must declare for the transport to be defined; otherwise <code>NotApplicable</code>.</li>
<li>A budget family the Transformation accepts.</li>
</ul></dd>
<dt><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.glance_source">⟨component.glance_source⟩</span></dt>
<dd>Section 4 from Surname and Othersurname (2020)</dd>
</dl>
</section>

## Parameters

<dl class="dp-parameters">
<dt id="param-inner_share"><span class="dp-parameter-name">inner_share</span> <span class="dp-parameter-type">: float in (0, 1), <span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.parameter_default">⟨component.parameter_default⟩</span></span></dt>
<dd><p>The description declared on the field.</p><p class="dp-parameter-note">Corresponds to a symbol in the cited work; why its default was chosen.</p></dd>
<dt id="param-centre"><span class="dp-parameter-name">centre</span> <span class="dp-parameter-type">: bool, <span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.parameter_default">⟨component.parameter_default⟩</span></span></dt>
<dd><p>The description declared on the field.</p></dd>
<dt id="param-residual_rule"><span class="dp-parameter-name">residual_rule</span> <span class="dp-parameter-type">: str, <span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.parameter_default">⟨component.parameter_default⟩</span></span></dt>
<dd><p>The description declared on the field.</p></dd>
</dl>

<details class="dp-fold" markdown="1">
<summary><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.implementation_notes">⟨component.implementation_notes⟩</span></summary>

What the transformation does to the estimand, what it assumes, and the
case a reader is most likely to get wrong.

The authored method note moved here from the task guide, where a
Transformation's reduction was explained at length.

Deviations from the cited paper:

- One modification, one bullet.


</details>

## References

Surname, A. B., &amp; Othersurname, C. (2020). <span data-verbatim>Title of the second cited paper</span>. Venue Name. <a href="https://arxiv.org/abs/0000.00000">https://arxiv.org/abs/0000.00000</a>
{ .dp-reference }

## Example

```python
from dp_filler.tasks.task_name import FillerTransformation

transformation = FillerTransformation(inner_share=0.5)
output = transformation.run(instance.input, instance.metadata, budget, inner=inner_algorithm)
```

## API

```python
class FillerTransformation(
    inner_share: float = 0.5,
    centre: bool = True,
    residual_rule: str = "pooled",
)
```

### `run`

```python
run(input: TaskNameInput, metadata: TaskNameMetadata, budget: Budget, inner: Algorithm) -> TaskNameOutput
```

The method's docstring first paragraph.

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col"><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.api_parameter_column">⟨component.api_parameter_column⟩</span></th><th scope="col"><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.api_type_column">⟨component.api_type_column⟩</span></th><th scope="col">Description</th></tr></thead>
<tbody>
<tr><td><code>input</code></td><td><code>TaskNameInput</code></td><td>The parameter description.</td></tr>
<tr><td><code>metadata</code></td><td><code>TaskNameMetadata</code></td><td>The parameter description.</td></tr>
<tr><td><code>budget</code></td><td><code>Budget</code></td><td>The parameter description.</td></tr>
<tr><td><code>inner</code></td><td><code>Algorithm</code></td><td>The parameter description.</td></tr>
</tbody>
</table>
</div>

### `transported_metadata`

```python
transported_metadata(metadata: TaskNameMetadata) -> InnerTaskMetadata
```

The method's docstring first paragraph.

<p class="dp-roster-link"><a href="../t22-roster-landing/?transformation=FillerTransformation"><span class="dp-authored-copy dp-copy-slot" data-copy-slot="component.roster_transformation_many">⟨component.roster_transformation_many⟩</span> →</a></p>
