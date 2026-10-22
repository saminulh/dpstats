<div class="dp-task-page" hidden></div>

# Task Name: definition

## Metadata

```python
from dp_filler.tasks.task_name import TaskNameMetadata
```

What is public about the instance before any method runs, and
why a method may rely on it.

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col">Field</th><th scope="col">Type</th><th scope="col">Description</th></tr></thead>
<tbody>
<tr><td><code>sample_size</code></td><td><code>int</code></td><td>The field's full meaning: what it holds, in what order, and what a row is.</td></tr>
<tr><td><code>dimension</code></td><td><code>int</code></td><td>The field's full meaning: what it holds, in what order, and what a row is.</td></tr>
<tr><td><code>bounds</code></td><td><code>tuple[float, float]</code></td><td>The field's full meaning: what it holds, in what order, and what a row is.</td></tr>
<tr><td><code>share_public</code></td><td><code>float</code></td><td>The field's full meaning: what it holds, in what order, and what a row is.</td></tr>
</tbody>
</table>
</div>

## Input

```python
from dp_filler.tasks.task_name import TaskNameInput
```

What the analyst hands the method, in plain language, and what
one row of it stands for.

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col">Field</th><th scope="col">Type</th><th scope="col">Description</th></tr></thead>
<tbody>
<tr><td><code>sample</code></td><td><code>Array</code></td><td>The field's full meaning: what it holds, in what order, and what a row is.</td></tr>
<tr><td><code>response</code></td><td><code>Array</code></td><td>The field's full meaning: what it holds, in what order, and what a row is.</td></tr>
<tr><td><code>weights</code></td><td><code>Array</code> (optional)</td><td>The field's full meaning: what it holds, in what order, and what a row is.</td></tr>
<tr><td><code>seed</code></td><td><code>int</code></td><td>The field's full meaning: what it holds, in what order, and what a row is.</td></tr>
</tbody>
</table>
</div>

## Output

```python
from dp_filler.tasks.task_name import TaskNameOutput
```

What a private release returns, and what a reader may do with
it.

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col">Field</th><th scope="col">Type</th><th scope="col">Description</th></tr></thead>
<tbody>
<tr><td><code>estimate</code></td><td><code>Array</code></td><td>The field's full meaning: what it holds, in what order, and what a row is.</td></tr>
<tr><td><code>scale</code></td><td><code>float</code></td><td>The field's full meaning: what it holds, in what order, and what a row is.</td></tr>
<tr><td><code>interval</code></td><td><code>tuple[float, float]</code></td><td>The field's full meaning: what it holds, in what order, and what a row is.</td></tr>
</tbody>
</table>
</div>

## Notes on the definition

Material about the declared fields themselves, such as a structured
design domain, how a domain is supplied, or how a configurable loss is
declared.
