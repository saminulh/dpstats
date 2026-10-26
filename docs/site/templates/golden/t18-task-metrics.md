<div class="dp-task-page" hidden></div>

# Task Name: metrics

```python
from dp_filler import list_built_in_metrics
list_built_in_metrics(TaskName)
```

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col">Metric</th><th scope="col">Required metric arguments</th><th scope="col">Description</th></tr></thead>
<tbody>
<tr><td><a href="#metric-one">metric_one</a></td><td>—</td><td>What metric one measures, over what, and how a trial that produces no estimate is treated.</td></tr>
<tr><td><a href="#metric-two">metric_two</a></td><td><a href="#metric-argument-sample"><code>sample</code></a>, <a href="#metric-argument-response"><code>response</code></a></td><td>What metric two measures, over what, and how a trial that produces no estimate is treated.</td></tr>
<tr><td><a href="#metric-three">metric_three</a></td><td><a href="#metric-argument-bounds"><code>bounds</code></a></td><td>What metric three measures, over what, and how a trial that produces no estimate is treated.</td></tr>
<tr><td><a href="#metric-four">metric_four</a></td><td><a href="#metric-argument-sample"><code>sample</code></a>, <a href="#metric-argument-bounds"><code>bounds</code></a></td><td>What metric four measures, over what, and how a trial that produces no estimate is treated.</td></tr>
</tbody>
</table>
</div>

## metric_one {#metric-one}

$$
M_1 \;=\; \frac{1}{S}\sum_{s=1}^{S}
\lVert \hat{\theta}_s - \theta^{\star} \rVert
$$

<span class="dp-authored-copy dp-copy-slot" data-copy-slot="task.required_arguments">⟨task.required_arguments⟩</span> —

## metric_two {#metric-two}

$$
M_2 \;=\; \frac{1}{n}\sum_{i=1}^{n}
\ell\!\left( \hat{\theta}; Z_i \right)
$$

<span class="dp-authored-copy dp-copy-slot" data-copy-slot="task.required_arguments">⟨task.required_arguments⟩</span> <a href="#metric-argument-sample"><code>sample</code></a>, <a href="#metric-argument-response"><code>response</code></a>

## metric_three {#metric-three}

<span class="dp-authored-copy dp-copy-slot" data-copy-slot="task.required_arguments">⟨task.required_arguments⟩</span> <a href="#metric-argument-bounds"><code>bounds</code></a>

## metric_four {#metric-four}

$$
M_4 \;=\; \frac{1}{S}\sum_{s=1}^{S}
\min\!\left\{\, 1,\;
\frac{\lVert \hat{\theta}_s - \theta^{\star} \rVert}{\tau(\text{bounds})}
\,\right\}
$$

<span class="dp-authored-copy dp-copy-slot" data-copy-slot="task.required_arguments">⟨task.required_arguments⟩</span> <a href="#metric-argument-sample"><code>sample</code></a>, <a href="#metric-argument-bounds"><code>bounds</code></a>

## <span class="dp-authored-copy dp-copy-slot" data-copy-slot="task.metric_arguments_heading">⟨task.metric_arguments_heading⟩</span> {#metric-arguments}

<p class="dp-note dp-authored-copy dp-copy-slot" data-copy-slot="task.metric_arguments_intro">⟨task.metric_arguments_intro⟩</p>

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col"><span class="dp-authored-copy dp-copy-slot" data-copy-slot="task.metric_argument_column">⟨task.metric_argument_column⟩</span></th><th scope="col">Description</th><th scope="col"><span class="dp-authored-copy dp-copy-slot" data-copy-slot="task.used_by_column">⟨task.used_by_column⟩</span></th></tr></thead>
<tbody>
<tr><td><span id="metric-argument-bounds"><code>bounds</code></span></td><td>—</td><td><a href="#metric-three">metric_three</a>, <a href="#metric-four">metric_four</a></td></tr>
<tr><td><span id="metric-argument-response"><code>response</code></span></td><td>—</td><td><a href="#metric-two">metric_two</a></td></tr>
<tr><td><span id="metric-argument-sample"><code>sample</code></span></td><td>—</td><td><a href="#metric-two">metric_two</a>, <a href="#metric-four">metric_four</a></td></tr>
</tbody>
</table>
</div>

## Baselines

<table class="dp-table dp-table--wide table table-vcenter table-hover">
<thead><tr><th scope="col">Baseline</th><th scope="col">Declared by</th><th scope="col">Definition</th><th scope="col">Applies to</th><th scope="col">Role</th></tr></thead>
<tbody>
<tr><th scope="row" id="baseline-filler_scale"><code>filler_scale</code></th><td><span class="dp-badge badge">Benchmark protocol</span></td><td>The description the protocol states for a value the source supplies.</td><td><span class="dp-sentence"><a href="#metric-one">metric_one</a></span></td><td>Headline baseline</td></tr>
<tr><th scope="row" id="baseline-filler_reference"><code>filler_reference</code></th><td><span class="dp-badge badge">Library</span></td><td>The description the library's metric argument registry declares.</td><td><span class="dp-sentence"><a href="#metric-one">metric_one</a>, <a href="#metric-two">metric_two</a></span></td><td>Offered on Explore</td></tr>
<tr><th scope="row" id="baseline-best_median"><code>best_median</code></th><td><span class="dp-badge badge">Benchmark protocol</span></td><td>The best median cost among the standard-class algorithms, per instance and budget.</td><td><span class="dp-sentence"><a href="#metric-one">metric_one</a>, <a href="#metric-two">metric_two</a></span></td><td>Offered on Explore</td></tr>
</tbody>
</table>
