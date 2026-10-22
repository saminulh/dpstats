<div class="dp-task-page" hidden></div>

# Task Name: methodology

<p><a href="../t15-methodology/">Filler protocol</a></p>

<p class="dp-authored-copy dp-copy-slot" data-copy-slot="protocol.ranking_rule">⟨protocol.ranking_rule⟩</p>

## Standings

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col">Rule</th><th scope="col">Value</th></tr></thead>
<tbody>
<tr><td>Weighting</td><td><code>provenance_ref</code></td></tr>
<tr><td>Eligibility</td><td><code>standard</code></td></tr>
</tbody>
</table>
</div>

## Baselines

<table class="dp-table dp-table--wide table table-vcenter table-hover">
<thead><tr><th scope="col">Baseline</th><th scope="col">Declared by</th><th scope="col">Definition</th><th scope="col">Applies to</th><th scope="col">Role</th></tr></thead>
<tbody>
<tr><th scope="row" id="baseline-filler_scale"><code>filler_scale</code></th><td><span class="dp-badge badge">Benchmark protocol</span></td><td>The description the protocol states for a value the source supplies.</td><td><span class="dp-sentence"><a href="../t18-task-metrics/#metric-one">metric_one</a></span></td><td>Headline baseline</td></tr>
<tr><th scope="row" id="baseline-filler_reference"><code>filler_reference</code></th><td><span class="dp-badge badge">Library</span></td><td>The description the library's metric argument registry declares.</td><td><span class="dp-sentence"><a href="../t18-task-metrics/#metric-one">metric_one</a>, <a href="../t18-task-metrics/#metric-two">metric_two</a></span></td><td>Offered on Explore</td></tr>
<tr><th scope="row" id="baseline-best_median"><code>best_median</code></th><td><span class="dp-badge badge">Benchmark protocol</span></td><td>The best median cost among the standard-class algorithms, per instance and budget.</td><td><span class="dp-sentence"><a href="../t18-task-metrics/#metric-one">metric_one</a>, <a href="../t18-task-metrics/#metric-two">metric_two</a></span></td><td>Offered on Explore</td></tr>
</tbody>
</table>

## Threshold grid

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col">Minimum</th><th scope="col">Maximum</th><th scope="col">Scale</th><th scope="col">Steps</th></tr></thead>
<tbody>
<tr><td class="dp-value">0</td><td class="dp-value">10</td><td>linear</td><td class="dp-value">1001</td></tr>
</tbody>
</table>
</div>

<p class="dp-build-source dp-authored-copy dp-copy-slot" data-copy-slot="shared.built_from">⟨shared.built_from⟩</p>
