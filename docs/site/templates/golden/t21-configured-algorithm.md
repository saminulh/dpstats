<div class="dp-task-page" hidden></div>

# Filler Method A

<a href="../t09-algorithm/"><span class="dp-badge dp-badge--standard badge" title="the standard add/remove differential privacy guarantee holds on all inputs and budgets.">standard</span></a>
<span class="dp-roster-swatch" data-dp-roster-marker='{&quot;core&quot;: &quot;FillerAlgorithm&quot;, &quot;class&quot;: &quot;standard&quot;, &quot;marker&quot;: &quot;circle&quot;}'></span>
<script>(function(){var el=document.querySelector('.dp-roster-swatch');function paint(){if(!el)return;if(!window.dpSeriesInk||!window.dpSeriesSwatch)return;var m=JSON.parse(el.dataset.dpRosterMarker);var ink=window.dpSeriesInk(m.core,m['class']);var emphasis=window.dpSeriesEmphasis();var swatch=window.dpSeriesSwatch({colour:ink.colour,width:emphasis.regularWidth,opacity:emphasis.regularOpacity,dash:window.dpSeriesDash(m['class']),marker:m.marker});swatch.classList.add("dp-roster-swatch-mark");el.replaceWith(swatch);}if(document.readyState==='loading'){document.addEventListener('DOMContentLoaded',paint);}else{paint();}})();</script>

[Task Name](../t03-task-overview/)
{ .dp-note }

What the release's own declaration says this configuration does.

## [Leaderboard](../t04-leaderboard/) ranks

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col">Budget</th><th scope="col">Rank</th><th scope="col">Metric</th><th scope="col"><span class="dp-authored-copy dp-copy-slot" data-copy-slot="leaderboard.column_headline">⟨leaderboard.column_headline⟩</span></th><th scope="col">Coverage</th></tr></thead>
<tbody>
<tr><td>ε = 1.0</td><td>Rank 1</td><td><a href="../t04-leaderboard/#profile-mean-estimation-l2-error-to-ground-truth">Metric one</a></td><td>1.42</td><td>90.0%</td></tr>
<tr><td>ε = 1.0</td><td>not ranked</td><td><a href="../t04-leaderboard/#profile-mean-estimation-l2-error-to-sample-mean">Metric two</a></td><td>—</td><td>—</td></tr>
</tbody>
</table>
</div>

## Results breakdown on the public corpus · [Explore](../t05-explore/)

<div class="dp-table-scroll table-responsive">
<table class="dp-table dp-table--wide table table-vcenter table-hover">
<thead><tr><th scope="col">Budget</th><th scope="col">Rank</th><th scope="col">Released</th><th scope="col">No release</th><th scope="col">Invalid regime</th><th scope="col">Algorithm error</th><th scope="col">Timeout</th><th scope="col">Memory limit</th></tr></thead>
<tbody>
<tr><td>ε = 1.0</td><td>Rank 1</td><td>82 (91.1%)</td><td>0</td><td>8 (8.9%)</td><td>0</td><td>0</td><td>0</td></tr>
</tbody>
</table>
</div>

## Recipe

### [FillerTransformation](../t10-transformation/)

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col">Name</th><th scope="col">Value</th><th scope="col">Description</th></tr></thead>
<tbody>
<tr><td><code>axis</code></td><td>'row'</td><td>Which axis the transform reduces.</td></tr>
</tbody>
</table>
</div>


### [FillerAlgorithm](../t09-algorithm/)

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col">Name</th><th scope="col">Value</th><th scope="col">Description</th></tr></thead>
<tbody>
<tr><td><code>probability</code></td><td>0.5</td><td>The parameter the search set opens at.</td></tr>
</tbody>
</table>
</div>


#### FillerLoss

<div class="dp-table-scroll table-responsive">
<table class="dp-table table table-vcenter table-hover">
<thead><tr><th scope="col">Name</th><th scope="col">Value</th><th scope="col">Description</th></tr></thead>
<tbody>
<tr><td><code>k</code></td><td>1.0</td><td>The loss family's own parameter.</td></tr>
</tbody>
</table>
</div>


<details><summary><span class="dp-authored-copy dp-copy-slot" data-copy-slot="roster.exact_configuration">⟨roster.exact_configuration⟩</span></summary>

```json
{
  "path": "dp_filler.algorithms.FillerAlgorithm"
}
```

</details>

## Source

### A Filler Recipe Source

Recipe source: Ada Filler, [A Filler Paper](https://doi.org/10.0000/filler), Algorithm 1.
