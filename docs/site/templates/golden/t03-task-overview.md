<div class="dp-task-page" hidden></div>

# Task Name

The quantity this task estimates, as the Task declares it.

```python
from dp_filler.tasks import BUILT_IN_TASKS
BUILT_IN_TASKS["task_name"]
```

Leaderboard ·
Explore ·
[Definition](../t06-definition/) ·
[Metrics](../t18-task-metrics/) ·
[Native algorithms](../t07-task-algorithms/) ·
[Transformations](../t08-task-transformations/)

## Problem formulation

The estimand in words: what quantity a release is estimating, on what
kind of sample, and what the analyst already knows about it.

A second paragraph, where the estimand needs one: the sampling
assumption, the bounded quantities, or what the analyst is allowed to
know in public.

$$
\theta^{\star}(P) \;=\; \arg\min_{\theta \in \Theta}\;
\mathbb{E}_{Z \sim P}\big[\, \ell(\theta; Z) \,\big]
$$

What a private release must output: the object, its shape, and
anything the output carries alongside the estimate.

## Metrics at a glance

| <span class="dp-authored-copy dp-copy-slot" data-copy-slot="task.metric_column">⟨task.metric_column⟩</span> | Description |
| --- | --- |
| [Metric one](../t18-task-metrics/#metric-one) | What metric one measures. |
| [Metric two](../t18-task-metrics/#metric-two) | What metric two measures. |
| [Metric three](../t18-task-metrics/#metric-three) | What metric three measures. |
