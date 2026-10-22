/* Read a task's panel datasets and derive the views a control chooses.

   The build emits data, not views: one index naming the algorithms, the
   instances, the panels and the baselines, and per panel one binary matrix of
   every trial's cost — a gzip-compressed little-endian Float64Array in row-major
   [algorithm][instance][seed] order, with NaN where a cell has fewer runs than
   the panel's seed count and throughout a cell no Attempt answered. Infinities
   are stored natively, because a failed run is a real cost no threshold covers.

   Every view a reader can reach is a derivation of that matrix, so the
   derivations live here — once, over typed arrays — and each one is a stated
   mirror of the library function that owns the same reading. Nothing here
   measures a store or decides a baseline: the vectors a cost is read against
   travel in the index. */
(function () {
  "use strict";

  var documents = new Map();

  /* One fetch per file per page, however many views read it. */
  function fetchOnce(url, read) {
    if (!documents.has(url)) {
      documents.set(
        url,
        fetch(url).then(function (response) {
          if (!response.ok) {
            throw new Error(response.status + " " + response.statusText);
          }
          return read(response);
        }).catch(function (error) {
          documents.delete(url);
          throw error;
        })
      );
    }
    return documents.get(url);
  }

  function load(url) {
    return fetchOnce(url, function (response) { return response.json(); });
  }

  /* An index names its panel files relative to itself, so a page embedding a
     Leaderboard from elsewhere (the landing page) resolves them against the
     index's own location, not the page's. */
  function loadIndex(url) {
    var location = new URL(url, document.baseURI);
    return load(location.href).then(function (index) {
      index.location = location.href;
      return index;
    });
  }

  function resolve(index, path) {
    return index.location ? new URL(path, index.location).href : path;
  }

  function loadMatrix(url) {
    return fetchOnce(url, function (response) {
      var stream = response.body.pipeThrough(new DecompressionStream("gzip"));
      return new Response(stream).arrayBuffer().then(function (buffer) { return new Float64Array(buffer); });
    });
  }

  /* One panel, as every view of it reads it: a release's published curves, or
     the run matrix a measured store's readings are all derived from. */
  function panelOf(index, entry) {
    if (entry.curves) {
      return load(resolve(index, entry.curves)).then(function (published) {
        return {
          metric: entry.metric,
          budget: entry.budget,
          published: published.curves,
          taus: published.taus,
        };
      });
    }
    return loadMatrix(resolve(index, entry.runs)).then(function (values) {
      return {
        metric: entry.metric,
        budget: entry.budget,
        values: values,
        seeds: entry.seeds,
        seedIds: entry.seed_ids,
        instances: index.instances.length,
        // JSON encodes non-finite baselines as numeric strings.
        baselines: Object.fromEntries(Object.entries(entry.baselines || {}).map(function (pair) {
          return [pair[0], pair[1].map(Number)];
        })),
        epsilon: entry.epsilon,
      };
    });
  }

  /* Mirror of dpstats.analysis.compute_error_ratio: division on
     extended-real costs is total. An exact zero cost against an exact zero
     denominator matches it; anything else against zero is unboundedly worse; an
     infinite cost stays infinite whatever it is divided by; a finite cost
     against an infinite denominator is zero, within every threshold, since it
     answered where the baseline failed; and a quotient no order can rank takes
     the worst score. */
  function ratio(value, denominator) {
    if (denominator === 0) return value === 0 ? 1 : Infinity;
    var quotient = value / denominator;
    if (value === Infinity || Number.isNaN(quotient)) return Infinity;
    return quotient;
  }

  /* Reading in the Metric's own units divides every cost by one, so this
     baseline ships as no vector at all. */
  var NO_BASELINE = "none";

  /* The baseline each instance's costs are read against. */
  function denominators(panel, baseline) {
    if (baseline === NO_BASELINE) return null;
    var vectors = panel.baselines || {};
    if (!Object.prototype.hasOwnProperty.call(vectors, baseline)) {
      throw new Error("Baseline " + baseline + " is unavailable for this panel.");
    }
    return vectors[baseline];
  }

  function denominator(vector, column) {
    return vector === null ? 1 : vector[column];
  }

  /* Mirror of dpstats.analysis.aggregate_trials: one order
     statistic over the trials of a trial group, on the extended reals, so an
     infinite cost keeps its place in the ordering and a trial group whose trials
     failed at least half the time summarizes as infinite rather than as a
     summary of the trials that survived. */
  function median(costs) {
    if (!costs.length) return null;
    var sorted = costs.slice().sort(function (left, right) { return left - right; });
    var half = sorted.length >> 1;
    if (sorted.length % 2) return sorted[half];
    var low = sorted[half - 1];
    var high = sorted[half];
    if (low === high) return low;
    // The midpoint of a finite and an infinite cost is that infinity.
    if (!Number.isFinite(low) || !Number.isFinite(high)) {
      return low === -Infinity && high === Infinity ? Infinity : (Number.isFinite(low) ? high : low);
    }
    return (low + high) / 2;
  }

  /* The one raw read of a matrix slot: a non-success run, or a declared cell
     nothing answered, is stored as a negative status code (index.status_codes
     names each one) in place of the extended-real cost it would otherwise
     carry, and reads back here as the +Infinity a failed run always was. */
  function rawCost(panel, index) {
    var value = panel.values[index];
    return value < 0 ? Infinity : value;
  }

  /* One cell's costs: every seed's own, or their median. A run the matrix pads
     with NaN is a seed this cell has no Attempt for, and a cell no Attempt
     answered at all carries none. */
  function cellCosts(panel, row, column, aggregation) {
    var start = (row * panel.instances + column) * panel.seeds;
    var present = [];
    for (var seat = 0; seat < panel.seeds; seat += 1) {
      if (!Number.isNaN(panel.values[start + seat])) present.push(rawCost(panel, start + seat));
    }
    if (aggregation === "individual") return present;
    var middle = median(present);
    return middle === null ? [] : [middle];
  }

  /* Which seed each of a cell's costs came from, for a reader inspecting one. */
  function cellSeats(panel, row, column) {
    var start = (row * panel.instances + column) * panel.seeds;
    var seats = [];
    for (var seat = 0; seat < panel.seeds; seat += 1) {
      if (!Number.isNaN(panel.values[start + seat])) seats.push(seat);
    }
    return seats;
  }

  /* One run's own status: null for a recorded success (whatever nonnegative
     or infinite cost it carries), or the negative code panel.values stores
     instead, keyed into index.status_codes for a label. */
  function cellStatus(panel, row, column, seat) {
    var value = panel.values[(row * panel.instances + column) * panel.seeds + seat];
    return value < 0 ? value : null;
  }

  /* The named outcomes summarize_outcomes reports, keyed by the negative
     status code panel.values stores in place of a cost (index.status_codes
     names the same codes for a reader; -6/not_run is declared but never
     recorded, so it carries no counts key of its own). */
  var OUTCOME_NAMES = {
    "-1": "no_release",
    "-2": "invalid_regime",
    "-3": "timeout",
    "-4": "memory_limit_exceeded",
    "-5": "algorithm_error",
  };
  var NOT_RUN_CODE = -6;

  /* Mirror of dpstats.analysis.summarize_outcomes: one
     algorithm row's outcome counts over the given instance columns, reading
     every seed slot rather than a cell's aggregated cost. A slot the matrix
     pads with NaN is not a declared run at all; a declared cell no Attempt
     answered (-6/not_run) counts toward declared but never recorded, since a
     machinery incident is not an outcome. Shares are read from the returned
     counts over `recorded`, unweighted by instance. */
  function outcomes(panel, row, columns) {
    var counts = {
      success: 0,
      no_release: 0,
      invalid_regime: 0,
      algorithm_error: 0,
      timeout: 0,
      memory_limit_exceeded: 0,
    };
    var recorded = 0;
    var declared = 0;
    for (var pick = 0; pick < columns.length; pick += 1) {
      var start = (row * panel.instances + columns[pick]) * panel.seeds;
      for (var seat = 0; seat < panel.seeds; seat += 1) {
        var value = panel.values[start + seat];
        if (Number.isNaN(value)) continue;
        declared += 1;
        if (value === NOT_RUN_CODE) continue;
        recorded += 1;
        counts[value < 0 ? OUTCOME_NAMES[String(value)] : "success"] += 1;
      }
    }
    return { counts: counts, recorded: recorded, declared: declared };
  }

  /* Mirror of dpstats.analysis.compute_baseline: one algorithm's
     median cost on each instance is what the others are read against there.
     An instance it never answered carries no denominator (null). */
  function algorithmMedians(panel, row) {
    var vector = new Array(panel.instances);
    for (var column = 0; column < panel.instances; column += 1) {
      var costs = cellCosts(panel, row, column, "individual");
      vector[column] = costs.length ? medianOf(costs) : null;
    }
    return vector;
  }

  function medianOf(values) {
    var sorted = values.slice().sort(function (left, right) { return left - right; });
    var half = sorted.length >> 1;
    return sorted.length % 2 ? sorted[half] : (sorted[half - 1] + sorted[half]) / 2;
  }

  var ALGORITHM_BASELINE = "algorithm:";

  /* The page's baseline: a named vector the index ships, or an algorithm the
     reader chose, whose medians are derived here from the same matrix. */
  function resolveBaseline(panel, baseline, index) {
    if (baseline.indexOf(ALGORITHM_BASELINE) !== 0) return denominators(panel, baseline);
    var uuid = baseline.slice(ALGORITHM_BASELINE.length);
    var row = index.algorithms.findIndex(function (entry) { return entry.uuid === uuid; });
    return row < 0 ? null : algorithmMedians(panel, row);
  }

  function baselineAlgorithm(baseline) {
    return baseline.indexOf(ALGORITHM_BASELINE) === 0 ? baseline.slice(ALGORITHM_BASELINE.length) : null;
  }

  function firstCovering(taus, value) {
    var last = taus.length - 1;
    if (!(value <= taus[last])) return taus.length;
    var low = 0;
    var high = last;
    while (low < high) {
      var middle = (low + high) >> 1;
      if (taus[middle] >= value) high = middle;
      else low = middle + 1;
    }
    return low;
  }

  /* Mirror of dpstats.analysis.compute_profile_curves: at threshold tau,
     this share of the group's runs cost no more than tau times their baseline.
     The share is taken inside each instance first and then averaged across the
     instances that carry a cost, so an instance with many runs does not outvote
     the rest, and a run that failed carries an infinite error ratio that no
     finite threshold covers. Each instance carries the share the protocol's
     weighting gives it, and that share is split evenly across its own runs. */
  function curve(panel, row, columns, vector, aggregation, taus, shares) {
    var starts = new Float64Array(taus.length + 1);
    var total = 0;
    for (var pick = 0; pick < columns.length; pick += 1) {
      var column = columns[pick];
      var costs = cellCosts(panel, row, column, aggregation);
      if (!costs.length) continue;
      // An instance the baseline never answered has no denominator there.
      var against = denominator(vector, column);
      if (against === null) continue;
      var share = shares && shares.length ? shares[column] : 1;
      total += share;
      var weight = share / costs.length;
      for (var index = 0; index < costs.length; index += 1) {
        var value = ratio(costs[index], against);
        starts[firstCovering(taus, value)] += weight;
      }
    }
    var fractions = new Float64Array(taus.length);
    var running = 0;
    for (var stop = 0; stop < taus.length; stop += 1) {
      running += starts[stop];
      fractions[stop] = total === 0 ? 0 : running / total;
    }
    return { fractions: fractions, instances: total };
  }

  /* One algorithm's success curve at every error ratio it reaches, exactly:
     its distinct finite ratios ascending (x), each with its weighted share of
     trials at or below it (y), under curve()'s own weighting. The standings
     read it where the page holds every trial. */
  function exactCurve(panel, row, columns, vector, aggregation, shares) {
    var values = [];
    var weights = [];
    var total = 0;
    for (var pick = 0; pick < columns.length; pick += 1) {
      var column = columns[pick];
      var costs = cellCosts(panel, row, column, aggregation);
      if (!costs.length) continue;
      var against = denominator(vector, column);
      if (against === null) continue;
      var share = shares && shares.length ? shares[column] : 1;
      total += share;
      for (var index = 0; index < costs.length; index += 1) {
        var value = ratio(costs[index], against);
        if (!isFinite(value)) continue;
        values.push(value);
        weights.push(share / costs.length);
      }
    }
    var order = values.map(function (_, index) { return index; });
    order.sort(function (a, b) { return values[a] - values[b]; });
    var x = [];
    var y = [];
    var running = 0;
    order.forEach(function (index) {
      running += weights[index];
      var fraction = total === 0 ? 0 : running / total;
      if (x.length && x[x.length - 1] === values[index]) y[y.length - 1] = fraction;
      else { x.push(values[index]); y.push(fraction); }
    });
    return { x: x, y: y };
  }

  /* The one ranking rule, mirroring panel_data.headline_order. A curve's
     reading at the quantile p is the first error ratio at which its share
     reaches p (Infinity where it never does), then the share it reaches at
     all and the first ratio at which it reaches that, which break ties among
     the infinite ones. A share summed in floating point may fall a rounding
     short of the quantile it reaches. */
  var ROUNDING = 1e-12;
  function firstReaching(curve, share) {
    for (var index = 0; index < curve.x.length; index += 1) {
      if (curve.y[index] >= share - ROUNDING) return curve.x[index];
    }
    return Infinity;
  }
  function reachedBy(curve) {
    var share = curve.y.length ? Math.max.apply(null, curve.y) : 0;
    return [share, share > 0 ? firstReaching(curve, share) : Infinity];
  }
  /* reached: [share, tau at share], from reachedBy or a shipped reading. */
  function readingAt(curve, p, reached) {
    return [firstReaching(curve, p), reached[0], reached[1]];
  }
  /* A shipped reading as numbers: an infinite τ travels as "Infinity". */
  function shippedReading(entry) {
    return entry.map(Number);
  }
  /* A reading's headline cell: its error ratio, or, where it never reaches
     the quantile, the slot naming the share it does reach. */
  window.dpHeadlineText = function (reading) {
    if (isFinite(reading[0])) return window.dpNumber(reading[0]);
    return window.dpSiteCopy("leaderboard.unreached", {
      percent: String(Math.round(reading[1] * 100)),
    });
  };

  function compareReadings(a, b) {
    var left = [a[0], -a[1], a[2]];
    var right = [b[0], -b[1], b[2]];
    for (var index = 0; index < 3; index += 1) {
      if (left[index] < right[index]) return -1;
      if (left[index] > right[index]) return 1;
    }
    return 0;
  }

  /* Each baseline-relative cost (the ratio mirror above) beside one public
     feature of the instance or the budget it ran under. */
  function featurePoints(panel, row, columns, vector, aggregation, feature) {
    var points = [];
    for (var pick = 0; pick < columns.length; pick += 1) {
      var column = columns[pick];
      var value = feature(column);
      if (value === null || value === undefined) continue;
      var against = denominator(vector, column);
      if (against === null) continue;
      var costs = cellCosts(panel, row, column, aggregation);
      var seats = aggregation === "individual" ? cellSeats(panel, row, column) : null;
      for (var seat = 0; seat < costs.length; seat += 1) {
        points.push({
          instance: column,
          seed: seats === null ? null : seats[seat],
          feature: value,
          value: ratio(costs[seat], against),
        });
      }
    }
    return points;
  }

  /* Keep both ends of every flat segment and every exact value change. The
     remaining points describe the same piecewise-linear chart. */
  function curvePoints(taus, fractions) {
    if (!taus.length) return { x: [], y: [] };
    var kept = [0];
    for (var index = 1; index < fractions.length; index += 1) {
      if (fractions[index] === fractions[index - 1]) continue;
      if (kept[kept.length - 1] !== index - 1) kept.push(index - 1);
      kept.push(index);
    }
    if (kept[kept.length - 1] !== fractions.length - 1) {
      kept.push(fractions.length - 1);
    }
    return {
      x: kept.map(function (index) { return taus[index]; }),
      y: kept.map(function (index) { return fractions[index]; }),
    };
  }

  /* Chart chrome sits on a transparent plot so the reader's colour scheme shows
     through, so it takes one mid-tone that reads on both surfaces. */
  var INK = "#898781";
  var GRID = "rgba(137,135,129,0.22)";
  /* What a curve counts, by the trial aggregation it is read under: every
     trial, or one median-over-trials cost per instance. The question and formula
     are site-copy slots whose fields are READING_FIELDS: {term} a definable
     reader term shown as the site-copy slot {copy} or the symbol {word},
     {slot} a place the page fills with the metric or baseline name. */
  var METRIC = { slot: "metric" };
  var BASELINE = { slot: "baseline" };
  var TAU = { term: "tau", word: "τ" };
  var READING_FIELDS = {
    success_rate: { term: "success_rate", copy: "shared.term_success_rate" },
    trial: { term: "trial", copy: "shared.term_trial" },
    trials: { term: "trial", copy: "shared.term_trials" },
    instances: { term: "instance", copy: "shared.term_instances" },
    metric: METRIC,
    tau: TAU,
    baseline: BASELINE,
  };
  var COUNTED = {
    individual: {
      question: "shared.profile_question_individual",
      formula: "shared.profile_formula_individual",
    },
    median: {
      question: "shared.profile_question_median",
      formula: "shared.profile_formula_median",
    },
  };

  function counted(aggregation) {
    return COUNTED[aggregation] || COUNTED.individual;
  }

  /* The plotting payload for one budget's chart. The page applies the
     series rule — colour by core, dash by privacy interpretation, marker by
     configuration, weight by the current reading. Every algorithm is a trace;
     the leading ranked ones open drawn and the rest open in the legend. The
     drawing page writes each curve's hover text. */
  function figure(spec) {
    var data = [];
    var drawn = 0;
    spec.series.forEach(function (series) {
      /* Which classes are present is the page's class filter's decision (a
         hidden series is one it dropped); among the present ones the leading
         ranked open drawn. */
      var opens = false;
      if (!series.hidden) {
        opens = drawn < spec.leading;
        drawn += 1;
      }
      var points = curvePoints(spec.taus, series.fractions);
      /* The curve passes through its exact headline point, so the quantile
         guide at the protocol's quantile meets it where the standings read. */
      // A point past the grid's top is not joined: no curve is drawn there.
      var exact = series.headline ? Number(series.headline[0]) : Infinity;
      var top = spec.taus.length ? Number(spec.taus[spec.taus.length - 1]) : -Infinity;
      if (isFinite(exact) && exact <= top && points.x.indexOf(exact) < 0) {
        var at = 0;
        while (at < points.x.length && points.x[at] < exact) at += 1;
        points.x.splice(at, 0, exact);
        points.y.splice(at, 0, spec.quantile);
      }
      data.push({
        type: "scatter",
        mode: "lines+markers",
        name: series.label,
        legendgroup: series.key,
        showlegend: true,
        visible: series.hidden ? false : opens ? true : "legendonly",
        /* Colour, dash, marker and weight are the page's one series rule,
           applied at draw time against the current reading; the payload
           carries only what that rule is keyed on. */
        meta: {
          slot: series.slot,
          diagnostic: series.diagnostic,
          core: series.core,
          interpretation: series.interpretation,
          marker: series.marker,
          reached: series.reached,
        },
        line: { shape: "linear" },
        x: points.x,
        y: points.y,
      });
    });
    return {
      aggregation: spec.aggregation || "individual",
      units: spec.units,
      tauScale: spec.tauScale || "linear",
      data: data,
      layout: {
        title: { text: spec.title, font: { size: 13, color: INK } },
        xaxis: {
          type: "log",
          title: {
            text: window.dpSiteCopy("shared.threshold_label"),
          },
          zeroline: false,
          gridcolor: GRID,
        },
        yaxis: {
          title: { text: window.dpSiteCopy("shared.term_success_rate") },
          range: [0, window.dpProfilePad()],
          tickformat: ".0%",
          zeroline: false,
          gridcolor: GRID,
        },
        // The guide marks the quantile the standings beside the chart are read
        // at, across the plot, and follows the slider.
        shapes: [
          {
            type: "line",
            xref: "paper",
            x0: 0,
            x1: 1,
            y0: spec.guide,
            y1: spec.guide,
            line: { color: INK, width: 1, dash: "dot" },
          },
        ],
        hovermode: "closest",
        margin: { l: 64, r: 16, t: 36 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: "inherit", color: INK },
      },
    };
  }

  /* What a reader picked one baseline as: the catalogue's own label, or the
     label of the algorithm they chose to read against. */
  function baselineLabel(index, metric, baseline) {
    var catalogue = (index.baselines || {})[metric];
    var options = (catalogue && catalogue.options) || [];
    for (var position = 0; position < options.length; position += 1) {
      if (options[position].key === baseline) return options[position].label;
    }
    var uuid = baselineAlgorithm(baseline);
    if (!uuid) return baseline;
    var entry = index.algorithms.filter(function (row) { return row.uuid === uuid; })[0];
    return entry ? entry.label : baseline;
  }

  /* What a reading's costs are: the Metric's own name, and the name of what
     it is divided by, or "" where no baseline divides it. */
  function units(index, metric, baseline) {
    var derived = index.metric_labels || {};
    return {
      metric: derived[metric] || metric,
      baseline: !baseline || baseline === NO_BASELINE ? "" : baselineLabel(index, metric, baseline),
    };
  }

  /* One instance feature, as the comparison reads it: the sizes come from the
     instance's public metadata and ε from the budget the panel declares. */
  function featureReader(index, panel, name) {
    if (name === "epsilon") {
      if (!panel.epsilon) return null;
      return function (column) { return panel.epsilon[column]; };
    }
    return function (column) { return index.instances[column][name]; };
  }

  /* A positive, finite value: the only kind a logarithmic axis places. */
  function positive(value) {
    return value > 0 && Number.isFinite(value);
  }

  /* One feature's weighted mid-rank quantile map over a whole instance set,
     never a reader's filter, so a position never moves when a filter changes.
     For value v, q(v) = W(< v) + W(= v)/2: a tied group shares one position
     and its weight is split evenly across it. An instance with no value is
     left out. `read(column)` gives each column's value and `weightOf(column)`
     its share. */
  function quantileMap(read, count, weightOf) {
    var entries = [];
    var total = 0;
    for (var column = 0; column < count; column += 1) {
      var value = read(column);
      if (value === null || value === undefined) continue;
      var weight = weightOf(column);
      entries.push({ value: value, weight: weight });
      total += weight;
    }
    entries.sort(function (left, right) { return left.value - right.value; });
    var valueToQ = new Map();
    var ticks = [];
    var cumulative = 0;
    var index = 0;
    while (index < entries.length) {
      var next = index;
      var groupWeight = 0;
      while (next < entries.length && entries[next].value === entries[index].value) {
        groupWeight += entries[next].weight;
        next += 1;
      }
      var q = total ? (cumulative + groupWeight / 2) / total : 0;
      valueToQ.set(entries[index].value, q);
      ticks.push({
        value: entries[index].value,
        q: q,
        lo: cumulative / (total || 1),
        hi: (cumulative + groupWeight) / (total || 1),
        count: next - index,
      });
      cumulative += groupWeight;
      index = next;
    }
    return { valueToQ: valueToQ, ticks: ticks, total: total };
  }

  /* The feature value at quantile `target`: the tie group whose own [lo, hi)
     span holds it, or the last group for target 1. */
  function valueAtQuantile(map, target) {
    if (!map.ticks.length) return null;
    var found = map.ticks.find(function (tick) { return target < tick.hi; });
    return (found || map.ticks[map.ticks.length - 1]).value;
  }

  /* The weighted median (target 0.5) or a weighted quartile of a small
     value/weight sample: cumulative weight to the first entry reaching
     target * total, the nearest-rank reading quantileMap uses. +Infinity
     sorts and cumulates like any other value, so a window dominated by
     failures returns +Infinity. */
  function weightedQuantile(entries, target) {
    if (!entries.length) return null;
    var sorted = entries.slice().sort(function (left, right) { return left.value - right.value; });
    var total = sorted.reduce(function (sum, entry) { return sum + entry.weight; }, 0);
    if (!total) return null;
    var cumulative = 0;
    for (var index = 0; index < sorted.length; index += 1) {
      cumulative += sorted[index].weight;
      if (cumulative >= target * total) return sorted[index].value;
    }
    return sorted[sorted.length - 1].value;
  }

  /* One algorithm's featurePoints as a running-median sample: each point at
     its feature value's position in `map`, an instance's weight split evenly
     across its own points. A point whose value the map does not rank is
     left out. */
  function featureSample(points, map, weightOf) {
    var perColumn = new Map();
    points.forEach(function (point) {
      perColumn.set(point.instance, (perColumn.get(point.instance) || 0) + 1);
    });
    return points
      .map(function (point) {
        return {
          q: map.valueToQ.get(point.feature),
          value: point.value,
          weight: weightOf(point.instance) / perColumn.get(point.instance),
        };
      })
      .filter(function (point) { return point.q !== undefined; });
  }

  /* Mirror of dpstats.analysis.trace_running_median: the running
     weighted median and quartiles of one algorithm's points at `positions`
     evenly spaced positions in weight-quantile space, each over a window of
     `window` of the benchmark's weight centred on it (one-sided at the
     edges). `sample` holds {q, value, weight} points; a position whose window
     holds less than half its nominal weight of this sample draws nothing
     (null). The page reads both constants from its payload. */
  function runningMedian(sample, window, positions) {
    var half = window / 2;
    var sampleTotal = sample.reduce(function (sum, point) { return sum + point.weight; }, 0);
    var out = { q: [], median: [], lower: [], upper: [] };
    for (var index = 0; index < positions; index += 1) {
      var p = index / (positions - 1);
      var lo = Math.max(0, p - half);
      var hi = Math.min(1, p + half);
      var within = sample.filter(function (point) { return point.q >= lo && point.q <= hi; });
      var held = within.reduce(function (sum, point) { return sum + point.weight; }, 0);
      out.q.push(p);
      if (!sampleTotal || held / sampleTotal < half) {
        out.median.push(null); out.lower.push(null); out.upper.push(null);
        continue;
      }
      out.median.push(weightedQuantile(within, 0.5));
      out.lower.push(weightedQuantile(within, 0.25));
      out.upper.push(weightedQuantile(within, 0.75));
    }
    return out;
  }

  /* Mirror of dpstats.analysis.compute_robust_range: Tukey's fences
     `fence` interquartile ranges beyond the weighted quartiles of a
     {value, weight} sample, cut to its smallest and largest value; null for
     an empty sample. */
  function robustRange(sample, fence) {
    if (!sample.length) return null;
    var q1 = weightedQuantile(sample, 0.25);
    var q3 = weightedQuantile(sample, 0.75);
    var values = sample.map(function (entry) { return entry.value; });
    return [
      Math.max(q1 - fence * (q3 - q1), Math.min.apply(null, values)),
      Math.min(q3 + fence * (q3 - q1), Math.max.apply(null, values)),
    ];
  }

  /* How many points one algorithm's trials make: trials on one instance
     with the same ratio are one point. Mirror of plot_error_ratios'
     collapsing; `points` holds {instance, value}. */
  function collapsedCount(points) {
    return new Set(points.map(function (point) { return point.instance + "|" + point.value; })).size;
  }

  /* Mirror of the property axis plot_error_ratios opens on: every placed
     coordinate, with `padding` of the span left beyond each end; [-1, 1]
     with none. */
  function fittedRange(coordinates, padding) {
    if (!coordinates.length) return [-1, 1];
    var lo = Math.min.apply(null, coordinates);
    var hi = Math.max.apply(null, coordinates);
    var pad = (hi - lo) * padding || 0.5;
    return [lo - pad, hi + pad];
  }

  /* Mirror of the ratio axis plot_error_ratios opens on: robustRange over a
     {value, weight} sample on the axis's own scale, widened to hold a ratio
     of one, from zero on a linear axis, with `padding` of the span left
     beyond each end. */
  function openingRatioRange(sample, fence, padding, log) {
    var one = log ? 0 : 1;
    var lo = one, hi = one;
    if (sample.length) {
      var fenced = robustRange(sample, fence);
      lo = Math.min(one, fenced[0]);
      hi = Math.max(one, fenced[1]);
    }
    if (!log) lo = 0;
    var pad = (hi - lo) * padding || 0.5;
    return [lo - pad, hi + pad];
  }

  /* The outcome categories of one algorithm, bottom to top: the two success
     strengths, every other terminal outcome, then a declared cell nothing
     answered. Keyed by the status_codes names the index ships. */
  var OUTCOME_STACK = [
    "success_within", "success_beyond", "invalid_regime", "no_release",
    "timeout", "memory_limit_exceeded", "algorithm_error", "not_run",
  ];
  /* Two algorithms read against the same baseline: the share of an
     instance both reach the threshold on, each alone, and neither. */
  var PAIR_STACK = ["both", "a_only", "b_only", "neither"];

  function withinThreshold(value, threshold) {
    return positive(threshold) && Number.isFinite(value) && value <= threshold;
  }

  /* One instance's weight, split into the outcome categories: whole weight
     to one success bucket under median aggregation, or split evenly across
     its recorded seed slots under each-trial aggregation, each slot's own
     status deciding its bucket. A median dominated by failures splits the
     whole weight across the outcomes its seeds carry, in their proportion,
     a success seat counting as beyond the threshold. Returns false where the
     instance has no denominator or no recorded slot. */
  function distributeOutcome(panel, statusCodes, row, column, vector, aggregation, weight, threshold, add) {
    var against = denominator(vector, column);
    var seats = cellSeats(panel, row, column);
    if (against === null || !seats.length) return false;
    var label = function (status) { return (statusCodes || {})[status] || "not_run"; };
    var bucket = function (value) {
      return withinThreshold(value, threshold) ? "success_within" : "success_beyond";
    };
    var costs = cellCosts(panel, row, column, "individual");
    if (aggregation === "median") {
      var middle = median(costs);
      if (middle !== null && Number.isFinite(middle)) {
        add(bucket(ratio(middle, against)), weight);
        return true;
      }
      var counts = {};
      seats.forEach(function (seat) {
        var status = cellStatus(panel, row, column, seat);
        var key = status === null ? "success_beyond" : label(status);
        counts[key] = (counts[key] || 0) + 1;
      });
      Object.keys(counts).forEach(function (key) { add(key, weight * counts[key] / seats.length); });
      return true;
    }
    var per = weight / seats.length;
    seats.forEach(function (seat, index) {
      var status = cellStatus(panel, row, column, seat);
      add(status === null ? bucket(ratio(costs[index], against)) : label(status), per);
    });
    return true;
  }

  /* One algorithm's success rate on one instance, against the page's
     baseline: the share of its recorded trials within the threshold under
     each-trial aggregation, or 1 or 0 as its median is within or not under
     median aggregation. A failed trial counts as not within. Null where it
     has no recorded trial. */
  function successRate(panel, row, column, against, aggregation, threshold) {
    var costs = cellCosts(panel, row, column, "individual");
    if (!costs.length) return null;
    var within = function (cost) { return withinThreshold(ratio(cost, against), threshold); };
    if (aggregation === "median") {
      var middle = median(costs);
      return Number.isFinite(middle) && within(middle) ? 1 : 0;
    }
    return costs.filter(within).length / costs.length;
  }

  /* One instance's weight split between two algorithms' pair categories,
     both read against the page's own baseline. Each algorithm's trials are
     its own randomness, so no trial of one is matched to a trial of the
     other: with p_A and p_B their own success rates, both takes
     min(p_A, p_B), A only p_A - min, B only p_B - min and neither
     1 - max(p_A, p_B) of the weight. Returns false, omitting the instance,
     where either algorithm has no recorded trial or there is no
     denominator. */
  function distributePair(panel, rowA, rowB, column, vector, aggregation, weight, threshold, add) {
    var against = denominator(vector, column);
    if (against === null) return false;
    var rateA = successRate(panel, rowA, column, against, aggregation, threshold);
    var rateB = successRate(panel, rowB, column, against, aggregation, threshold);
    if (rateA === null || rateB === null) return false;
    var both = Math.min(rateA, rateB);
    add("both", weight * both);
    add("a_only", weight * (rateA - both));
    add("b_only", weight * (rateB - both));
    add("neither", weight * (1 - Math.max(rateA, rateB)));
    return true;
  }

  /* Mirror of dpstats.analysis.bin_outcomes: the outcome
     histogram over one instance feature, `spec.bins` equal bins of the
     feature (logarithmic or linear) or equal-weight quantile bins whose
     inner edges snap to the nearest tie-group boundary, each bin's instance
     weight split by distributeOutcome, or by distributePair when `rowB` is
     not negative. Shares are of the whole selection's weight, including the
     instances no bin can place (no feature value, or none the scale places,
     no denominator, or no recorded trial of either algorithm), which
     `excluded` sums.

     spec: {panel, statusCodes, row, rowB, columns, vector, aggregation,
     threshold, read, scale, bins, map (the feature's quantileMap), weightOf}.
     Returns {keys, edges and the placed data's lo and hi (in axis
     coordinates), weights, members (each bin's instance columns), total,
     excluded, snapped}. */
  function outcomeHistogram(spec) {
    var paired = spec.rowB >= 0;
    var keys = paired ? PAIR_STACK : OUTCOME_STACK;
    var quantileScale = spec.scale === "quantile";
    var coordinate = function (value) {
      if (spec.scale === "linear") return value;
      if (quantileScale) return spec.map.valueToQ.get(value);
      return Math.log10(value);
    };
    var placeable = function (value) {
      return spec.scale === "log" ? positive(value)
        : quantileScale ? spec.map.valueToQ.has(value)
        : Number.isFinite(value);
    };
    var total = spec.columns.reduce(function (sum, column) { return sum + spec.weightOf(column); }, 0) || 1;
    var placed = [];
    var excluded = 0;
    spec.columns.forEach(function (column) {
      var value = spec.read(column);
      var weight = spec.weightOf(column);
      if (value === null || value === undefined || !placeable(value)) { excluded += weight; return; }
      placed.push({ column: column, coord: coordinate(value), weight: weight });
    });
    var bins = spec.bins;
    var edges;
    var snapped = false;
    var lo = 0;
    var hi = 1;
    if (quantileScale) {
      edges = [];
      for (var index = 0; index <= bins; index += 1) {
        if (index === 0 || index === bins) { edges.push(index / bins); continue; }
        var nominal = index / bins;
        var nearest = nominal;
        var best = Infinity;
        spec.map.ticks.forEach(function (tick) {
          var difference = Math.abs(tick.hi - nominal);
          if (difference < best) { best = difference; nearest = tick.hi; }
        });
        // Snapping matters only where a tie group of more than one instance
        // straddles the nominal edge.
        if (spec.map.ticks.some(function (tick) {
          return tick.count > 1 && tick.lo < nominal && nominal < tick.hi;
        })) snapped = true;
        edges.push(nearest);
      }
    } else {
      var coords = placed.map(function (entry) { return entry.coord; });
      lo = coords.length ? Math.min.apply(null, coords) : 0;
      hi = coords.length ? Math.max.apply(null, coords) : 1;
      var step = (hi - lo) / bins || 1;
      edges = [];
      for (var edge = 0; edge <= bins; edge += 1) edges.push(lo + edge * step);
    }
    // Right-open bins over the edge array, which quantile snapping may make
    // uneven; the last bin also takes its upper edge.
    var binOf = function (coord) {
      for (var candidate = 1; candidate < bins; candidate += 1) {
        if (coord < edges[candidate]) return candidate - 1;
      }
      return bins - 1;
    };
    var weights = {};
    keys.forEach(function (key) { weights[key] = new Array(bins).fill(0); });
    var members = [];
    for (var slot = 0; slot < bins; slot += 1) members.push([]);
    placed.forEach(function (entry) {
      var bin = Math.min(bins - 1, Math.max(0, binOf(entry.coord)));
      var add = function (key, share) { weights[key][bin] += share; };
      var done = paired
        ? distributePair(spec.panel, spec.row, spec.rowB, entry.column, spec.vector,
          spec.aggregation, entry.weight, spec.threshold, add)
        : distributeOutcome(spec.panel, spec.statusCodes, spec.row, entry.column, spec.vector,
          spec.aggregation, entry.weight, spec.threshold, add);
      if (!done) excluded += entry.weight;
      else members[bin].push(entry.column);
    });
    return {
      keys: keys, edges: edges, lo: lo, hi: hi, weights: weights, members: members,
      total: total, excluded: excluded, snapped: snapped,
    };
  }

  /* The configured algorithm the legend, the family chips and the standings share. */
  function roster(index) {
    return index.algorithms.map(function (entry) {
      return {
        uuid: entry.uuid,
        algorithm: entry.label,
        family: entry.family,
        slot: entry.slot,
        diagnostic: entry.diagnostic,
        interpretation: entry["class"],
        data_bounds: entry.data_bounds,
        fwl_demean: entry.fwl_demean,
        core: entry.core,
        marker: entry.marker,
        transformations: entry.transformations,
      };
    });
  }


  window.dpPanel = {
    readingFields: READING_FIELDS,
    loadIndex: loadIndex,
    resolveBaseline: resolveBaseline,
    baselineAlgorithm: baselineAlgorithm,
    algorithmMedians: algorithmMedians,
    load: load,
    loadMatrix: loadMatrix,
    panelOf: panelOf,
    ratio: ratio,
    rawCost: rawCost,
    median: median,
    denominators: denominators,
    denominator: denominator,
    cellCosts: cellCosts,
    cellSeats: cellSeats,
    cellStatus: cellStatus,
    outcomes: outcomes,
    firstCovering: firstCovering,
    curve: curve,
    featurePoints: featurePoints,
    curvePoints: curvePoints,
    exactCurve: exactCurve,
    reachedBy: reachedBy,
    readingAt: readingAt,
    shippedReading: shippedReading,
    compareReadings: compareReadings,
    figure: figure,
    counted: counted,
    units: units,
    baselineLabel: baselineLabel,
    featureReader: featureReader,
    quantileMap: quantileMap,
    valueAtQuantile: valueAtQuantile,
    weightedQuantile: weightedQuantile,
    featureSample: featureSample,
    runningMedian: runningMedian,
    robustRange: robustRange,
    collapsedCount: collapsedCount,
    fittedRange: fittedRange,
    openingRatioRange: openingRatioRange,
    distributeOutcome: distributeOutcome,
    distributePair: distributePair,
    outcomeHistogram: outcomeHistogram,
    roster: roster,
  };
})();
