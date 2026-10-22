/* Read one panel's costs against one public feature of the instance, as a
   scatter (Error ratio) or as stacked outcome shares (Trial
   outcomes).

   Every point here is derived from the same panel matrices the profiles read,
   through `panel-data.js`'s mirrors: the feature view pairs each ratio
   with one public feature. Grouping coincident
   border markers changes their area and inspection only; every point stays in
   the plot.

   Each chart owns its algorithm selection. Error ratio draws up to five
   chosen algorithms in five configuration-colour slots; Outcomes draws one
   algorithm, or that algorithm against one it is compared with, in the first
   two slots. The shipped series rule supplies their colours, markers and
   class dashes; profiles retain their core-colour rule. */
(() => {
  "use strict";
  const finite = value => value > 0 && Number.isFinite(value);
  // The site's own number format, ∞ included.
  const display = value => window.dpNumber(value);
  const escape = value => String(value).replace(/[&<>"']/g, char =>
    ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[char]);
  const selections = new WeakMap();
  const copy = (key, fields) => window.dpSiteCopy(key, fields);
  /* An instance property as the index names it: a site-copy slot read when
     the page draws, or the bare symbol it is written as. */
  const featureLabel = spec => spec.slot ? copy(spec.slot) : spec.symbol;
  /* The error ratio reading's constants and the chart style, as the build
     wrote them for this page and for its notebook (explore_reading.py,
     chart_style.py): the scatter reads both from here, never from literals. */
  const shipped = (() => {
    const tag = document.querySelector("script[data-explore-chart]");
    return tag ? JSON.parse(tag.textContent) : null;
  })();

  /* Plotly's own colour parser accepts rgb()/hex/named colours only — not
     every CSS colour function a custom property can resolve to. Even asking
     the DOM to compute a color-mix() value back does not reliably land on
     rgb(): a browser may serialize it as color(srgb ...), which Plotly
     cannot parse either. A 1x1 canvas always rasterizes to raw 8-bit pixels,
     so painting the raw string there and reading the pixel back is the one
     path guaranteed to produce rgb(...); it is the single source both a
     drawn trace's marker.color and its legend swatch read, so they cannot
     diverge. */
  const resolveColour = (() => {
    let ctx = null;
    return raw => {
      if (!ctx) ctx = document.createElement("canvas").getContext("2d");
      ctx.fillStyle = "#000";
      ctx.fillStyle = raw;
      ctx.clearRect(0, 0, 1, 1);
      ctx.fillRect(0, 0, 1, 1);
      const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
      return a === 255 ? `rgb(${r}, ${g}, ${b})` : `rgba(${r}, ${g}, ${b}, ${(a / 255).toFixed(3)})`;
    };
  })();

  function connect(root) {
    const section = root.closest(".dp-metric");
    const page = root.closest("[data-playground]");
    const profile = section.querySelector(".dp-chart");
    const baseline = page.querySelector('[data-shared-baseline]');
    const aggregation = page.querySelector('[data-trial-aggregation]');
    const median = () => aggregation.value === 'median';
    const feature = root.querySelector("[data-comparison-feature]");
    const scale = root.querySelector("[data-comparison-scale]");
    const yScale = root.querySelector("[data-comparison-yscale]");
    const listBox = root.querySelector("[data-comparison-selection]");
    const roster = root.querySelector("[data-comparison-algorithms]");
    const slotsBox = root.querySelector("[data-outcome-slots]");
    const status = root.querySelector("[data-comparison-status]");
    const emptyStatus = root.querySelector("[data-comparison-empty]");
    const retry = root.querySelector("[data-comparison-retry]");
    const scatterOnly = root.querySelectorAll("[data-comparison-scatter-only]");
    const histogramOnly = root.querySelectorAll("[data-comparison-histogram-only]");
    const plot = root.querySelector(".dp-comparison-plot");
    const scatterAriaLabel = plot.getAttribute("aria-label") || "";
    const output = root.querySelector("[data-comparison-runs]");
    const histNote = root.querySelector("[data-comparison-histogram-note]");
    const histogramEmpty = root.querySelector("[data-comparison-histogram-empty]");
    const hideHistogram = hidden => {
      histNote.hidden = hidden;
      if (hidden) hideHistogramDetails();
    };
    /* "Error ratio" and "Trial outcomes" are two tabs over this one
       panel and its feature controls; which chart is drawn and visible, and
       which of the two selections is shown, follows playground.js's own
       subview marker. */
    const subview = () => root.dataset.comparisonSubview === "outcomes" ? "outcomes" : "chart";
    /* The reader's choices live on the page, where playground.js reads them
       from and writes them to the address; null there means the default. */
    const chosen = () => page.dpComparisonSelection;
    const announce = () => page.dispatchEvent(new Event("dp-comparison-selection"));
    /* Error ratio reads at a handful of algorithms, so five is a hard cap
       rather than a suggestion: at five the box says so and the sixth pick
       replaces nothing. */
    const PICK_CAP = 5;
    // Error ratio's list is one per page, kept across metric sections.
    if (!selections.has(page)) {
      selections.set(page, {selected: new Set(), slots: new Map(), initialized: false});
    }
    const selection = selections.get(page);
    const {selected, slots} = selection;
    const nextSlot = () => Array.from({length: PICK_CAP}, (_, index) => index)
      .find(index => ![...slots.values()].includes(index));
    function selectAlgorithm(id) {
      if (selected.size >= PICK_CAP) return false;
      if (!slots.has(id)) slots.set(id, nextSlot());
      selected.add(id);
      return true;
    }
    function removeAlgorithm(id) {
      selected.delete(id);
      slots.delete(id);
    }
    // The list as the reader has made it, for the address.
    const remember = () => { chosen().feature = [...selected]; };
    // Search previews the next free slot without reserving it. Existing
    // selections keep their slot across budgets, metrics and filters.
    function treatment(algorithm, slot = slots.get(algorithm.id) ?? nextSlot()) {
      const style = window.dpComparisonStyle(slot, algorithm.class);
      const emphasis = window.dpSeriesEmphasis();
      return {
        colour: style.colour,
        width: emphasis.regularWidth,
        opacity: emphasis.regularOpacity,
        dash: window.dpSeriesDash(algorithm.class),
        marker: style.marker,
      };
    }
    function algorithmOf(id) {
      return panel.algorithms.find(algorithm => algorithm.id === id);
    }
    function colour(id) {
      const algorithm = algorithmOf(id);
      return algorithm ? treatment(algorithm).colour : "";
    }
    // `ranges` and `histRanges` are each view's current axis ranges (null:
    // its opening view). `revision` counts the readings that reopen both
    // views; `opened` names the view and revision the container was last
    // plotted afresh for, and `plotted` whether it holds a plot at all.
    let budget, panel, state, plotted = false, opened = null, revision = 0,
      ranges = null, histRanges = null, vector = null, reader = null;
    /* A reading that changes what the axes hold reopens both views. */
    const reopen = () => { ranges = null; histRanges = null; revision += 1; };
    // The panel and baseline `vector` was last resolved for; null (raw costs) is a resolved vector too.
    let resolvedFor = [];
    let drawing = Promise.resolve();
    let rows = [], wanted = [], initial = [[-1, 1], [-1, 1]], groups = new Map(), clicks = 0, ring = -1;
    const logY = () => !yScale || yScale.value !== "linear";
    const coordinate = (value, axis) => {
      if (axis === 0) {
        if (scale.value === "linear") return value;
        if (scale.value === "quantile") {
          const found = quantileMap(feature.value).valueToQ.get(value);
          return found === undefined ? NaN : found;
        }
      }
      return axis === 1 && !logY() ? value : Math.log10(value);
    };
    // Whether a value has a place on its axis: a log axis has none for 0 or
    // ∞, a linear error-ratio axis has one for 0.
    const placed = (value, axis) => {
      if (axis === 1 && !logY()) return Number.isFinite(value) && value >= 0;
      return finite(value) && Number.isFinite(coordinate(value, axis));
    };
    // The prompt is a text slot, so it is restored as the markup it shipped as.
    const inspection = output.innerHTML;
    const baselineId = () => state ? window.dpPanel.baselineAlgorithm(state.baseline) : null;
    const named = id => {
      const entry = panel.algorithms.find(algorithm => algorithm.id === id);
      return entry ? window.dpQualifiedAlgorithmLabel(entry.label, entry.class) : id;
    };
    const rowOf = id => state.roster.findIndex(entry => entry.uuid === id);
    // The page owns class and baseline eligibility, including when its class
    // controls are in a dialog outside the page's DOM subtree.
    const drawable = algorithm => state.rows.includes(rowOf(algorithm.id));
    // The baseline algorithm is never read against itself.
    const eligible = algorithm => drawable(algorithm) && algorithm.id !== baselineId();

    /* The baseline algorithm is never shown against itself: drop it from the
       list and, if nothing drawable is left, show the first drawable one. */
    function ensureShown() {
      const against = baselineId();
      if (!against || !selected.has(against)) return;
      removeAlgorithm(against);
      if (![...selected].some(id => panel.algorithms.some(algorithm => algorithm.id === id && drawable(algorithm)))) {
        const first = panel.algorithms.find(eligible);
        if (first) selectAlgorithm(first.id);
      }
      if (chosen().feature) remember();
    }

    /* The candidate the standings rank first, over the reader's current
       instances and quantile: each one's exact curve read through the one
       ranking rule (panel-data.js), ties keeping the candidates' order. */
    function leader(candidates) {
      if (!candidates.length) return null;
      const p = Number(profile.dataset.quantile);
      const shares = (state.index.instances || []).map((_, column) => weightOf(column));
      let best = null, bestReading = null;
      candidates.forEach(algorithm => {
        const row = rowOf(algorithm.id);
        if (row < 0) return;
        const curve = window.dpPanel.exactCurve(state.panel, row, state.instances || [], vector,
          median() ? "median" : "individual", shares);
        const reading = window.dpPanel.readingAt(curve, p, window.dpPanel.reachedBy(curve));
        if (!bestReading || window.dpPanel.compareReadings(reading, bestReading) < 0) {
          best = algorithm;
          bestReading = reading;
        }
      });
      return best || candidates[0];
    }

    /* Outcomes' algorithm, required, and the one it is compared with, if any.
       Until the reader chooses, the algorithm is the success rate leader, kept
       while it stays drawable. A choice this reading cannot draw stands aside
       without being forgotten, and returns when it can be drawn again. */
    let outcomesDefault = null, outcomesDefaultReading = [];
    function outcomesPair() {
      const usable = id => {
        const algorithm = id && algorithmOf(id);
        return algorithm && eligible(algorithm) ? algorithm : null;
      };
      // The leader is read once the instances and their baseline are resolved,
      // and again only when it stops being drawable or the panel or baseline changes.
      const resolved = resolvedFor[0] === state.panel && resolvedFor[1] === state.baseline;
      if (resolved && (state.instances || []).length &&
          (!usable(outcomesDefault) || outcomesDefaultReading[0] !== state.panel || outcomesDefaultReading[1] !== state.baseline)) {
        outcomesDefault = leader(panel.algorithms.filter(eligible))?.id ?? null;
        outcomesDefaultReading = [state.panel, state.baseline];
      }
      const a = usable(chosen().outcomes) || usable(outcomesDefault);
      const b = usable(chosen().compare);
      return {a, b: a && b && b.id !== a.id ? b : null};
    }

    /* What this chart's notebook needs beyond the page's own reading: the
       property and its scale, the algorithms the open view draws, the view's
       texts with their fields left for the figure to fill, and the chart
       style in the reader's number separators. The error ratio view adds its
       ratio scale and each drawn algorithm's light-theme colour, marker and
       name; the outcomes view the two compared algorithms' colours. */
    root.dpNotebookSelection = () => {
      if (!panel) return {};
      const shown = {FEATURE: feature.value, SCALE: scale.value};
      const style = JSON.parse(JSON.stringify(shipped.style));
      const look = style.layout.meta;
      // The reader's separators, in the locale dpNumber writes in.
      const parts = new Intl.NumberFormat(undefined).formatToParts(12345.6);
      const part = type => parts.find(entry => entry.type === type)?.value;
      look.number_separators = {decimal: part("decimal") || ".", group: part("group") || ","};
      const lightColour = (algorithm, slot) =>
        algorithm.class === look.reference.class ? look.reference.color : look.slots[slot].color;
      if (subview() === "outcomes") {
        const {a, b} = outcomesPair();
        if (a) look.outcomes.colors.a_only = lightColour(a, 0);
        if (b) look.outcomes.colors.b_only = lightColour(b, 1);
        return {...shown, ALGORITHM: a ? a.id : null, COMPARE_WITH: b ? b.id : null,
          LABELS: {
            xlabel: scale.value === "quantile"
              ? copy("explore.histogram_x_quantile", {feature: panel.features[feature.value]})
              : panel.features[feature.value],
            ylabel: copy("explore.histogram_y_axis"),
            success_hover: copy("explore.bar_hover_success", {share: "{share}"}),
            stack_hover: copy("explore.bar_hover_stack", {share: "{share}"}),
            pair_hover: a && b ? `${a.label} ≤ τ: {a} · ${b.label} ≤ τ: {b}` : "A ≤ τ: {a} · B ≤ τ: {b}",
          },
          CHART_STYLE: style};
      }
      const drawn = panel.algorithms.filter(algorithm => selected.has(algorithm.id) && drawable(algorithm));
      drawn.forEach(algorithm => {
        const slot = look.slots[slots.get(algorithm.id)];
        look.series[algorithm.id] = {
          color: algorithm.class === look.reference.class ? look.reference.color : slot.color,
          symbol: slot.symbol,
          name: named(algorithm.id),
        };
      });
      const counted = median() ? "explore.median_point_count" : "explore.trial_count";
      return {...shown,
        ALGORITHMS: drawn.map(algorithm => algorithm.id),
        Y_SCALE: logY() ? "log" : "linear",
        LABELS: {
          feature: panel.features[feature.value],
          xlabel: xTitle(),
          ylabel: yTitle(),
          point_hover: copy("explore.point_hover", {property: "{property}", x: "{x}", y: "{y}"}),
          point_beyond_view: copy("explore.point_beyond_view"),
          count_one: copy(counted + "_one", {count: "{count}"}),
          count: copy(counted, {count: "{count}"}),
          running_median: copy("explore.running_median", {algorithm: "{algorithm}"}),
        },
        CHART_STYLE: style};
    };

    /* Which features this panel can offer: the properties the task declares,
       and ε only where the budget supplies a finite one on every instance. */
    function features(current) {
      const offered = {}, unavailable = {};
      Object.entries(current.index.features).forEach(([name, spec]) => {
        const label = featureLabel(spec);
        if (name === "epsilon" && !current.panel.epsilon) {
          unavailable[name] = {label, reason: "explore.feature_unavailable_epsilon"};
          return;
        }
        /* An infinite condition number has no place on a logarithmic feature
           axis, and dropping those instances would plot a different cohort
           than the one selected. */
        if (name === "design_condition_number" &&
            current.index.instances.some(instance => !finite(Number(instance[name])))) {
          unavailable[name] = {label, reason: "explore.feature_unavailable_condition_number"};
          return;
        }
        // A feature with one distinct value in this panel (for example ε
        // when a single budget is selected) has nothing to plot against and
        // degenerates to a single bar; offer it as unavailable, with a
        // reason, the same as every other disabled option here.
        const read = window.dpPanel.featureReader(current.index, current.panel, name);
        const seen = new Set();
        for (let column = 0; column < current.index.instances.length && seen.size < 2; column += 1) {
          const value = read ? read(column) : null;
          if (value !== null && value !== undefined) seen.add(value);
        }
        if (seen.size <= 1) {
          unavailable[name] = {label, reason: "explore.feature_unavailable_constant"};
          return;
        }
        offered[name] = label;
      });
      return {offered, unavailable};
    }

    function baselineLabels() {
      const labels = {};
      baseline.querySelectorAll('input[type="radio"]').forEach(input => {
        if (!input.value) return;
        const label = baseline.querySelector(`label[for="${input.id}"]`);
        labels[input.value] = label ? label.textContent : input.value;
      });
      return labels;
    }

    /* One reading of the loaded panel, named the way this view talks about it. */
    function panelView(current) {
      const offered = features(current);
      return {
        metric: current.metric,
        note: current.index.notes[current.metric] || "",
        algorithms: current.roster.map(entry => ({
          id: entry.uuid, label: entry.algorithm, class: entry.interpretation,
          slot: entry.slot, diagnostic: entry.diagnostic, core: entry.core,
          marker: entry.marker, transformations: entry.transformations,
        })),
        instances: current.index.instances.map(entry => ({
          id: entry.uuid, label: entry.label, n: entry.n, d: entry.d,
        })),
        features: offered.offered,
        feature_unavailable: offered.unavailable,
        baselines: baselineLabels(),
      };
    }

    function configure(current) {
      const nextBudget = current.metric + "/" + current.budget;
      panel = panelView(current);
      if (!selection.initialized) {
        const opened = chosen().feature;
        const first = panel.algorithms.find(drawable);
        if (opened) opened.forEach(selectAlgorithm);
        else if (first) selectAlgorithm(first.id);
        selection.initialized = Boolean(opened || first);
      }
      if (budget === nextBudget) {
        renderRoster();
        return;
      }
      budget = nextBudget;
      reopen();
      quantileCache.clear();
      const previous = feature.value;
      feature.replaceChildren();
      Object.entries(panel.features).forEach(([value, label]) => feature.add(new Option(label, value)));
      Object.entries(panel.feature_unavailable).forEach(([value, {label, reason: key}]) => {
        const option = new Option(label, value);
        option.disabled = true;
        const reason = window.dpSiteCopy(key);
        if (reason) option.title = reason;
        feature.add(option);
      });
      feature.value = previous in panel.features ? previous : "n";
      renderRoster();
    }

    const picker = window.dpAlgorithmPicker(listBox.querySelector('[data-algorithm-picker]'), {
      trigger: listBox.querySelector('[data-comparison-add]'),
      algorithms: () => panel ? panel.algorithms.filter(drawable) : [],
      selected: () => selected,
      limit: PICK_CAP,
      treatment: treatment,
      choose: id => {
        if (!selectAlgorithm(id)) return;
        remember();
        announce();
      },
      remove: unpick,
    });

    function unpick(id) {
      removeAlgorithm(id);
      remember();
      announce();
    }

    /* Outcomes' two slots: each names its algorithm on the trigger that opens
       the shared search, and only the comparison can be cleared. The slot is
       the role: the algorithm draws in the first comparison colour, the one
       it is compared with in the second. */
    const outcomeSlots = [["algorithm", "outcomes"], ["compare", "compare"]].map(([role, key], slot) => {
      const box = slotsBox.querySelector(`[data-outcome-slot="${role}"]`);
      const trigger = box.querySelector("[data-outcome-trigger]");
      const clear = box.querySelector("[data-outcome-clear]");
      if (clear) clear.addEventListener("click", () => {
        chosen()[key] = null;
        announce();
        trigger.focus({preventScroll: true});
      });
      const search = window.dpAlgorithmPicker(box.querySelector("[data-algorithm-picker]"), {
        trigger,
        algorithms: () => panel ? panel.algorithms.filter(eligible) : [],
        selected: () => {
          const pair = panel && state ? outcomesPair() : {};
          return new Set([pair.a, pair.b].filter(Boolean).map(algorithm => algorithm.id));
        },
        treatment: algorithm => treatment(algorithm, slot),
        choose: id => {
          chosen()[key] = id;
          announce();
        },
      });
      return {trigger, clear, search, slot};
    });
    function renderSlots() {
      const pair = outcomesPair();
      [pair.a, pair.b].forEach((algorithm, index) => {
        const {trigger, clear, search, slot} = outcomeSlots[index];
        trigger.replaceChildren();
        if (algorithm) {
          // The bars take the two slot colours only once there is a pair.
          if (pair.b) trigger.append(window.dpSeriesSwatch(treatment(algorithm, slot)));
          trigger.append(algorithm.label, " ", window.dpInterpretationBadge(algorithm.class));
        } else {
          trigger.append(window.dpSiteCopySpan("explore.outcome_none"));
        }
        if (clear) clear.hidden = !algorithm;
        search.refresh();
      });
    }

    /* Hovering a legend row raises that algorithm's traces and steps every
       other algorithm back, the way the profile chart does: on the drawn
       traces in place, so a hover redraws nothing; the next draw restores
       the data's own opacity. */
    function raise(active) {
      if (!plotted || !plot.data) return;
      plot.querySelectorAll(".overplot .trace.scatter").forEach(group => {
        const datum = group.__data__ && group.__data__[0];
        const trace = datum && datum.trace;
        if (!trace) return;
        const own = String(trace.opacity === undefined ? 1 : trace.opacity);
        const opacity = !active || !trace.legendgroup || trace.legendgroup === active ? own : "0.2";
        if (group.style.opacity !== opacity) group.style.opacity = opacity;
      });
    }
    /* The plotted set is the legend: one row per drawn algorithm, carrying the
       swatch the chart draws it with and its own remove control. */
    function renderRoster() {
      roster.replaceChildren();
      [...selected].map(algorithmOf).filter(algorithm => algorithm && drawable(algorithm))
        .forEach(algorithm => {
          const row = window.dpPlottedRow({
            uuid: algorithm.id,
            label: algorithm.label,
            interpretation: algorithm.class,
            treatment: treatment(algorithm),
            onRemove: () => unpick(algorithm.id),
            onDetails: opener => window.dpAlgorithmDialog({
              host: page,
              uuid: algorithm.id,
              title: algorithm.label,
              treatment: treatment(algorithm),
              interpretation: algorithm.class,
              opener,
            }),
          });
          ["pointerover", "focusin"].forEach(type => row.addEventListener(type, () => raise(algorithm.id)));
          ["pointerout", "focusout"].forEach(type => row.addEventListener(type, () => raise(null)));
          roster.append(row);
        });
      /* The baseline is drawn but not chosen, so it is named without a remove
         control, the way Home and the task leaderboard name their rows. */
      const against = baselineId();
      if (against) {
        const row = window.dpPlottedRow({
          uuid: against,
          label: copy("explore.plotted_baseline", {algorithm: named(against)}),
          treatment: null,
        });
        row.classList.add("dp-plotted-baseline");
        roster.append(row);
      }
      root.querySelector("[data-comparison-note]").textContent = panel.note;
      root.querySelector('[data-comparison-selected-count]').textContent =
        copy("explore.selected_count", {selected: selected.size, cap: PICK_CAP});
      listBox.querySelector('[data-comparison-add]').disabled = selected.size >= PICK_CAP;
      picker.refresh();
      renderSlots();
    }

    // Each view shows only its own chart's selection.
    function showSelection() {
      const outcomes = subview() === "outcomes";
      listBox.hidden = outcomes;
      slotsBox.hidden = !outcomes;
    }

    /* Each instance's protocol share, exactly as the profile curve reads it
       (dpstats.analysis.compute_profile_curves): the release's weight
       where one is shipped, one share otherwise. */
    function weightOf(column) {
      const table = state.index.weights;
      return table && table.length ? table[column] : 1;
    }

    /* Each feature's weighted quantile map over the full instance set
       (panel-data.js), cached per panel and feature. */
    const quantileCache = new Map();
    function quantileMap(name) {
      if (!quantileCache.has(name)) {
        const read = window.dpPanel.featureReader(state.index, state.panel, name);
        quantileCache.set(name, window.dpPanel.quantileMap(read, state.index.instances.length, weightOf));
      }
      return quantileCache.get(name);
    }
    function valueAtQuantile(name, target) {
      return window.dpPanel.valueAtQuantile(quantileMap(name), target);
    }

    /* Bottom to top: the two success strengths, then every other terminal
       ResultStatus, then a declared cell nothing answered. Keyed by the
       status_codes name the index ships (see cellStatus/label), each with
       its name and colour token as the shared chart style lists them
       (chart_style.py), which the notebook draws with too. */
    const outcomesLook = shipped.style.layout.meta.outcomes;
    const stackOf = entries => entries.map(entry =>
      ({...entry, hatched: outcomesLook.hatched.includes(entry.key)}));
    const HIST_STACK = stackOf(outcomesLook.stack);
    const withLabel = entry => ({...entry, label: entry.slot ? copy(entry.slot) : entry.text});

    /* One instance's weight split into HIST_STACK's outcomes (panel-data.js). */
    function distribute(row, column, instanceWeight, threshold, add) {
      return window.dpPanel.distributeOutcome(state.panel, state.index.status_codes, row, column,
        vector, median() ? "median" : "individual", instanceWeight, threshold, add);
    }

    /* The one figure shape every view of this plot draws in: size, margins,
       fonts, background and drag/hover behaviour. The selected-algorithm
       list above it names the series; Outcomes adds Plotly's compact key.
       A view overrides only
       what is inherent to it: axis types/titles, barmode, threshold/parity
       lines. */
    function baseLayout(ink, style, token) {
      const width = plot.clientWidth;
      return {
        width, height: width < 480 ? 320 : Math.min(520, Math.max(350, width * 0.6)), autosize: true,
        // b leaves room for a two-line log axis (minor ticks plus decade
        // labels) under the title, on every view that shares this layout.
        margin: {l: 65, r: 30, t: 28, b: 100}, paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: token("--dp-chart-plot"),
        font: {color: ink, size: parseFloat(style.fontSize), family: style.fontFamily},
        showlegend: false, hovermode: "closest",
        // The shared chart style's drag mode and modebar.
        dragmode: shipped.style.layout.dragmode, modebar: shipped.style.layout.modebar,
      };
    }
    function baseAxis(ink, grid) {
      const axis = {showline: true, mirror: true, linecolor: ink, gridcolor: grid, zeroline: false, automargin: true};
      // At 375px the log axis's minor-tick labels ("5 2 ... 5 2") are too
      // small to read and look like stray data; drop them there and keep
      // only the decade labels, which automargin still sizes for.
      if (plot.clientWidth < 480) axis.minor = {showticklabels: false};
      return axis;
    }
    /* A quantile-scale feature axis is drawn linear over [0, 1]; its ticks
       name the feature value at that quantile (target 0, 0.1, ..., 1) rather
       than the quantile number itself. */
    function formatFeatureValue(value) {
      if (value === Infinity) return "∞";
      return Number.isInteger(value) && Math.abs(value) < 1e6 ? value.toLocaleString() : window.dpNumber(value);
    }
    function quantileAxisExtra(name, narrow) {
      // The axis title already names the feature, so a tick need only carry
      // its value. At narrow widths (375px mobile) label every other decile
      // so adjacent quantile ticks do not collide.
      const step = narrow ? 2 : 1;
      const tickvals = Array.from({length: 11}, (_, index) => index / 10).filter((_, index) => index % step === 0);
      const ticktext = tickvals.map(target => formatFeatureValue(valueAtQuantile(name, target)));
      return {type: "linear", tickmode: "array", tickvals, ticktext};
    }
    const plotConfig = {responsive: true, displaylogo: false, doubleClick: false};
    const sameRange = (a, b) => a[0] === b[0] && a[1] === b[1];

    /* Draw one view. Within one opening of a view the plot is updated in
       place, and its uirevision keeps the reader's zoom, pan and drag mode.
       A view opened afresh (first drawn, switched to, or reopened by a
       reading) is plotted anew on its opening ranges, which the modebar's
       reset returns to, and then moved to the ranges the reader left it at.
       `opening` holds the x range, and the y range where the view sets one. */
    async function plotView(traces, layout, opening) {
      const key = `${subview()}|${revision}`;
      layout.uirevision = key;
      if (plotted && opened === key) {
        await Plotly.react(plot, traces, layout, plotConfig);
        return;
      }
      const current = [layout.xaxis.range, opening[1] ? layout.yaxis.range : null];
      await Plotly.newPlot(plot, traces, {
        ...layout,
        xaxis: {...layout.xaxis, range: opening[0]},
        ...(opening[1] ? {yaxis: {...layout.yaxis, range: opening[1]}} : {}),
      }, plotConfig);
      plotted = true;
      opened = key;
      listen();
      const moved = {
        ...(sameRange(current[0], opening[0]) ? {} : {"xaxis.range": current[0]}),
        ...(!opening[1] || sameRange(current[1], opening[1]) ? {} : {"yaxis.range": current[1]}),
      };
      if (Object.keys(moved).length) await Plotly.relayout(plot, moved);
    }

    /* A fresh plot carries no listeners, so each one binds them again. */
    function listen() {
      plot.on("plotly_click", event => {
        if (!event.points?.length) return;
        if (subview() === "outcomes") {
          if (histExplain.hidden) return;
          explainedBin = event.points[0].pointNumber;
          explainBin();
          highlightExplainedBin();
          return;
        }
        const same = groups.get(event.points[0].customdata) || [];
        if (!same.length) return;
        clicks += 1;
        const point = event.points[0];
        Plotly.restyle(plot, {x: [[point.x]], y: [[point.y]], xaxis: [point.data.xaxis], yaxis: [point.data.yaxis]}, [ring]);
        explainPoint(same, clicks);
      });
      plot.on("plotly_relayout", event => {
        if (!Object.keys(event).some(key => /^([xy]axis)\.range/.test(key))) return;
        // The scatter and the histogram share one container, so each view
        // keeps its own ranges.
        if (subview() === "outcomes") {
          histRanges = plot.layout.xaxis.range.slice();
        } else {
          ranges = [plot.layout.xaxis.range.slice(), plot.layout.yaxis.range.slice()];
        }
        repaint();
      });
    }

    const histGuide = root.querySelector("[data-comparison-histogram-guide]");
    const histExplain = root.querySelector("[data-comparison-histogram-explain]");
    // Each help label and the reading guide open the site's one contextual help.
    root.querySelectorAll("[data-help-trigger]").forEach(trigger => {
      const content = document.getElementById(trigger.getAttribute("popovertarget"));
      if (content) window.dpContextualHelp(trigger, content);
    });
    const histBin = root.querySelector("[data-histogram-bin]");
    const histBreakdown = root.querySelector("[data-histogram-breakdown]");
    const histSize = root.querySelector("[data-histogram-size]");
    const histInstances = root.querySelector("[data-histogram-instances]");
    const histDataBox = root.querySelector("[data-comparison-histogram-data]");
    const histDataHead = root.querySelector("[data-comparison-histogram-data-head]");
    const histDataBody = root.querySelector("[data-comparison-histogram-data-body]");
    // The category table and contextual help accompany the Outcomes view.
    function showHistogramDetails() {
      histGuide.hidden = false;
      histExplain.hidden = false;
      histDataBox.hidden = false;
    }
    function hideHistogramDetails() {
      histGuide.hidden = true;
      histExplain.hidden = true;
      histDataBox.hidden = true;
    }

    // The explanation reads the very same shares as the bars and their
    // hovers. It never reads panel costs or assigns an outcome of its own.
    let histogram = null, explainedBin = null;
    const percentage = value => (value * 100).toFixed(1) + "%";
    /* One bar's outcome mix beside the same mix over every instance shown, so
       a reader sees how this bar departs from the whole. */
    function explainBin() {
      if (!histogram) return;
      const {entries, shares, totals} = histogram;
      const index = explainedBin;
      const total = totals[index];
      const everything = totals.reduce((sum, value) => sum + value, 0);
      histBin.value = String(index);
      histBreakdown.replaceChildren(...entries.map(entry => {
        const row = document.createElement("tr");
        const label = document.createElement("th");
        label.scope = "row";
        label.textContent = entry.label;
        row.append(label);
        const overall = shares[entry.key].reduce((sum, value) => sum + value, 0);
        [total ? percentage(shares[entry.key][index] / total) : "—",
          everything ? percentage(overall / everything) : "—"].forEach(text => {
          const cell = document.createElement("td");
          cell.textContent = text;
          row.append(cell);
        });
        return row;
      }));
      histSize.replaceChildren(total
        ? window.dpSiteCopySpan("explore.bin_share", {share: percentage(total / everything)})
        : window.dpSiteCopySpan("explore.bin_empty"));
      listBinInstances(histogram.members[index] || []);
    }
    /* The bin's instances, as the bars place them, each a link to its page
       previewing its card; the first BIN_SHOWN, then all on request. A label
       becomes a link only once its card is found, so no link reaches an
       instance without a page. */
    const BIN_SHOWN = 20;
    let binListed = 0;
    function listBinInstances(columns, all = false) {
      const listing = binListed += 1;
      const instances = columns.map(column => state.index.instances[column])
        .sort((a, b) => a.label.localeCompare(b.label));
      if (!instances.length) { histInstances.replaceChildren(); return; }
      const heading = document.createElement("h4");
      heading.className = "dp-bin-instances-heading";
      heading.append(window.dpSiteCopySpan("explore.bin_instances_heading", {count: instances.length.toLocaleString()}));
      const list = document.createElement("ul");
      list.className = "dp-bin-instances";
      (all ? instances : instances.slice(0, BIN_SHOWN)).forEach((instance, at) => {
        const item = document.createElement("li");
        const label = document.createElement("code");
        label.textContent = instance.label;
        item.append(label);
        list.append(item);
        const href = new URL(`../instances/${instance.uuid}/`, state.index.location).href;
        window.dpInstanceCard(href).then(() => {
          if (listing !== binListed) return;
          const link = document.createElement("a");
          link.href = href;
          link.append(label);
          item.replaceChildren(link);
          window.dpInstancePreview(link, `${root.id || "dp-comparison"}-bin-instance-${listing}-${at}`);
        }, () => {});
      });
      histInstances.replaceChildren(heading, list);
      if (!all && instances.length > BIN_SHOWN) {
        const more = document.createElement("button");
        more.type = "button";
        more.className = "btn btn-sm";
        more.append(window.dpSiteCopySpan("explore.bin_show_all", {count: instances.length.toLocaleString()}));
        more.addEventListener("click", () => listBinInstances(columns, true));
        histInstances.append(more);
      }
    }
    function explainedShape() {
      if (!histogram || histExplain.hidden || subview() !== "outcomes") return [];
      return [{type: "rect", xref: "x", yref: "paper",
        x0: histogram.edges[explainedBin], x1: histogram.edges[explainedBin + 1], y0: 0, y1: 1,
        line: {color: getComputedStyle(plot).color, width: 1}, fillcolor: "rgba(0,0,0,0)"}];
    }
    function highlightExplainedBin() {
      if (plotted && subview() === "outcomes" && !histExplain.hidden && !plot.hidden) {
        Plotly.relayout(plot, {shapes: explainedShape()});
      }
    }
    histBin.addEventListener("change", () => {
      explainedBin = Number(histBin.value);
      explainBin();
      highlightExplainedBin();
    });

    function updateHistogramExplanation(entries, bounds, shares, edges, members) {
      const pair = outcomesPair();
      const context = JSON.stringify([feature.value, scale.value, pair.a && pair.a.id, pair.b && pair.b.id,
        state.baseline, state.aggregation, state.instances]);
      const totals = bounds.map((_, index) => entries.reduce((sum, entry) => sum + shares[entry.key][index], 0));
      if (!histogram || histogram.panel !== state.panel || histogram.context !== context) {
        explainedBin = totals.indexOf(Math.max(...totals));
      }
      histogram = {entries, bounds, shares, totals, edges, members, context, panel: state.panel};
      // Keep the native select (and its keyboard focus) while changing only
      // its options when bin boundaries change.
      if (histBin.options.length !== bounds.length || bounds.some((bound, index) => histBin.options[index].text !== bound)) {
        histBin.replaceChildren(...bounds.map((bound, index) => new Option(bound, String(index))));
      }
      root.querySelector("[data-histogram-bin-label]").textContent =
        copy("explore.bin_label", {feature: panel.features[feature.value]});
      explainBin();
    }
    /* A compact table equivalent of the chart, for the accessibility
       disclosure: one row per bin, one column per outcome category, in the
       same shares the bars draw. */
    function updateHistData(present, bounds, shares) {
      histDataHead.replaceChildren();
      const headRow = document.createElement("tr");
      const first = document.createElement("th");
      first.scope = "col";
      first.append(window.dpSiteCopySpan("explore.bin"));
      headRow.append(first);
      present.forEach(entry => {
        const cell = document.createElement("th");
        cell.scope = "col";
        cell.textContent = entry.label;
        headRow.append(cell);
      });
      histDataHead.append(headRow);
      histDataBody.replaceChildren();
      bounds.forEach((bound, index) => {
        const row = document.createElement("tr");
        const label = document.createElement("td");
        label.textContent = bound;
        row.append(label);
        present.forEach(entry => {
          const cell = document.createElement("td");
          cell.textContent = percentage(shares[entry.key][index]);
          row.append(cell);
        });
        histDataBody.append(row);
      });
    }

    /* The paired stack, in draw order: both within tau, then each algorithm
       alone, then neither. Per instance, both is the smaller
       of the two algorithms' own success rates, each alone the excess of its
       rate over that, and neither the rest; no trial of one algorithm is
       matched to a trial of the other. An instance either algorithm has no
       trial on is omitted. The outcome-type breakdown does not apply
       here; any non-success is simply not within tau.
       "A only"/"B only" carry no page token: each takes its slot's colour,
       the same colour its swatch shows on the Algorithm and Compare with
       choices. */
    const PAIR_STACK = stackOf(outcomesLook.pairs);

    /* One bin's range, in the units the reader picked: the feature value
       under Logarithmic/Linear, or the underlying feature value at each
       quantile edge under Quantile (never a bare quantile fraction, which
       names no feature unit). */
    function binBounds(edges, index, raw, quantileScale) {
      if (quantileScale) {
        return `${formatFeatureValue(valueAtQuantile(feature.value, edges[index]))}–${formatFeatureValue(valueAtQuantile(feature.value, edges[index + 1]))}`;
      }
      return `${window.dpNumber(raw(edges[index]))}–${window.dpNumber(raw(edges[index + 1]))}`;
    }

    async function drawHistogram() {
      const {a: algorithmA, b: algorithmB} = outcomesPair();
      const row = rowOf(algorithmA.id);
      hideHistogram(false);
      const paired = Boolean(algorithmB);
      const rowB = paired ? rowOf(algorithmB.id) : -1;
      const threshold = Number(profile.dataset.threshold);
      // Bottom to top: HIST_STACK for one algorithm's own outcome types, or
      // PAIR_STACK's both/A-only/B-only/neither once a comparison is
      // chosen. A is the Algorithm slot and B the Compare with slot; the
      // compact chart key never repeats their configured names. Detailed
      // hovers retain those identities.
      const entries = (paired ? PAIR_STACK : HIST_STACK).map(withLabel);
      // Bins, their shares and what is excluded come from panel-data.js, the
      // same computation the downloaded notebook is checked against.
      const quantileScale = scale.value === "quantile";
      const histogram = window.dpPanel.outcomeHistogram({
        panel: state.panel, statusCodes: state.index.status_codes, row, rowB,
        columns: state.instances, vector, aggregation: median() ? "median" : "individual",
        threshold, read: reader, scale: scale.value,
        bins: quantileScale ? shipped.reading.QUANTILE_BINS : shipped.reading.OUTCOME_BINS,
        map: quantileScale ? quantileMap(feature.value) : null, weightOf,
      });
      const {edges, lo, hi, total, excluded, snapped, weights} = histogram;
      const BINS = edges.length - 1;
      const raw = value => scale.value === "log" ? 10 ** value : value;
      histNote.replaceChildren();
      if (excluded) {
        histNote.append(window.dpSiteCopySpan("explore.histogram_excluded_share", {share: percentage(excluded / total)}));
      }
      [...(excluded ? ["explore.histogram_excluded_reason"] : []),
        ...(snapped ? ["explore.histogram_snapped_note"] : []),
        ...(paired ? ["explore.histogram_paired_note"] : [])].forEach(key => {
        const slot = document.createElement("span");
        if (window.dpSiteCopyFill(slot, key)) histNote.append(" ", slot);
      });
      const centers = edges.slice(0, BINS).map((edge, index) => raw((edge + edges[index + 1]) / 2));
      const widths = edges.slice(0, BINS).map((edge, index) => raw(edges[index + 1]) - raw(edge));
      const bounds = edges.slice(0, BINS).map((edge, index) => binBounds(edges, index, raw, quantileScale));
      const style = getComputedStyle(plot);
      const token = name => style.getPropertyValue(name).trim();
      const ink = style.color;
      // "A only"/"B only" have no page token: they read the drawing algorithms' own series colours.
      const entryColour = entry => entry.token ? resolveColour(token(entry.token))
        : treatment(entry.key === "a_only" ? algorithmA : algorithmB, entry.key === "a_only" ? 0 : 1).colour;
      const shares = {};
      entries.forEach(entry => { shares[entry.key] = weights[entry.key].map(value => value / total); });
      const withinA = paired
        ? shares.both.map((value, index) => value + shares.a_only[index])
        : shares.success_within;
      const withinB = paired ? shares.both.map((value, index) => value + shares.b_only[index]) : null;
      const present = entries.filter(entry => shares[entry.key].some(value => value > 0));
      const traces = present.map(entry => {
        const barColour = entryColour(entry);
        return {
          type: "bar", name: escape(entry.label), showlegend: true, x: centers, y: shares[entry.key], width: widths,
          marker: {
            color: barColour,
            ...(entry.hatched ? {pattern: {...outcomesLook.hatch, fgcolor: ink, bgcolor: barColour}} : {}),
          },
          hovertext: bounds.map((bound, index) => {
            const totalInBin = present.reduce((sum, other) => sum + shares[other.key][index], 0);
            const pct = value => totalInBin ? percentage(value / totalInBin) : "–";
            return `${escape(bound)}<br>${escape(entry.label)}: ${percentage(shares[entry.key][index])}` +
              (paired
                ? `<br>${escape(algorithmA.label)} ≤ τ: ${pct(withinA[index])} · ${escape(algorithmB.label)} ≤ τ: ${pct(withinB[index])}`
                : `<br>${escape(copy("explore.bar_hover_success", {share: pct(withinA[index])}))}`) +
              `<br>${escape(copy("explore.bar_hover_stack", {share: percentage(shares[entry.key][index])}))}`;
          }),
          hovertemplate: "%{hovertext}<extra></extra>"};
      });
      showHistogramDetails();
      updateHistData(present, bounds, shares);
      updateHistogramExplanation(present, bounds, shares, edges.map(raw), histogram.members);
      const described = {categories: present.length, feature: panel.features[feature.value], bins: BINS};
      plot.setAttribute("aria-label", finite(threshold)
        ? window.dpSiteCopy("explore.histogram_label", {...described, tau: window.dpTau(threshold)})
        : window.dpSiteCopy("explore.histogram_label_no_threshold", described));
      const padded = (() => {
        const pad = (hi - lo) * shipped.reading.OUTCOME_PADDING || 0.5;
        return [lo - pad, hi + pad];
      })();
      const layout = {
        ...baseLayout(ink, style, token), barmode: "stack", bargap: outcomesLook.bargap,
        shapes: explainedShape(),
        showlegend: true,
        legend: {...outcomesLook.legend, font: {color: ink}},
        xaxis: {...baseAxis(ink, token("--dp-chart-grid")), type: scale.value,
          range: histRanges || (quantileScale ? [0, 1] : padded),
          ...(quantileScale ? quantileAxisExtra(feature.value, plot.clientWidth < 480) : {}),
          title: {text: escape(quantileScale
            ? copy("explore.histogram_x_quantile", {feature: panel.features[feature.value]})
            : panel.features[feature.value])}},
        yaxis: {...baseAxis(ink, token("--dp-chart-grid")), rangemode: "tozero", tickformat: ".0%",
          title: {text: escape(copy("explore.histogram_y_axis"))}},
      };
      await plotView(traces, layout, [quantileScale ? [0, 1] : padded]);
    }

    /* The clicked point, in the panel beside the scatter: its reading, each
       algorithm with its count, then its instances' cards, whose titles link
       the instance pages. An instance without a card (so without a page)
       shows its label only. Every trial at one point shares its property
       value and its error, so those are stated once. */
    const prompt = [...output.childNodes];
    function showPrompt() { output.replaceChildren(...prompt.map(node => node.cloneNode(true))); }
    // One value, or the span of a merged spot's values (a spot on the edge
    // gathers everything beyond the view at that position).
    const spanOf = values => {
      const finiteOnes = values.filter(Number.isFinite);
      if (finiteOnes.length !== values.length || !finiteOnes.length) return display(values[0]);
      const lo = Math.min(...finiteOnes), hi = Math.max(...finiteOnes);
      return lo === hi ? display(lo) : `${display(lo)} – ${display(hi)}`;
    };
    function explainPoint(same, click) {
      const facts = document.createElement("div");
      facts.className = "dp-point-facts";
      const reading = document.createElement("div");
      reading.append(window.dpSiteCopySpan("explore.point_reading",
        {property: panel.features[feature.value], x: spanOf(same.map(row => row.x)), y: spanOf(same.map(row => row.y))}));
      facts.append(reading);
      const counts = new Map();
      same.forEach(row => {
        const seen = counts.get(row.algorithm.id) || {rows: 0, instances: new Set()};
        seen.rows += 1;
        seen.instances.add(row.instance.id);
        counts.set(row.algorithm.id, seen);
      });
      counts.forEach(({rows: count, instances: on}, id) => {
        const entry = document.createElement("div");
        const tally = document.createElement("span");
        tally.className = "text-secondary";
        const counted = median() ? ["explore.point_median", {}]
          : [count === 1 ? "explore.trial_count_one" : "explore.trial_count", {count: count.toLocaleString()}];
        tally.append(" · ", window.dpSiteCopy(...counted));
        if (on.size > 1) tally.append(" ", window.dpSiteCopy("explore.point_on_instances", {count: on.size.toLocaleString()}));
        const entryName = panel.algorithms.find(algorithm => algorithm.id === id);
        entry.append(entryName ? entryName.label : id, tally);
        facts.append(entry);
      });
      output.replaceChildren(facts);
      const instances = [...new Map(same.map(row => [row.instance.id, row.instance])).values()];
      const SHOWN = 3;
      if (instances.length > SHOWN) {
        const more = document.createElement("div");
        more.className = "dp-point-more text-secondary";
        more.append(window.dpSiteCopySpan("explore.point_more_instances", {count: (instances.length - SHOWN).toLocaleString()}));
        output.append(more);
      }
      instances.slice(0, SHOWN).forEach(instance => {
        const href = new URL(`../instances/${instance.id}/`, state.index.location).href;
        window.dpInstanceCard(href).then(card => card, () => {
          const label = document.createElement("code");
          label.textContent = instance.label;
          return label;
        }).then(shown => {
          if (click !== clicks) return;
          const more = output.querySelector(".dp-point-more");
          if (more) more.before(shown); else output.append(shown);
        });
      });
    }

    /* The opening x range holds every placed instance property (fittedRange). */
    function fittedX() {
      return window.dpPanel.fittedRange(
        rows.filter(row => placed(row.x, 0)).map(row => coordinate(row.x, 0)),
        shipped.reading.RANGE_PADDING[0]);
    }
    /* The opening y range (openingRatioRange): Tukey's far-out fences around
       the weighted quartiles of the plotted error ratios, on the axis's own
       scale, always holding a ratio of 1; a linear axis starts at 0. Each
       instance's weight splits across one algorithm's points there, as the
       running median's does; values beyond the range sit on the plot's edge. */
    function robustY() {
      const perCell = new Map();
      rows.forEach(row => {
        const cell = `${row.algorithm.id}/${row.column}`;
        perCell.set(cell, (perCell.get(cell) || 0) + 1);
      });
      const sample = rows.filter(row => placed(row.y, 1)).map(row => ({
        value: coordinate(row.y, 1),
        weight: weightOf(row.column) / perCell.get(`${row.algorithm.id}/${row.column}`),
      }));
      return window.dpPanel.openingRatioRange(sample, shipped.reading.RANGE_FENCE,
        shipped.reading.RANGE_PADDING[1], logY());
    }

    /* The scatter's axis titles, as drawn and as its notebook writes them. */
    function xTitle() {
      return scale.value === "quantile"
        ? copy("explore.scatter_x_quantile", {feature: panel.features[feature.value]})
        : panel.features[feature.value];
    }
    function yTitle() {
      return baseline.value !== 'none'
        ? copy("explore.scatter_y_axis", {baseline: panel.baselines[baseline.value]})
        : panel.metric.replaceAll("_", " ");
    }

    function load() {
      if (!root.open || section.hidden || section.dataset.loading === 'true') return;
      state = section.dpState;
      if (!state) {
        status.replaceChildren(window.dpSiteCopySpan('explore.comparison_loading'));
        return;
      }
      configure(state);
      if (feature.value in panel.feature_unavailable) {
        plot.hidden = true;
        hideHistogram(true);
        const reason = document.createElement("span");
        window.dpSiteCopyFill(reason, panel.feature_unavailable[feature.value].reason);
        status.replaceChildren(reason);
        return;
      }
      ensureShown();
      wanted = panel.algorithms.filter(algorithm => selected.has(algorithm.id) && drawable(algorithm));
      renderRoster();
      vector = window.dpPanel.resolveBaseline(state.panel, state.baseline, state.index);
      resolvedFor = [state.panel, state.baseline];
      reader = window.dpPanel.featureReader(state.index, state.panel, feature.value);
      rows = [];
      wanted.forEach(algorithm => {
        const row = rowOf(algorithm.id);
        if (row < 0) return;
        window.dpPanel.featurePoints(
          state.panel, row, state.instances, vector, state.aggregation, reader,
        ).forEach(point => {
          rows.push({
            algorithm: algorithm,
            instance: panel.instances[point.instance],
            column: point.instance,
            seed: point.seed === null ? null : state.panel.seedIds[point.seed],
            x: point.feature,
            y: point.value,
          });
        });
      });
      initial = [fittedX(), robustY()];
      if (!ranges) ranges = initial.map(range => range.slice());
      output.innerHTML = inspection;
      repaint();
    }

    function repaint() {
      drawing = drawing.then(draw).then(() => { if (retry) retry.hidden = true; }).catch(error => {
        status.replaceChildren(window.dpSiteCopySpan('explore.comparison_draw_error', {error: error.message}));
        // Retry only reaches the reader on an actual failed load, never as a
        // standing link beside a working chart.
        if (retry) retry.hidden = false;
        console.error(error);
      });
    }

    async function draw() {
      if (!root.open || section.hidden || section.dataset.loading === 'true' || !ranges) return;
      const showChart = subview() === "chart";
      showSelection();
      // Scatter-only chrome (the seed/median note, the borders note, the
      // Selected point details disclosure) never applies to the histogram.
      scatterOnly.forEach(el => { el.hidden = !showChart; });
      histogramOnly.forEach(el => { el.hidden = showChart; });
      // Algorithm names live in each chart's own selection, not repeated in the status.
      const counted = (one, many, count) => copy(count === 1 ? one : many, {count: count.toLocaleString()});
      status.textContent = showChart
        ? (median()
          ? counted("explore.median_point_count_one", "explore.median_point_count", rows.length)
          : counted("explore.trial_count_one", "explore.trial_count", rows.length)) +
          " · " + counted("explore.status_selected_algorithms_one", "explore.status_selected_algorithms", wanted.length)
        : counted("explore.status_instances_one", "explore.status_instances", state.instances.length);
      const nothingChosen = !showChart && !outcomesPair().a;
      histogramEmpty.hidden = !nothingChosen;
      // Selecting nothing is a valid state on both tabs: the chart draws
      // empty and says so, the same copy success rate uses, with the
      // selection and controls left in place.
      const blank = !state.instances.length
        ? "explore.empty_no_instances"
        : !showChart ? ""
        : !wanted.length ? "explore.empty_no_algorithms"
        : !rows.length ? "explore.empty_no_feature_values"
        : "";
      emptyStatus.hidden = !blank;
      if (blank) window.dpSiteCopyFill(emptyStatus, blank);
      // One shared container draws both views, so it stays visible for
      // either; only which traces it holds differs.
      plot.hidden = Boolean(blank) || nothingChosen;
      if (plot.hidden) { hideHistogram(true); return; }
      if (!showChart) {
        // Trial outcomes draws in the shared container.
        await drawHistogram();
        return;
      }
      if (scatterAriaLabel) plot.setAttribute("aria-label", scatterAriaLabel);
      const style = getComputedStyle(plot);
      const token = name => style.getPropertyValue(name).trim();
      const ink = style.color, grid = token("--dp-chart-grid");
      const look = shipped.style.layout.meta;
      const traces = [], annotations = [];
      groups = new Map();
      const totals = new Map();
      /* Where a ratio sits: inside the view, past its bottom or top edge (a
         finite value the view leaves out), or one the axis cannot place (∞,
         or 0 on a log axis). Everything below the view sits on the bottom
         edge and everything above it on the top one; only the ratio axis
         holds values beyond the view, so a property value outside the view
         is not drawn. */
      const sideOf = value => {
        if (!placed(value, 1)) return value === Infinity || !logY() ? "inf" : "zero";
        const at = coordinate(value, 1);
        return at < ranges[1][0] ? "below" : at > ranges[1][1] ? "above" : "in";
      };
      const EDGE = {zero: 0, below: 0, above: 1, inf: 1};
      const placeOf = row => {
        const side = sideOf(row.y);
        const at = [scale.value === "quantile" ? coordinate(row.x, 0) : row.x, side !== "in" ? EDGE[side] : row.y];
        return {side, at, key: `${side}|${at[0]}|${at[1]}`};
      };
      const inView = row => {
        if (!placed(row.x, 0)) return false;
        const at = coordinate(row.x, 0);
        return at >= ranges[0][0] && at <= ranges[0][1];
      };
      rows.forEach(row => {
        totals.set(row.algorithm.id, (totals.get(row.algorithm.id) || 0) + 1);
        if (!inView(row)) return;
        const key = placeOf(row).key;
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(row);
      });
      const across = ranges[0].map(value => scale.value === "log" ? 10 ** value : value);
      traces.push({type: "scatter", mode: "lines", showlegend: false, hoverinfo: "skip",
        x: across, y: [1, 1], line: {...look.reference_line, color: ink}});
      const active = panel.algorithms.filter(algorithm => totals.has(algorithm.id));
      const colours = new Map(active.map(algorithm => [algorithm.id, colour(algorithm.id)]));
      // Running weighted median: RUNNING_POSITIONS positions evenly spaced
      // in the full set's weight-quantile space
      // (quantileMap, also the histogram's own bin-edge mapping), a
      // RUNNING_WINDOW-of-benchmark-weight window around each, and the study-weighted
      // median/25th/75th percentile of the plotted error over the window.
      // A failure counts as +Infinity, so a failure-majority window draws
      // at the axis's own ∞ border, as an edge/corner scatter point does.
      // The window is always defined in weight units; only the plotted x
      // differs per scale, matching how the scatter itself places points.
      const qmap = quantileMap(feature.value);
      const xOf = q => scale.value === "quantile" ? q : valueAtQuantile(feature.value, q);
      const fromY = c => logY() ? 10 ** c : c;
      const yTop = fromY(ranges[1][1]), yBottom = fromY(ranges[1][0]);
      const onAxis = value => placed(value, 1) ? value : value === Infinity ? yTop : yBottom;
      active.forEach(algorithm => {
        const own = rows.filter(row => row.algorithm.id === algorithm.id)
          .map(row => ({instance: row.column, feature: row.x, value: row.y}));
        const running = window.dpPanel.runningMedian(window.dpPanel.featureSample(own, qmap, weightOf),
          shipped.reading.RUNNING_WINDOW, shipped.reading.RUNNING_POSITIONS);
        const shown = running.median.map(value => value !== null);
        const xs = running.q.map((q, index) => shown[index] ? xOf(q) : null);
        const medians = running.median.map(value => value === null ? null : onAxis(value));
        const lowers = running.lower.map(value => value === null ? null : onAxis(value));
        const uppers = running.upper.map(value => value === null ? null : onAxis(value));
        const band = resolveColour(
          `color-mix(in srgb, ${colours.get(algorithm.id)} ${look.running_median.band_opacity * 100}%, transparent)`);
        traces.push({type: "scatter", mode: "lines", showlegend: false, legendgroup: algorithm.id,
          hoverinfo: "skip", x: xs, y: uppers, line: {width: 0}, connectgaps: false});
        traces.push({type: "scatter", mode: "lines", showlegend: false, legendgroup: algorithm.id,
          hoverinfo: "skip", x: xs, y: lowers, fill: "tonexty", fillcolor: band,
          line: {width: 0}, connectgaps: false});
        const runningName = escape(copy("explore.running_median", {algorithm: named(algorithm.id)}));
        traces.push({type: "scatter", mode: "lines", showlegend: false, legendgroup: algorithm.id,
          name: runningName,
          x: xs, y: medians, line: {color: colours.get(algorithm.id), width: look.running_median.width}, connectgaps: false,
          hovertemplate: `${runningName}<extra></extra>`});
      });
      /* One algorithm's trials on one instance with the same ratio are one
         point (dpPanel.collapsedCount); every marker's area grows with the
         points it stands for, on one scale for the whole chart: a single
         point is the smallest dot, the busiest spot the largest, wherever
         each sits. Hovers and the point panel count trials. */
      const collapsed = (group, id) => window.dpPanel.collapsedCount(
        group.filter(row => row.algorithm.id === id).map(row => ({instance: row.column, value: row.y})));
      let busiest = 1;
      groups.forEach(group => {
        new Set(group.map(row => row.algorithm.id)).forEach(id => { busiest = Math.max(busiest, collapsed(group, id)); });
      });
      const diameter = count => busiest > 1
        ? look.marker.size + look.marker.size_span * Math.sqrt((count - 1) / (busiest - 1)) : look.marker.size;
      const hover = new Map();
      const clickHint = window.dpSiteCopy("explore.scatter_click_hint")
        ? "<br>" + escape(window.dpSiteCopy("explore.scatter_click_hint")) : "";
      groups.forEach((group, key) => {
        const count = group.length;
        const beyond = ["below", "above"].includes(placeOf(group[0]).side);
        hover.set(key, escape(copy("explore.point_hover",
          {property: panel.features[feature.value], x: spanOf(group.map(row => row.x)), y: spanOf(group.map(row => row.y))})) +
          (beyond ? "<br>" + escape(copy("explore.point_beyond_view")) : "") + "<br>" +
          escape(window.dpSiteCopy(median() ? (count === 1 ? "explore.median_point_count_one" : "explore.median_point_count")
            : (count === 1 ? "explore.trial_count_one" : "explore.trial_count"), {count: count.toLocaleString()})) +
          clickHint);
      });
      active.forEach(algorithm => {
        const bySymbol = new Map();
        groups.forEach((group, key) => {
          const own = group.filter(row => row.algorithm.id === algorithm.id);
          if (!own.length) return;
          const {side, at} = placeOf(own[0]);
          // A value beyond the view is a filled dot on its edge; ∞ and 0 an open ring.
          const symbol = side === "in" ? treatment(algorithm).marker
            : side === "below" || side === "above" ? "circle" : "circle-open";
          const trace = `${side === "in"}|${symbol}`;
          if (!bySymbol.has(trace)) bySymbol.set(trace, {inside: side === "in", symbol, points: []});
          bySymbol.get(trace).points.push({own, at, key, count: collapsed(group, algorithm.id)});
        });
        const colour = colours.get(algorithm.id);
        bySymbol.forEach(({inside, symbol, points}) => {
          const ring = symbol === "circle-open";
          traces.push({type: "scatter", mode: "markers", showlegend: false, legendgroup: algorithm.id,
            name: escape(named(algorithm.id)), xaxis: "x", yaxis: inside ? "y" : "y2", cliponaxis: false,
            x: points.map(point => point.at[0]), y: points.map(point => point.at[1]),
            marker: {color: colour, symbol,
              opacity: inside ? look.marker.opacity : look.marker.edge_opacity,
              size: points.map(point => diameter(point.count)), sizemode: "diameter",
              line: {width: ring ? look.marker.ring_width : 0}},
            customdata: points.map(point => point.key),
            hovertext: points.map(point => hover.get(point.key)),
            hovertemplate: "%{hovertext}<extra></extra>"});
        });
      });
      // One count per edge, per algorithm in its colour, at the edge's end.
      const edges = new Map();
      groups.forEach(group => {
        const {side, at} = placeOf(group[0]);
        if (side === "in") return;
        if (!edges.has(at[1])) edges.set(at[1], new Map());
        const counts = edges.get(at[1]);
        group.forEach(row => counts.set(row.algorithm.id, (counts.get(row.algorithm.id) || 0) + 1));
      });
      edges.forEach((counts, y) => {
        annotations.push({xref: "paper", yref: "paper", x: look.edge_count.at, y,
          xanchor: "right", yanchor: y < 0.5 ? "bottom" : "top",
          yshift: y < 0.5 ? look.edge_count.shift : -look.edge_count.shift, showarrow: false,
          text: active.filter(algorithm => counts.has(algorithm.id)).map(algorithm =>
            `<span style="color:${colours.get(algorithm.id)}">${counts.get(algorithm.id)}</span>`).join(" · "),
          font: {color: ink}});
      });
      const width = plot.clientWidth;
      /* The shipped chart style is the layout's defaults; the reader's theme
         paints its colours over them. */
      // The template sets the axes' ticks; a narrow chart takes fewer.
      const axis = {linecolor: ink, gridcolor: grid, ...(width < 450 ? {nticks: 3} : {}),
        // At 375px the log axis's minor-tick labels are too small to read.
        ...(width < 480 ? {minor: {showticklabels: false}} : {})};
      const featureQuantile = scale.value === "quantile";
      const layout = {
        template: shipped.style,
        width, height: width < 480 ? 320 : Math.min(520, Math.max(350, width * 0.6)), autosize: true,
        plot_bgcolor: token("--dp-chart-plot"), font: {color: ink},
        xaxis: {...axis, type: scale.value, range: ranges[0],
          ...(featureQuantile ? quantileAxisExtra(feature.value, plot.clientWidth < 480) : {}),
          title: {text: escape(xTitle())}},
        yaxis: {...axis, type: logY() ? "log" : "linear", range: ranges[1], title: {text: escape(yTitle())}},
        yaxis2: {overlaying: "y", range: [0, 1], visible: false, fixedrange: true}, annotations,
      };
      // The clicked point's ring, drawn last so it sits over the markers; a
      // redraw clears it, and the panel returns to its prompt.
      traces.push({type: "scatter", mode: "markers", x: [], y: [], xaxis: "x", yaxis: "y", showlegend: false,
        hoverinfo: "skip", cliponaxis: false, marker: {symbol: "circle-open", size: 20, color: ink, line: {width: 2}}});
      ring = traces.length - 1;
      clicks += 1;
      showPrompt();
      hideHistogram(true);
      renderRoster();
      await plotView(traces, layout, initial);
    }
    baseline.addEventListener("change", () => { reopen(); load(); });
    aggregation.addEventListener('change', () => { reopen(); load(); });
    [feature, scale].forEach(control => control.addEventListener("change", () => { reopen(); load(); }));
    if (yScale) yScale.addEventListener("change", () => { reopen(); load(); });
    root.querySelector("[data-comparison-retry]").addEventListener("click", load);
    page.addEventListener('dp-comparison-selection', () => {
      // Every metric section's panel shares this event, including ones never
      // opened yet, whose `panel` is still null because configure() has not run.
      if (panel) renderRoster();
      load();
    });
    root.addEventListener("toggle", event => { if (event.target === root) load(); });
    // Switching between "Error ratio" and "Trial outcomes" redraws
    // the now-visible chart from the same loaded rows; it never refetches.
    root.addEventListener("dp-comparison-subview", () => {
      showSelection();
      if (plotted) repaint();
    });
    page.addEventListener("dp-panel-ready", event => {
      if (event.detail.section === section) load();
    });
    /* The page's one threshold. The scatter does not read it; the outcome
       histogram has no τ axis, so it takes a slider on the page's τ grid,
       whose bars follow it as it moves. */
    const tauNow = () => Number(profile.dataset.threshold);
    if (page.matches("[data-playground]") && window.dpTauPacer) {
      // The histogram costs a few milliseconds a step, so its bars follow the slider.
      const pacer = window.dpTauPacer(root, true);
      const control = document.createElement("label");
      control.className = "dp-hist-tau";
      control.innerHTML = ' <input type="range" min="0" step="1"> <output></output>';
      control.prepend(window.dpSiteCopySpan("shared.threshold_label"));
      const slider = control.querySelector("input");
      const readout = control.querySelector("output");
      root.querySelector("[data-comparison-feature-controls]").append(control);
      const grid = () => page.dpTaus || [];
      const nearest = tau => grid().reduce((best, each, index) =>
        Math.abs(each - tau) < Math.abs(grid()[best] - tau) ? index : best, 0);
      const sync = () => {
        control.hidden = subview() !== "outcomes";
        if (!grid().length || !finite(tauNow())) return;
        slider.max = String(grid().length - 1);
        slider.value = String(nearest(tauNow()));
        readout.textContent = window.dpTau(grid()[Number(slider.value)]);
      };
      slider.addEventListener("input", () => {
        const tau = grid()[Number(slider.value)];
        readout.textContent = window.dpTau(tau);
        pacer.request(tau);
      });
      slider.addEventListener("change", () => pacer.settle());
      section.addEventListener("dp-threshold", sync);
      root.addEventListener("dp-comparison-subview", sync);
      sync();
    }
    // Only the histogram reads τ; the scatter catches up when it is shown,
    // since switching views redraws.
    section.addEventListener('dp-threshold', () => { if (subview() === "outcomes") repaint(); });
    new MutationObserver(() => {
      if (panel) renderRoster();
      repaint();
    }).observe(document.body, {attributes: true, attributeFilter: ["data-bs-theme"]});
    // The chart redraws when its own width changes: the window, the sidebar,
    // or the bar panel beside it appearing. A height change needs no redraw.
    let observedWidth = 0;
    new ResizeObserver(([entry]) => {
      const width = entry.contentRect.width;
      if (!width || width === observedWidth) return;
      observedWidth = width;
      if (plotted) requestAnimationFrame(repaint);
    }).observe(plot);
  }
  function init() { document.querySelectorAll(".dp-comparison").forEach(connect); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
