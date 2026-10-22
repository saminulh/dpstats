/* Move a Leaderboard's standings to the metric, budget and threshold its
   controls name.

   The page ships one index and, per panel, one dense matrix of median cost (or,
   for a release, its published curves verbatim). This file derives what a
   control chooses: the curve each algorithm draws, the share each keeps within
   the selected threshold, and the ranking over those shares. Each derivation is `panel-data.js`'s mirror of the library function
   that owns the same reading.

   Cells are filled as text rather than markup, so a label or value can never be
   read as HTML. The page is already correct without this file: every metric's
   panels ship rendered at their opening threshold, which is what a reader
   without scripting sees. */
(function () {
  "use strict";

  /* A Leaderboard reads each Metric against the one baseline its view
     names, so the catalogue's default is the only choice there is. */
  function officialBaseline(index, metric) {
    return index.baselines[metric].default;
  }

  function cell(className, text, exact) {
    var td = document.createElement("td");
    td.className = className;
    td.textContent = text;
    if (exact) td.title = exact;
    return td;
  }

  /* The columns a table declares, which is the only list a row is built from.
     A table that declares none is read from its own header, so the row and the
     header can never disagree. */
  function columnsOf(body) {
    var table = body.closest ? body.closest("table") : body.parentNode;
    if (!table) return [];
    var declared = table.getAttribute("data-columns");
    if (declared) return declared.split(",");
    var heads = table.tHead ? table.tHead.rows[0].cells : [];
    return [].slice.call(heads).map(function (_, position) {
      return String(position);
    });
  }

  function loadError(container, error) {
    var message = container.querySelector(':scope > .dp-load-error');
    if (!message) {
      message = document.createElement('p');
      message.className = 'dp-load-error';
      message.setAttribute('role', 'alert');
      container.prepend(message);
    }
    message.replaceChildren(window.dpSiteCopySpan('leaderboard.results_load_error', {error: error.message}));
  }

  function spec(section) {
    var tag = section.querySelector("script[data-results-section]");
    if (!tag.dpPayload) tag.dpPayload = JSON.parse(tag.textContent);
    return tag.dpPayload;
  }

  function payloadTag(section, chart, budget) {
    var selector =
      'script[data-profile-chart="' + chart + '"][data-budget="' + budget + '"]';
    var tag = section.querySelector(selector);
    if (!tag) {
      tag = document.createElement("script");
      tag.type = "application/json";
      tag.dataset.profileChart = chart;
      tag.dataset.budget = budget;
      section.appendChild(tag);
    }
    return tag;
  }

  /* One panel's curves: a release publishes them, and a measured store ships the
     matrices they are traced from — one median per instance, or every seed's own
     cost where a reader asks for individual trials. */
  function curvesOf(index, panel, aggregation) {
    var curves = {};
    if (panel.published) {
      index.algorithms.forEach(function (entry) {
        var published = panel.published[entry.uuid];
        if (published) curves[entry.uuid] = Float64Array.from(published);
      });
      return { curves: curves, taus: panel.taus };
    }
    var columns = [];
    for (var column = 0; column < panel.instances; column += 1) columns.push(column);
    var vector = window.dpPanel.denominators(panel, officialBaseline(index, panel.metric));
    index.algorithms.forEach(function (entry, row) {
      var traced = window.dpPanel.curve(
        panel,
        row,
        columns,
        vector,
        aggregation,
        index.taus,
        index.weights
      );
      if (traced.instances) curves[entry.uuid] = traced.fractions;
    });
    return { curves: curves, taus: index.taus };
  }

  function connect(page) {
    var sections = [].slice.call(page.querySelectorAll(".dp-metric"));
    var slider = page.querySelector('.dp-filters input[type="range"]');
    var readout = page.querySelector(".dp-filters output");
    if (!sections.length || !slider || !readout) return;
    window.dpPanel
      .loadIndex(page.dataset.index)
      .then(function (index) {
        start(page, index, sections, slider, readout);
      })
      .catch(function (error) {
        loadError(page, error);
        console.error("Leaderboard could not load its index: " + error.message);
      });
  }

  function start(page, index, sections, slider, readout) {
    var roster = window.dpPanel.roster(index).filter(function (row) {
      return !row.diagnostic;
    });
    var byUuid = {};
    roster.forEach(function (row) { byUuid[row.uuid] = row; });
    var taus = index.taus;
    // The chart's tau tab previews on the same grid the standings are read at.
    page.dpTaus = taus.map(Number);
    var loaded = {};
    var files = {};
    var requests = {};
    /* The Leaderboard reads every trial; the median-over-trials reading is
       Explore's. */
    function aggregation() {
      return "individual";
    }

    var controls = window.dpResultsControls(page, sections, index);
    var metricSelect = controls.select;

    function active() {
      for (var position = 0; position < sections.length; position += 1) {
        if (!sections[position].hidden) return sections[position];
      }
      return sections[0];
    }

    function budgetOf(section) {
      return section.querySelector(".dp-chart").dataset.budget;
    }

    /* The official slice the standings are read in: "all" of the task's
       Instances, or one the release names. The slice selector holds it. */
    var sliceSelect = page.querySelector("[data-slice-select]");
    var slices = (index.slices || []).map(function (entry) { return entry.key; });
    var slice = "all";
    var sliceSize = sliceSelect && sliceSelect.querySelector("[data-slice-sizes]");
    var sliceSizes = sliceSize ? JSON.parse(sliceSize.dataset.sliceSizes) : {};
    // The selected slice's size, beside the selector, follows the selection.
    function showSliceSize() {
      if (sliceSize && sliceSizes[slice]) sliceSize.textContent = sliceSizes[slice];
    }
    function sliceReadings(key) {
      return (index.headline[slice] || {})[key] || {};
    }

    /* The index owns every data path; a section names only its own budgets. */
    function panelOf(section, budget) {
      return index.panels.filter(function (panel) {
        return panel.metric === section.dataset.metric && panel.budget === budget &&
          (panel.slice || "all") === slice;
      })[0];
    }

    function source(panel) {
      return panel.curves || panel.runs;
    }

    /* One curve set per panel and aggregation, kept once it is traced. */
    function key(panel) {
      return source(panel) + "/" + aggregation();
    }

    function held(section) {
      return loaded[key(panelOf(section, budgetOf(section)))];
    }

    /* Each curve's reading at quantile p, through the one ranking rule. At
       the protocol's quantile it is the exact reading the release ships; at
       any other p, the grid's (exploration only). The share reached and the
       τ that reaches it do not depend on p, so they are always the shipped ones. */
    function readingsAt(section, traced, p) {
      var curves = traced.curves;
      var grid = traced.taus.map(Number);
      var shipped = sliceReadings(section.dataset.metric + "/" + budgetOf(section));
      var readings = {};
      Object.keys(curves).forEach(function (uuid) {
        if (!shipped[uuid]) return;
        var exact = window.dpPanel.shippedReading(shipped[uuid]);
        readings[uuid] = p === Number(index.headline_quantile)
          ? exact
          : window.dpPanel.readingAt({ x: grid, y: curves[uuid] }, p, [exact[1], exact[2]]);
      });
      return readings;
    }

    // Official ranks always use the complete eligible pool, independent of plot filters.
    function renderRows(body, readings) {
      var ranked = roster
        .filter(function (row) {
          return readings[row.uuid];
        })
        .map(function (row) {
          return { row: row, reading: readings[row.uuid] };
        });
      ranked.sort(function (left, right) {
        var order = window.dpPanel.compareReadings(left.reading, right.reading);
        if (order) return order;
        if (left.row.algorithm < right.row.algorithm) return -1;
        return left.row.algorithm > right.row.algorithm ? 1 : 0;
      });
      ranked.forEach(function (entry) {
        entry.rank =
          1 +
          ranked.filter(function (other) {
            return window.dpPanel.compareReadings(other.reading, entry.reading) < 0;
          }).length;
      });
      body.textContent = "";
      if (!ranked.length) {
        var empty = document.createElement("tr");
        var emptyCell = cell("dp-empty-row", "");
        var emptyText = document.createElement("span");
        if (window.dpSiteCopyFill(emptyText, "leaderboard.empty_panel")) emptyCell.append(emptyText);
        emptyCell.colSpan = columnsOf(body).length;
        empty.appendChild(emptyCell);
        body.appendChild(empty);
      }
      var columns = columnsOf(body);
      ranked.forEach(function (entry) {
        var row = entry.row;
        var tr = document.createElement("tr");
        tr.dataset.uuid = row.uuid;
        tr.dataset.reading = JSON.stringify(entry.reading.map(String));
        /* The table says which columns it has; a row never decides for itself,
           so a surface that renders fewer columns server-side keeps them. */
        columns.forEach(function (column) {
          if (column === "rank") {
            tr.appendChild(cell("dp-rank", entry.rank || "—"));
          } else if (column === "algorithm") {
            tr.appendChild(cell("dp-alg", row.algorithm));
          } else if (column === "headline") {
            tr.appendChild(
              cell("dp-value", window.dpHeadlineText(entry.reading), String(entry.reading[0]))
            );
          }
        });
        body.appendChild(tr);
      });
    }

    /* The slider holds the quantile p in whole percent. Moving it also
       reselects the leaders unless selection is custom. */
    function quantile() {
      return Number(slider.value) / 100;
    }
    function show(percent) {
      percent = Math.max(
        Number(slider.min),
        Math.min(Number(slider.max), percent)
      );
      slider.value = String(percent);
      var section = active();
      var budget = budgetOf(section);
      var traced = held(section);
      var p = quantile();
      if (traced) {
        var readings = readingsAt(section, traced, p);
        var bodies = [].slice.call(
          section.querySelectorAll(".dp-budget:not([hidden]) .dp-leaderboard tbody")
        );
        bodies.forEach(function (body) {
          renderRows(body, readings);
        });
        drawn(section, budget);
        section.dispatchEvent(new CustomEvent("dp-standings"));
      }
      readout.textContent = window.dpPercent(p) + "%";
      readout.title = String(p);
      section.dispatchEvent(
        new CustomEvent("dp-quantile", {
          bubbles: true,
          detail: { p: p },
        })
      );
    }

    function drawn(section, budget) {
      var panel = panelOf(section, budget);
      var traced = loaded[key(panel)];
      if (!traced) return;
      var chart = spec(section).chart;
      // The curves do not depend on the threshold: once this panel is drawn,
      // a threshold move only re-ranks and flips the curves that enter or
      // leave the shown set.
      var reading = key(panel) + "|" + budget;
      var element = document.getElementById(chart);
      if (section.dpDrawnReading === reading && element) {
        var labels = {};
        var all = readingsAt(section, traced, quantile());
        var readings = {};
        roster.forEach(function (row) {
          if (!all[row.uuid]) return;
          readings[row.uuid] = all[row.uuid];
          labels[row.uuid] = row.algorithm;
        });
        if (window.dpRerank(section, controls, element, readings, labels)) return;
      }
      section.dpDrawnReading = reading;
      var readings = readingsAt(section, traced, quantile());
      var shipped = sliceReadings(section.dataset.metric + "/" + budget);
      var series = roster
        .filter(function (row) { return readings[row.uuid]; })
        .map(function (row) {
          var exact = window.dpPanel.shippedReading(shipped[row.uuid]);
          return {
            key: row.uuid,
            label: row.algorithm,
            slot: row.slot,
            diagnostic: row.diagnostic,
            core: row.core,
            interpretation: row.interpretation,
            marker: row.marker,
            fractions: traced.curves[row.uuid],
            headline: exact,
            reached: [exact[1], exact[2]],
          };
        });
      // Eligible competitors open drawn best first.
      series.sort(function (left, right) {
        var order = window.dpPanel.compareReadings(readings[left.key], readings[right.key]);
        if (order) return order;
        return left.label < right.label ? -1 : left.label > right.label ? 1 : 0;
      });
      controls.selectTop(section, series.map(function (entry) { return entry.key; }));
      payloadTag(section, chart, budget).dpPayload = window.dpPanel.figure({
        title: budget,
        series: series,
        taus: traced.taus,
        leading: roster.length, // Visibility comes from the shared selection.
        units: window.dpPanel.units(
          index,
          section.dataset.metric,
          officialBaseline(index, section.dataset.metric)
        ),
        guide: quantile(),
        quantile: Number(index.headline_quantile),
        tauScale: index.tau_scale,
      });
      document.dispatchEvent(
        new CustomEvent("dp-chart", { detail: { chart: chart } })
      );
    }

    function fetchPanel(section) {
      var previousError = section.querySelector(':scope > .dp-load-error');
      if (previousError) previousError.remove();
      var budget = budgetOf(section);
      var panel = panelOf(section, budget);
      var reading = key(panel);
      if (loaded[reading]) {
        section.dataset.loading = "false";
        show(Number(slider.value));
        return;
      }
      section.dataset.loading = "true";
      if (requests[reading]) return;
      requests[reading] = (
        files[source(panel)] ||
        window.dpPanel.panelOf(index, panel).then(function (loadedPanel) {
          files[source(panel)] = Promise.resolve(loadedPanel);
          return loadedPanel;
        })
      )
        .then(function (loadedPanel) {
          loaded[reading] = curvesOf(index, loadedPanel, aggregation());
          section.dataset.loading = "false";
          if (section === active() && reading === key(panelOf(section, budgetOf(section)))) {
            show(Number(slider.value));
          }
        })
        .catch(function (error) {
          section.dataset.loading = "false";
          loadError(section, error);
          console.error(
            "Leaderboard could not load " +
              section.dataset.metric +
              " / " +
              budget +
              ": " +
              error.message
          );
        })
        .finally(function () { delete requests[reading]; });
    }

    function select(metric) {
      sections.forEach(function (section) {
        section.hidden = section.dataset.metric !== metric;
      });
      metricSelect.value = metric;
      var section = active();
      controls.show(section);
      show(Number(slider.value));
      fetchPanel(section);
    }

    metricSelect.addEventListener("change", function () {
      select(metricSelect.value);
    });
    sections.forEach(function (section) {
      window.dpConnectLegend(section, { leading: Number(controls.top.value), rows: roster });
    });
    controls.top.addEventListener("change", function () { show(Number(slider.value)); });
    slider.addEventListener("input", function () {
      show(Number(slider.value));
    });
    // A press or drag on the chart asks for a quantile; the page reads the
    // whole percent nearest it, so the chart and the range input agree.
    page.addEventListener("dp-quantile-seek", function (event) {
      var percent = Math.round(event.detail.value * 100);
      if (percent !== Number(slider.value)) show(percent);
    });
    page.addEventListener("dp-roster-change", function (event) {
      if (event.target === active()) {
        show(Number(slider.value));
      }
    });
    page.addEventListener("dp-budget-change", function () {
      fetchPanel(active());
    });
    /* Choosing a slice reads its own panel; the address names it, replacing
       the entry rather than adding one, so Back still leaves the page. */
    function chooseSlice(key) {
      if (slices.indexOf(key) < 0 || key === slice) return;
      slice = key;
      if (sliceSelect) {
        var radio = sliceSelect.querySelector('input[value="' + key + '"]');
        if (radio) radio.checked = true;
      }
      showSliceSize();
      var query = new URLSearchParams(window.location.search);
      if (key === "all") query.delete("slice"); else query.set("slice", key);
      var text = query.toString();
      try {
        window.history.replaceState(window.history.state, "",
          window.location.pathname + (text ? "?" + text : "") + window.location.hash);
      } catch (error) { /* sandboxed */ }
      fetchPanel(active());
    }
    if (sliceSelect) {
      sliceSelect.addEventListener("change", function (event) {
        if (event.target.name) chooseSlice(event.target.value);
      });
    }
    // The controls are inert until this point, so the chrome around them stays
    // hidden until they can actually move the standings.
    page.classList.add("is-interactive");
    // The page opens on the protocol's quantile, which the slider's markup holds.
    /* A link names the comparison it means: ?metric=&budget= open on it,
       and a value the page does not offer is ignored. */
    var wanted = new URLSearchParams(window.location.search);
    var opened = sections.filter(function (section) {
      return section.dataset.metric === wanted.get("metric");
    })[0];
    var budget = wanted.get("budget");
    if (budget) {
      sections.forEach(function (section) {
        if (section.dpSelectBudget) section.dpSelectBudget(budget, false);
      });
    }
    if (slices.indexOf(wanted.get("slice")) >= 0) {
      slice = wanted.get("slice");
      var opening = sliceSelect && sliceSelect.querySelector('input[value="' + slice + '"]');
      if (opening) opening.checked = true;
      showSliceSize();
    }
    select((opened || active()).dataset.metric);
  }

  function init() {
    document.querySelectorAll("[data-leaderboard]").forEach(connect);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
