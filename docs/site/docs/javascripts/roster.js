// A task's Roster landing: search, sort and three facets over the one table
// the page already renders in full. The state lives in the query string
// (?class=&core=&transformation=&q=&sort=), read on load and replaced on
// every change, so a filtered view can be linked. Comma-separated values are
// alternatives within a facet; facets combine with AND.
//
// A transformation key names one transformation class on this Roster: its
// plain class name, or `Name@task_slug` for a same-named class another task
// declares. A plain name the facet does not list matches every class of
// that name.
(function () {
  "use strict";

  var FACETS = ["class", "core", "transformation"];
  /* Each facet's active-filter chip, a slot naming the chosen value. */
  var LABELS = { "class": "roster.active_class", core: "roster.active_core", transformation: "roster.active_transformation" };

  function split(value) {
    return value
      ? value.split(",").map(function (item) { return item.trim(); }).filter(Boolean)
      : [];
  }

  function wire(box) {
    var controls = box.querySelector("[data-dp-roster-controls]");
    var tbody = box.querySelector("tbody");
    var search = box.querySelector("[data-dp-roster-search]");
    var sort = box.querySelector("[data-dp-roster-sort]");
    var coreSelect = box.querySelector("[data-dp-roster-core]");
    var active = box.querySelector("[data-dp-roster-active]");
    var count = box.querySelector("[data-dp-roster-count]");
    var fold = box.querySelector("[data-dp-roster-fold]");
    var chips = Array.prototype.slice.call(box.querySelectorAll("[data-dp-facet]"));
    var classOrder = chips
      .filter(function (chip) { return chip.getAttribute("data-dp-facet") === "class"; })
      .map(function (chip) { return chip.getAttribute("data-dp-value"); });
    var listed = {};
    chips.forEach(function (chip) {
      if (chip.getAttribute("data-dp-facet") === "transformation") {
        listed[chip.getAttribute("data-dp-value")] = chip.getAttribute("data-dp-label");
      }
    });
    var rows = Array.prototype.map.call(tbody.rows, function (tr, position) {
      var rank = tr.getAttribute("data-dp-rank");
      var headline = tr.getAttribute("data-dp-headline");
      return {
        tr: tr,
        position: position,
        label: tr.getAttribute("data-dp-label"),
        cls: tr.getAttribute("data-dp-class"),
        core: tr.getAttribute("data-dp-core"),
        keys: split((tr.getAttribute("data-dp-transformations") || "").replace(/ /g, ",")),
        rank: rank ? parseInt(rank, 10) : Infinity,
        // An unranked row has no headline value and sorts after every ranked one.
        headline: headline ? Number(headline) : Infinity,
        search: tr.getAttribute("data-dp-search") || "",
        steps: Array.prototype.slice.call(tr.querySelectorAll("[data-dp-key]"))
      };
    });
    var cores = Array.prototype.map.call(coreSelect.options, function (option) {
      return option.value;
    }).filter(Boolean);
    var state = { "class": [], core: [], transformation: [], q: "", sort: "rank" };

    function keyMatches(key, wanted) {
      if (key === wanted) return true;
      if (listed[wanted] !== undefined || wanted.indexOf("@") >= 0) return false;
      return key.split("@")[0] === wanted;
    }

    function hasTransformation(row, wanted) {
      return row.keys.some(function (key) { return keyMatches(key, wanted); });
    }

    function matches(row, skip) {
      if (skip !== "class" && state["class"].length && state["class"].indexOf(row.cls) < 0) return false;
      if (skip !== "core" && state.core.length && state.core.indexOf(row.core) < 0) return false;
      if (skip !== "transformation" && state.transformation.length &&
          !state.transformation.some(function (wanted) { return hasTransformation(row, wanted); })) {
        return false;
      }
      if (state.q) {
        var words = state.q.toLowerCase().split(/\s+/).filter(Boolean);
        if (!words.every(function (word) { return row.search.indexOf(word) >= 0; })) return false;
      }
      return true;
    }

    function countFor(facet, value) {
      return rows.filter(function (row) {
        if (!matches(row, facet)) return false;
        if (facet === "class") return row.cls === value;
        if (facet === "core") return row.core === value;
        return hasTransformation(row, value);
      }).length;
    }

    function shown(facet, value) {
      if (facet !== "transformation") return value;
      if (listed[value] !== undefined) return listed[value];
      var at = value.indexOf("@");
      return at < 0 ? value : window.dpSiteCopy("roster.other_task_class", {name: value.slice(0, at), task: value.slice(at + 1)});
    }

    function readQuery() {
      var params = new URLSearchParams(window.location.search);
      FACETS.forEach(function (facet) { state[facet] = split(params.get(facet)); });
      state.q = params.get("q") || "";
      var wanted = params.get("sort");
      state.sort = wanted === "headline" || wanted === "name" ? wanted : "rank";
    }

    function writeQuery() {
      var params = new URLSearchParams();
      FACETS.forEach(function (facet) {
        if (state[facet].length) params.set(facet, state[facet].join(","));
      });
      if (state.q) params.set("q", state.q);
      if (state.sort !== "rank") params.set("sort", state.sort);
      var query = params.toString().replace(/%2C/g, ",").replace(/%40/g, "@");
      history.replaceState(null, "", window.location.pathname + (query ? "?" + query : "") +
        window.location.hash);
    }

    function toggle(facet, value) {
      var at = state[facet].indexOf(value);
      if (at >= 0) state[facet].splice(at, 1); else state[facet].push(value);
      update();
    }

    /* One active filter as a chip: *key* is its slot, *value* the filter it names. */
    function removable(key, value, onRemove) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "dp-chip btn btn-sm";
      var text = window.dpSiteCopy(key, {value: value});
      button.setAttribute("aria-label", window.dpSiteCopy("roster.remove_filter", {filter: text}));
      button.appendChild(window.dpSiteCopySpan(key, {value: value}));
      var cross = document.createElement("span");
      cross.setAttribute("aria-hidden", "true");
      cross.textContent = "×";
      button.appendChild(cross);
      button.addEventListener("click", function () {
        onRemove();
        var next = active.querySelector("button");
        (next || search).focus();
      });
      return button;
    }

    function renderActive() {
      active.textContent = "";
      var any = false;
      FACETS.forEach(function (facet) {
        state[facet].slice().forEach(function (value) {
          any = true;
          active.appendChild(removable(LABELS[facet], shown(facet, value), function () {
            toggle(facet, value);
          }));
        });
      });
      if (state.q) {
        any = true;
        active.appendChild(removable("roster.active_search", state.q, function () {
          state.q = "";
          search.value = "";
          update();
        }));
      }
      if (!any) return;
      var label = document.createElement("span");
      label.className = "dp-active-label";
      label.append(window.dpSiteCopySpan("roster.active_filters"));
      active.insertBefore(label, active.firstChild);
      var clear = document.createElement("button");
      clear.type = "button";
      clear.className = "dp-action btn btn-sm";
      clear.append(window.dpSiteCopySpan("roster.clear_all"));
      clear.addEventListener("click", function () {
        FACETS.forEach(function (facet) { state[facet] = []; });
        state.q = "";
        search.value = "";
        update();
        search.focus();
      });
      active.appendChild(clear);
    }

    function renderControls() {
      chips.forEach(function (chip) {
        var facet = chip.getAttribute("data-dp-facet");
        var value = chip.getAttribute("data-dp-value");
        var chosen = state[facet].indexOf(value) >= 0;
        if (chip.type === "checkbox") chip.checked = chosen;
        else chip.setAttribute("aria-pressed", String(chosen));
        // The count is supporting text beside the option, never its name.
        chip.closest(".dp-chip-option, .dp-check-option").querySelector("[data-dp-count]").textContent =
          String(countFor(facet, value));
      });
      var anyOption = coreSelect.options[0];
      while (coreSelect.options.length > 1) coreSelect.remove(1);
      if (state.core.length > 1) {
        var several = document.createElement("option");
        several.value = state.core.join(",");
        several.textContent = state.core.join(", ");
        coreSelect.appendChild(several);
      }
      cores.forEach(function (core) {
        var option = document.createElement("option");
        option.value = core;
        option.textContent = core + " (" + countFor("core", core) + ")";
        coreSelect.appendChild(option);
      });
      coreSelect.value = state.core.join(",");
      if (coreSelect.selectedIndex < 0) anyOption.selected = true;
      sort.value = state.sort;
    }

    function compare(a, b) {
      if (state.sort === "name") return a.label.localeCompare(b.label);
      // The headline ranks lower first; equal values, infinities included, by name.
      if (state.sort === "headline") {
        return (a.headline === b.headline ? 0 : a.headline < b.headline ? -1 : 1) ||
          a.label.localeCompare(b.label);
      }
      return (a.rank - b.rank) || (classOrder.indexOf(a.cls) - classOrder.indexOf(b.cls)) ||
        a.position - b.position;
    }

    var empty = document.createElement("tr");
    var emptyCell = document.createElement("td");
    emptyCell.colSpan = tbody.rows[0] ? tbody.rows[0].cells.length : 1;
    emptyCell.className = "dp-empty";
    emptyCell.append(window.dpSiteCopySpan("roster.empty_filter"));
    empty.appendChild(emptyCell);

    function renderRows() {
      var shownRows = 0;
      var ordered = rows.slice().sort(compare);
      ordered.forEach(function (row) {
        var match = matches(row);
        row.tr.hidden = !match;
        if (match) shownRows += 1;
        row.steps.forEach(function (step) {
          var key = step.getAttribute("data-dp-key");
          step.classList.toggle("is-match", state.transformation.some(function (wanted) {
            return keyMatches(key, wanted);
          }));
        });
        tbody.appendChild(row.tr);
      });
      if (shownRows) {
        if (empty.parentNode) empty.parentNode.removeChild(empty);
      } else {
        tbody.appendChild(empty);
      }
      var showing = rows.length === 1 ? "roster.showing_one" : "roster.showing";
      count.replaceChildren(window.dpSiteCopySpan(showing, {shown: shownRows, total: rows.length}));
    }

    function update() {
      renderControls();
      renderActive();
      renderRows();
      writeQuery();
    }

    chips.forEach(function (chip) {
      chip.addEventListener("click", function () {
        toggle(chip.getAttribute("data-dp-facet"), chip.getAttribute("data-dp-value"));
      });
    });
    coreSelect.addEventListener("change", function () {
      state.core = split(coreSelect.value);
      update();
    });
    sort.addEventListener("change", function () {
      state.sort = sort.value;
      update();
    });
    var timer;
    search.addEventListener("input", function () {
      clearTimeout(timer);
      timer = setTimeout(function () {
        state.q = search.value.trim();
        update();
      }, 150);
    });

    // The facets stay open from 768px up; below it they start folded.
    var wide = window.matchMedia("(min-width: 768px)");
    function syncFold() { fold.open = wide.matches; }
    syncFold();
    if (wide.addEventListener) wide.addEventListener("change", syncFold);

    readQuery();
    search.value = state.q;
    controls.hidden = false;
    update();
  }

  document.querySelectorAll("[data-dp-roster-landing]").forEach(wire);
})();
