// The Corpus index's search, task and field filters, its column sorting, and
// the source page's copy control. Both pages are complete without this file:
// it narrows and reorders rows a reader already has, and reveals the controls
// only once they work. The selection rides the URL, so a filtered view is a
// link someone else can open.
(function () {
  "use strict";

  var KEYS = { q: "", task: "", field: "", sort: "", dir: "" };

  function cellText(row, index) {
    var cell = row.cells[index];
    return cell ? cell.textContent.trim() : "";
  }

  function readState() {
    var params = new URLSearchParams(location.search);
    var state = {};
    Object.keys(KEYS).forEach(function (key) {
      state[key] = params.get(key) || KEYS[key];
    });
    return state;
  }

  function writeState(state) {
    var params = new URLSearchParams(location.search);
    Object.keys(KEYS).forEach(function (key) {
      if (state[key]) params.set(key, state[key]);
      else params.delete(key);
    });
    var query = params.toString();
    history.replaceState(null, "", query ? "?" + query : location.pathname);
  }

  function matches(row, state) {
    var tasks = (row.getAttribute("data-tasks") || "").split(" ");
    var fields = (row.getAttribute("data-fields") || "").split(" ");
    var haystack = row.getAttribute("data-search") || "";
    if (state.task && tasks.indexOf(state.task) < 0) return false;
    if (state.field && fields.indexOf(state.field) < 0) return false;
    if (state.q && haystack.indexOf(state.q.toLowerCase()) < 0) return false;
    return true;
  }

  function columnIndex(table, name) {
    var heads = Array.prototype.map.call(table.tHead.rows[0].cells, function (cell) {
      return cell.textContent.trim();
    });
    return heads.indexOf(name);
  }

  // With a task chosen, a row's instance count and sample sizes are that task's.
  function narrowCounts(table, row, state) {
    [["Instances", "data-instances-"], ["Sample sizes", "data-samples-"]].forEach(function (pair) {
      var cell = row.cells[columnIndex(table, pair[0])];
      if (!cell) return;
      if (cell.dataset.dpAll === undefined) cell.dataset.dpAll = cell.textContent;
      var own = state.task ? row.getAttribute(pair[1] + state.task) : null;
      cell.textContent = own === null ? cell.dataset.dpAll : own;
    });
  }

  function apply(table, empty, state) {
    var shown = 0;
    Array.prototype.forEach.call(table.tBodies[0].rows, function (row) {
      var keep = matches(row, state);
      row.hidden = !keep;
      if (keep) {
        shown += 1;
        narrowCounts(table, row, state);
      }
    });
    if (empty) empty.hidden = shown !== 0;
    var scroller = table.parentNode;
    if (scroller) scroller.hidden = shown === 0;
  }

  function wire(group) {
    var table = group.parentNode.querySelector(".dp-table");
    if (!table || !table.tHead || !table.tBodies.length) return;
    var empty = group.parentNode.querySelector("[data-dp-corpus-empty]");
    var search = group.querySelector("[data-dp-corpus-search]");
    var task = group.querySelector("[data-dp-corpus-task]");
    var field = group.querySelector("[data-dp-corpus-field]");
    var clear = group.querySelector("[data-dp-corpus-clear]");
    var state = readState();
    if (search) search.value = state.q;
    if (task) task.value = state.task;
    if (field) field.value = state.field;

    function changed() {
      apply(table, empty, state);
      writeState(state);
    }
    if (search)
      search.addEventListener("input", function () {
        state.q = search.value;
        changed();
      });
    if (task)
      task.addEventListener("change", function () {
        state.task = task.value;
        changed();
      });
    if (field)
      field.addEventListener("change", function () {
        state.field = field.value;
        changed();
      });
    if (clear)
      clear.addEventListener("click", function () {
        state.q = state.task = state.field = "";
        if (search) search.value = "";
        if (task) task.value = "";
        if (field) field.value = "";
        changed();
      });
    if (window.dpTableSort)
      window.dpTableSort(table, {
        initial: state.sort
          ? { index: Number(state.sort), ascending: state.dir !== "desc" }
          : null,
        onSort: function (index, ascending) {
          state.sort = String(index);
          state.dir = ascending ? "asc" : "desc";
          changed();
        },
      });
    apply(table, empty, state);
    group.hidden = false;
  }

  // Each button copies the citation it sits beside.
  function wireCopy(button) {
    var citation = button.closest("[data-dp-reference]").querySelector("[data-dp-citation]");
    if (!citation || !navigator.clipboard) return;
    button.hidden = false;
    button.addEventListener("click", function () {
      navigator.clipboard.writeText(citation.textContent.trim()).catch(function () {
        var selection = window.getSelection();
        if (!selection || !selection.selectAllChildren) return;
        selection.removeAllRanges();
        selection.selectAllChildren(citation);
      });
    });
  }

  function start() {
    document.querySelectorAll("[data-dp-corpus-filter]").forEach(wire);
    document.querySelectorAll("[data-dp-copy]").forEach(wireCopy);
  }
  start();
})();
