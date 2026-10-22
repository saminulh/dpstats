// One sortable-table implementation for every generated table on this site:
// the Corpus index, a task's Algorithms table and its Transformations table. The
// table is readable and complete without it; it only reorders rows a reader
// already has.
//
// A header becomes a button so it reads and behaves as the control it is.
// Exactly one header carries aria-sort at a time, and its value is the
// direction the rows are actually in.
//
// A column whose cells hold a declared vocabulary rather than free text sorts
// in that vocabulary's own order — privacy interpretations run from strongest
// to weakest, not alphabetically — which the header declares as
// `data-dp-order`. A cell the vocabulary does not name sorts after the ones it
// does, by its text.
(function () {
  "use strict";

  function cellText(row, index) {
    var cell = row.cells[index];
    return cell ? cell.textContent.trim() : "";
  }

  // A cell holding counts or a span of them is compared as the numbers it
  // shows: its group separators would otherwise end the first numeric run and
  // sort every three-digit value after every four-digit one.
  function sortKey(text) {
    return /^[\d.,\s\u2013\u2014-]+$/.test(text) ? text.replace(/,/g, "") : text;
  }

  function declaredOrder(head) {
    var declared = head.getAttribute("data-dp-order");
    return declared ? declared.split(",") : null;
  }

  function compare(order, left, right) {
    if (order) {
      var first = order.indexOf(left);
      var second = order.indexOf(right);
      if (first < 0) first = order.length;
      if (second < 0) second = order.length;
      if (first !== second) return first - second;
    }
    return sortKey(left).localeCompare(sortKey(right), undefined, {
      numeric: true,
    });
  }

  function sortRows(table, index, order, ascending) {
    var body = table.tBodies[0];
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (left, right) {
      var result = compare(order, cellText(left, index), cellText(right, index));
      return ascending ? result : -result;
    });
    rows.forEach(function (row) {
      body.appendChild(row);
    });
  }

  // `options.initial` restores a sort a page carries in its own state, and
  // `options.onSort` hands that page back the sort a reader just chose. A page
  // keeping no sort state passes neither.
  function wireSort(table, options) {
    if (!table || !table.tHead || !table.tBodies.length) return;
    var settings = options || {};
    var heads = Array.prototype.slice.call(table.tHead.rows[0].cells);
    heads.forEach(function (head, index) {
      var order = declaredOrder(head);
      var button = document.createElement("button");
      button.type = "button";
      button.className = "dp-sort";
      // The heading's own nodes move into the button, so a text slot keeps its marking.
      button.append.apply(button, Array.prototype.slice.call(head.childNodes));
      head.appendChild(button);
      function apply(ascending) {
        heads.forEach(function (other) {
          other.removeAttribute("aria-sort");
        });
        sortRows(table, index, order, ascending);
        head.setAttribute("aria-sort", ascending ? "ascending" : "descending");
      }
      button.addEventListener("click", function () {
        // Clicking the sorted column reverses it; clicking another starts it
        // ascending, and the column it leaves stops claiming a direction.
        var ascending = head.getAttribute("aria-sort") !== "ascending";
        apply(ascending);
        if (settings.onSort) settings.onSort(index, ascending);
      });
      if (settings.initial && settings.initial.index === index) {
        apply(settings.initial.ascending);
      }
    });
  }

  window.dpTableSort = wireSort;
})();
