// The Algorithms and Transformations tables' privacy-interpretation chips.
// Sorting is the shared table-sort implementation every generated table uses.
// Every table is readable and complete without this file; it only narrows the
// rows a reader already has.
(function () {
  "use strict";

  function cellText(row, index) {
    var cell = row.cells[index];
    return cell ? cell.textContent.trim() : "";
  }

  // The column holding each row's class, or -1 where the table has none.
  function classIndex(table) {
    var columns = Array.prototype.slice.call(table.tHead.rows[0].cells);
    return columns.findIndex(function (cell) {
      return cell.textContent.trim() === "Privacy interpretation";
    });
  }

  // The chips narrow every table on the page. A group left with no matching
  // row is hidden whole, heading included.
  function wireFilter(group) {
    var tables = Array.prototype.slice.call(
      group.parentNode.querySelectorAll(".dp-table")
    );
    // The chips are native radios; the checked one names the class shown.
    group.addEventListener("change", function (event) {
      var wanted = event.target.getAttribute("data-dp-class");
      tables.forEach(function (table) {
        var index = classIndex(table);
        if (index < 0) return;
        var shown = 0;
        Array.prototype.forEach.call(table.tBodies[0].rows, function (row) {
          var match = wanted === "All" || cellText(row, index) === wanted;
          row.hidden = !match;
          if (match) shown += 1;
        });
        var section = table.closest(".dp-table-group");
        if (section) section.hidden = shown === 0;
      });
    });
  }

  function start() {
    document.querySelectorAll("[data-dp-sortable] .dp-table").forEach(function (table) {
      window.dpTableSort(table);
    });
    document.querySelectorAll("[data-dp-class-filter]").forEach(function (group) {
      wireFilter(group);
      group.parentNode.querySelectorAll(".dp-table").forEach(function (table) {
        if (window.dpTableSort) window.dpTableSort(table);
      });
    });
  }
  start();
})();
