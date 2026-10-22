/* Long tables open short (style guide W110–W112): a table of more than
   LONG rows a reader can see shows its first SHORT of them and a "Show all N"
   button naming the total. Search, filters and sorting keep working on every
   row: each change refolds what is visible, in the table's own order, and a
   narrowed table of LONG rows or fewer shows them all. Once expanded, a table
   keeps its header in view while it scrolls (W111). The rankings tables page
   their own rows and are left alone. */
(function () {
  var LONG = 50;
  var SHORT = 25;

  function setup(table) {
    var body = table.tBodies[0];
    if (!body || body.rows.length <= LONG) return;
    var wrap = table.closest('.dp-table-scroll') || table;
    var more = document.createElement('button');
    more.type = 'button';
    more.className = 'btn btn-sm dp-table-more';
    wrap.after(more);
    var expanded = false;
    function refold() {
      var visible = Array.prototype.filter.call(body.rows, function (row) { return !row.hidden; });
      var fold = !expanded && visible.length > LONG;
      Array.prototype.forEach.call(body.rows, function (row) {
        row.classList.toggle('dp-row-folded', fold && !row.hidden && visible.indexOf(row) >= SHORT);
      });
      more.hidden = !fold;
      if (fold) window.dpSiteCopyFill(more, 'site.table_show_all', { count: visible.length.toLocaleString() });
      wrap.classList.toggle('dp-table-pinned', !fold && visible.length > LONG);
    }
    more.addEventListener('click', function () {
      expanded = true;
      refold();
    });
    // Filters toggle rows' `hidden`; sorting re-appends them.
    new MutationObserver(refold).observe(body, {
      subtree: true, childList: true, attributes: true, attributeFilter: ['hidden'],
    });
    refold();
  }

  function init() {
    document.querySelectorAll('.dp-content table:not(.dp-leaderboard)').forEach(setup);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
