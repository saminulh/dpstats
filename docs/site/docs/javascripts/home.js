/* Task navigation only: each panel is the same leaderboard as its full page. */
(function () {
  "use strict";
  var home = document.querySelector("[data-home-leaderboards]");
  if (!home) return;
  var panels = [].slice.call(home.querySelectorAll('.dp-home-panel'));

  /* Bootstrap's tabs switch the panels; a panel's charts redraw once it shows. */
  home.addEventListener("shown.bs.tab", function (event) {
    var panel = home.querySelector(event.target.getAttribute("data-bs-target"));
    if (!panel) return;
    panel.querySelectorAll(".dp-chart").forEach(function (chart) {
      document.dispatchEvent(new CustomEvent("dp-chart", { detail: { chart: chart.id } }));
    });
  });
  /* Each panel's chart heading carries the budget choice, and its standings
     heading links to the full Leaderboard with Explore beside it, once the
     results controller has built them. */
  function arrange(panel) {
    if (panel.dataset.arranged) return true;
    var heading = panel.querySelector(".dp-profile-column > .dp-profile-heading");
    var shared = panel.querySelector(".dp-shared-controls");
    var stripHead = panel.querySelector(".dp-plotted-head");
    if (!heading || !shared || !stripHead) return false;
    var row = document.createElement("div");
    row.className = "dp-home-chart-head";
    heading.before(row);
    row.append(heading, shared);
    var links = {};
    panel.querySelectorAll(".dp-home-task-heading a").forEach(function (link) {
      links[link.textContent.trim()] = link.getAttribute("href");
    });
    if (links.Leaderboard) {
      var title = document.createElement("a");
      title.className = "dp-home-board-link";
      title.href = links.Leaderboard;
      var arrow = document.createElement("span");
      arrow.className = "dp-home-board-icon";
      arrow.setAttribute("aria-hidden", "true");
      arrow.textContent = "\u2192";
      title.append(window.dpSiteCopySpan("home.board_link"), " ", arrow);
      title.title = window.dpSiteCopy("home.board_link_title");
      stripHead.prepend(title);
    }
    if (links.Explore) {
      var explore = document.createElement("a");
      explore.className = "dp-home-explore-link";
      explore.href = links.Explore;
      explore.append(window.dpSiteCopySpan("home.explore_link"), " \u2192");
      stripHead.append(explore);
    }
    panel.dataset.arranged = "true";
    return true;
  }
  panels.forEach(function (panel) {
    if (arrange(panel)) return;
    var watch = new MutationObserver(function () {
      if (arrange(panel)) watch.disconnect();
    });
    watch.observe(panel, { childList: true, subtree: true });
  });
  home.classList.add("is-interactive");
})();
