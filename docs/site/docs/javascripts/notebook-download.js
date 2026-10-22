/* Hand a reader the view they are looking at, as a notebook they can run.

   The build ships one notebook per view of a results surface beside its page.
   Its parameter cells are tagged and carry their names and build-time values;
   each opens with one comment line and one assignment per name. This file
   fetches the open view's document, rewrites those assignments with the
   selection the page's controls hold, and offers the result as a file.
   Nothing here computes a result or writes the notebook's body.

   The control stays hidden until a controller wires it. */
(function () {
  "use strict";

  /* One parameter value as Python source, mirroring the build's own writer:
     JSON spells everything but these three as Python does. */
  function literal(value) {
    if (value === null) return "None";
    if (value === true) return "True";
    if (value === false) return "False";
    if (Array.isArray(value)) return "[" + value.map(literal).join(", ") + "]";
    if (typeof value === "object") {
      return "{" + Object.keys(value).map(function (key) {
        return JSON.stringify(key) + ": " + literal(value[key]);
      }).join(", ") + "}";
    }
    return JSON.stringify(value);
  }

  /* Rewrite one tagged cell's assignments with the reader's values, keeping
     its comment line and its body; returns its values. */
  function rewrite(cell, chosen) {
    var values = cell.metadata.dp_parameters;
    var names = Object.keys(values);
    names.forEach(function (name) {
      if (name in chosen) values[name] = chosen[name];
    });
    cell.source = [cell.source[0]]
      .concat(names.map(function (name) { return name + " = " + literal(values[name]) + "\n"; }))
      .concat(cell.source.slice(1 + names.length));
    return values;
  }

  var EMPTY_MEANS_NOTHING = ["ALGORITHMS"];
  // A view drawing one chosen algorithm has nothing to recreate without one.
  var REQUIRED = ["ALGORITHM"];

  function offer(name, text) {
    var url = URL.createObjectURL(
      new Blob([text], { type: "application/x-ipynb+json" })
    );
    var link = document.createElement("a");
    link.href = url;
    link.download = name;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function slug(text) {
    return String(text).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  }

  /* Offer the open view's notebook, with the reader's selection in it.

     `selection` returns the parameters the page currently holds, by the names
     the notebook declares, and VIEW, the open view; a name the page does not
     hold keeps the build's own value. */
  window.dpNotebookDownload = function (page, selection) {
    var button = page.querySelector("[data-notebook-download]");
    var notebooks = page.dataset.notebooks ? JSON.parse(page.dataset.notebooks) : null;
    if (!button || !notebooks) return;
    var status = page.querySelector("[data-notebook-status]");
    var group = button.closest(".dp-notebook-control");
    if (group) group.hidden = false;
    var held = {};
    button.addEventListener("click", function () {
      if (status) status.textContent = "";
      /* An empty list is a selection of nothing, which the notebook would
         refuse; null is every one and is never what an empty list means. */
      var chosen = selection() || {};
      var template = notebooks[chosen.VIEW || "profiles"];
      if (!template) {
        if (status) status.replaceChildren(window.dpSiteCopySpan("explore.notebook_no_view"));
        return;
      }
      if (EMPTY_MEANS_NOTHING.some(function (name) {
        return Array.isArray(chosen[name]) && chosen[name].length === 0;
      }) || REQUIRED.some(function (name) {
        return Object.prototype.hasOwnProperty.call(chosen, name) && !chosen[name];
      })) {
        if (status) status.replaceChildren(window.dpSiteCopySpan("explore.notebook_nothing_selected"));
        return;
      }
      button.disabled = true;
      (held[template] ||
        fetch(template).then(function (response) {
          if (!response.ok) throw new Error(response.status + " " + response.statusText);
          return response.text();
        })
      )
        .then(function (text) {
          held[template] = Promise.resolve(text);
          var document_ = JSON.parse(text);
          var values = {};
          document_.cells.forEach(function (cell) {
            if (cell.metadata.dp_parameters) Object.assign(values, rewrite(cell, chosen));
          });
          offer(
            [slug(values.TASK), slug(values.METRIC), template.split("/").pop()].join("-"),
            JSON.stringify(document_, null, 1) + "\n"
          );
        })
        .catch(function (error) {
          if (status) status.replaceChildren(window.dpSiteCopySpan("explore.notebook_build_error", {error: error.message}));
          console.error("Notebook download failed: " + error.message);
        })
        .finally(function () {
          button.disabled = false;
        });
    });
  };
})();
