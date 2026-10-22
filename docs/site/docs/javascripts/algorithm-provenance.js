/* Configuration identity, not the label, owns this disclosure. */
(function () {
  "use strict";
  var script = document.currentScript;
  var siteRoot = new URL("../", script.src);

  function element(tag, text, className) {
    var node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  /* One of the maintainer's text slots as a muted line, or nothing where blank. */
  function muted(parent, key) {
    var line = element("p", undefined, "dp-provenance-muted");
    if (window.dpSiteCopyFill(line, key)) parent.appendChild(line);
  }

  function link(label, url) {
    var anchor = element("a", label);
    anchor.href = url;
    return anchor;
  }

  /* One configuration as the steps it runs, in order: its transformations, then
     the algorithm they feed. A step shows its name, its kind, its one-line
     summary and the values this configuration runs with; the step's own page,
     which its name links, explains the rest. The order and the kinds carry the
     structure, so no captions are needed. */
  /* Docstring text marks code spans with single or double backticks; show them as code
     (text nodes only, never parsed as HTML). */
  function withCode(text, className) {
    var node = element("div", undefined, className);
    String(text).split(/(``[^`]+``|`[^`]+`)/).forEach(function (part) {
      var code = /^``([^`]+)``$/.exec(part) || /^`([^`]+)`$/.exec(part);
      if (code) node.appendChild(element("code", code[1]));
      else if (part) node.appendChild(document.createTextNode(part));
    });
    return node;
  }

  function parameters(node) {
    if (!node.hyperparameters.length) return null;
    /* What this configuration runs with; a value at its default recedes, a
       changed one stands out and names the default. */
    var list = element("div", undefined, "dp-step-params");
    node.hyperparameters.forEach(function (row) {
      var item = element("div", undefined, "dp-param");
      item.appendChild(element("code", row.name, "dp-param-name"));
      var cell = element("span", undefined, "dp-param-cell");
      var value = element("span", String(row.value), "dp-param-value");
      cell.appendChild(value);
      if (String(row.value) === String(row.default)) {
        value.classList.add("dp-value-default");
        value.title = window.dpSiteCopy("component.default_value_title");
      } else {
        value.classList.add("dp-value-changed");
        var changed = element("span", undefined, "dp-param-default");
        window.dpSiteCopyFill(changed, "component.changed_default", {default: row.default});
        cell.appendChild(changed);
      }
      item.appendChild(cell);
      list.appendChild(item);
    });
    return list;
  }

  function isTransformation(child) { return child.field.startsWith("transformations["); }

  function steps(node) {
    if (node.kind === "Composition") {
      var items = [];
      node.children.filter(isTransformation).forEach(function (child) { items = items.concat(steps(child.component)); });
      node.children.filter(function (child) { return !isTransformation(child); }).forEach(function (child) {
        items = items.concat(steps(child.component));
      });
      return items;
    }
    var step = element("li", undefined, "dp-step");
    step.dataset.kind = node.kind;
    var head = element("div", undefined, "dp-step-head");
    head.appendChild(node.docs ? link(node.name, new URL(node.docs, siteRoot)) : element("span", node.name));
    head.firstChild.classList.add("dp-step-name");
    head.appendChild(element("span", node.kind, "dp-step-kind"));
    step.appendChild(head);
    if (node.summary) step.appendChild(withCode(node.summary, "dp-step-summary"));
    if (!node.docs) muted(step, "component.no_implementation_docs");
    var values = parameters(node);
    if (values) step.appendChild(values);
    var nested = [];
    node.children.filter(isTransformation).forEach(function (child) { nested = nested.concat(steps(child.component)); });
    node.children.filter(function (child) { return !isTransformation(child); }).forEach(function (child) {
      nested = nested.concat(steps(child.component));
    });
    if (nested.length) {
      var inner = element("ol", undefined, "dp-pipeline dp-pipeline-nested");
      nested.forEach(function (item) { inner.appendChild(item); });
      step.appendChild(inner);
    }
    return [step];
  }

  function component(node, host) {
    var list = element("ol", undefined, "dp-pipeline");
    steps(node).forEach(function (item) { list.appendChild(item); });
    host.appendChild(list);
  }

  var provenance = new Map();

  /* One fetch per task URL: Home contains several task rosters. The configurations are the largest
     payload and only a Details trigger reads them. */
  function rows(page) {
    var url = page.dataset.algorithmProvenance;
    if (!provenance.has(url)) {
      provenance.set(url, fetch(url).then(function (response) {
        if (!response.ok) {
          throw new Error(response.status + " " + response.statusText);
        }
        return response.json();
      }).catch(function (error) {
        provenance.delete(url);
        throw error;
      }));
    }
    return provenance.get(url);
  }

  /* The dialog body for one configuration: the release description, the
     recipe it runs, and a link to its Roster page for the rest. */
  function details(payload, uuid, rosterHref) {
    var fragment = document.createDocumentFragment();
    var entry = payload[uuid];
    if (entry && entry.declaration && entry.declaration.description) {
      fragment.appendChild(element("p", entry.declaration.description));
    }
    if (entry) component(entry.component, fragment);
    else muted(fragment, "component.no_structured_configuration");
    var full = link(undefined, rosterHref);
    full.append(window.dpSiteCopySpan("component.full_details"));
    fragment.appendChild(full);
    return fragment;
  }

  window.dpProvenanceDetails = function (page, uuid) {
    return rows(page).then(function (payload) {
      var task = page.dataset.task || "";
      var rosterHref = new URL(task + "/roster/" + uuid + "/", siteRoot);
      return details(payload, uuid, rosterHref);
    });
  };

  /* Every algorithm cell on a standings table carries a Details trigger,
     which opens the one shared dialog (window.dpAlgorithmDialog) through the
     section's own dpOpenDetails, wired by profile-charts.js. */
  function mount(page) {
    page.querySelectorAll(".dp-leaderboard tr[data-uuid]").forEach(function (row) {
      var cell = row.querySelector(".dp-alg");
      if (!cell || cell.querySelector(".dp-provenance-trigger")) return;
      // One text node: a table holds hundreds of these buttons.
      var button = element("button", "ⓘ " + window.dpSiteCopy("component.details_button"), "dp-provenance-trigger");
      button.type = "button";
      button.setAttribute("aria-label", window.dpSiteCopy("component.details_aria", {algorithm: cell.textContent.trim()}));
      cell.appendChild(button);
    });
  }

  document.querySelectorAll("[data-leaderboard], [data-playground]").forEach(function (page) {
    if (!page.dataset.algorithmProvenance) return;
    mount(page);
    var observer = new MutationObserver(function () { mount(page); });
    page.querySelectorAll(".dp-leaderboard tbody").forEach(function (body) {
      observer.observe(body, {childList:true, subtree:true});
    });
  });
  document.addEventListener("click", function (event) {
    var button = event.target.closest(".dp-provenance-trigger");
    if (!button) return;
    // dpOpenDetails is wired per rendered results section (dpConnectLegend's
    // own scope, one .dp-metric per metric on Leaderboard/Explore); a page
    // with no such section (this build's own fixtures) wires it on the
    // outer [data-leaderboard]/[data-playground] instead.
    var section = button.closest(".dp-metric") || button.closest("[data-leaderboard], [data-playground]");
    var row = button.closest("tr[data-uuid]");
    if (section && row && section.dpOpenDetails) section.dpOpenDetails(row.dataset.uuid, button);
  });
})();
