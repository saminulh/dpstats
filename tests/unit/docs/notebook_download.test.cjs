"use strict";
/* The download writes the reader's selection into the notebook's parameter
   cells, and offers no file for a selection of nothing. */
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const SOURCE = path.join(
  __dirname, "../../../docs/site/docs/javascripts/notebook-download.js"
);

// A parameter cell as the build writes it: a comment, one assignment per
// name, then its body.
function cell(parameters, body = []) {
  return {
    cell_type: "code",
    metadata: { tags: ["parameters"], dp_parameters: parameters },
    source: ["# comment\n"]
      .concat(Object.keys(parameters).map((name) => `${name} = ${JSON.stringify(parameters[name])}\n`))
      .concat(body),
  };
}

function template(...cells) {
  return JSON.stringify({ cells, metadata: {} });
}

// Each view's notebook, as a page ships them beside itself.
const NOTEBOOKS = {
  profiles: "recreate.ipynb",
  outcomes: "recreate-outcomes.ipynb",
  feature: "recreate-feature.ipynb",
};
const TEMPLATES = {
  "recreate.ipynb": template(
    cell({ TASK: "t", METRIC: "m", ALGORITHMS: null, INSTANCE_FILTERS: {} }),
    cell({ PLOTTED_ALGORITHMS: null }, ["\n", "plot()\n"]),
  ),
  "recreate-outcomes.ipynb": template(
    cell({ TASK: "t", METRIC: "m", ALGORITHM: "a", COMPARE_WITH: null }),
  ),
  "recreate-feature.ipynb": template(
    cell({ TASK: "t", METRIC: "m", ALGORITHMS: null }),
    cell({ CHART_STYLE: {} }),
  ),
};

function harness(current) {
  const offered = [];
  const names = [];
  let click = null;
  // The status shows the slot the page would fill; the slot's text is the page's.
  const status = { textContent: "", replaceChildren(node) { this.textContent = node.slot; } };
  const button = {
    disabled: false,
    closest: () => null,
    addEventListener: (_, handler) => { click = handler; },
  };
  const page = {
    dataset: { notebooks: JSON.stringify(NOTEBOOKS) },
    querySelector: (selector) =>
      selector === "[data-notebook-download]" ? button
        : selector === "[data-notebook-status]" ? status : null,
  };
  const context = {
    window: { dpSiteCopySpan: (key) => ({ slot: key }) },
    console,
    JSON,
    Object,
    Array,
    Promise,
    Error,
    String,
    Blob: function (parts) { offered.push(parts.join("")); },
    URL: { createObjectURL: () => "blob:x", revokeObjectURL: () => {} },
    fetch: (url) => Promise.resolve({ ok: true, text: () => Promise.resolve(TEMPLATES[url]) }),
    document: {
      createElement: () => ({
        click() {}, remove() {},
        set download(name) { names.push(name); },
      }),
      body: { appendChild() {} },
    },
  };
  new Function(...Object.keys(context), fs.readFileSync(SOURCE, "utf8")).apply(
    null, Object.values(context)
  );
  context.window.dpNotebookDownload(page, () => current);
  return { offered, names, status, press: () => { click(); return new Promise((r) => setTimeout(r, 20)); } };
}

test("an empty algorithm list offers no notebook", async () => {
  const held = harness({ ALGORITHMS: [] });
  await held.press();
  assert.deepStrictEqual(held.offered, []);
  assert.strictEqual(held.status.textContent, "explore.notebook_nothing_selected");
});

test("a picked list is written as a list, and null stays every one", async () => {
  const held = harness({ ALGORITHMS: ["a", "b"] });
  await held.press();
  const document = JSON.parse(held.offered[0]);
  assert.deepStrictEqual(document.cells[0].source.slice(1, 4), [
    'TASK = "t"\n', 'METRIC = "m"\n', 'ALGORITHMS = ["a", "b"]\n',
  ]);
  assert.strictEqual(document.cells[1].source.join(""), "# comment\nPLOTTED_ALGORITHMS = None\n\nplot()\n");
});

test("every parameter cell takes the reader's values and keeps its body", async () => {
  const filters = { n: [10, null], design_condition_number: [null, 1000] };
  const held = harness({
    ALGORITHMS: ["a"], INSTANCE_FILTERS: filters, PLOTTED_ALGORITHMS: ["a"],
  });
  await held.press();
  const document = JSON.parse(held.offered[0]);
  const opening = document.cells[0].source.join("");
  assert.ok(opening.includes('INSTANCE_FILTERS = {"n": [10, None], "design_condition_number": [None, 1000]}\n'), opening);
  assert.deepStrictEqual(document.cells[0].metadata.dp_parameters.INSTANCE_FILTERS, filters);
  assert.strictEqual(document.cells[1].source.join(""), '# comment\nPLOTTED_ALGORITHMS = ["a"]\n\nplot()\n');
});

test("the open view hands over its own notebook", async () => {
  const held = harness({ VIEW: "outcomes", ALGORITHM: "b", COMPARE_WITH: "c" });
  await held.press();
  const source = JSON.parse(held.offered[0]).cells[0].source.join("");
  assert.match(source, /ALGORITHM = "b"\n/);
  assert.match(source, /COMPARE_WITH = "c"\n/);
  assert.deepStrictEqual(held.names, ["t-m-recreate-outcomes.ipynb"]);
});

test("a view drawing one algorithm offers nothing without one", async () => {
  const held = harness({ VIEW: "outcomes", ALGORITHM: null });
  await held.press();
  assert.deepStrictEqual(held.offered, []);
  assert.strictEqual(held.status.textContent, "explore.notebook_nothing_selected");
});

// The same mapping test_recreate_notebook.py writes with the build's writer.
const STYLE = { layout: { showlegend: false, meta: { series: {}, glyph: "∞", x: null } } };
const STYLE_SOURCE = 'CHART_STYLE = {"layout": {"showlegend": False, "meta": {"series": {}, "glyph": "∞", "x": None}}}\n';

test("a mapping is written as a Python dict, as the build writes it", async () => {
  const held = harness({ VIEW: "feature", ALGORITHMS: ["a"], CHART_STYLE: STYLE });
  await held.press();
  const source = JSON.parse(held.offered[0]).cells[1].source.join("");
  assert.ok(source.includes(STYLE_SOURCE), source);
});
