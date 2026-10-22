// What the site quality checks look at. Limits live in budgets.json; what
// each measurement means, and how to feel it yourself, is README.md.

// Pages whose first load is measured (Lighthouse CI), as paths from the site
// root, or a pattern for a page whose address changes with the release (the
// first match in the build). The Linear Regression pages are the heaviest task.
export const LOAD_PAGES = {
  home: "",
  leaderboard: "linear_regression/leaderboard/",
  explore: "linear_regression/public-results/",
  algorithm: "linear_regression/algorithms/AdaSSP/",
  configured_algorithm: /^linear_regression\/roster\/[^/]+\/$/,
  tutorial: "library/",
};

// The page the heavy interactions run on.
export const EXPLORE = LOAD_PAGES.explore;

// One page per template, found in the build's sitemap: the first URL whose
// path matches. A pattern that matches nothing fails the run, so a renamed or
// removed template is noticed rather than silently left unchecked.
export const TEMPLATES = {
  home: /^$/,
  "task overview": /^linear_regression\/$/,
  leaderboard: /^linear_regression\/leaderboard\/$/,
  explore: /^linear_regression\/public-results\/$/,
  protocol: /^linear_regression\/methodology\/$/,
  definition: /^linear_regression\/definition\/$/,
  metrics: /^linear_regression\/metrics\/$/,
  "native algorithms": /^linear_regression\/algorithms\/$/,
  transformations: /^linear_regression\/transformations\/$/,
  "algorithm page": /^linear_regression\/algorithms\/AdaSSP\/$/,
  "transformation page": /^linear_regression\/transformations\/Standardize\/$/,
  "roster landing": /^linear_regression\/roster\/$/,
  "configured algorithm": /^linear_regression\/roster\/[^/]+\/$/,
  "corpus landing": /^corpus\/$/,
  "corpus source": /^corpus\/[^/]+\/$/,
  instance: /^linear_regression\/instances\/[^/]+\/$/,
  "user guide": /^library\/$/,
  "api reference": /^library\/reference\/[^/]+\/$/,
  submit: /^submit\/$/,
};

// States a reader opens that the static page does not show: each is a page,
// a label, the click or hover that opens it and, for search, the query typed.
export const OPEN_STATES = [
  { template: "explore", label: "algorithm filter open", click: ".dp-plotted-filter" },
  { template: "explore", label: "search open", click: "[data-bs-target='#search-panel']" },
  { template: "explore", label: "search results", click: "[data-bs-target='#search-panel']", query: "AdaSSP" },
  { template: "explore", label: "instance filter open", click: "[aria-controls='instance-filters']" },
  { template: "native algorithms", label: "class help open", click: "button.dp-term" },
  { template: "corpus source", label: "instance card open", hover: "[data-dp-instance-cards] a[href*='/instances/']" },
];

export const THEMES = ["light", "dark"];

// The site's three named widths (style guide W100), each checked at its
// narrowest common size: desktop from 992px, tablet from 768px, phone below.
export const WIDTHS = {
  desktop: { width: 1440, height: 900 },
  tablet: { width: 768, height: 1024 },
  phone: { width: 320, height: 640 },
};

// Screenshots (tests/visual.spec.js): every template and open state in both
// themes at the desktop size, and every template once more at phone width.
// Elements whose content changes between identical builds are masked.
export const DESKTOP = { width: 1440, height: 900 };
export const PHONE = { width: 390, height: 844 };
export const SCREENSHOT_MASK = [];

// The colour pairs a reader depends on, resolved in the browser in each theme
// and held to WCAG 2.2: 4.5:1 for text (1.4.3), 3:1 for controls and chart
// marks (1.4.11). Each is [foreground token, background token]; a translucent
// background is laid over --dp-bg first. A new colour token that carries text
// or marks adds its pair here.
export const CONTRAST_FLOORS = { text: 4.5, ui: 3, mark: 3 };
export const CONTRAST_PAIRS = {
  text: [
    ["--tblr-body-color", "--dp-bg"],
    ["--tblr-body-color", "--tblr-bg-surface-secondary"],
    ["--tblr-body-color", "--tblr-bg-surface"],
    ["--tblr-body-color", "--dp-warn-soft"],
    ["--tblr-secondary", "--dp-bg"],
    ["--tblr-secondary", "--tblr-bg-surface-secondary"],
    ["--tblr-secondary", "--tblr-bg-surface"],
    ["--tblr-primary", "--dp-accent-soft"],
    ["--tblr-link-color", "--dp-bg"],
    ["--dp-warn", "--dp-warn-soft"],
    ["--dp-class-diagnostic", "--dp-bg"],
    ["--dp-class-theoretical", "--dp-bg"],
    ["--dp-class-informal", "--dp-bg"],
    ["--dp-class-nonprivate", "--dp-bg"],
  ],
  ui: [
    ["--dp-control-border", "--dp-bg"],
    ["--tblr-primary", "--dp-bg"],
    ["--dp-warn", "--dp-bg"],
  ],
  mark: [
    ["--dp-chart-diagnostic", "--dp-chart-plot"],
    ["--dp-chart-other", "--dp-chart-plot"],
    ["--dp-hist-success-strong", "--dp-chart-plot"],
    ["--dp-hist-success-soft", "--dp-chart-plot"],
    ["--dp-hist-pair-both", "--dp-chart-plot"],
    ["--dp-hist-pair-neither", "--dp-chart-plot"],
    ["--dp-hist-fail-1", "--dp-chart-plot"],
    ["--dp-hist-fail-2", "--dp-chart-plot"],
    ["--dp-hist-fail-3", "--dp-chart-plot"],
    ["--dp-hist-fail-4", "--dp-chart-plot"],
    ["--dp-hist-fail-5", "--dp-chart-plot"],
  ],
};

// A typical laptop on home broadband.
export const MACHINE = { cpuSlowdown: 4, latencyMs: 40, downloadKbps: 40000, uploadKbps: 10000 };

// Runs per measurement; each reported value is the median.
export const REPEATS = 3;
