// The one page a quality run leaves behind: what passed, what failed, against
// which limit, and how each headline number has moved over past runs.
import { existsSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { LOAD_PAGES } from "./quality.config.js";

const LOAD_AUDITS = {
  "first-contentful-paint": ["First content", "ms"],
  "largest-contentful-paint": ["Largest content", "ms"],
  "total-blocking-time": ["Blocked", "ms"],
  "cumulative-layout-shift": ["Layout shift", ""],
  "total-byte-weight": ["Downloaded", "KB"],
};

const EXPLORE_LABELS = {
  "load.chart_ready_ms": "First load: chart ready",
  "load.blocked_ms": "First load: page blocked",
  "load.layout_shift": "First load: layout shift",
  "load.heap_mb": "First load: memory (MB)",
  "switch_budget_new.blocked_ms": "Switch ε, new: page blocked",
  "switch_budget_new.settled_ms": "Switch ε, new: done after",
  "switch_budget_new.fetched_kb": "Switch ε, new: downloaded at click (KB)",
  "switch_budget_seen.blocked_ms": "Switch ε, seen: page blocked",
  "switch_budget_seen.settled_ms": "Switch ε, seen: done after",
  "show_all_curves.blocked_ms": "Show all curves: page blocked",
  "pan_all_curves.fps": "Pan the chart, all curves shown: frames per second",
  "pan_all_curves.slow_frames": "Pan the chart, all curves shown: frames over 50 ms",
  "pan_all_curves.blocked_ms": "Pan the chart, all curves shown: page blocked",
  "hover_all_curves.fps": "Hover over curves, all shown: frames per second",
  "hover_all_curves.slow_frames": "Hover over curves, all shown: frames over 50 ms",
  "hover_all_curves.blocked_ms": "Hover over curves, all shown: page blocked",
  "hover_all_curves.restyles": "Hover over curves, all shown: full chart restyles (0 expected)",
  "hover_all_curves.browser_mb": "Hover over curves, all shown: whole browser's memory (MB)",
  "switch_budget_all_curves.blocked_ms": "Switch ε, all curves shown: page blocked",
  "back_to_10_curves.blocked_ms": "Back to 10 curves: page blocked",
  "tau_drag.fps": "Drag τ: frames per second",
  "tau_drag.slow_frames": "Drag τ: frames over 50 ms",
  "tau_drag.release_blocked_ms": "Drag τ: page blocked after release",
  "tau_drag.start_ms": "Drag τ: press to the tab's first move (ms)",
  "tau_drag.live_updates": "Drag τ: thresholds taken while held (the curves re-rank live)",
  "switch_metric.blocked_ms": "Switch metric: page blocked",
  "hover.fps": "Hover over curves: frames per second",
  "hover.blocked_ms": "Hover over curves: page blocked",
  "hover.restyles": "Hover over curves: full chart restyles (0 expected)",
  "comparison_hover.fps": "Hover the comparison legend: frames per second",
  "comparison_hover.blocked_ms": "Hover the comparison legend: page blocked",
  "comparison_hover.redraws": "Hover the comparison legend: chart redraws (0 expected)",
  "interaction.inp_ms": "Slowest response to an input (INP; 0 means every response was under 40 ms)",
  "memory.growth_after_40_switches_mb": "Memory growth over 40 ε switches (MB)",
};

const escape = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const read = (path) => (existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null);
const fmt = (v) => (v === null || v === undefined || Number.isNaN(v) ? "—" : typeof v === "number" ? v.toLocaleString("en-US") : String(v));

function verdict(value, limit) {
  if (!limit || value === null || value === undefined) return "";
  if (limit.max !== undefined) return value <= limit.max ? "pass" : "fail";
  if (limit.min !== undefined) return value >= limit.min ? "pass" : "fail";
  return "";
}

/** The representative Lighthouse run per page: its numbers and report path. */
function lighthouse(out, loadPages) {
  const dir = join(out, "lighthouse");
  const manifest = read(join(dir, "manifest.json"));
  if (!manifest) return {};
  const pages = {};
  for (const entry of manifest.filter((e) => e.isRepresentativeRun)) {
    const name = Object.entries(loadPages).find(([, path]) => new URL(entry.url).pathname === "/" + path);
    if (!name) continue;
    const lhr = read(entry.jsonPath);
    pages[name[0]] = {
      report: relative(out, entry.htmlPath),
      values: Object.fromEntries(Object.keys(LOAD_AUDITS).map((id) => {
        const raw = lhr.audits[id] ? lhr.audits[id].numericValue : null;
        return [id, raw === null ? null : id === "total-byte-weight" ? Math.round(raw / 1024) : id === "cumulative-layout-shift" ? Math.round(raw * 1000) / 1000 : Math.round(raw)];
      })),
    };
  }
  return pages;
}

function spark(points) {
  const values = points.filter((v) => typeof v === "number");
  if (values.length < 2) return "";
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  const step = 100 / (points.length - 1);
  const path = points.map((v, i) => (typeof v === "number" ? `${i ? "L" : "M"}${(i * step).toFixed(1)},${(22 - ((v - lo) / span) * 20).toFixed(1)}` : "")).join(" ");
  return `<svg class="spark" viewBox="0 0 100 24" preserveAspectRatio="none" aria-hidden="true"><path d="${path}"/></svg>`;
}

/** Render the report and the one history line this run adds. */
export function renderReport({ report, out, budgets, historyPath }) {
  const load = lighthouse(out, report.load_pages || {});
  const explore = (read(join(out, "interactions.json")) || {}).medians || {};
  const a11y = read(join(out, "accessibility.json")) || {};
  const regressions = read(join(out, "accessibility-regressions.json")) || [];
  const stylesheet = read(join(out, "stylesheet.json")) || { findings: null, byRule: {}, regressions: [] };
  const prose = read(join(out, "prose.json")) || { severities: {}, regressions: [] };
  const size = read(join(out, "site-size.json"));
  const memory = read(join(out, "load-memory.json")) || {};
  const idle = read(join(out, "idle.json")) || {};
  const results = read(join(out, "playwright.json"));
  const checks = [];
  const walk = (suite) => {
    (suite.specs || []).forEach((spec) => checks.push({ title: spec.title, file: spec.file, ok: spec.ok }));
    (suite.suites || []).forEach(walk);
  };
  if (results) results.suites.forEach(walk);

  const shots = checks.filter((c) => /visual/.test(c.file));
  const changed = shots.filter((c) => !c.ok);
  const headline = {
    when: report.when, label: report.label, commit: report.commit, dirty: report.dirty,
    build: report.build || null,
    load: Object.fromEntries(Object.entries(load).map(([k, v]) => [k, v.values])),
    explore,
    accessibility: {
      findings: Object.values(a11y).reduce((n, f) => n + Object.values(f.violations).reduce((a, b) => a + b, 0), 0),
      beyond_baseline: regressions.length,
    },
    stylesheet: { findings: stylesheet.findings, beyond_baseline: stylesheet.regressions.length },
    prose: { ...prose.severities, beyond_baseline: prose.regressions.length },
    site: size ? { total_mb: size.total_mb, largest_page_kb: size.largest_page.kb, largest_file_kb: size.largest_file.kb } : null,
    load_memory: memory,
    idle,
    screenshots: { taken: shots.length, changed: changed.length, accepted: Boolean(report.screenshotsAccepted) },
    failures: report.failures,
  };
  const history = existsSync(historyPath)
    ? readFileSync(historyPath, "utf8").trim().split("\n").filter(Boolean).map((l) => JSON.parse(l))
    : [];
  const series = [...history.slice(-29), headline];
  const trend = (pick) => spark(series.map((h) => { try { return pick(h); } catch { return null; } }));

  const row = (cells, cls = "") => `<tr${cls ? ` class="${cls}"` : ""}>${cells.map((c) => `<td>${c}</td>`).join("")}</tr>`;
  const status = (v) => (v ? `<span class="pill ${v}">${v}</span>` : "");

  const buildRows = report.build
    ? Object.entries(report.build).map(([kind, b]) => {
      const limit = (budgets.build || {})[kind] || {};
      const verdicts = [verdict(b.elapsed_s, { max: limit.elapsed_s }), verdict(b.peak_memory_gb, { max: limit.peak_memory_gb })];
      return row([`${kind} build`, `${fmt(b.elapsed_s)} s`, `${fmt(b.peak_memory_gb)} GB`,
        `≤ ${fmt(limit.elapsed_s)} s, ≤ ${fmt(limit.peak_memory_gb)} GB`,
        status(verdicts.includes("fail") ? "fail" : "pass"), trend((h) => h.build[kind].elapsed_s)]);
    })
    : [row(["Not measured this run (pass --build)", "", "", "", "", ""])];

  const loadRows = Object.keys(LOAD_PAGES).flatMap((page) => {
    const entry = load[page];
    if (!entry) return [row([page, "not measured", "", "", ""])];
    return Object.entries(LOAD_AUDITS).map(([id, [label, unit]], i) => {
      const limitRaw = (budgets.load[page] || {})[id];
      const limit = limitRaw === undefined ? null : { max: id === "total-byte-weight" ? Math.round(limitRaw / 1024) : limitRaw };
      const value = entry.values[id];
      return row([i ? "" : `<a href="${escape(entry.report)}">${escape(page)}</a>`, label, `${fmt(value)} ${unit}`,
        limit ? `≤ ${fmt(limit.max)} ${unit}` : "", status(verdict(value, limit)), trend((h) => h.load[page][id])]);
    });
  });

  const exploreRows = Object.entries(EXPLORE_LABELS).map(([key, label]) => {
    const limit = budgets.explore[key];
    const value = explore[key];
    return row([label, fmt(value), limit ? (limit.max !== undefined ? `≤ ${fmt(limit.max)}` : `≥ ${fmt(limit.min)}`) : "",
      status(verdict(value, limit)), trend((h) => h.explore[key])]);
  });

  const a11yRows = Object.entries(a11y).map(([where, f]) => {
    const count = Object.values(f.violations).reduce((a, b) => a + b, 0);
    const worse = regressions.filter((r) => r.startsWith(where + ":")).length;
    const rules = f.details.map((d) => `<a href="${escape(d.url)}">${escape(d.id)}</a> (${d.nodes})`).join(", ");
    return row([escape(where), fmt(count), worse ? status("fail") : status("pass"), rules || "none", fmt(f.needsReview.length)]);
  });

  const stylesheetRows = [
    ...Object.entries(stylesheet.byRule).map(([rule, n]) => row([escape(rule), fmt(n), "", ""])),
    ...stylesheet.regressions.map((r) => row([escape(r), "", status("fail"), ""])),
    row(["All findings", fmt(stylesheet.findings), status(stylesheet.regressions.length ? "fail" : "pass"),
      trend((h) => h.stylesheet.findings)]),
  ];

  const memoryRows = Object.keys(LOAD_PAGES).map((page) => {
    const m = memory[page] || {};
    const limit = (budgets.load_memory || {})[page] || {};
    const ok = [verdict(m.peak_heap_mb, { max: limit.peak_heap_mb }), verdict(m.dom_nodes, { max: limit.dom_nodes })];
    return row([escape(page), fmt(m.peak_heap_mb), limit.peak_heap_mb !== undefined ? `≤ ${fmt(limit.peak_heap_mb)}` : "",
      fmt(m.dom_nodes), limit.dom_nodes !== undefined ? `≤ ${fmt(limit.dom_nodes)}` : "",
      status(ok.includes("fail") ? "fail" : ok.includes("pass") ? "pass" : ""), trend((h) => h.load_memory[page].peak_heap_mb)]);
  });
  const IDLE_KEYS = ["frames", "timers", "fetches", "redraws", "animations", "blocked_ms", "heap_growth_mb"];
  const idleRows = Object.keys(LOAD_PAGES).map((page) => {
    const r = idle[page];
    if (!r) return [row([escape(page), "not measured", "", "", "", "", "", "", "", ""])];
    const limit = (budgets.idle || {})[page] || {};
    const verdicts = IDLE_KEYS.map((key) => verdict(r[key], limit[key] !== undefined ? { max: limit[key] } : null));
    const cell = (key) => `${fmt(r[key])}${limit[key] !== undefined ? ` <span class="meta">≤ ${fmt(limit[key])}</span>` : ""}`;
    return row([escape(page), ...IDLE_KEYS.map(cell),
      status(verdicts.includes("fail") ? "fail" : verdicts.includes("pass") ? "pass" : ""), trend((h) => h.idle[page].heap_growth_mb)]);
  });
  const siteLimits = budgets.site || {};
  const sizeRows = size ? [
    ["Largest page", `${fmt(size.largest_page.kb)} KB <span class="meta">${escape(size.largest_page.path)}</span>`, size.largest_page.kb, siteLimits.largest_page_kb, "KB", (h) => h.site.largest_page_kb],
    ["Largest file", `${fmt(size.largest_file.kb)} KB <span class="meta">${escape(size.largest_file.path)}</span>`, size.largest_file.kb, siteLimits.largest_file_kb, "KB", (h) => h.site.largest_file_kb],
    ["Whole site", `${fmt(size.total_mb)} MB in ${fmt(size.files)} files (${fmt(size.pages)} pages)`, size.total_mb, siteLimits.total_mb, "MB", (h) => h.site.total_mb],
  ].map(([label, shown, value, limit, unit, pick]) => row([label, shown, limit !== undefined ? `≤ ${fmt(limit)} ${unit}` : "",
    status(verdict(value, limit !== undefined ? { max: limit } : null)), trend(pick)])) : [row(["Not measured", "", "", "", ""])];

  const proseRows = [
    ...["error", "warning", "suggestion"].map((level) => row([level === "error" ? "Errors (held to the baseline)" : `${level[0].toUpperCase()}${level.slice(1)}s (reported)`,
      fmt(prose.severities[level]), level === "error" ? status(prose.regressions.length ? "fail" : "pass") : "", trend((h) => h.prose[level])])),
    ...prose.regressions.map((r) => row([escape(r), "", status("fail"), ""])),
  ];

  const checkRows = checks.filter((c) => !/accessibility|idle|load-memory|visual/.test(c.file)).map((c) => row([escape(c.title), status(c.ok ? "pass" : "fail")]));
  const shotRows = changed.length
    ? changed.map((c) => row([escape(c.title.replace(/^screenshot: /, "")), status("fail")]))
    : [row([`All ${shots.length} match the accepted images`, status(shots.length ? "pass" : "")])];
  const ok = !report.failures.length;

  const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Site quality report</title>
<style>
:root { --bg: #f7f8fa; --surface: #fff; --text: #1d2433; --muted: #5d6678; --border: #dfe3ea; --pass: #1d7a46; --pass-bg: #e3f4ea; --fail: #b3261e; --fail-bg: #fde8e6; --line: #2f6fd6; }
@media (prefers-color-scheme: dark) { :root { --bg: #131826; --surface: #1b2233; --text: #e4e8f0; --muted: #9aa4b8; --border: #2c3447; --pass: #7fd3a1; --pass-bg: #173826; --fail: #ff8a80; --fail-bg: #3d1f1d; --line: #8cb4e0; } }
body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, -apple-system, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 28px 0 8px; }
.meta, .note { color: var(--muted); font-size: 13px; }
.banner { padding: 12px 16px; border-radius: 10px; margin: 16px 0; font-weight: 600; }
.banner.pass { background: var(--pass-bg); color: var(--pass); } .banner.fail { background: var(--fail-bg); color: var(--fail); }
.scroll { overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
td, th { padding: 7px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: 12px; }
tr:last-child td { border-bottom: 0; }
.pill { font-size: 12px; padding: 1px 8px; border-radius: 999px; font-weight: 600; }
.pill.pass { background: var(--pass-bg); color: var(--pass); } .pill.fail { background: var(--fail-bg); color: var(--fail); }
.spark { width: 100px; height: 24px; } .spark path { fill: none; stroke: var(--line); stroke-width: 1.5; vector-effect: non-scaling-stroke; }
a { color: var(--line); }
</style></head><body><main>
<h1>Site quality report</h1>
<p class="meta">${escape(report.when)} · library ${escape(report.commit)}${report.dirty ? " (uncommitted site changes)" : ""}${report.label ? " · " + escape(report.label) : ""} · site ${escape(report.site)}</p>
<div class="banner ${ok ? "pass" : "fail"}">${ok ? "Every check passed." : "Failed: " + escape(report.failures.join("; "))}</div>
<p class="note">Browser runs simulate a typical laptop on home broadband (CPU ${report.machine.cpuSlowdown}× slower, ${report.machine.latencyMs} ms latency, ${report.machine.downloadKbps / 1000} Mbit/s). Each number is the median of ${report.repeats} runs. Trends show the last 30 runs on this machine. Limits live in <code>docs/site/quality/budgets.json</code>; what each measurement means, and how to feel it yourself, is <code>docs/site/quality/README.md</code>.${report.dashboard ? ` Lighthouse history: <a href="${escape(report.dashboard)}">dashboard</a>.` : ""} Full browser test log: <a href="playwright-report/index.html">Playwright report</a>.</p>
<h2>Build</h2>
<div class="scroll"><table><tr><th>Build</th><th>Time</th><th>Peak memory</th><th>Limit</th><th></th><th>Trend (time)</th></tr>${buildRows.join("")}</table></div>
<h2>First load (Lighthouse)</h2>
<div class="scroll"><table><tr><th>Page (full report)</th><th>Measurement</th><th>Value</th><th>Limit</th><th></th><th>Trend</th></tr>${loadRows.join("")}</table></div>
<h2>Memory while loading</h2>
<p class="note">The page's JavaScript memory sampled every 100 ms from the first request until it settles, and the elements it built.</p>
<div class="scroll"><table><tr><th>Page</th><th>Peak memory (MB)</th><th>Limit</th><th>Elements</th><th>Limit</th><th></th><th>Trend (peak)</th></tr>${memoryRows.join("")}</table></div>
<h2>At rest after loading</h2>
<p class="note">The same pages left alone for 30 s after they settle: what the page's own code ran meanwhile (each a count; a page at rest runs nothing), how long it was blocked, and how much its memory grew after a forced collection. One run per page.</p>
<div class="scroll"><table><tr><th>Page</th><th>Frame callbacks</th><th>Timers</th><th>Fetches</th><th>Chart redraws</th><th>Animations playing</th><th>Blocked (ms)</th><th>Memory growth (MB)</th><th></th><th>Trend (growth)</th></tr>${idleRows.join("")}</table></div>
<h2>Site size (every file in the build)</h2>
<div class="scroll"><table><tr><th>Measurement</th><th>Value</th><th>Limit</th><th></th><th>Trend</th></tr>${sizeRows.join("")}</table></div>
<h2>Explore: heavy operations</h2>
<div class="scroll"><table><tr><th>Operation</th><th>Value (ms unless stated)</th><th>Limit</th><th></th><th>Trend</th></tr>${exploreRows.join("")}</table></div>
<h2>Correctness checks</h2>
<div class="scroll"><table><tr><th>Check</th><th></th></tr>${checkRows.join("")}</table></div>
<h2>Screenshots</h2>
<p class="note">${report.screenshotsAccepted ? "This run accepted the pages' current look as the new reference. " : ""}Each page is compared pixel for pixel with the last accepted image, kept per machine in <code>QUALITY_HOME/screenshots</code>. A changed page shows its expected, actual and difference images in the <a href="playwright-report/index.html">Playwright report</a>; when the change is intended, rerun with <code>--update-screenshots</code>.</p>
<div class="scroll"><table><tr><th>Changed page</th><th></th></tr>${shotRows.join("")}</table></div>
<h2>Stylesheet (Stylelint)</h2>
<p class="note">Colours and type sizes come from tokens. Known findings are listed in <code>stylesheet-baseline.json</code> and may only shrink.</p>
<div class="scroll"><table><tr><th>Rule</th><th>Findings</th><th></th><th>Trend</th></tr>${stylesheetRows.join("")}</table></div>
<h2>Prose (Vale)</h2>
<p class="note">Google's developer documentation style (STYLE S10) over the authored docs, tutorials and the built API reference. Errors that exist today are listed in <code>prose-baseline.json</code> and may only shrink; warnings and suggestions are for the writer. Run <code>npm --prefix docs/site/quality run lint:prose -- --site &lt;built site&gt;</code> for the full list.</p>
<div class="scroll"><table><tr><th>Alerts</th><th>Count</th><th></th><th>Trend</th></tr>${proseRows.join("")}</table></div>
<h2>Accessibility (axe, WCAG 2.2 AA rules)</h2>
<p class="note">Known findings are listed in <code>a11y-baseline.json</code> and may only shrink; a page fails when it has a finding the baseline does not. "Needs review" items are ones axe cannot decide; they belong to the manual checklist.</p>
<div class="scroll"><table><tr><th>Page and theme</th><th>Findings</th><th>Beyond baseline</th><th>Rules (failing elements)</th><th>Needs review</th></tr>${a11yRows.join("")}</table></div>
</main></body></html>
`;
  return { html, headline };
}
