// The measuring side of the browser checks: observers installed before any
// page script runs, and the interactions a reader performs on Explore.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { MACHINE } from "../quality.config.js";

// The package exports no path to its browser bundle, so it is read from disk.
const WEB_VITALS = readFileSync(
  new URL("../node_modules/web-vitals/dist/web-vitals.attribution.iife.js", import.meta.url),
  "utf8",
);

// Installed before page scripts: long tasks, long animation frames, fetches,
// paints, layout shifts and every chart redraw, so a measurement can ask when
// the page last did anything after a moment, and how long it was blocked.
const PROBE = `
(() => {
  const P = window.__q = { lt: [], laf: [], res: [], plot: [], paint: {}, lcp: 0, cls: 0, chartReady: null, inp: [] };
  const watch = (type, fn) => { try { new PerformanceObserver(l => l.getEntries().forEach(fn)).observe({ type, buffered: true }); } catch (e) {} };
  watch('longtask', e => P.lt.push([e.startTime, e.duration]));
  watch('long-animation-frame', e => P.laf.push([e.startTime, e.duration, e.blockingDuration || 0]));
  watch('resource', e => P.res.push(e.responseEnd));
  watch('paint', e => { P.paint[e.name] = e.startTime; });
  watch('largest-contentful-paint', e => { P.lcp = e.startTime; });
  watch('layout-shift', e => { if (!e.hadRecentInput) P.cls += e.value; });
  setInterval(() => document.querySelectorAll('.dp-chart').forEach(gd => {
    if (gd.__q || typeof gd.on !== 'function') return;
    gd.__q = true;
    gd.on('plotly_afterplot', () => {
      const t = performance.now();
      P.plot.push(t);
      if (P.chartReady === null && gd.querySelector('.main-svg')) P.chartReady = t;
    });
    if (P.chartReady === null && gd.querySelector('.main-svg')) P.chartReady = performance.now();
  }), 50);
  P.settle = (t0, quiet) => new Promise(resolve => {
    const last = () => Math.max(t0,
      ...P.lt.filter(([s]) => s >= t0).map(([s, d]) => s + d),
      ...P.res.filter(t => t >= t0), ...P.plot.filter(t => t >= t0));
    const tick = () => {
      if (performance.now() - last() < quiet) return void setTimeout(tick, 50);
      const after = P.laf.filter(([s]) => s >= t0);
      resolve({
        settled_ms: Math.round(last() - t0),
        blocked_ms: Math.round(P.lt.filter(([s]) => s >= t0).reduce((a, [, d]) => a + Math.max(0, d - 50), 0)),
        worst_frame_ms: Math.round(after.reduce((a, [, d]) => Math.max(a, d), 0)),
        redraws: P.plot.filter(t => t >= t0).length,
      });
    };
    setTimeout(tick, quiet);
  });
  P.frames = [];
  P.record = on => {
    P.recording = on;
    if (!on) return;
    P.frames = [];
    const step = t => { if (!P.recording) return; P.frames.push(t); requestAnimationFrame(step); };
    requestAnimationFrame(step);
  };
})();
`;

// INP as Chrome reports it to real visitors, per interaction.
const VITALS = `${WEB_VITALS}
window.webVitals.onINP(m => { window.__q && window.__q.inp.push(Math.round(m.value)); }, { reportAllChanges: true });
`;

/** A cold page on the configured machine, with the probe installed. */
export async function openPage(browser) {
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  await context.addInitScript(PROBE);
  await context.addInitScript(VITALS);
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Performance.enable");
  await cdp.send("Network.enable");
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: MACHINE.cpuSlowdown });
  await cdp.send("Network.emulateNetworkConditions", {
    offline: false,
    latency: MACHINE.latencyMs,
    downloadThroughput: (MACHINE.downloadKbps * 1000) / 8,
    uploadThroughput: (MACHINE.uploadKbps * 1000) / 8,
  });
  const transfer = { bytes: 0 };
  cdp.on("Network.loadingFinished", (e) => { transfer.bytes += e.encodedDataLength; });
  return { context, page, cdp, transfer };
}

export const settle = (page, t0, quiet = 700) =>
  page.evaluate(([a, q]) => window.__q.settle(a, q), [t0, quiet]);

export const now = (page) => page.evaluate(() => performance.now());

/** Live JavaScript heap after a forced collection, in MB. */
export async function heapMb(cdp) {
  await cdp.send("HeapProfiler.collectGarbage");
  const { metrics } = await cdp.send("Performance.getMetrics");
  const used = metrics.find((m) => m.name === "JSHeapUsedSize").value;
  return Math.round(used / 1e5) / 10;
}

/** Run one reader action with real input (a click or a selection, so the
 * browser counts it as an interaction) and measure until the page settles. */
export async function act(page, input) {
  const t0 = await now(page);
  await input(page);
  return settle(page, t0);
}

// The radios are drawn as buttons over a hidden input; a reader clicks the label.
export const pickBudget = (epsilon) => (page) =>
  page.locator(`label[for$="budget-subinversesamplesizedelta-eps-${epsilon}"]:visible`).first().click();

export const curveLimit = (value) => (page) =>
  page.locator("select:visible", { has: page.locator("option[value='all']") }).first().selectOption(value);

async function metricLabel(page, checked) {
  const id = await page.evaluate((want) => {
    const r = [...document.querySelectorAll('input[type=radio][name$="metric"]')].find((r) => r.checked === want && r.labels.length && r.labels[0].offsetParent);
    return r ? r.id : null;
  }, checked);
  return page.locator(`label[for="${id}"]`);
}
export const otherMetric = async (page) => (await metricLabel(page, false)).click();
export const firstMetric = async (page) => {
  const id = await page.evaluate(() => document.querySelector('input[type=radio][name$="metric"]').id);
  await page.locator(`label[for="${id}"]`).click();
};

function frameStats(frames) {
  const gaps = frames.slice(1).map((t, i) => t - frames[i]).sort((a, b) => a - b);
  if (!gaps.length) return { fps: 0, slow_frames: 0 };
  const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
  return { fps: Math.round(10000 / mean) / 10, slow_frames: gaps.filter((g) => g > 50).length };
}

/** The whole browser's resident memory in MB, summed over every process it
 * reports (browser, GPU, renderers, network): what a task manager shows. */
export async function browserMemoryMb(browser) {
  const session = await browser.newBrowserCDPSession();
  const { processInfo } = await session.send("SystemInfo.getProcessInfo");
  await session.detach();
  const kb = execFileSync("ps", ["-o", "rss=", "-p", processInfo.map((p) => p.id).join(",")], { encoding: "utf8" });
  return Math.round(kb.trim().split(/\s+/).reduce((a, k) => a + Number(k), 0) / 1024);
}

/** Explore's Error ratio view with a second algorithm added, then its legend
 * rows hovered back and forth: frames, blocked time, and how many times the
 * scatter redrew (a hover only dims, so it should redraw nothing). */
export async function comparisonHover(page) {
  await page.click("#playground-view-feature");
  await settle(page, await now(page));
  await page.locator('[data-comparison-add]').filter({ visible: true }).first().click();
  const search = page.locator("[data-comparison-selection] [data-algorithm-picker] [role=combobox]").filter({ visible: true }).first();
  await search.fill("Ada");
  await search.press("ArrowDown");
  await search.press("Enter");
  await settle(page, await now(page));
  const rows = page.locator("[data-comparison-selection] [data-uuid]").filter({ visible: true });
  if ((await rows.count()) < 2) throw new Error("the Error ratio view did not plot two algorithms");
  const boxes = [await rows.nth(0).boundingBox(), await rows.nth(1).boundingBox()];
  await page.evaluate(() => {
    const plot = [...document.querySelectorAll(".dp-comparison-plot")].find((p) => p.data && p.offsetParent);
    window.__q.comparisonRedraws = 0;
    const count = () => { window.__q.comparisonRedraws += 1; };
    plot.on("plotly_afterplot", count);
    plot.on("plotly_restyle", count);
  });
  const t0 = await now(page);
  await page.evaluate(() => window.__q.record(true));
  for (let step = 0; step < 30; step += 1) {
    const box = boxes[step % 2];
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    await page.waitForTimeout(40);
  }
  const frames = await page.evaluate(() => { window.__q.record(false); return window.__q.frames; });
  await page.mouse.move(0, 0);
  const done = await settle(page, t0);
  const redraws = await page.evaluate(() => window.__q.comparisonRedraws);
  return { ...frameStats(frames), blocked_ms: done.blocked_ms, redraws };
}

/** The visible chart's plot area, scrolled into view first: a pointer moved
 * over a chart below the window touches nothing. */
async function plotArea(page) {
  await page.evaluate(() => {
    const c = [...document.querySelectorAll(".dp-chart")].find((c) => c.offsetParent);
    if (c) c.scrollIntoView({ block: "center" });
  });
  await page.waitForTimeout(300);
  return page.evaluate(() => {
    const c = [...document.querySelectorAll(".dp-chart")].find((c) => c.offsetParent);
    const d = c && c.querySelector(".nsewdrag");
    if (!d) return null;
    const r = d.getBoundingClientRect();
    return { x: r.x, y: r.y, width: r.width, height: r.height };
  });
}

/** Drag the τ tab there and back, as a hand would (a move every 10 ms). */
export async function dragTau(page) {
  const box = await plotArea(page);
  const handle = await page.evaluate(() => {
    const h = [...document.querySelectorAll(".dp-tau-handle")].find((h) => h.offsetParent);
    if (!h) return null;
    const r = h.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  });
  if (!handle || !box) throw new Error("no τ tab to drag");
  // When the tab first moves after the press, and how many thresholds the
  // page takes while it is held: a live drag re-ranks as it goes.
  await page.evaluate(() => {
    const h = [...document.querySelectorAll(".dp-tau-handle")].find((h) => h.offsetParent);
    const q = window.__q;
    q.tabMoved = null;
    q.liveUpdates = 0;
    q.counting = true;
    new MutationObserver(() => { if (q.tabMoved === null && q.pressed) q.tabMoved = performance.now(); })
      .observe(h, { attributes: true, attributeFilter: ["style"] });
    document.addEventListener("dp-threshold", () => { if (q.counting) q.liveUpdates += 1; });
  });
  await page.mouse.move(handle.x, handle.y);
  await page.evaluate(() => window.__q.record(true));
  const pressed = await page.evaluate(() => (window.__q.pressed = performance.now()));
  await page.mouse.down();
  for (let step = 0; step < 120; step += 1) {
    const along = step < 60 ? step / 60 : (120 - step) / 60;
    await page.mouse.move(box.x + box.width * (0.1 + 0.8 * along), handle.y);
    await page.waitForTimeout(10);
  }
  const held = await page.evaluate(() => { window.__q.counting = false; return window.__q.liveUpdates; });
  const t0 = await now(page);
  await page.mouse.up();
  const frames = await page.evaluate(() => { window.__q.record(false); return window.__q.frames; });
  const released = await settle(page, t0);
  const moved = await page.evaluate(() => window.__q.tabMoved);
  return { ...frameStats(frames), release_blocked_ms: released.blocked_ms,
    start_ms: moved === null ? null : Math.round(moved - pressed), live_updates: held };
}

/** Pan the chart for about a second, as a reader dragging it does: frames
 * per second and how long the page was blocked. */
export async function panChart(page) {
  const box = await plotArea(page);
  if (!box) throw new Error("no chart to pan");
  const x = box.x + box.width * 0.6, y = box.y + box.height * 0.5;
  const t0 = await now(page);
  await page.evaluate(() => window.__q.record(true));
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 0; step < 60; step += 1) {
    await page.mouse.move(x - step * 3, y + (step % 10) * 3);
    await page.waitForTimeout(16);
  }
  await page.mouse.up();
  const frames = await page.evaluate(() => { window.__q.record(false); return window.__q.frames; });
  const done = await settle(page, t0);
  return { ...frameStats(frames), blocked_ms: done.blocked_ms };
}

/** Sweep the pointer across the plotted curves. */
export async function hover(page) {
  const box = await plotArea(page);
  if (!box) throw new Error("no chart to hover");
  const t0 = await now(page);
  // A hover only restyles the drawn curves in place; a Plotly restyle here
  // means the in-place path missed a curve and fell back to a full redraw.
  await page.evaluate(() => {
    const q = window.__q;
    q.restyles = 0;
    if (!q.restyleCounted) {
      q.restyleCounted = true;
      const restyle = Plotly.restyle;
      Plotly.restyle = function () { q.restyles += 1; return restyle.apply(this, arguments); };
    }
    q.record(true);
  });
  for (let step = 0; step < 80; step += 1) {
    await page.mouse.move(
      box.x + box.width * (0.05 + (0.9 * step) / 80),
      box.y + box.height * (0.3 + 0.4 * ((step % 20) / 20)),
    );
    await page.waitForTimeout(25);
  }
  const frames = await page.evaluate(() => { window.__q.record(false); return window.__q.frames; });
  const restyles = await page.evaluate(() => window.__q.restyles);
  await page.mouse.move(0, 0);
  const done = await settle(page, t0);
  return { ...frameStats(frames), blocked_ms: done.blocked_ms, restyles };
}

/** The chart drawn on screen: whether it shows, with axes, and how many curves. */
export const chartState = (page) => page.evaluate(() => {
  const c = [...document.querySelectorAll(".dp-chart")].find((c) => c.data) || document.querySelector(".dp-chart");
  const shown = (c && c.data || []).filter((t) => t.visible !== false && t.visible !== "legendonly").length;
  return {
    curves: shown,
    visible: !!(c && c.offsetParent && c.querySelector(".main-svg")),
    axes: !!(c && c.offsetParent && c.querySelector(".xtick")),
  };
});

export function median(values) {
  const sorted = values.filter((v) => typeof v === "number").sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
}

/** Open one of OPEN_STATES: its click or hover and, for search, its query with the results in. */
export async function openState(page, state) {
  if (state.hover) {
    await page.locator(state.hover).first().hover();
    await page.locator(":popover-open").first().waitFor();
    return;
  }
  await page.locator(state.click).first().click();
  if (!state.query) return;
  await page.locator("pagefind-input input").fill(state.query);
  await page.locator("pagefind-results .pf-result").first().waitFor();
}
