// The heavy things a reader does on Explore, each measured as the median of
// REPEATS cold runs and held to its limit in budgets.json.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { test, expect } from "@playwright/test";
import { EXPLORE, REPEATS } from "../quality.config.js";
import {
  act, browserMemoryMb, chartState, comparisonHover, curveLimit, dragTau, firstMetric, heapMb, hover,
  median, now, openPage, otherMetric, panChart, pickBudget, settle,
} from "./probe.js";

const OUT = process.env.QUALITY_OUT || "out";
const BUDGETS = JSON.parse(readFileSync(new URL("../budgets.json", import.meta.url), "utf8"));

test.describe.configure({ mode: "serial" });

/** One cold visit to Explore and every heavy operation on it, in order. */
async function oneRun(browser) {
  const { context, page, cdp, transfer } = await openPage(browser);
  await page.goto(EXPLORE, { waitUntil: "load" });
  const loaded = await settle(page, 0, 1000);
  const probe = await page.evaluate(() => ({ chartReady: window.__q.chartReady, cls: window.__q.cls }));
  const run = {
    "load.chart_ready_ms": Math.round(probe.chartReady ?? NaN),
    "load.blocked_ms": loaded.blocked_ms,
    "load.layout_shift": Math.round(probe.cls * 1000) / 1000,
    "load.heap_mb": await heapMb(cdp),
  };
  let before = transfer.bytes;
  let r = await act(page, pickBudget("10-0"));
  run["switch_budget_new.blocked_ms"] = r.blocked_ms;
  run["switch_budget_new.settled_ms"] = r.settled_ms;
  run["switch_budget_new.fetched_kb"] = Math.round((transfer.bytes - before) / 1024);
  r = await act(page, pickBudget("1-0"));
  run["switch_budget_seen.blocked_ms"] = r.blocked_ms;
  run["switch_budget_seen.settled_ms"] = r.settled_ms;
  r = await act(page, curveLimit("all"));
  run["show_all_curves.blocked_ms"] = r.blocked_ms;
  const pan = await panChart(page);
  run["pan_all_curves.fps"] = pan.fps;
  run["pan_all_curves.slow_frames"] = pan.slow_frames;
  run["pan_all_curves.blocked_ms"] = pan.blocked_ms;
  const sweepAll = await hover(page);
  run["hover_all_curves.fps"] = sweepAll.fps;
  run["hover_all_curves.slow_frames"] = sweepAll.slow_frames;
  run["hover_all_curves.blocked_ms"] = sweepAll.blocked_ms;
  run["hover_all_curves.restyles"] = sweepAll.restyles;
  await heapMb(cdp);
  run["hover_all_curves.browser_mb"] = await browserMemoryMb(browser);
  r = await act(page, pickBudget("10-0"));
  run["switch_budget_all_curves.blocked_ms"] = r.blocked_ms;
  await act(page, pickBudget("1-0"));
  r = await act(page, curveLimit("10"));
  run["back_to_10_curves.blocked_ms"] = r.blocked_ms;
  const drag = await dragTau(page);
  run["tau_drag.fps"] = drag.fps;
  run["tau_drag.slow_frames"] = drag.slow_frames;
  run["tau_drag.release_blocked_ms"] = drag.release_blocked_ms;
  run["tau_drag.start_ms"] = drag.start_ms;
  run["tau_drag.live_updates"] = drag.live_updates;
  r = await act(page, otherMetric);
  run["switch_metric.blocked_ms"] = r.blocked_ms;
  await firstMetric(page);
  await settle(page, await now(page));
  const sweep = await hover(page);
  run["hover.fps"] = sweep.fps;
  run["hover.blocked_ms"] = sweep.blocked_ms;
  run["hover.restyles"] = sweep.restyles;
  run["interaction.inp_ms"] = await page.evaluate(() => Math.max(0, ...window.__q.inp));
  const heapBefore = await heapMb(cdp);
  // Forty switches through whichever budgets the release offers.
  const offered = await page.locator('label[for*="budget-subinversesamplesizedelta-eps-"]:visible').evaluateAll(
    (labels) => [...new Set(labels.map((label) => label.htmlFor.split("-eps-")[1]))]);
  for (let switches = 0; switches < 40; switches += 1) {
    await act(page, pickBudget(offered[(switches + 1) % offered.length]));
  }
  run["memory.growth_after_40_switches_mb"] = Math.round((await heapMb(cdp) - heapBefore) * 10) / 10;
  const compared = await comparisonHover(page);
  run["comparison_hover.fps"] = compared.fps;
  run["comparison_hover.blocked_ms"] = compared.blocked_ms;
  run["comparison_hover.redraws"] = compared.redraws;
  await context.close();
  return run;
}

test("Explore stays within its interaction budgets", async ({ browser }) => {
  test.setTimeout(15 * 60 * 1000);
  const runs = [];
  for (let i = 0; i < REPEATS; i += 1) runs.push(await oneRun(browser));
  const medians = Object.fromEntries(Object.keys(runs[0]).map((key) => [key, median(runs.map((run) => run[key]))]));
  mkdirSync(OUT, { recursive: true });
  writeFileSync(`${OUT}/interactions.json`, JSON.stringify({ medians, runs }, null, 2));
  for (const [key, limit] of Object.entries(BUDGETS.explore)) {
    const value = medians[key];
    expect.soft(value, `${key} has no measurement`).not.toBeNull();
    if (limit.max !== undefined) expect.soft(value, `${key} = ${value}, limit ${limit.max}`).toBeLessThanOrEqual(limit.max);
    if (limit.min !== undefined) expect.soft(value, `${key} = ${value}, limit ${limit.min}`).toBeGreaterThanOrEqual(limit.min);
  }
});

test("Switching ε never repeats an algorithm's name in its row", async ({ browser }) => {
  const { context, page } = await openPage(browser);
  await page.goto(EXPLORE, { waitUntil: "load" });
  await settle(page, 0, 1000);
  for (let round = 0; round < 3; round += 1) {
    for (const epsilon of ["10-0", "1-0"]) await act(page, pickBudget(epsilon));
  }
  const repeated = await page.evaluate(() => [...document.querySelectorAll(".dp-leaderboard td.dp-alg")]
    .filter((cell) => [...cell.childNodes].filter((n) => n.nodeType === 3 && n.data.trim()).length > 1)
    .map((cell) => cell.innerText.slice(0, 80)));
  expect(repeated, "algorithm cells whose name appears more than once").toEqual([]);
  await context.close();
});

test("Hiding every curve keeps the chart, empty, at its scale", async ({ browser }) => {
  const { context, page } = await openPage(browser);
  await page.goto(EXPLORE, { waitUntil: "load" });
  await settle(page, 0, 1000);
  for (let i = 0; i < 40; i += 1) {
    const clicked = await page.evaluate(() => {
      const b = [...document.querySelectorAll("button")].find((x) => x.offsetParent &&
        /^Hide /.test(x.getAttribute("aria-label") || x.textContent.trim()));
      if (b) b.click();
      return !!b;
    });
    if (!clicked) break;
    await page.waitForTimeout(60);
  }
  await page.waitForTimeout(1000);
  const chart = await chartState(page);
  expect(chart.curves, "every curve hidden").toBe(0);
  expect(chart.visible && chart.axes, "the chart and its axes stay on screen").toBe(true);
  await context.close();
});

test("Filtering out every Instance keeps the chart, empty, at its scale", async ({ browser }) => {
  const { context, page } = await openPage(browser);
  await page.goto(EXPLORE, { waitUntil: "load" });
  await settle(page, 0, 1000);
  await page.evaluate(() => {
    // Narrow the first axis to its lowest cut and the second to its highest: no Instance is in both.
    const [first, second] = document.querySelectorAll(".dp-numeric-slider");
    first.noUiSlider.set([0, 0]);
    second.noUiSlider.set([null, Infinity]);
    second.noUiSlider.set([Infinity, null]);
  });
  await page.waitForTimeout(1200);
  const left = await page.evaluate(() => {
    const n = document.querySelector("[aria-controls='instance-filters'] .dp-control-count");
    return n && /^[\d,]+ of [\d,]+$/.test(n.textContent.trim()) ? parseInt(n.textContent.replace(/,/g, ""), 10) : null;
  });
  expect(left, "no Instance left").toBe(0);
  const chart = await chartState(page);
  expect(chart.visible && chart.axes, "the chart and its axes stay on screen").toBe(true);
  await context.close();
});

/** The threshold the chart's standings are read at, as the page states it. */
const readAt = (page) => page.evaluate(() => {
  const section = [...document.querySelectorAll("section.dp-metric")].find((s) => !s.hidden);
  const heading = section && section.querySelector(".dp-score-heading");
  return heading ? heading.textContent : null;
});

test("Holding the τ tab still updates the reading before release", async ({ browser }) => {
  const { context, page } = await openPage(browser);
  await page.goto(EXPLORE, { waitUntil: "load" });
  await settle(page, 0, 1000);
  const start = await readAt(page);
  const handle = await page.locator(".dp-tau-handle:visible").first().boundingBox();
  const area = await page.locator(".dp-chart .nsewdrag").first().boundingBox();
  await page.mouse.move(handle.x + handle.width / 2, handle.y + handle.height / 2);
  await page.mouse.down();
  for (let i = 1; i <= 20; i += 1) {
    await page.mouse.move(handle.x + (area.x + area.width * 0.6 - handle.x) * (i / 20), handle.y + handle.height / 2);
    await page.waitForTimeout(15);
  }
  await page.waitForTimeout(1500);
  const held = await readAt(page);
  await page.mouse.up();
  expect(held, "the reading follows the held tab").not.toBe(start);
  await context.close();
});

test("An emptied chart returns to its own scale, not a reader's zoom", async ({ browser }) => {
  const { context, page } = await openPage(browser);
  await page.goto(EXPLORE, { waitUntil: "load" });
  await settle(page, 0, 1000);
  const range = () => page.evaluate(() => [...document.querySelectorAll(".dp-chart")].find((c) => c.data)._fullLayout.xaxis.range.slice());
  const initial = await range();
  await page.evaluate(() => Plotly.relayout([...document.querySelectorAll(".dp-chart")].find((c) => c.data), { "xaxis.range": [0.2, 0.4] }));
  await page.waitForTimeout(500);
  await page.evaluate(() => {
    // Narrow the first axis to its lowest cut and the second to its highest: no Instance is in both.
    const [first, second] = document.querySelectorAll(".dp-numeric-slider");
    first.noUiSlider.set([0, 0]);
    second.noUiSlider.set([null, Infinity]);
    second.noUiSlider.set([Infinity, null]);
  });
  await page.waitForTimeout(1500);
  expect(await range()).toEqual(initial);
  await context.close();
});
