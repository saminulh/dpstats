// Each measured page left alone after it loads. A page at rest should run
// nothing: a frame callback, timer or fetch that fires while nobody touches
// the page is a loop, and a loop costs every reader who leaves a tab open.
// Counted over 30 s after the page settles: frame callbacks, timers and
// fetches the page's own code runs, chart redraws, time the page was blocked,
// animations still playing, and how much its memory grew after a forced
// collection. One run per page: a loop fires in every run.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { test, expect } from "@playwright/test";
import { loadPages } from "../pages.mjs";
import { heapMb, now, openPage, settle } from "./probe.js";

const OUT = process.env.QUALITY_OUT || "out";
const BUDGETS = JSON.parse(readFileSync(new URL("../budgets.json", import.meta.url), "utf8"));
const REST_MS = 30_000;
const measured = {};

// Installed before page scripts, after the probe: each frame callback, timer
// and fetch is counted while the window is open. The probe registered its own
// interval before this ran, so that one is not counted.
const COUNTER = `
(() => {
  const I = window.__idle = { on: false, frames: 0, timers: 0, fetches: 0 };
  const raf = window.requestAnimationFrame;
  window.requestAnimationFrame = function (fn) {
    return raf.call(window, function (t) { if (I.on) I.frames += 1; return fn.call(this, t); });
  };
  for (const name of ["setTimeout", "setInterval"]) {
    const original = window[name];
    window[name] = function (fn, ...rest) {
      const counted = typeof fn === "function"
        ? function () { if (I.on) I.timers += 1; return fn.apply(this, arguments); }
        : fn;
      return original.call(window, counted, ...rest);
    };
  }
  const fetch = window.fetch;
  window.fetch = function () { if (I.on) I.fetches += 1; return fetch.apply(this, arguments); };
})();
`;

async function oneRest(browser, path) {
  const { context, page, cdp } = await openPage(browser);
  await context.addInitScript(COUNTER);
  await page.goto(path, { waitUntil: "load" });
  await settle(page, 0, 1000);
  const heapBefore = await heapMb(cdp);
  const t0 = await now(page);
  await page.evaluate(() => { window.__idle.on = true; });
  await page.waitForTimeout(REST_MS);
  const counts = await page.evaluate((start) => {
    const I = window.__idle, q = window.__q;
    I.on = false;
    return {
      frames: I.frames,
      timers: I.timers,
      fetches: I.fetches,
      redraws: q.plot.filter((t) => t >= start).length,
      blocked_ms: Math.round(q.lt.filter(([s]) => s >= start).reduce((a, [, d]) => a + Math.max(0, d - 50), 0)),
      animations: document.getAnimations().filter((a) => a.playState === "running").length,
    };
  }, t0);
  const heapAfter = await heapMb(cdp);
  await context.close();
  return { ...counts, heap_growth_mb: Math.round((heapAfter - heapBefore) * 10) / 10 };
}

for (const [name, path] of Object.entries(loadPages(process.env.QUALITY_SITE))) {
  test(`${name}: at rest after loading`, async ({ browser }) => {
    const rest = await oneRest(browser, path);
    measured[name] = rest;
    for (const [key, limit] of Object.entries((BUDGETS.idle || {})[name] || {})) {
      expect.soft(rest[key], `${name} ${key} = ${rest[key]}, limit ${limit}`).toBeLessThanOrEqual(limit);
    }
  });
}

test.afterAll(() => {
  mkdirSync(OUT, { recursive: true });
  writeFileSync(`${OUT}/idle.json`, JSON.stringify(measured, null, 2));
});
