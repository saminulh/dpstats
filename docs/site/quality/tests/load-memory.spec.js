// Memory while each measured page first loads: the JavaScript heap sampled
// every 100 ms until the page settles, so a page that balloons and falls back
// is caught, not only what it holds afterwards; and the elements it builds.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { test, expect } from "@playwright/test";
import { REPEATS } from "../quality.config.js";
import { loadPages } from "../pages.mjs";
import { median, openPage, settle } from "./probe.js";

const OUT = process.env.QUALITY_OUT || "out";
const BUDGETS = JSON.parse(readFileSync(new URL("../budgets.json", import.meta.url), "utf8"));
const measured = {};

async function oneLoad(browser, path) {
  const { context, page, cdp } = await openPage(browser);
  let peak = 0;
  let sampling = true;
  const sample = async () => {
    while (sampling) {
      try {
        const { metrics } = await cdp.send("Performance.getMetrics");
        const used = metrics.find((m) => m.name === "JSHeapUsedSize").value;
        if (used > peak) peak = used;
      } catch {
        return;
      }
      await new Promise((r) => setTimeout(r, 100));
    }
  };
  const sampler = sample();
  await page.goto(path, { waitUntil: "load" });
  await settle(page, 0, 1000);
  sampling = false;
  await sampler;
  const { metrics } = await cdp.send("Performance.getMetrics");
  const nodes = metrics.find((m) => m.name === "Nodes").value;
  await context.close();
  return { peak_heap_mb: Math.round(peak / 1e5) / 10, dom_nodes: nodes };
}

for (const [name, path] of Object.entries(loadPages(process.env.QUALITY_SITE))) {
  test(`${name}: memory while loading`, async ({ browser }) => {
    const runs = [];
    for (let i = 0; i < REPEATS; i += 1) runs.push(await oneLoad(browser, path));
    const medians = { peak_heap_mb: median(runs.map((r) => r.peak_heap_mb)), dom_nodes: median(runs.map((r) => r.dom_nodes)) };
    measured[name] = medians;
    for (const [key, limit] of Object.entries((BUDGETS.load_memory || {})[name] || {})) {
      expect.soft(medians[key], `${name} ${key} = ${medians[key]}, limit ${limit}`).toBeLessThanOrEqual(limit);
    }
  });
}

test.afterAll(() => {
  mkdirSync(OUT, { recursive: true });
  writeFileSync(`${OUT}/load-memory.json`, JSON.stringify(measured, null, 2));
});
