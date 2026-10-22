// Automated accessibility (axe, WCAG 2.2 A/AA rules) on one page per template,
// in both themes, plus the states a reader opens. Known findings are listed in
// a11y-baseline.json and may only shrink: a new rule failing, or more failures
// of a known one, fails the run. QUALITY_UPDATE_BASELINE=1 rewrites the list.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { OPEN_STATES, THEMES } from "../quality.config.js";
import { templatePages } from "../pages.mjs";
import { openState } from "./probe.js";

const SITE = process.env.QUALITY_SITE;
const OUT = process.env.QUALITY_OUT || "out";
const BASELINE_PATH = new URL("../a11y-baseline.json", import.meta.url);
const UPDATE = process.env.QUALITY_UPDATE_BASELINE === "1";
const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

const findings = {};
// The findings are compared with the baseline once, after every page: the
// file's tests stay in one worker, in order, when the others run in parallel.
test.describe.configure({ mode: "default" });

/** One axe pass, summarised as rule id → failing node count. */
async function audit(page) {
  const result = await new AxeBuilder({ page }).withTags(TAGS).analyze();
  return {
    violations: Object.fromEntries(result.violations.map((v) => [v.id, v.nodes.length])),
    details: result.violations.map((v) => ({ id: v.id, impact: v.impact, help: v.help, url: v.helpUrl, nodes: v.nodes.length,
      examples: v.nodes.slice(0, 3).map((n) => n.target.join(" ")) })),
    needsReview: result.incomplete.map((v) => v.id),
  };
}

async function visit(browser, path, theme) {
  // Settled as the screenshots are: no motion, the network quiet and fonts in,
  // so a table still drawing or a colour mid-transition is not read as a finding.
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: "reduce" });
  await context.addInitScript((t) => { try { localStorage.setItem("dp-theme-choice", t); } catch (e) {} }, theme);
  const page = await context.newPage();
  await page.goto(path, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(1500);
  return { context, page };
}

for (const { name, path } of templatePages(SITE)) {
  for (const theme of THEMES) {
    test(`${name} (${theme})`, async ({ browser }) => {
      const { context, page } = await visit(browser, path, theme);
      findings[`${name} | ${theme}`] = await audit(page);
      await context.close();
    });
  }
}

for (const state of OPEN_STATES) {
  for (const theme of THEMES) {
    test(`${state.template}: ${state.label} (${theme})`, async ({ browser }) => {
      const { path } = templatePages(SITE).find((t) => t.name === state.template);
      const { context, page } = await visit(browser, path, theme);
      await openState(page, state);
      await page.waitForTimeout(600);
      findings[`${state.template}: ${state.label} | ${theme}`] = await audit(page);
      await context.close();
    });
  }
}

test.afterAll(() => {
  mkdirSync(OUT, { recursive: true });
  writeFileSync(`${OUT}/accessibility.json`, JSON.stringify(findings, null, 2));
  const current = Object.fromEntries(Object.entries(findings).map(([k, v]) => [k, v.violations]));
  if (UPDATE || !existsSync(BASELINE_PATH)) {
    writeFileSync(BASELINE_PATH, JSON.stringify(current, null, 2) + "\n");
    return;
  }
  const baseline = JSON.parse(readFileSync(BASELINE_PATH, "utf8"));
  const worse = [];
  for (const [where, rules] of Object.entries(current)) {
    for (const [rule, count] of Object.entries(rules)) {
      const known = (baseline[where] || {})[rule] || 0;
      if (count > known) worse.push(`${where}: ${rule} ${known} → ${count}`);
    }
  }
  writeFileSync(`${OUT}/accessibility-regressions.json`, JSON.stringify(worse, null, 2));
  expect(worse, "accessibility findings beyond the baseline").toEqual([]);
});
