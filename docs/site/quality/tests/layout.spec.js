// Page layout at the site's three named widths (style guide W17, W18): the
// page itself never scrolls sideways, anything wider than the window sits in
// its own named, focusable region, running prose keeps a readable line, and a
// display equation centres on the text column it sits in, and a chart's
// drawing stays inside its box, with the panel beside it no taller, after the
// sidebar folds and the window narrows (W123).
import { test, expect } from "@playwright/test";
import { templatePages } from "../pages.mjs";
import { OPEN_STATES, WIDTHS } from "../quality.config.js";
import { openState } from "./probe.js";

const SITE = process.env.QUALITY_SITE;
// About 75 characters is the measure; a line may run a little past it.
const MAX_PROSE_CHARACTERS = 85;

/** What overflows the window outside a named, focusable region. */
function overflowProblems() {
  const width = document.documentElement.clientWidth;
  const problems = [];
  if (document.documentElement.scrollWidth > width + 1) problems.push(`the page scrolls sideways (${document.documentElement.scrollWidth} > ${width})`);
  const regions = [...document.querySelectorAll("[role=region][aria-label][tabindex], pre[tabindex], .js-plotly-plot")];
  for (const element of document.querySelectorAll(".dp-content *")) {
    const box = element.getBoundingClientRect();
    if (!box.width || box.right <= width + 1) continue;
    if (regions.some((region) => region.contains(element))) continue;
    // Report only the outermost offender, not every child of it.
    if (problems.some((p) => p.element && p.element.contains(element))) continue;
    const name = element.tagName.toLowerCase() + (element.className && typeof element.className === "string" ? "." + element.className.trim().split(/\s+/).join(".") : "");
    problems.push({ element, text: `${name} reaches ${Math.round(box.right)}px` });
  }
  return problems.map((p) => (typeof p === "string" ? p : p.text));
}

/** The longest running-prose line, in characters of the paragraph's own font. */
function longestProse() {
  let longest = { characters: 0, where: "" };
  for (const paragraph of document.querySelectorAll(".dp-content p")) {
    // Results pages carry short leads above full-width charts; their text spans the page (W17).
    if (!paragraph.offsetParent || paragraph.closest("table, figure, .dp-plotted, nav, header, footer, dialog, .dp-results-page")) continue;
    if ((paragraph.textContent || "").trim().length < 120) continue;
    const probe = document.createElement("span");
    probe.textContent = "x".repeat(10);
    probe.style.font = getComputedStyle(paragraph).font;
    probe.style.position = "absolute";
    document.body.append(probe);
    const characters = paragraph.getBoundingClientRect().width / (probe.getBoundingClientRect().width / 10);
    probe.remove();
    if (characters > longest.characters) longest = { characters: Math.round(characters), where: (paragraph.textContent || "").trim().slice(0, 40) };
  }
  return longest;
}

/** Display equations whose centre is off the centre of the text column. */
function offCentreEquations() {
  const prose = document.querySelector(".dp-content p");
  if (!prose) return [];
  const column = prose.getBoundingClientRect();
  return [...document.querySelectorAll(".dp-content div.arithmatex")].filter((block) => block.offsetParent).flatMap((block) => {
    const math = block.querySelector("mjx-container svg, mjx-container mjx-math") || block;
    const box = math.getBoundingClientRect();
    const off = Math.abs((box.left + box.right) / 2 - (column.left + column.right) / 2);
    return off > 4 ? [`an equation is centred ${Math.round(off)}px off its text column`] : [];
  });
}

/** Each shown chart's drawing height, and whether the drawing stays inside its own box. */
function chartBounds() {
  return [...document.querySelectorAll(".js-plotly-plot")].filter((plot) => plot.getBoundingClientRect().height).map((plot) => {
    const box = plot.getBoundingClientRect();
    const drawing = plot.querySelector(".main-svg").getBoundingClientRect();
    return { height: Math.round(drawing.height), box: Math.round(box.height), inside: drawing.bottom <= box.bottom + 1 && drawing.right <= box.right + 1 };
  });
}

/** Panels beside a chart that run taller than the chart they sit beside. */
function railsPastCharts() {
  return [...document.querySelectorAll(".dp-chart-shell .dp-plotted")].filter((rail) => rail.getBoundingClientRect().height).flatMap((rail) => {
    const plot = [...rail.closest(".dp-chart-shell").querySelectorAll(".js-plotly-plot")].find((each) => each.getBoundingClientRect().height);
    if (!plot) return [];
    const railHeight = Math.round(rail.getBoundingClientRect().height);
    const plotHeight = Math.round(plot.getBoundingClientRect().height);
    return railHeight > plotHeight + 1 ? [`a panel beside a ${plotHeight}px chart is ${railHeight}px tall`] : [];
  });
}

/** Open each state a reader can open on this page, from a fresh load, and check the panels beside its charts. */
async function railsInOpenStates(page, name, path) {
  const problems = [];
  for (const state of OPEN_STATES.filter((each) => each.template === name)) {
    await page.goto(path, { waitUntil: "networkidle" });
    await page.waitForTimeout(800);
    await openState(page, state);
    await page.waitForTimeout(800);
    problems.push(...(await page.evaluate(railsPastCharts)).map((problem) => `${state.label}: ${problem}`));
  }
  return problems;
}

/** Fold the sidebar and narrow the window, as a reader would; every chart keeps its height inside its box. */
async function chartsAfterResizing(page) {
  const before = await page.evaluate(chartBounds);
  const toggle = page.locator("#sidebar-toggle");
  if (await toggle.isVisible()) {
    await toggle.click();
    await page.waitForTimeout(800);
  }
  const viewport = page.viewportSize();
  await page.setViewportSize({ width: viewport.width - 200, height: viewport.height });
  await page.waitForTimeout(800);
  const after = await page.evaluate(chartBounds);
  const rails = await page.evaluate(railsPastCharts);
  return rails.concat(after.flatMap((chart, index) =>
    chart.inside && chart.height === (before[index] || chart).height ? [] : [`a chart drawn ${(before[index] || chart).height}px tall draws ${chart.height}px in a ${chart.box}px box after resizing`]));
}

for (const { name, path } of templatePages(SITE)) {
  for (const [label, viewport] of Object.entries(WIDTHS)) {
    test(`layout: ${name} at ${label} (${viewport.width}px)`, async ({ browser }) => {
      const context = await browser.newContext({ viewport, reducedMotion: "reduce" });
      const page = await context.newPage();
      await page.goto(path, { waitUntil: "networkidle" });
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(800);
      const problems = [...(await page.evaluate(overflowProblems)), ...(label === "desktop" ? await page.evaluate(offCentreEquations) : [])];
      const prose = label === "desktop" ? await page.evaluate(longestProse) : { characters: 0 };
      if (label === "desktop") {
        problems.push(...(await page.evaluate(railsPastCharts)), ...(await chartsAfterResizing(page)));
        await page.setViewportSize(viewport);
        problems.push(...(await railsInOpenStates(page, name, path)));
      }
      await context.close();
      expect(problems, problems.join("\n")).toEqual([]);
      expect(prose.characters, `running prose is ${prose.characters} characters a line at "${prose.where}…"`).toBeLessThanOrEqual(MAX_PROSE_CHARACTERS);
    });
  }
}
