// Colour contrast at the token level: every pair in CONTRAST_PAIRS, resolved
// by the browser in each theme, so Tabler variables, color-mix() and the
// cascade layers decide each value exactly as they do on a page. axe checks
// the text it finds on rendered pages; this checks every pair, including the
// chart marks axe cannot read.
import { test, expect } from "@playwright/test";
import { CONTRAST_FLOORS, CONTRAST_PAIRS, THEMES } from "../quality.config.js";

/** Each token in the page as opaque sRGB bytes, laid over --dp-bg. */
async function resolve(page, tokens) {
  return page.evaluate((names) => {
    const probe = document.createElement("div");
    document.querySelector(".dp-content, main, body").appendChild(probe);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 1;
    const ink = canvas.getContext("2d", { willReadFrequently: true });
    const paint = (name) => {
      probe.style.backgroundColor = "";
      probe.style.backgroundColor = `var(${name})`;
      return getComputedStyle(probe).backgroundColor;
    };
    const base = paint("--dp-bg");
    const out = {};
    for (const name of names) {
      if (!getComputedStyle(probe).getPropertyValue(name).trim()) { out[name] = null; continue; }
      ink.clearRect(0, 0, 1, 1);
      ink.fillStyle = base;
      ink.fillRect(0, 0, 1, 1);
      ink.fillStyle = paint(name);
      ink.fillRect(0, 0, 1, 1);
      out[name] = Array.from(ink.getImageData(0, 0, 1, 1).data.slice(0, 3));
    }
    probe.remove();
    return out;
  }, tokens);
}

/** WCAG 2.x contrast ratio between two sRGB byte triples. */
function ratio(one, two) {
  const lum = (rgb) => rgb.map((v) => v / 255)
    .map((c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4))
    .reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0);
  const [a, b] = [lum(one), lum(two)].sort((x, y) => y - x);
  return (a + 0.05) / (b + 0.05);
}

const hex = (rgb) => `#${rgb.map((v) => v.toString(16).padStart(2, "0")).join("")}`;

for (const theme of THEMES) {
  test(`colour pairs (${theme})`, async ({ browser }) => {
    const context = await browser.newContext();
    await context.addInitScript((t) => { try { localStorage.setItem("dp-theme-choice", t); } catch (e) {} }, theme);
    const page = await context.newPage();
    await page.goto("", { waitUntil: "load" });
    await expect(page.locator("html")).toHaveAttribute("data-bs-theme", theme);
    const tokens = [...new Set(Object.values(CONTRAST_PAIRS).flat(2))];
    const colours = await resolve(page, tokens);
    const failures = [];
    for (const [kind, pairs] of Object.entries(CONTRAST_PAIRS)) {
      for (const [fg, bg] of pairs) {
        if (!colours[fg] || !colours[bg]) { failures.push(`${fg} on ${bg}: token not defined`); continue; }
        const r = ratio(colours[fg], colours[bg]);
        if (r < CONTRAST_FLOORS[kind]) {
          failures.push(`${kind} ${fg} ${hex(colours[fg])} on ${bg} ${hex(colours[bg])}: ${r.toFixed(2)}:1, needs ${CONTRAST_FLOORS[kind]}:1`);
        }
      }
    }
    await context.close();
    expect(failures, failures.join("\n")).toEqual([]);
  });
}
