// Visual regression: one full-page screenshot per template in each theme, the
// states a reader opens, and every template at phone width, each compared with
// the last accepted image. Accepted images live in QUALITY_HOME/screenshots,
// outside every checkout, since they are specific to this machine's fonts and
// to the build they were taken from. The first run records them; a changed
// page fails with a diff in the Playwright report, and
// `run.mjs --update-screenshots` accepts the new look.
import { test, expect } from "@playwright/test";
import { DESKTOP, OPEN_STATES, PHONE, SCREENSHOT_MASK, THEMES } from "../quality.config.js";
import { templatePages } from "../pages.mjs";
import { openState } from "./probe.js";

const SITE = process.env.QUALITY_SITE;

/** Open a page in a theme and wait until its charts have stopped drawing. */
async function settle(browser, path, theme, viewport) {
  const context = await browser.newContext({ viewport, deviceScaleFactor: 1, reducedMotion: "reduce" });
  await context.addInitScript((t) => { try { localStorage.setItem("dp-theme-choice", t); } catch (e) {} }, theme);
  const page = await context.newPage();
  await page.goto(path, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  // A results panel says when it has drawn; then the page must stop changing,
  // so a table still filling or an emphasis pass still running is never
  // photographed.
  await page.waitForFunction(() => [...document.querySelectorAll(".dp-metric")]
    .filter((panel) => !panel.hidden).every((panel) => panel.dataset.loading !== "true"), null, { timeout: 30_000 });
  await quiet(page);
  return { context, page };
}

/** Wait until the page has made no change to its document for a full second. */
async function quiet(page) {
  await page.evaluate(() => new Promise((done) => {
    let timer = setTimeout(finish, 1000);
    const stop = setTimeout(finish, 30_000);
    const watcher = new MutationObserver(() => { clearTimeout(timer); timer = setTimeout(finish, 1000); });
    watcher.observe(document.documentElement, { subtree: true, childList: true, attributes: true, characterData: true });
    function finish() { watcher.disconnect(); clearTimeout(timer); clearTimeout(stop); done(); }
  }));
}

/** Grow the window to the whole page, so the page is laid out once at the
    size it is photographed at: a full-page capture stretches the window
    mid-capture, and anything sized by the window's height then re-lays out. */
async function wholePage(page) {
  const width = page.viewportSize().width;
  for (let attempt = 0; attempt < 4; attempt++) {
    const height = await page.evaluate(() => document.documentElement.scrollHeight);
    if (height === page.viewportSize().height) return;
    await page.setViewportSize({ width, height });
    await quiet(page);
  }
}

async function shoot(page, name, fullPage = true) {
  // The "more below" fade is applied on an animation frame after a resize, so
  // whether a capture catches it is timing; it carries no information.
  await page.addStyleTag({ content: ".dp-more-below { -webkit-mask-image: none !important; mask-image: none !important; }" });
  if (fullPage) await wholePage(page);
  await expect(page).toHaveScreenshot(`${name}.png`, {
    fullPage: false,
    animations: "disabled",
    caret: "hide",
    mask: SCREENSHOT_MASK.map((selector) => page.locator(selector)),
    maxDiffPixels: 0,
    threshold: 0,
  });
}

const slug = (text) => text.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
const pages = templatePages(SITE);

for (const { name, path } of pages) {
  for (const theme of THEMES) {
    test(`screenshot: ${name} (${theme})`, async ({ browser }) => {
      const { context, page } = await settle(browser, path, theme, DESKTOP);
      await shoot(page, `${slug(name)}-${theme}`);
      await context.close();
    });
  }
  test(`screenshot: ${name} (phone)`, async ({ browser }) => {
    const { context, page } = await settle(browser, path, "light", PHONE);
    await shoot(page, `${slug(name)}-phone`);
    await context.close();
  });
}

for (const state of OPEN_STATES) {
  for (const theme of THEMES) {
    test(`screenshot: ${state.template}, ${state.label} (${theme})`, async ({ browser }) => {
      const { path } = pages.find((t) => t.name === state.template);
      const { context, page } = await settle(browser, path, theme, DESKTOP);
      await openState(page, state);
      await quiet(page);
      // An opened overlay is photographed in the window it opens over: a
      // panel that locks page scrolling makes a full-page capture's height vary.
      await shoot(page, `${slug(state.template)}-${slug(state.label)}-${theme}`, false);
      await context.close();
    });
  }
}
