// The standalone browser regressions in tests/unit/docs/*.test.html, run as
// they are when opened by hand: each page drives the real site scripts and
// reports PASS or FAIL in its title. One same-origin host serves the
// checkout, with /site/ mapped onto the built site under test, so the pages
// that read the built site see the one this run checks.
import { existsSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { test, expect } from "@playwright/test";

const LIBRARY = resolve(dirname(fileURLToPath(import.meta.url)), "../../../..");
const PAGES = join(LIBRARY, "tests/unit/docs");
const SITE = process.env.QUALITY_SITE;
// A .localhost name is a secure context, as the hand-run localhost server is.
const HOST = "http://checkout.localhost";

/** The file a request on the checkout host names, or null. */
function fileFor(url) {
  const path = decodeURIComponent(new URL(url).pathname);
  const root = path.startsWith("/site/") ? SITE : LIBRARY;
  let file = join(root, path.startsWith("/site/") ? path.slice("/site".length) : path);
  if (!file.startsWith(root)) return null;
  if (existsSync(file) && statSync(file).isDirectory()) file = join(file, "index.html");
  return existsSync(file) ? file : null;
}

for (const name of readdirSync(PAGES).filter((f) => f.endsWith(".test.html")).sort()) {
  test(`browser page: ${name}`, async ({ browser }) => {
    test.setTimeout(180_000);
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    await context.route(`${HOST}/**`, (route) => {
      const file = fileFor(route.request().url());
      return file ? route.fulfill({ path: file }) : route.fulfill({ status: 404, body: "not found" });
    });
    const page = await context.newPage();
    await page.goto(`${HOST}/tests/unit/docs/${name}`);
    await expect(page).toHaveTitle(/^(PASS|FAIL)/, { timeout: 150_000 });
    const report = (await page.locator("#results").textContent()) || "";
    await context.close();
    expect(report.split("\n").filter((line) => line.startsWith("FAIL")), report).toEqual([]);
  });
}
