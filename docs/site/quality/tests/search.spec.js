// Site search returns the right page for ten fixed queries. Each query is typed
// into the search panel of the served build, so the check covers the index,
// the ranking site.js sets and the results the reader sees. An expectation is
// the least a reader should get; a ranking change that breaks one changes the
// ranking, never the expectation.
import { test, expect } from "@playwright/test";

const METHODOLOGY = "/methodology/";
const QUERIES = [
  ["privacy bud", "the Privacy budgets tutorial in the top 3", (r) => top(r, 3).includes("/library/tutorials/privacy_budgets_and_accounting/")],
  ["success rate", "the Methodology page in the top 3", (r) => top(r, 3).includes(METHODOLOGY)],
  ["AdaSSP", "the AdaSSP algorithm page first", (r) => r[0]?.path === "/linear_regression/algorithms/AdaSSP/"],
  ["Huber", "a Huber page first", (r) => /Huber/.test(r[0]?.title)],
  ["problem instance", "the Methodology page in the top 3", (r) => top(r, 3).includes(METHODOLOGY)],
  ["baseline", "the Metrics and baselines tutorial in the top 5", (r) => top(r, 5).includes("/library/tutorials/metrics_and_baselines/")],
  ["tau", "the Methodology page in the top 5", (r) => top(r, 5).includes(METHODOLOGY)],
  ["Kendall", "both DPKendall pages in the top 5", (r) => ["linear_regression", "single_coefficient_estimation"]
    .every((task) => top(r, 5).includes(`/${task}/transformations/DPKendall/`))],
  ["corpus", "the Corpus landing first", (r) => r[0]?.path === "/corpus/"],
  ["install", "the Install tutorial first", (r) => r[0]?.path === "/library/tutorials/install_and_fetch/"],
];

const top = (results, n) => results.slice(0, n).map((result) => result.path);

/** The first five results for one query, once the panel shows them. */
async function search(page, query) {
  await page.locator("pagefind-input input").fill(query);
  await expect(page.locator("pagefind-summary")).toContainText(query);
  const results = page.locator("pagefind-results .pf-result");
  await expect.poll(async () => {
    const shown = await results.count();
    return shown > 0 && (await results.locator(".pf-result-link").count()) >= Math.min(shown, 5);
  }).toBe(true);
  return results.evaluateAll((items) => items.slice(0, 5).map((item) => {
    const link = item.querySelector(".pf-result-link");
    return {
      path: new URL(link.href).pathname,
      title: link.textContent.trim(),
      section: item.querySelector(".dp-search-section")?.textContent.trim(),
    };
  }));
}

test("ten fixed queries find their page", async ({ page }) => {
  await page.goto("/");
  await page.click("[data-bs-target='#search-panel']");
  const failures = [];
  for (const [query, expected, holds] of QUERIES) {
    const results = await search(page, query);
    const paths = top(results, 5);
    const listed = results.map((r) => `${r.title} [${r.section}]`).join(" | ");
    if (!holds(results)) failures.push(`"${query}": expected ${expected}; got ${listed}`);
    if (new Set(paths).size !== paths.length) failures.push(`"${query}": a page appears twice in ${listed}`);
    if (results.some((r) => !r.section)) failures.push(`"${query}": a result has no section label in ${listed}`);
  }
  expect(failures).toEqual([]);
});

test("the section filter narrows results", async ({ page }) => {
  await page.goto("/");
  await page.click("[data-bs-target='#search-panel']");
  await search(page, "privacy bud");
  const filters = page.locator("pagefind-filter-pane");
  await filters.locator("summary").filter({ hasText: "Section" }).click();
  await filters.getByRole("checkbox", { name: /^API/ }).check();
  await expect.poll(async () => {
    const sections = await page.locator("pagefind-results .pf-result .dp-search-section").allTextContents();
    return sections.length > 0 && sections.every((section) => section.trim() === "API");
  }).toBe(true);
});
