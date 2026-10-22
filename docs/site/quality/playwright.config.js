// The browser checks: Explore interaction budgets, colour contrast,
// accessibility and screenshots. run.mjs serves the site and sets
// QUALITY_BASE_URL, QUALITY_SITE, QUALITY_OUT and QUALITY_HOME.
import { homedir } from "node:os";
import { join } from "node:path";
import { defineConfig } from "@playwright/test";

const OUT = process.env.QUALITY_OUT || "out";
const HOME = process.env.QUALITY_HOME || join(homedir(), ".dp-site-quality");
// One worker by default: a timing check shares no machine with a second
// browser. `run.mjs --part functional` may ask for more, since those checks
// measure no time.
const WORKERS = Number(process.env.QUALITY_WORKERS || 1);

export default defineConfig({
  testDir: "tests",
  workers: WORKERS,
  fullyParallel: WORKERS > 1,
  timeout: 120_000,
  reporter: [
    ["list"],
    ["json", { outputFile: `${OUT}/playwright.json` }],
    ["html", { outputFolder: `${OUT}/playwright-report`, open: "never" }],
  ],
  use: {
    baseURL: process.env.QUALITY_BASE_URL,
    browserName: "chromium",
    headless: true,
  },
  outputDir: `${OUT}/playwright-artifacts`,
  // Accepted screenshots: per machine, outside the checkout (tests/visual.spec.js).
  snapshotPathTemplate: `${HOME}/screenshots/{arg}{ext}`,
  updateSnapshots: "none",
});
