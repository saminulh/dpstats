#!/usr/bin/env node
// The site quality run: one command for every check, one report.
//
//   node run.mjs --site <built site> [--label <name>]
//   node run.mjs --site <dir> --build --release <distribution dir> [--revision <rev>] [--cold]
//   add --update-baseline to shrink the finding baselines, --update-screenshots
//   to accept the pages' current look
//   --part timing runs only what measures time (Lighthouse and the interaction,
//   idle and load-memory budgets), one browser at a time; --part functional
//   runs the rest, across QUALITY_WORKERS browsers, on any machine
//
// Builds the site first when asked (time and peak memory), serves it
// compressed as a static host does, runs Lighthouse CI against the load
// budgets, Stylelint, Vale and the Playwright checks (Explore interaction budgets,
// colour contrast, accessibility), appends the headline numbers to the run history, and
// writes out/index.html. Exits non-zero when any budget or check fails.
// History and the dashboard database live in QUALITY_HOME (default
// ~/.dp-site-quality), outside every checkout.
import { spawn, spawnSync } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "node:net";
import { parseArgs } from "node:util";
import { chromium } from "@playwright/test";
import { MACHINE, REPEATS } from "./quality.config.js";
import { loadPages } from "./pages.mjs";
import { renderReport } from "./report.mjs";
import { lintProse } from "./prose.mjs";
import { lintStylesheet } from "./stylesheet.mjs";
import { scanSite, sizeFailures } from "./sitesize.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const LIBRARY = resolve(HERE, "../../..");
const HOME = process.env.QUALITY_HOME || join(homedir(), ".dp-site-quality");
const OUT = join(HERE, "out");
const BIN = join(HERE, "node_modules/.bin");

const { values: args } = parseArgs({
  options: {
    site: { type: "string" },
    label: { type: "string", default: "" },
    build: { type: "boolean", default: false },
    cold: { type: "boolean", default: false },
    release: { type: "string" },
    revision: { type: "string", default: "" },
    mkdocs: { type: "string" },
    "update-baseline": { type: "boolean", default: false },
    "update-screenshots": { type: "boolean", default: false },
    "skip-lighthouse": { type: "boolean", default: false },
    part: { type: "string", default: "all" },
  },
});
// The checks that measure time: their budgets hold on the machine they were set
// on, with no second browser beside them.
const TIMING_TESTS = ["tests/interactions.spec.js", "tests/idle.spec.js", "tests/load-memory.spec.js"];
const PARTS = ["all", "functional", "timing"];
if (!PARTS.includes(args.part)) {
  console.error(`--part is one of ${PARTS.join(", ")}`);
  process.exit(2);
}
const TIMING = args.part !== "functional";
const FUNCTIONAL = args.part !== "timing";
if (!args.site) {
  console.error("usage: node run.mjs --site <built site dir> [--build --release <dir>] [--label <name>]");
  process.exit(2);
}
const SITE = resolve(args.site);
const BUDGETS = JSON.parse(readFileSync(join(HERE, "budgets.json"), "utf8"));
rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });
mkdirSync(HOME, { recursive: true });

const git = (...a) => spawnSync("git", ["-C", LIBRARY, ...a], { encoding: "utf8" }).stdout.trim();
const report = {
  when: new Date().toISOString(),
  label: args.label,
  commit: git("rev-parse", "--short", "HEAD"),
  dirty: git("status", "--porcelain", "--", "docs/site").length > 0,
  machine: MACHINE,
  repeats: REPEATS,
  part: args.part,
  site: SITE,
};

const failures = [];

// --- Build: wall time and peak memory, cold (prepared data rebuilt) and warm.
// Limits in budgets.json "build"; the reference release is the scale guard.
function timedBuild() {
  const mkdocs = args.mkdocs || join(LIBRARY, ".venv/bin/mkdocs");
  const env = {
    ...process.env,
    PYTHONPATH: LIBRARY,
    DOCS_LOCAL_RELEASE_ROOT: resolve(args.release),
    DOCS_LIB_DOCS_DIR: join(LIBRARY, "docs"),
    ...(args.revision ? { DOCS_RECREATE_RELEASE_REVISION: args.revision } : {}),
  };
  const start = Date.now();
  const run = spawnSync("/usr/bin/time", ["-l", mkdocs, "build", "--strict", "-f", "docs/site/mkdocs.yml", "-d", SITE],
    { cwd: LIBRARY, env, encoding: "utf8", maxBuffer: 1 << 28 });
  if (run.status !== 0) {
    console.error(run.stderr.slice(-4000));
    throw new Error("the site build failed");
  }
  const peak = /(\d+)\s+maximum resident set size/.exec(run.stderr);
  return { elapsed_s: Math.round((Date.now() - start) / 100) / 10, peak_memory_gb: peak ? Math.round(Number(peak[1]) / 1e7) / 100 : null };
}
if (args.build) {
  if (!args.release) throw new Error("--build needs --release <distribution dir>");
  report.build = {};
  if (args.cold) {
    rmSync(join(LIBRARY, "docs/site/.prepared"), { recursive: true, force: true });
    report.build.cold = timedBuild();
  }
  report.build.warm = timedBuild();
  for (const [kind, measured] of Object.entries(report.build)) {
    const limit = (BUDGETS.build || {})[kind] || {};
    for (const key of ["elapsed_s", "peak_memory_gb"]) {
      if (limit[key] !== undefined && measured[key] > limit[key]) failures.push(`the ${kind} build's ${key} (${measured[key]} > ${limit[key]})`);
    }
  }
}
if (!existsSync(join(SITE, "index.html"))) throw new Error(`${SITE} holds no built site`);
const LOAD_PAGES = loadPages(SITE);
report.load_pages = LOAD_PAGES;

// --- Serve the site compressed, as a static host does.
const port = await new Promise((done) => {
  const probe = createServer().listen(0, "127.0.0.1", () => { const p = probe.address().port; probe.close(() => done(p)); });
});
const BASE = `http://127.0.0.1:${port}/`;
const server = spawn(join(BIN, "serve"), ["-l", `tcp://127.0.0.1:${port}`, "--no-request-logging", "--no-clipboard", SITE],
  { stdio: "ignore" });
for (let i = 0; i < 100; i += 1) {
  if (await fetch(BASE).then((r) => r.ok, () => false)) break;
  await new Promise((r) => setTimeout(r, 100));
}

// --- The whole site's size: every page and file, not only the measured ones.
if (FUNCTIONAL) {
  const size = scanSite(SITE);
  writeFileSync(join(OUT, "site-size.json"), JSON.stringify(size, null, 2));
  const tooBig = sizeFailures(size, BUDGETS.site);
  if (tooBig.length) failures.push("a site-size limit (" + tooBig.join("; ") + ")");
}

// --- Lighthouse CI: first loads of the named pages, against the load budgets.
// Lighthouse runs Chrome's headless shell, which Playwright installs beside its
// Chromium: full Chrome paints only while the Mac's display is awake, so a run
// left unattended would fail every page with "no first paint".
function headlessShell() {
  let dir = dirname(chromium.executablePath());
  while (!/^chromium-\d+$/.test(basename(dir))) {
    if (dirname(dir) === dir) throw new Error("cannot find Playwright's Chromium install");
    dir = dirname(dir);
  }
  const shell = join(dirname(dir), basename(dir).replace("chromium-", "chromium_headless_shell-"));
  const find = (at) => {
    for (const name of existsSync(at) ? readdirSync(at) : []) {
      const path = join(at, name);
      if (/^chrome-headless-shell(\.exe)?$/.test(name) && !statSync(path).isDirectory()) return path;
      if (statSync(path).isDirectory()) { const found = find(path); if (found) return found; }
    }
    return null;
  };
  const found = find(shell);
  if (!found) throw new Error(`no Chrome headless shell under ${shell}: run \`npx playwright install chromium\` in docs/site/quality`);
  return found;
}
const lhciDir = join(OUT, "lighthouse");
function lighthouseConfig() {
  const matrix = Object.entries(LOAD_PAGES).map(([name, path]) => ({
    matchingUrlPattern: `^${(BASE + path).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`,
    assertions: Object.fromEntries(Object.entries(BUDGETS.load[name] || {}).map(([audit, limit]) =>
      [audit, ["error", { maxNumericValue: limit, aggregationMethod: "median" }]])),
  }));
  return {
    ci: {
      collect: {
        url: Object.values(LOAD_PAGES).map((path) => BASE + path),
        numberOfRuns: REPEATS,
        chromePath: headlessShell(),
        settings: {
          formFactor: "desktop",
          screenEmulation: { mobile: false, width: 1440, height: 900, deviceScaleFactor: 1, disabled: false },
          throttlingMethod: "devtools",
          throttling: {
            cpuSlowdownMultiplier: MACHINE.cpuSlowdown,
            requestLatencyMs: MACHINE.latencyMs,
            downloadThroughputKbps: MACHINE.downloadKbps,
            uploadThroughputKbps: MACHINE.uploadKbps,
          },
          onlyCategories: ["performance"],
        },
      },
      assert: { assertMatrix: matrix },
      upload: { target: "filesystem", outputDir: lhciDir },
    },
  };
}
if (TIMING && !args["skip-lighthouse"]) {
  const config = join(OUT, "lighthouserc.json");
  writeFileSync(config, JSON.stringify(lighthouseConfig(), null, 2));
  const lhci = (...a) => spawnSync(join(BIN, "lhci"), [...a, `--config=${config}`], { cwd: OUT, stdio: "inherit" }).status;
  rmSync(join(OUT, ".lighthouseci"), { recursive: true, force: true });
  if (lhci("collect") !== 0) failures.push("Lighthouse could not load the pages");
  else {
    if (lhci("assert") !== 0) failures.push("a page-load budget");
    lhci("upload");
    // The local dashboard keeps the history of every run, when it is running.
    const tokenFile = join(HOME, "lhci-project.json");
    if (existsSync(tokenFile)) {
      const { token, server: url } = JSON.parse(readFileSync(tokenFile, "utf8"));
      if (await fetch(url).then((r) => r.ok, () => false)) {
        spawnSync(join(BIN, "lhci"), ["upload", `--config=${config}`, "--target=lhci", `--serverBaseUrl=${url}`, `--token=${token}`,
          "--ignoreDuplicateBuildFailure"], { cwd: OUT, stdio: "inherit",
          env: { ...process.env, LHCI_BUILD_CONTEXT__CURRENT_HASH: git("rev-parse", "HEAD"),
            LHCI_BUILD_CONTEXT__COMMIT_MESSAGE: args.label || git("log", "-1", "--format=%s"),
            LHCI_BUILD_CONTEXT__CURRENT_BRANCH: git("rev-parse", "--abbrev-ref", "HEAD") } });
        report.dashboard = url;
      }
    }
  }
}

// --- Stylelint: literal colours and type sizes, against a shrinking baseline.
if (FUNCTIONAL) {
  const stylesheet = await lintStylesheet({ update: args["update-baseline"] });
  writeFileSync(join(OUT, "stylesheet.json"), JSON.stringify(stylesheet, null, 2));
  if (stylesheet.regressions.length) failures.push("the stylesheet lint");
}

// --- Vale: prose errors against a shrinking baseline; the rest reported.
if (FUNCTIONAL) {
  const prose = lintProse({ site: SITE, update: args["update-baseline"] });
  writeFileSync(join(OUT, "prose.json"), JSON.stringify(prose, null, 2));
  if (prose.regressions.length) failures.push("the prose lint");
}

// --- Playwright: Explore interaction budgets, colour contrast and accessibility.
// Screenshots are accepted when asked, and on this machine's first run.
report.screenshotsAccepted = args["update-screenshots"] || !existsSync(join(HOME, "screenshots"));
const specs = readdirSync(join(HERE, "tests")).filter((name) => name.endsWith(".spec.js")).map((name) => `tests/${name}`);
const selected = args.part === "all" ? [] : specs.filter((spec) => TIMING_TESTS.includes(spec) === TIMING);
const tests = spawnSync(join(BIN, "playwright"), ["test", ...selected, ...(report.screenshotsAccepted ? ["--update-snapshots=changed"] : [])], {
  cwd: HERE,
  stdio: "inherit",
  env: { ...process.env, QUALITY_BASE_URL: BASE, QUALITY_SITE: SITE, QUALITY_OUT: OUT, QUALITY_HOME: HOME,
    // Only a functional-only run may share the machine between browsers.
    QUALITY_WORKERS: args.part === "functional" ? process.env.QUALITY_WORKERS || "1" : "1",
    ...(args["update-baseline"] ? { QUALITY_UPDATE_BASELINE: "1" } : {}) },
});
if (tests.status !== 0) failures.push("a browser check");
server.kill();

// --- Report and history.
report.failures = failures;
const page = renderReport({ report, out: OUT, budgets: BUDGETS, historyPath: join(HOME, "history.jsonl") });
appendFileSync(join(HOME, "history.jsonl"), JSON.stringify(page.headline) + "\n");
writeFileSync(join(OUT, "index.html"), page.html);
writeFileSync(join(OUT, "summary.json"), JSON.stringify(page.headline, null, 2));
console.log(`\nreport: ${join(OUT, "index.html")}`);
console.log(failures.length ? `FAILED: ${failures.join("; ")}` : "every check passed");
process.exit(failures.length ? 1 : 0);
