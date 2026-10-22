#!/usr/bin/env node
// Vale over the site's prose, against a baseline that only shrinks.
//
//   node prose.mjs [--site <built site>] [--update-baseline]
//
// Lints the library's authored docs (docs/content, reference, tasks and
// submission.md, and the top-level tutorials notebooks) and, given a built site, its generated API reference. Only
// error-level alerts are held to prose-baseline.json, counted per file and
// rule: a new one fails, and --update-baseline rewrites the list after fixes
// but refuses to grow it. Warnings and suggestions are counted and reported.
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const HERE = dirname(fileURLToPath(import.meta.url));
const LIBRARY = resolve(HERE, "../../..");
const VALE = join(HERE, "node_modules/.bin/vale");
const BASELINE = join(HERE, "prose-baseline.json");

/** Where an alert is, stable across checkouts and builds. */
function place(file, site) {
  const path = resolve(HERE, file);
  return site && path.startsWith(site) ? `site/${relative(site, path)}` : relative(LIBRARY, path);
}

/** Lint the prose; return counts by severity and the errors beyond the baseline. */
export function lintProse({ site = null, update = false } = {}) {
  // Without a site the API reference is not linted, so its entries would drop out.
  if (update && !site) throw new Error("--update-baseline needs --site: the baseline covers the API reference");
  if (!existsSync(VALE)) {
    throw new Error("Vale is not installed here: run `npm --prefix docs/site/quality ci` once after pulling.");
  }
  if (!existsSync(join(HERE, "vale-styles/Google"))) {
    const sync = spawnSync(VALE, ["sync"], { cwd: HERE, encoding: "utf8" });
    if (sync.status !== 0) throw new Error(`vale sync failed: ${sync.error?.message || sync.stderr}`);
  }
  // Authored prose only: the site's own machinery under docs/site is not.
  const targets = ["content", "reference", "tasks", "submission.md"].map((p) => join(LIBRARY, "docs", p));
  targets.push(join(LIBRARY, "tutorials"));
  if (site) targets.push(join(site, "library/reference"));
  const run = spawnSync(VALE, ["--output=JSON", "--glob=!**/_executed*/**", ...targets],
    { cwd: HERE, encoding: "utf8", maxBuffer: 1 << 28 });
  if (run.status > 1) throw new Error(`vale failed: ${run.stderr}`);
  const alerts = JSON.parse(run.stdout || "{}");
  const severities = { error: 0, warning: 0, suggestion: 0 };
  const errors = {};
  for (const [file, found] of Object.entries(alerts)) {
    for (const alert of found) {
      severities[alert.Severity] += 1;
      if (alert.Severity !== "error") continue;
      const key = `${place(file, site)} :: ${alert.Check}`;
      errors[key] = (errors[key] || 0) + 1;
    }
  }
  // With no baseline yet, --update-baseline records today's errors once.
  const baseline = existsSync(BASELINE) ? JSON.parse(readFileSync(BASELINE, "utf8")) : update ? errors : {};
  const regressions = Object.entries(errors)
    .filter(([key, n]) => n > (baseline[key] || 0))
    .map(([key, n]) => `${key} ×${n} (baseline ${baseline[key] || 0})`);
  if (update) {
    if (regressions.length) throw new Error(`the baseline only shrinks; fix these first:\n${regressions.join("\n")}`);
    writeFileSync(BASELINE, JSON.stringify(Object.fromEntries(Object.entries(errors).sort()), null, 2) + "\n");
  }
  return { files: Object.keys(alerts).length, severities, regressions };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({ options: { site: { type: "string" }, "update-baseline": { type: "boolean", default: false } } });
  const result = lintProse({ site: values.site ? resolve(values.site) : null, update: values["update-baseline"] });
  console.log(`prose: ${result.files} files with alerts`, result.severities);
  for (const line of result.regressions) console.error(`beyond baseline: ${line}`);
  process.exit(result.regressions.length ? 1 : 0);
}
