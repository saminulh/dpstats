#!/usr/bin/env node
// Stylelint over the site stylesheet, against a baseline that only shrinks.
//
//   node stylesheet.mjs [--update-baseline]
//
// A finding is keyed by its rule and message, so moving a declaration does not
// change the baseline but writing a new raw colour or size does. The run
// fails when any key is counted more often than the baseline allows;
// --update-baseline rewrites the baseline after findings are fixed, and
// refuses to add to it. Colours written as token values on :root or a theme
// are the tokens themselves and are not findings.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import postcss from "postcss";
import stylelint from "stylelint";

const HERE = dirname(fileURLToPath(import.meta.url));
const FILES = [resolve(HERE, "../docs/stylesheets/*.css")];
const BASELINE = join(HERE, "stylesheet-baseline.json");
// A token block: a rule on the root or a theme, whatever else its selector lists.
const TOKEN_SELECTOR = /^\s*(:root|\[data-bs-theme(=[^\]]*)?\])\s*$/;

/** The lines of a stylesheet that declare a token, where literal colours belong. */
function tokenLines(source) {
  const lines = new Set();
  postcss.parse(readFileSync(source, "utf8")).walkDecls(/^--/, (decl) => {
    if (decl.parent.type === "rule" && decl.parent.selectors.every((s) => TOKEN_SELECTOR.test(s))) lines.add(decl.source.start.line);
  });
  return lines;
}

/** Lint the stylesheet; return the findings counted by key and any beyond the baseline. */
export async function lintStylesheet({ update = false } = {}) {
  const { results } = await stylelint.lint({ files: FILES, configFile: join(HERE, "stylelint.config.js") });
  const counts = {};
  const examples = {};
  for (const result of results) {
    const tokens = tokenLines(result.source);
    for (const warning of result.warnings) {
      if (tokens.has(warning.line) && warning.rule !== "declaration-property-value-allowed-list") continue;
      const key = `${warning.rule}: ${warning.text.replace(/\s*\([^)]*\)$/, "")}`;
      counts[key] = (counts[key] || 0) + 1;
      (examples[key] ||= []).push(`${relative(HERE, result.source)}:${warning.line}`);
    }
  }
  // With no baseline yet, --update-baseline records today's findings once.
  const baseline = existsSync(BASELINE) ? JSON.parse(readFileSync(BASELINE, "utf8")) : update ? counts : {};
  const regressions = Object.entries(counts)
    .filter(([key, n]) => n > (baseline[key] || 0))
    .map(([key, n]) => `${key} ×${n} (baseline ${baseline[key] || 0}) at ${examples[key].join(", ")}`);
  if (update) {
    if (regressions.length) throw new Error(`the baseline only shrinks; fix these first:\n${regressions.join("\n")}`);
    const sorted = Object.fromEntries(Object.entries(counts).sort(([a], [b]) => a.localeCompare(b)));
    writeFileSync(BASELINE, JSON.stringify(sorted, null, 2) + "\n");
  }
  const byRule = {};
  for (const [key, n] of Object.entries(counts)) byRule[key.split(":")[0]] = (byRule[key.split(":")[0]] || 0) + n;
  return { findings: Object.values(counts).reduce((a, b) => a + b, 0), byRule, regressions };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const result = await lintStylesheet({ update: process.argv.includes("--update-baseline") });
  console.log(`stylesheet: ${result.findings} findings`, result.byRule);
  for (const line of result.regressions) console.error(`beyond baseline: ${line}`);
  process.exit(result.regressions.length ? 1 : 0);
}
