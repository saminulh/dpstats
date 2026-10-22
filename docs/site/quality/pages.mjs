// The pages a built site holds, found by path pattern, so a check can name a
// kind of page (a configured algorithm) whose address changes with the release.
import { readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { LOAD_PAGES, TEMPLATES } from "./quality.config.js";

/** Every page the build wrote, as a path from the site root ("" is Home). */
export function sitePages(site) {
  if (!site) throw new Error("QUALITY_SITE must name the built site directory");
  const found = [];
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      if (statSync(path).isDirectory()) walk(path);
      else if (name === "index.html") found.push(relative(site, dir).split("\\").join("/"));
    }
  };
  walk(site);
  return found.map((p) => (p ? `${p}/` : "")).sort();
}

/** The first page matching *pattern*; a pattern matching nothing fails, so a
 * renamed or removed kind of page is noticed rather than left unchecked. */
function firstMatch(pages, name, pattern) {
  const path = pages.find((p) => pattern.test(p));
  if (path === undefined) throw new Error(`no page in the build matches "${name}" (${pattern})`);
  return path;
}

/** One page per template, as [{ name, path }]. */
export function templatePages(site) {
  const pages = sitePages(site);
  return Object.entries(TEMPLATES).map(([name, pattern]) => ({ name, path: firstMatch(pages, name, pattern) }));
}

/** The pages whose first load is measured, as { name: path }, patterns resolved. */
export function loadPages(site) {
  const pages = sitePages(site);
  return Object.fromEntries(Object.entries(LOAD_PAGES).map(([name, where]) =>
    [name, where instanceof RegExp ? firstMatch(pages, name, where) : where]));
}
