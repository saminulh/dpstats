// The whole built site's size, file by file, so a page or data file that
// balloons anywhere is caught, not only on the pages the browser checks visit.
import { readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

/** Every file under *site*: the largest page, the largest file, the total. */
export function scanSite(site, top = 10) {
  const files = [];
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const path = join(dir, name);
      const stat = statSync(path);
      if (stat.isDirectory()) walk(path);
      else files.push({ path: relative(site, path).split("\\").join("/"), bytes: stat.size });
    }
  };
  walk(site);
  files.sort((a, b) => b.bytes - a.bytes);
  const pages = files.filter((f) => f.path.endsWith(".html"));
  const kb = (f) => (f ? { path: f.path, kb: Math.round(f.bytes / 1024) } : null);
  return {
    pages: pages.length,
    files: files.length,
    total_mb: Math.round(files.reduce((a, f) => a + f.bytes, 0) / 1e5) / 10,
    largest_page: kb(pages[0]),
    largest_file: kb(files[0]),
    largest_pages: pages.slice(0, top).map(kb),
    largest_files: files.slice(0, top).map(kb),
  };
}

/** Each site-size limit the scan breaks, in words. */
export function sizeFailures(scan, limits = {}) {
  const broken = [];
  if (limits.largest_page_kb !== undefined && scan.largest_page.kb > limits.largest_page_kb)
    broken.push(`largest page ${scan.largest_page.path} is ${scan.largest_page.kb} KB, limit ${limits.largest_page_kb}`);
  if (limits.largest_file_kb !== undefined && scan.largest_file.kb > limits.largest_file_kb)
    broken.push(`largest file ${scan.largest_file.path} is ${scan.largest_file.kb} KB, limit ${limits.largest_file_kb}`);
  if (limits.total_mb !== undefined && scan.total_mb > limits.total_mb)
    broken.push(`site is ${scan.total_mb} MB, limit ${limits.total_mb}`);
  return broken;
}
