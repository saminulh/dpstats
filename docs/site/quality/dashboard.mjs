#!/usr/bin/env node
// The local Lighthouse dashboard: the history of every page-load run, with
// build-to-build comparison. Keeps running until stopped; while it runs,
// run.mjs uploads each run to it. Its database lives in QUALITY_HOME
// (default ~/.dp-site-quality), outside every checkout.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { createServer } from "node:net";

const HERE = dirname(fileURLToPath(import.meta.url));
const HOME = process.env.QUALITY_HOME || join(homedir(), ".dp-site-quality");
const PORT = Number(process.env.QUALITY_DASHBOARD_PORT || 9041);
const URL_BASE = `http://127.0.0.1:${PORT}`;
const TOKEN_FILE = join(HOME, "lhci-project.json");
mkdirSync(HOME, { recursive: true });

// Another program on the port would answer in the dashboard's place.
const free = await new Promise((done) => {
  const probe = createServer().once("error", () => done(false)).listen(PORT, "127.0.0.1", () => probe.close(() => done(true)));
});
if (!free) {
  console.error(`port ${PORT} is in use; set QUALITY_DASHBOARD_PORT to a free one`);
  process.exit(1);
}

const server = spawn(join(HERE, "node_modules/.bin/lhci"), [
  "server",
  `--port=${PORT}`,
  "--storage.storageMethod=sql",
  "--storage.sqlDialect=sqlite",
  `--storage.sqlDatabasePath=${join(HOME, "lhci.sqlite")}`,
], { stdio: "inherit" });

process.on("exit", () => server.kill());
let up = false;
for (let i = 0; i < 100 && !up; i += 1) {
  up = await fetch(`${URL_BASE}/v1/projects`).then((r) => r.ok, () => false);
  if (!up) await new Promise((r) => setTimeout(r, 200));
}
if (!up) {
  console.error("the dashboard server did not start");
  process.exit(1);
}

// One project, created the first time; its upload token is kept beside the database.
const known = existsSync(TOKEN_FILE) ? JSON.parse(readFileSync(TOKEN_FILE, "utf8")) : null;
const projects = await fetch(`${URL_BASE}/v1/projects`).then((r) => r.json());
if (!known || !projects.some((p) => p.id === known.id)) {
  const created = await fetch(`${URL_BASE}/v1/projects`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name: "DP Platform site", externalUrl: "", baseBranch: "dev" }),
  }).then((r) => r.json());
  writeFileSync(TOKEN_FILE, JSON.stringify({ id: created.id, token: created.token, adminToken: created.adminToken, server: URL_BASE }, null, 2));
}
console.log(`\nLighthouse dashboard: ${URL_BASE}  (Ctrl-C to stop)`);
process.on("SIGINT", () => { server.kill(); process.exit(0); });
