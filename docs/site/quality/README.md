# Site quality

One command checks whether the documentation site is fast, accessible and
working, against limits kept in this folder, and leaves one report to read.
Every check here is something you can also feel yourself; this page says how.

## Running it

Once per checkout, install the tools (Node 22 or newer):

```bash
npm --prefix docs/site/quality ci
```

Then, against a built site:

```bash
node docs/site/quality/run.mjs --site <built site directory> --label "<what this build is>"
```

To also build the site first and time the build (warm; add `--cold` to
rebuild the prepared results too, about two minutes longer):

```bash
node docs/site/quality/run.mjs --site <output directory> --build --release <distribution release directory> --mkdocs .venv/bin/mkdocs
```

The checks split in two. `--part timing` runs only those that measure time
(Lighthouse and the Explore interaction, idle and load-memory budgets), one
browser at a time, on the machine their limits were set on. `--part
functional` runs the rest: site size, the stylesheet and prose lints,
contrast, accessibility, layout, search, screenshots and the browser test
pages. They measure no time, so `QUALITY_WORKERS=<n>` spreads them over `n`
browsers on any machine. With no `--part`, both halves run, one browser at a
time.

The run exits non-zero when any limit or check fails. Read
`docs/site/quality/out/index.html`: what passed, what failed against which
limit, and each headline number's trend over past runs. The full browser log
is `out/playwright-report/index.html`, and each page's Lighthouse report is
linked from the load table.

For history across runs with build-to-build comparison of page loads, keep
the Lighthouse dashboard running in another terminal while you run checks:

```bash
npm --prefix docs/site/quality run dashboard
```

It serves at <http://127.0.0.1:9041>. Run history and the dashboard's
database live in `~/.dp-site-quality/` (set `QUALITY_HOME` to move them),
outside every checkout.

## How the numbers are taken

- **Machine.** The browser plays a typical laptop on home broadband: CPU four
  times slower than this Mac, 40 ms network latency, 40 Mbit/s down
  (`quality.config.js`). Compare only runs taken on the same machine.
- **Like a real host.** The site is served compressed, so downloaded sizes are
  what a reader transfers.
- **Repeats.** Each measurement is the median of three runs.
- **Settled and blocked.** An interaction is done when nothing has happened
  for 0.7 s: no long task, no download, no chart redraw. "Blocked" is how long
  the page could not respond during it: the part of every long task beyond
  50 ms.

As a rough guide to what you feel: blocking much past 200 ms reads as lag,
and anything drawn at under about 50 frames a second reads as jitter.

## What is measured

### Build

| Measurement | How to feel it yourself |
| --- | --- |
| Warm build: time and peak memory, prepared results reused | Run the strict build twice; time the second |
| Cold build (`--cold`): prepared results rebuilt from the release | Delete `docs/site/.prepared/` first, then build; watch Activity Monitor |

### First load (Lighthouse): Home, Leaderboard, Explore, an algorithm page, a configured algorithm's page, the user guide

| Measurement | How to feel it yourself |
| --- | --- |
| First content: the page stops being blank | Reload with the cache disabled (DevTools → Network) |
| Largest content: the page looks complete apart from charts | Same reload |
| Blocked: how long loading froze the page | Try to scroll or click while it loads |
| Layout shift: how much the page jumps as it loads | Watch the controls move when the chart arrives |
| Downloaded | DevTools → Network, the transferred total |

### Memory while loading: the same six pages

| Measurement | How to feel it yourself |
| --- | --- |
| Peak JavaScript memory from the first request until the page settles, sampled every 100 ms: a page that balloons and falls back is caught | DevTools → Performance, record a reload with "Memory" ticked |
| Elements the page builds | DevTools → Performance monitor, "DOM Nodes" |

### At rest after loading: the same six pages

Each page is left alone for 30 seconds after it settles. A page at rest
should run nothing: a frame callback, timer or fetch that fires while nobody
touches the page is a loop, and a loop costs every reader who leaves the tab
open. One run per page, since a loop fires in every run.

| Measurement | How to feel it yourself |
| --- | --- |
| Frame callbacks, timers and fetches the page's own code ran; chart redraws; animations still playing | Chrome's Task Manager (Window → Task Manager): a tab at rest shows 0% CPU |
| Blocked: long tasks while nothing happened | DevTools → Performance, record 30 s without touching the page: the timeline stays empty |
| Memory growth after a forced collection | DevTools → Memory, a heap snapshot now and one a minute later |

### Site size: every file in the build

| Measurement | How to feel it yourself |
| --- | --- |
| Largest page, largest file, whole site | `du -sh` on the built site; sort its files by size |

These read the build directory, so they cover every page, not only the
measured ones: a page or data file that grows to hundreds of megabytes
anywhere fails the run.

### Explore: the heavy operations

| Measurement | How to feel it yourself |
| --- | --- |
| First load: chart ready, blocked, layout shift, memory | Open Explore cold |
| Switch ε to one not yet shown: blocked, done after, downloaded at the click | Click a privacy budget you have not opened yet |
| Switch back to an ε already shown | Click the first budget again |
| Show all curves; back to 10 | Change the curve limit |
| With all curves shown: pan, hover and switch ε (frames per second, frames over 50 ms, blocked) | Show all curves, then drag the chart, sweep the pointer across it, and click another budget |
| With all curves shown, after hovering: the whole browser's memory, every process it runs | Chrome's Task Manager (Window → Task Manager) while you hover |
| Drag τ: frames per second, frames over 50 ms, blocked after letting go, press to the tab's first move, thresholds taken while held (the curves re-rank and recolour live) | Drag the τ tab left and right, then let go |
| Switch metric | Click another metric |
| Hover over curves: frames per second, blocked | Sweep the pointer across the chart |
| Slowest response to an input (INP, as Chrome reports it for real visitors) | Click around; the slowest reaction is the one you remember |
| Memory growth over 40 ε switches: a leak shows as growth | Click through the budgets for a minute; DevTools → Memory |
| Comparison view, two algorithms plotted, legend rows hovered back and forth: frames per second, blocked, chart redraws (a hover only dims, so none) | Open the comparison tab, add a second algorithm, and run the pointer up and down its list |

### Checks that must hold

- Hiding every curve keeps the chart on screen, empty, at its usual scale.
- Filtering out every Instance does the same, beside the
  `explore.empty_no_instances` text slot.
- Accessibility: one page per template, in light and dark, plus the states a
  reader opens (the algorithm filter, search empty and with results, a help
  popover), checked by axe
  against the WCAG 2.2 A and AA rules it can decide automatically.
- Search: ten fixed queries typed into the search panel each find their page,
  no page twice in a top five, every result labelled with its site section,
  and the section filter narrows results (`tests/search.spec.js`). A failing
  query is answered by the global ranking settings in `site.js`, never by
  weakening its expectation.
- Layout: every template at the site's three widths (`WIDTHS` in
  `quality.config.js`: desktop 1440, tablet 768, phone 320 px). The page never
  scrolls sideways; anything wider than the window is a table, code block or
  chart in its own named, focusable region; and at desktop width no running
  paragraph is longer than about 85 characters a line.
- Colour contrast: every colour pair in `CONTRAST_PAIRS`
  (`quality.config.js`), resolved by the browser in light and dark, reaches
  4.5:1 for text and 3:1 for controls and chart marks. axe checks only the
  text it finds on a page; this covers every pair, chart marks included. A new
  colour token that carries text or marks adds its pair there.
- The standalone browser pages in `tests/unit/docs/*.test.html`: each drives
  the real site scripts and reports PASS or FAIL. The run opens every one on a
  single local host that serves the checkout and maps `site/` onto the build
  under test, so they check the same build as everything else here.
- Screenshots: every template in light and dark, the open states above, and
  every template at phone width (390 px), each full page, compared pixel for
  pixel with the last accepted image. Images are accepted per machine and kept
  in `QUALITY_HOME/screenshots`, since fonts render differently elsewhere; the
  first run on a machine records them. When a page changes, the Playwright
  report shows the expected, actual and difference images side by side. If
  the change is intended, accept it with `--update-screenshots`; a new
  release or rebuilt data changes the charts, so accept after checking the
  report. Before each image the results panels finish drawing and the page
  makes no change for a second; the window is then sized to the whole page, so
  nothing sized by the window's height re-lays out mid-capture; an opened
  overlay is captured in the window it opens over. Animations and the "more
  below" fade are off, and an element whose content changes between
  identical builds is masked (`SCREENSHOT_MASK` in `quality.config.js`; none
  today).
- Chart colours (`tests/unit/docs/test_series_style.py`, run with the unit
  tests): each series colour stands 3:1 on the plot, the cores most often
  compared stay apart under simulated colour blindness, and each histogram
  outcome stays apart from the one it is stacked on.

### Stylesheet

Stylelint (`stylelint.config.js`) holds `docs/site/docs/stylesheets/` to the
tokens: no literal colour (hex, `rgb()`, `hsl()` or a named colour) outside a
token declared on `:root` or a theme, a `font-size` only from `--dp-size-*`,
and a `font-family` only from a token. Run it alone with
`npm --prefix docs/site/quality run lint:css`.

### Prose

Vale (`.vale.ini`) applies Google's developer documentation style (STYLE
S10) to the authored docs (`docs/content`, `reference`, `tasks`, `tutorials`
and `submission.md`, notebooks included) and to the built API reference.
Tutorials keep their walkthrough "we", and one readability score is a
suggestion only. Only errors fail; warnings and suggestions
are for the writer. `docs/content` is the maintainer's prose: a finding there
is reported to the maintainer, who rewrites it. Run it alone with
`npm --prefix docs/site/quality run lint:prose -- --site <built site>`; the
first run downloads the pinned Google and Readability styles into
`vale-styles/`.

## Limits

`budgets.json` holds every limit. The `build` entries bound the cold and warm
builds' time and peak memory on the reference release; they are the site's
scale guard, since no smaller stand-in shows a per-algorithm reload. The `load` entries are Lighthouse audit ids
per page, in milliseconds (bytes for `total-byte-weight`); the `explore`
entries take `max` or `min`; `load_memory` limits each page's peak memory
(MB) and elements; `idle` limits what each page does while left alone after
loading (counts of frame callbacks, timers, fetches, chart redraws and
animations, blocked time in ms, memory growth in MB); `site` limits the largest page and file (KB) and the
whole site (MB). A limit starts at today's median plus about 20%
and only ever tightens: a change that needs a looser limit says so, and why,
in the same diff.

Accessibility findings that exist today are listed in `a11y-baseline.json`.
A page fails when it has a finding the list does not, or more of one; fixing
findings and rerunning with `--update-baseline` shrinks the list. Never grow it
by hand.

Prose errors that exist today are listed in `prose-baseline.json`, counted
per file and rule, and shrink the same way (`node prose.mjs --site <built
site> --update-baseline`).

Stylesheet findings that exist today are listed in
`stylesheet-baseline.json`, counted by rule and value, and shrink the same
way: fix them, then rerun with `--update-baseline` (or
`node stylesheet.mjs --update-baseline`), which refuses to grow the list.

## What the tools cannot check

Run through this list by hand when a page's template or interaction changes:

- Use the page with the keyboard alone: every control reachable, focus always
  visible, nothing trapped.
- Zoom to 200%, then narrow the window to phone width: nothing cut off,
  nothing needs sideways scrolling outside a table.
- On one results page, with VoiceOver: the controls and the chart's table say
  what they are.
- Colour is never the only signal: curves and states are told apart by dash,
  marker or label too.
- Read the "needs review" items in the accessibility table: axe found them
  but could not decide them.
- Help that appears on hover or focus: Escape dismisses it without moving
  the pointer, the pointer can move onto it without it closing, and it stays
  until dismissed or the pointer and focus leave.
- Every element the change touched, in each state it has: normal,
  selected or open, and unavailable.
- Desktop first: compare the changed template's desktop and phone
  screenshots. The phone layout keeps the desktop content, order and
  controls, and changes only their arrangement.
- Numbers agree: for three sampled rows, the value in the table and in the
  chart's tooltip show the same precision.

## Where things are

- `quality.config.js`: the pages, templates, opened states and machine;
  `pages.mjs` finds them in the build.
- `budgets.json`, `a11y-baseline.json`, `stylesheet-baseline.json`: the limits.
- `stylelint.config.js`, `stylesheet.mjs`: the stylesheet lint.
- `.vale.ini`, `prose.mjs`, `prose-baseline.json`: the prose lint.
- `tests/`: the browser checks (interaction budgets, contrast, accessibility,
  search, screenshots, the standalone browser pages, load memory, pages at rest) and
  the measuring code they share (`probe.js`).
- `run.mjs`: the one command; `report.mjs`: the report; `dashboard.mjs`: the
  Lighthouse dashboard.
