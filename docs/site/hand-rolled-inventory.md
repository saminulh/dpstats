# Site-specific presentation inventory

This records the standard facilities used by the Explore shared-controls
round and the application behavior left in site code. It is not a completed
audit of unchanged site scripts or build steps.

| Surface | Standard facilities | Site-specific behavior and why it remains |
| --- | --- | --- |
| `explore_controls.py` | Native `details`/`summary`, labelled form controls, buttons, lists and row-template radio inputs; the existing Tabler control classes | Templates bind task-specific identities, accessible relationships and the four Explore views. A shared above-chart selection list carries comparison identity and A/B choices. Native HTML is used directly; no replacement disclosure or select widget is implemented. |
| `docs/javascripts/algorithm-picker.js` | Native auto popovers and CSS anchors own the overlay; vendored `@github/combobox-nav` 3.0.2 owns option navigation, active-descendant tracking and keyboard/pointer commitment | The page supplies the eligible roster and owns selection. The picker ranks exact labels, core prefixes and substrings; caps displayed suggestions; renders full labels and interpretation badges; and hands a choice back to the page. Neither HTML `datalist` nor Tabler's basic dropdown provides this ranking, rich option content and external plotted-set policy. The standard combobox is used for its supported behavior, not rewritten. |
| `playground.py`, `docs/javascripts/playground.js` and `docs/javascripts/comparison-charts.js` (shared-controls integration) | The templates and picker above, existing native radios/selects and Plotly | The existing evaluation state and plotted set remain the owners. Wiring a chosen baseline or configured algorithm into that state is application behavior; a second widget-owned selection model would duplicate it. Native `details` retains disclosure state across view switches. |
| `docs/javascripts/comparison-charts.js` (selected-configuration identity) | The shipped series palette and swatch renderer, native radio groups, Plotly's colour/marker properties and compact outcome legend | The page's existing selected set retains one of five colour/marker slots per configuration until removal. Plotly's default trace-order colours would recolour retained configurations after a removal and cannot share identity with the HTML selector. Only that slot lifetime and the existing A/B selection policy are site-owned; no custom legend renderer or geometry-based picker positioning is added. |

The picker follows the [APG combobox pattern](https://www.w3.org/WAI/ARIA/apg/patterns/combobox/)
through [GitHub's combobox implementation](https://github.com/github/combobox-nav).
The adapter disables Tab-to-commit, leaves Tab as normal focus navigation and
announces that Enter accepts the first suggestion. Browser popovers handle
light dismissal; site code cancels pending search work and returns picker
focus on dismissal. No measured popup coordinates or scroll/resize positioning
listeners remain.

Version, licence, hashes and upgrade instructions for the unmodified module
live in `docs/javascripts/vendor/README.txt`. The focused Explore browser pages
exercise the shared picker through the generated pages, including selection
limits, eligibility, keyboard behavior, dismissal and recovery.
