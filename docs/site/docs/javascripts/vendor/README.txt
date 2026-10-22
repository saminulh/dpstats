# Vendored browser runtimes

Tabler owns the navigation shell and standard controls:

- `tabler-1.5.1.min.js` and `../../stylesheets/vendor/tabler-1.5.1.min.css`
  are the `@tabler/core` 1.5.1 distribution (MIT, `tabler-LICENSE.txt`),
  fetched from `https://cdn.jsdelivr.net/npm/@tabler/core@1.5.1/dist/`.
  The JavaScript is unmodified, SHA256
  `d4c4c2768f166c308391e0cea44db593056f12b84a4eeef46d55e35ca46d6e60`.
  The stylesheet (fetched SHA256
  `6aa5677e9cfc2620405bf97a98074ba43ff06a411cb35c5337acfb56d124c273`) has its
  133 rules using `:has()` removed by `docs/site/trim_vendor_css.py`, giving
  SHA256 `8a896c93f29a3ececdf61503b4ac94bea0f988c89ced01b0e20f30ebb55b94fb`.
  Those rules style components the site does not use, and while any is
  present Chrome restyles the whole page after every change, which made a
  panned chart fall to under 20 frames a second. The one value they set here,
  `--tblr-sidebar-width`, is restated in `extra.css`; every other computed
  style is unchanged. A new Tabler version is trimmed the same way, and
  `tests/unit/docs/test_vendor_stylesheet.py` fails until it is.
- `overrides/icons.html` embeds the selected Tabler Icons 3.35.0 SVG paths
  (MIT, `tabler-icons-LICENSE.txt`), without an icon-font download.

- `nouislider-15.8.1.min.js` and `../../stylesheets/vendor/nouislider-15.8.1.min.css`
  are the `nouislider` npm package, version 15.8.1 exactly (MIT,
  `nouislider-LICENSE`), the range slider Tabler documents, fetched from
  `https://cdn.jsdelivr.net/npm/nouislider@15.8.1/dist/`; SHA256
  `79222927cb7695f403144b58e31da6a7082d82e08d310b0cd776264d311abe7d` and
  `9dc9155cfd6331d1f5629881ac0bc63928243fa4c1d83b17ce4921b5200be0ae`.
  `../../stylesheets/vendor/tabler-vendors-1.5.1.min.css` is `@tabler/core`
  1.5.1's own theme for it and the other libraries Tabler documents, fetched
  from the same distribution, SHA256
  `e5ab0139d4ef0aedfd80a48bb197fc816edae572d6509934bf0edd107668b38a`. Only the
  Explore page loads these three files.

The normal build reads these checked-in files; it never downloads its theme.

- `combobox-nav-3.0.2.js` is the unmodified ES module `dist/index.js` from
  `@github/combobox-nav` 3.0.2 (MIT, `combobox-nav-LICENSE.txt`), fetched with
  `npm pack @github/combobox-nav@3.0.2 --ignore-scripts` from the npm registry.
  SHA256 `43604aad0758257f2f78fcecf60c1805a1c60f3f2bfdb92554de4584fee0f2a5`.
  Explore's algorithm picker imports it locally; it makes no CDN request and
  adds no build-time package installation. To upgrade, replace the module and
  licence with the pinned package files, update the versioned import and hash,
  and run the Explore picker browser regressions. The site's remaining
  selection policy is described in `../../../hand-rolled-inventory.md`.

- `plotly-basic-3.7.0.min.js` — the `plotly.js-basic-dist-min` npm package,
  version 3.7.0 exactly (MIT, `plotly-basic-LICENSE`), sha256
  `c23b03591a6bdad0bd0f47a0c5c52305a5519812b73ce254be65cd841553c9c2`.

A page that draws charts loads this bundle locally, so the built site makes no
CDN or other network request to render them. To upgrade: replace the file with
the new pinned version, then update the filename and this hash here and at
every page that names the bundle, and re-run browser QA.
