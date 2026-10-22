/* The property views' notebooks against the page's own computation; run with
   node --test this file.

   test_recreate_notebook.py runs each view's notebook over the fixture
   release and dumps what it plots beside the panel matrix, naming the file in
   DP_NOTEBOOK_VIEWS. This file recomputes each case with the real dpPanel the
   page draws from, so a notebook and its page cannot plot different numbers.
   Without the file the check is skipped. */
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(path.resolve(__dirname, '../../../docs/site/docs/javascripts/panel-data.js'), 'utf8');
const PRECISION = 1e-9;

function loader(fetch) {
  const context = { window: {}, fetch, Response, DecompressionStream };
  vm.runInNewContext(source, context);
  return context.window.dpPanel;
}

function number(value) {
  return value === null ? null : Number(value);
}

function close(actual, expected, where) {
  if (expected === null || actual === null) {
    assert.equal(actual, expected, where);
    return;
  }
  if (!Number.isFinite(expected) || !Number.isFinite(actual)) {
    assert.equal(actual, expected, where);
    return;
  }
  assert.ok(Math.abs(actual - expected) <= PRECISION * Math.max(1, Math.abs(expected)),
    `${where}: ${actual} != ${expected}`);
}

const expectationsPath = process.env.DP_NOTEBOOK_VIEWS;

test('each property notebook plots what the page plots',
  { skip: !expectationsPath && 'DP_NOTEBOOK_VIEWS names no expectations' }, async () => {
    const expected = JSON.parse(readFileSync(expectationsPath, 'utf8'));
    const { index } = expected;
    const data = loader(async () => new Response(Buffer.from(expected.runs, 'base64')));
    const panel = await data.panelOf(index, index.panels[expected.position]);
    const columns = index.instances.map((_, column) => column);
    const uuidOf = column => index.instances[column].uuid;
    const rowOf = uuid => index.algorithms.findIndex(entry => entry.uuid === uuid);
    const weightOf = column => (index.weights && index.weights.length ? index.weights[column] : 1);

    expected.cases.forEach((reading, position) => {
      const where = `case ${position} (${reading.view}, ${reading.scale}, ${reading.aggregation})`;
      const vector = data.resolveBaseline(panel, reading.baseline, index);
      const read = data.featureReader(index, panel, reading.feature);
      const map = data.quantileMap(read, index.instances.length, weightOf);

      if (reading.view === 'feature') {
        assert.ok(reading.algorithms.length, where + ': no algorithm drawn');
        reading.algorithms.forEach(uuid => {
          const points = data.featurePoints(panel, rowOf(uuid), columns, vector, reading.aggregation, read);
          const page = points.map(point => [uuidOf(point.instance), point.feature, point.value])
            .sort((a, b) => a[0].localeCompare(b[0]) || a[2] - b[2]);
          const notebook = reading.points.filter(point => point[0] === uuid)
            .map(point => [point[1], number(point[2]), number(point[3])])
            .sort((a, b) => a[0].localeCompare(b[0]) || a[2] - b[2]);
          assert.equal(notebook.length, page.length, `${where}: ${uuid} point count`);
          page.forEach((point, index_) => {
            assert.equal(notebook[index_][0], point[0], `${where}: ${uuid} instance`);
            close(notebook[index_][1], point[1], `${where}: ${uuid} feature`);
            close(notebook[index_][2], point[2], `${where}: ${uuid} ratio`);
          });
          // One algorithm's trials on one instance with one ratio are one
          // point: the failing algorithm's ratio is the same on every seed.
          close(reading.drawn[uuid][0], data.collapsedCount(points), `${where}: ${uuid} points drawn`);
          close(reading.drawn[uuid][1], points.length, `${where}: ${uuid} trials drawn`);
          // The constants are the ones the notebook was written with.
          const running = data.runningMedian(data.featureSample(points, map, weightOf),
            reading.window, reading.positions);
          [['q', 'position'], ['median', 'median'], ['lower', 'lower'], ['upper', 'upper']].forEach(([column, named]) => {
            assert.equal(reading.running[uuid][named].length, running[column].length, `${where}: ${uuid} positions`);
            running[column].forEach((value, stop) =>
              close(number(reading.running[uuid][named][stop]), value,
                `${where}: ${uuid} running ${column} at ${stop}`));
          });
        });
        // The opening ratio range: the scatter's sample, each instance's
        // weight split across one algorithm's points there, on the axis's scale.
        const sample = [];
        reading.algorithms.forEach(uuid => {
          const points = data.featurePoints(panel, rowOf(uuid), columns, vector, reading.aggregation, read);
          const perColumn = new Map();
          points.forEach(point => perColumn.set(point.instance, (perColumn.get(point.instance) || 0) + 1));
          points.forEach(point => {
            const log = reading.y_scale === 'log';
            const placed = log ? point.value > 0 && Number.isFinite(point.value)
              : Number.isFinite(point.value) && point.value >= 0;
            if (!placed) return;
            sample.push({
              value: log ? Math.log10(point.value) : point.value,
              weight: weightOf(point.instance) / perColumn.get(point.instance),
            });
          });
        });
        const range = data.robustRange(sample, reading.fence);
        range.forEach((end, side) => close(reading.robust_range[side], end, `${where}: robust range end ${side}`));
        // The view the figure opens on: the fences widened to hold a ratio of
        // one, from zero on a linear axis, then padded; and every placed
        // property value, padded.
        data.openingRatioRange(sample, reading.fence, reading.padding[1], reading.y_scale === 'log')
          .forEach((end, side) => close(reading.y_range[side], end, `${where}: opening ratio range end ${side}`));
        const across = [];
        reading.algorithms.forEach(uuid => {
          data.featurePoints(panel, rowOf(uuid), columns, vector, reading.aggregation, read).forEach(point => {
            if (!(point.feature > 0 && Number.isFinite(point.feature))) return;
            across.push(reading.scale === 'log' ? Math.log10(point.feature)
              : reading.scale === 'quantile' ? map.valueToQ.get(point.feature) : point.feature);
          });
        });
        data.fittedRange(across, reading.padding[0])
          .forEach((end, side) => close(reading.x_range[side], end, `${where}: opening property range end ${side}`));
        assert.ok(reading.figure, where + ': the notebook drew no Plotly figure');
        return;
      }

      const histogram = data.outcomeHistogram({
        panel, statusCodes: index.status_codes, row: rowOf(reading.algorithm),
        rowB: reading.compare_with ? rowOf(reading.compare_with) : -1,
        columns, vector, aggregation: reading.aggregation, threshold: reading.tau,
        read, scale: reading.scale, bins: reading.bins, map, weightOf,
      });
      // The library's categories are the recorded outcomes; the page's add
      // a declared cell nothing answered, which no Result records.
      const extra = Array.from(histogram.keys).filter(key => !reading.keys.includes(key));
      assert.deepEqual(extra, reading.compare_with ? [] : ['not_run'], where + ': categories');
      assert.equal(histogram.keys.length, reading.keys.length + extra.length, where + ': category count');
      assert.ok(reading.figure, where + ': the notebook drew no Plotly figure');
      // Both sides keep their edges in axis coordinates (log10 on a log scale).
      histogram.edges.forEach((edge, stop) => close(reading.edges[stop], edge, `${where}: edge ${stop}`));
      close(reading.excluded, histogram.excluded / histogram.total, where + ': excluded share');
      histogram.keys.forEach(key => histogram.weights[key].forEach((weight, bin) =>
        close(reading.shares[key] ? reading.shares[key][bin] : 0, weight / histogram.total,
          `${where}: ${key} in bin ${bin}`)));
      histogram.members.forEach((members, bin) =>
        assert.deepEqual(reading.members[bin], Array.from(members, uuidOf).sort(), `${where}: members of bin ${bin}`));
    });
  });

/* The page's own number format, from the site runtime that defines it. */
function siteNumber() {
  const site = readFileSync(path.resolve(__dirname, '../../../docs/site/docs/javascripts/site.js'), 'utf8');
  const start = site.indexOf('var SUPERSCRIPTS');
  const end = site.indexOf('window.dpParseNumber');
  const context = { window: {}, Intl, Number, Math, String };
  vm.runInNewContext(site.slice(start, end), context);
  return context.window.dpNumber;
}

test('a reader in another locale reads the same hovers on the page and in the notebook',
  { skip: !expectationsPath && 'DP_NOTEBOOK_VIEWS names no expectations' }, () => {
    const { locale } = JSON.parse(readFileSync(expectationsPath, 'utf8'));
    // The page writes these separators into the notebook's style on download.
    const parts = new Intl.NumberFormat(undefined).formatToParts(12345.6);
    const part = type => parts.find(entry => entry.type === type).value;
    assert.deepEqual({ decimal: part('decimal'), group: part('group') }, locale.separators,
      'the page reads in the locale the notebook was written for');
    const dpNumber = siteNumber();
    const escape = value => String(value).replace(/[&<>"']/g, char =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
    const fill = (text, fields) => Object.entries(fields)
      .reduce((written, [name, value]) => written.replaceAll(`{${name}}`, value), text);
    const { labels } = locale;
    assert.ok(locale.markers.some(([, , , hover]) => hover.includes(',')),
      'some hover writes the decimal comma');
    locale.markers.forEach(([x, y, trials, hover]) => {
      const page = [
        fill(labels.point_hover, { property: labels.feature, x: dpNumber(x), y: dpNumber(y) }),
        fill(trials === 1 ? labels.count_one : labels.count, { count: trials.toLocaleString() }),
      ].map(escape).join('<br>');
      assert.equal(hover, page, `hover at ${x}, ${y}`);
    });
  });
