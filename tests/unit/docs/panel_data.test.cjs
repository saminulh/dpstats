/* Public browser data-loader contract; run with node --test this file.

   The library-reading check needs the expectations test_panel_data.py dumps
   for the fixture release's matrix: that test runs this file with
   DP_PANEL_EXPECTATIONS naming them, and without it the check is skipped. */
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const path = require('node:path');
const { test } = require('node:test');
const { gzipSync } = require('node:zlib');
const vm = require('node:vm');

const source = readFileSync(path.resolve(__dirname, '../../../docs/site/docs/javascripts/panel-data.js'), 'utf8');

function loader(fetch) {
  const context = { window: {}, fetch, Response, DecompressionStream };
  vm.runInNewContext(source, context);
  return context.window.dpPanel;
}

for (const operation of ['load', 'loadMatrix']) {
  const response = () => operation === 'load'
    ? new Response(JSON.stringify({ value: 7 }))
    : new Response(gzipSync(Buffer.from(new Float64Array([7]).buffer)));
  test(operation + ' shares concurrent and completed reads', async () => {
    let reads = 0;
    const data = loader(async () => { reads += 1; return response(); });
    const [first, second] = await Promise.all([data[operation]('/data'), data[operation]('/data')]);
    assert.equal(first, second);
    assert.equal(await data[operation]('/data'), first);
    assert.equal(reads, 1);
    assert.equal(operation === 'load' ? first.value : first[0], 7);
  });

  test(operation + ' permits an explicit retry after failure', async () => {
    let reads = 0;
    const data = loader(async () => ++reads === 1
      ? { ok: false, status: 503, statusText: 'Unavailable' }
      : response());
    await assert.rejects(data[operation]('/data'), /503 Unavailable/);
    const result = await data[operation]('/data');
    assert.equal(operation === 'load' ? result.value : result[0], 7);
    assert.equal(reads, 2);
  });
}

test('panel loading preserves infinite and zero baseline values', async () => {
  const data = loader(async () => new Response(
    gzipSync(Buffer.from(new Float64Array([Infinity, 0, 2]).buffer))));
  const panel = await data.panelOf({ instances: [0, 1, 2] }, {
    runs: '/costs', seeds: 1, seed_ids: [0],
    baselines: { best_median: ['Infinity', 0, 2] },
  });
  assert.deepEqual(Array.from(data.denominators(panel, 'best_median')), [Infinity, 0, 2]);
  assert.equal(panel.values[0], Infinity);
  assert.equal(data.denominators(panel, 'none'), null);
  assert.throws(() => data.denominators(panel, 'published_se'), /unavailable/);
});

test('matrix loading retains finite costs beyond single precision', async () => {
  const data = loader(async () => new Response(
    gzipSync(Buffer.from(new Float64Array([1e40, 1 - 1e-10]).buffer))));
  const values = await data.loadMatrix('/costs');
  assert.equal(values[0], 1e40);
  assert.equal(values[1], 1 - 1e-10);
});

/* Ten trials per algorithm on one instance, read with no baseline at a
   threshold of one: a cost of 0.5 is within it, 5 beyond it and -5 (an
   algorithm error) not within it. A is within on seeds 0-6 (p_A = 0.7), B on
   seeds 6-9 (p_B = 0.4), so the two share only seed 6: matching same-seed
   trials would give both 0.1, where the two rates alone give both 0.4. */
function pairPanel(rowA, rowB) {
  return { instances: 1, seeds: 10, values: new Float64Array([...rowA, ...rowB]) };
}
const WITHIN_A = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5, -5, 5];
const WITHIN_B = [5, -5, 5, 5, 5, 5, 0.5, 0.5, 0.5, 0.5];

function pairSplit(panel, aggregation) {
  const data = loader(async () => { throw new Error('no read'); });
  const split = {};
  const done = data.distributePair(panel, 0, 1, 0, null, aggregation, 2, 1,
    (key, share) => { split[key] = (split[key] || 0) + share; });
  return { done, split };
}

test('a pair splits an instance by the two success rates, not by seed', () => {
  const { done, split } = pairSplit(pairPanel(WITHIN_A, WITHIN_B), 'individual');
  assert.equal(done, true);
  // Shares of the instance's weight of 2.
  const expected = { both: 0.4, a_only: 0.3, b_only: 0, neither: 0.3 };
  Object.entries(expected).forEach(([key, share]) =>
    close(split[key] / 2, share, 1e-12, key));
});

test('under median aggregation each algorithm is within or not as a whole', () => {
  const { split } = pairSplit(pairPanel(WITHIN_A, WITHIN_B), 'median');
  close(split.a_only, 2, 1e-12, 'a_only');
  [split.both, split.b_only, split.neither].forEach((share, at) => close(share, 0, 1e-12, 'other ' + at));
});

test('an instance one algorithm has no trial on is omitted', () => {
  ['individual', 'median'].forEach((aggregation) => {
    const { done, split } = pairSplit(pairPanel(WITHIN_A, new Array(10).fill(NaN)), aggregation);
    assert.equal(done, false, aggregation);
    assert.deepEqual(split, {}, aggregation);
  });
});

function close(actual, expected, precision, what) {
  const wanted = Number(expected);
  if (!Number.isFinite(wanted)) {
    assert.equal(actual, wanted, what);
    return;
  }
  assert.ok(Math.abs(actual - wanted) <= precision * Math.max(1, Math.abs(wanted)),
    what + ': ' + actual + ' is not ' + wanted);
}

const expectationsPath = process.env.DP_PANEL_EXPECTATIONS;

test('the fixture matrix reads as the library reads its store',
  { skip: !expectationsPath && 'DP_PANEL_EXPECTATIONS names no expectations' }, async () => {
    const expected = JSON.parse(readFileSync(expectationsPath, 'utf8'));
    const { index, precision, taus } = expected;
    const data = loader(async () => new Response(Buffer.from(expected.runs, 'base64')));
    const panel = await data.panelOf(index, index.panels[expected.position]);
    const columns = index.instances.map((_, column) => column);

    for (const [aggregation, curves] of Object.entries(expected.curves)) {
      curves.forEach((fractions, row) => {
        const drawn = data.curve(panel, row, columns, null, aggregation, taus).fractions;
        fractions.forEach((fraction, stop) =>
          close(drawn[stop], fraction, precision, aggregation + ' curve of row ' + row));
      });
    }

    for (const [row, column, value] of expected.medians) {
      const middle = data.cellCosts(panel, row, column, 'median');
      assert.equal(middle.length, 1);
      close(middle[0], value, precision, 'median of cell ' + row + ',' + column);
    }

    // The browser's curve agrees with each shipped reading: it reaches the
    // quantile at the headline τ, and its share reached at the largest τ.
    const { quantile, baseline, standings } = expected.headline;
    const vector = data.denominators(panel, baseline);
    for (const [row, [tau, share]] of Object.entries(standings)) {
      const at = (threshold) => data.curve(panel, Number(row), columns, vector, 'individual',
        [threshold], index.weights).fractions[0];
      if (Number(tau) !== Infinity) {
        assert.ok(at(Number(tau)) >= quantile - 1e-12, 'headline τ of row ' + row);
      }
      close(at(Number.MAX_VALUE), share, precision, 'share reached by row ' + row);
    }
  });
