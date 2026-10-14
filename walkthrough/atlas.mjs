// Atlas Explorer: REPLAY of stored bid atlases (MOONSHOTS.md sections 1, 3, 4, 5).
// Pure helpers are exported for Node tests; the UI only boots inside a browser.
// Nothing here runs a model, sends a transaction, or touches real credits.

// ----------------------------------------------------------------------------- pure math

/** Index of the leaf whose interval (lo, hi] contains lambda, or -1. Leaves sorted by lo. */
export function leafIndexAt(leaves, lambda) {
  if (!(lambda > 0 && lambda <= 1)) return -1;
  let lo = 0, hi = leaves.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1, leaf = leaves[mid];
    if (lambda <= leaf.lo) hi = mid - 1;
    else if (lambda > leaf.hi) lo = mid + 1;
    else return mid;
  }
  return -1;
}

/**
 * Exact simulated charges for a lambda inside the leaf (MOONSHOTS.md section 1).
 * With S = a + b: a contest won by A charges A S*(1-lambda)*u/(1-u);
 * one won by B charges B S*lambda*(1-u)/u. Shared steps are free.
 */
export function chargesAt(leaf, lambda, S = 1) {
  if (!(lambda > leaf.lo && lambda <= leaf.hi)) {
    throw new RangeError(`lambda ${lambda} is outside this leaf's interval (${leaf.lo}, ${leaf.hi}].`);
  }
  if (!(Number.isFinite(S) && S > 0)) throw new RangeError('The total budget S must be positive.');
  let a = 0, b = 0, wonA = 0, wonB = 0;
  for (const step of leaf.steps) {
    if (step.branch === 'A') { wonA += 1; a += S * (1 - lambda) * step.u / (1 - step.u); }
    else if (step.branch === 'B') { wonB += 1; b += S * lambda * (1 - step.u) / step.u; }
  }
  return { a, b, wonA, wonB, contested: wonA + wonB, shared: leaf.steps.length - wonA - wonB };
}

/** Splits whose uniform equals the leaf's lower / upper boundary. */
export function boundarySplits(atlas, leaf) {
  const near = (x, y) => Math.abs(x - y) <= 1e-15 * Math.max(1, Math.abs(y));
  const find = (u) => atlas.splits.find(s => near(s.u, u)) || null;
  return { lower: leaf.lo > 0 ? find(leaf.lo) : null, upper: leaf.hi < 1 ? find(leaf.hi) : null };
}

/** Check that leaves are sorted, contiguous and partition (0, 1]. Returns a list of problems. */
export function partitionProblems(leaves) {
  const problems = [];
  if (!leaves.length) return ['The atlas has no leaves.'];
  if (leaves[0].lo !== 0) problems.push(`The first leaf starts at ${leaves[0].lo}, not 0.`);
  if (leaves[leaves.length - 1].hi !== 1) problems.push('The last leaf does not end at 1.');
  for (let i = 0; i < leaves.length; i += 1) {
    if (!(leaves[i].hi > leaves[i].lo)) problems.push(`Leaf ${leaves[i].id} is empty.`);
    if (i && leaves[i].lo !== leaves[i - 1].hi) problems.push(`Gap or overlap before leaf ${leaves[i].id}.`);
  }
  return problems;
}

/** Outcome keys in first-seen order. Only boolean and numeric outcomes can be drawn as lanes. */
export function outcomeKeys(leaves) {
  const keys = [];
  for (const leaf of leaves) {
    for (const [key, value] of Object.entries(leaf.outcomes || {})) {
      if ((typeof value === 'boolean' || typeof value === 'number') && !keys.includes(key)) keys.push(key);
    }
  }
  return keys;
}

/**
 * The shared user task to show above an atlas. Fixtures store `manifest.prompts` as a
 * string or {user | shared}; the text generators store per-advertiser prompt records
 * (a list) plus `manifest.user_prompt`.
 */
export function promptText(manifest = {}) {
  const prompts = manifest.prompts;
  if (typeof prompts === 'string') return prompts;
  if (prompts && !Array.isArray(prompts) && (prompts.user || prompts.shared)) return prompts.user || prompts.shared;
  return typeof manifest.user_prompt === 'string' ? manifest.user_prompt : null;
}

/** Width / height of an image atlas, from its manifest (width/height or image_size); 1.6 if unknown. */
export function imageAspect(manifest = {}) {
  const [w, h] = Array.isArray(manifest.image_size) ? manifest.image_size : [manifest.width, manifest.height];
  return Number(w) > 0 && Number(h) > 0 ? Math.min(3, Math.max(1 / 3, w / h)) : 1.6;
}

/** Simplex atlases list their cells as `leaves` (MOONSHOTS.md section 4); raw generator output says `cells`. */
export function atlasLeaves(atlas) {
  return atlas.leaves || atlas.cells || [];
}

/** Exact measure, switch count and monotonicity of one outcome along lambda. */
export function outcomeProfile(leaves, key) {
  const values = leaves.map(l => (l.outcomes || {})[key]);
  const numeric = values.every(v => typeof v === 'number');
  if (numeric) {
    let switches = 0, last = 0;
    for (let i = 1; i < values.length; i += 1) {
      const sign = Math.sign(values[i] - values[i - 1]);
      if (sign && last && sign !== last) switches += 1;
      if (sign) last = sign;
    }
    const lo = Math.min(...values), hi = Math.max(...values);
    const direction = switches ? 'mixed' : last > 0 ? 'up' : last < 0 ? 'down' : 'constant';
    return { type: 'number', values, switches, monotone: switches === 0, direction, min: lo, max: hi };
  }
  let measure = 0, switches = 0;
  leaves.forEach((leaf, i) => {
    if (values[i] === true) measure += leaf.hi - leaf.lo;
    if (i && values[i] !== values[i - 1]) switches += 1;
  });
  let direction = 'constant';
  if (switches === 1) direction = values[values.length - 1] === true ? 'up' : 'down';
  else if (switches > 1) direction = 'mixed';
  return { type: 'boolean', values, measure, switches, monotone: switches <= 1, direction };
}

/** Word-level LCS diff: which pieces of `text` are absent from `other`. Whitespace never marks. */
export function wordDiff(text, other) {
  const split = s => (s || '').match(/\s+|[\p{L}\p{N}'’]+|[^\s\p{L}\p{N}]/gu) || [];
  const a = split(text), b = split(other);
  const n = a.length, m = b.length;
  const table = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      table[i][j] = a[i] === b[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const out = [];
  let i = 0, j = 0;
  while (i < n) {
    if (j < m && a[i] === b[j]) { out.push({ text: a[i], changed: false }); i += 1; j += 1; }
    else if (j < m && table[i][j + 1] >= table[i + 1][j]) j += 1;
    else { out.push({ text: a[i], changed: !/^\s+$/.test(a[i]) }); i += 1; }
  }
  return out;
}

// Simplex helpers. Barycentric points are [l0, l1, l2]; area uses the (l0, l1) chart.
export function polygonArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i += 1) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    s += p[0] * q[1] - q[0] * p[1];
  }
  return Math.abs(s) / 2;
}

export function pointInPolygon(poly, point, eps = 1e-12) {
  let sign = 0;
  for (let i = 0; i < poly.length; i += 1) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    const cross = (q[0] - p[0]) * (point[1] - p[1]) - (q[1] - p[1]) * (point[0] - p[0]);
    if (Math.abs(cross) <= eps) continue;
    const s = Math.sign(cross);
    if (sign && s !== sign) return false;
    sign = s;
  }
  return true;
}

export function cellIndexAt(leaves, point) {
  return leaves.findIndex(leaf => pointInPolygon(leaf.polygon, point));
}

/** Exponential-race charges: the winner pays its critical bid E_i * max_{j != i} b_j / E_j, with b = S * lambda. */
export function raceCharges(leaf, lambda, S = 1) {
  const n = lambda.length, charges = new Array(n).fill(0), wins = new Array(n).fill(0);
  for (const step of leaf.steps) {
    if (step.winner == null || !step.E) continue;
    const i = step.winner;
    wins[i] += 1;
    // i wins iff E_i/b_i < min_j E_j/b_j, so its threshold bid is E_i * max_j b_j/E_j.
    let critical = 0;
    for (let j = 0; j < n; j += 1) if (j !== i) critical = Math.max(critical, (S * lambda[j]) / step.E[j]);
    charges[i] += step.E[i] * critical;
  }
  return { charges, wins, contested: wins.reduce((x, y) => x + y, 0) };
}

/** Human labels for the run mode (MOONSHOTS.md section 5). Viewing any stored atlas is REPLAY. */
export function modeInfo(mode) {
  return {
    synthetic_fixture: { badge: 'SYNTHETIC FIXTURE', tone: 'fixture', note: 'Hand-made toy data for UI development — not model output.', short: 'fixture' },
    synthetic_model_test: { badge: 'SYNTHETIC MODEL TEST · REPLAY', tone: 'fixture', note: 'Tiny random model: checks mechanics, not text or image quality.', short: 'model test' },
    live_model: { badge: 'REPLAY · LIVE MODEL OUTPUT', tone: 'replay', note: 'Stored output of real local model inference, shown without repair.', short: 'live model' },
  }[mode] || { badge: 'UNLABELED MODE', tone: 'danger', note: `Unknown mode “${mode}”: treat as unverified.`, short: 'unlabeled' };
}

// ---------------------------------------------------------------------------------- UI
const SVGNS = 'http://www.w3.org/2000/svg';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (x, d = 4) => Number.isFinite(x) ? x.toFixed(d) : '—';
const pct = x => `${(100 * x).toFixed(x < 0.001 && x > 0 ? 3 : 1)}%`;
const credits = x => x >= 100 ? x.toFixed(1) : x >= 1 ? x.toFixed(2) : x.toFixed(3);
const BIDDER_CLASSES = ['a', 'b', 'c'];
const state = {
  index: [], entry: null, atlas: null, atlasUrl: null, leaves: [], keys: [],
  lambda: 0.5, S: 100, leaf: -1, diff: 'lower', split: null, hoverLeaf: null,
  point: [1 / 3, 1 / 3, 1 / 3], cell: -1, colorBy: 'brand', reduced: false,
};
let $, $$;

function boot() {
  $ = (s, root = document) => root.querySelector(s);
  $$ = (s, root = document) => [...root.querySelectorAll(s)];
  const media = window.matchMedia('(prefers-reduced-motion: reduce)');
  state.reduced = media.matches;
  media.addEventListener?.('change', e => { state.reduced = e.matches; });
  initTheme();
  loadIndex().catch(error => fail(error));
}

function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem('atlas-theme'); } catch { /* storage unavailable */ }
  if (saved === 'light' || saved === 'dark') document.documentElement.dataset.theme = saved;
  const button = $('#theme-toggle');
  const current = () => document.documentElement.dataset.theme
    || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const label = () => { button.setAttribute('aria-label', `Switch to ${current() === 'dark' ? 'light' : 'dark'} theme`); button.dataset.mode = current(); };
  label();
  button.addEventListener('click', () => {
    const next = current() === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('atlas-theme', next); } catch { /* ignore */ }
    label();
  });
}

function fail(error) {
  $('#atlas-root').innerHTML = `<section class="panel error-panel"><h2>Could not open the atlas</h2><p>${esc(error.message || error)}</p><p class="muted">Serve this page with <code>python -m token_auction_lab.walkthrough</code> and open <code>/atlas.html</code>.</p></section>`;
}

async function getJSON(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.json();
}

function safeAtlasPath(path) {
  return typeof path === 'string' && /^[\w][\w.\-/]*\.json$/.test(path) && !path.includes('..') ? path : null;
}

async function loadIndex() {
  const index = await getJSON('/atlas_data/index.json');
  state.index = index.atlases || [];
  const params = new URLSearchParams(location.search);
  const wanted = safeAtlasPath(params.get('atlas'));
  const select = $('#atlas-select');
  select.innerHTML = state.index.map((e, i) => `<option value="${i}">${esc(e.title)} · ${esc(e.kind)}${e.schema === 'simplex-atlas-v1' ? ' · 3 bidders' : ''} · ${esc(modeInfo(e.mode).short)}</option>`).join('');
  let entry = state.index.find(e => e.path === wanted);
  if (!entry && wanted) { entry = { title: wanted, path: wanted, kind: '?', mode: '?' }; state.index.push(entry); select.insertAdjacentHTML('beforeend', `<option value="${state.index.length - 1}">${esc(wanted)}</option>`); }
  entry ||= state.index.find(e => e.mode === 'live_model') || state.index[0];
  if (!entry) throw new Error('atlas_data/index.json lists no atlases.');
  select.value = String(state.index.indexOf(entry));
  select.addEventListener('change', () => openAtlas(state.index[Number(select.value)], null));
  const lambda = Number(params.get('lambda'));
  await openAtlas(entry, lambda > 0 && lambda <= 1 ? lambda : null);
}

async function openAtlas(entry, lambda) {
  const url = new URL(`/atlas_data/${entry.path}`, location.origin);
  const atlas = await getJSON(url);
  if (!['bid-atlas-v1', 'simplex-atlas-v1'].includes(atlas.schema)) throw new Error(`Unsupported atlas schema “${atlas.schema}”.`);
  atlas.leaves = atlasLeaves(atlas);
  Object.assign(state, { entry, atlas, atlasUrl: url, split: null, hoverLeaf: null });
  const params = new URLSearchParams(location.search);
  params.set('atlas', entry.path);
  if (lambda == null) params.delete('lambda');
  history.replaceState(null, '', `?${params}`);
  renderLabels();
  renderHero();
  if (atlas.schema === 'simplex-atlas-v1') mountSimplex();
  else mountPair(lambda);
  document.title = `${atlas.title || entry.title} · Atlas Explorer`;
}

function renderLabels() {
  const { atlas } = state, info = modeInfo(atlas.mode);
  const badge = $('#mode-badge');
  badge.textContent = info.badge;
  badge.className = `badge mode tone-${info.tone}`;
  badge.title = info.note;
  $('#mode-note').textContent = info.note;
  const manifest = atlas.manifest || {};
  $('#meta').textContent = [`seed ${atlas.seed}`, `${atlas.max_steps} max steps`, manifest.model || manifest.model_id].filter(Boolean).join(' · ');
}

function renderHero() {
  const { atlas, entry } = state;
  const simplex = atlas.schema === 'simplex-atlas-v1';
  const prompt = promptText(atlas.manifest || {});
  const longest = Math.max(...atlas.leaves.map(l => l.steps.length));
  const verified = (atlas.verification || []);
  const ok = verified.filter(v => v.matches_direct_run).length;
  const brands = atlas.bidders.map((b, i) => `<span class="who ${BIDDER_CLASSES[i]}"><i></i>${esc(b)}</span>`).join(simplex ? ' ' : ' <span class="vs">vs</span> ');
  $('#hero').innerHTML = `
    <div class="hero-copy">
    <p class="eyebrow">${simplex ? 'SIMPLEX ATLAS · THREE BIDDERS · ONE SEED' : 'BID ATLAS · ONE SEED · EVERY BID RATIO'}</p>
    <h1>${esc(atlas.title || entry.title)}</h1>
    <p class="brands">${brands}</p>
    ${prompt ? `<blockquote class="prompt"><span>${atlas.kind === 'image' ? 'Image prompt' : 'User asks'}</span>${esc(prompt)}</blockquote>` : ''}
    </div>
    <dl class="hero-stats">
      <div><dt>${simplex ? 'cells' : 'distinct answers'}</dt><dd>${atlas.leaves.length}</dd></div>
      ${simplex
        ? `<div><dt>split events</dt><dd>${atlas.splits.length}</dd></div><div><dt>longest run</dt><dd>${longest}<small> steps</small></dd></div>`
        : `<div><dt>bound 1 + T</dt><dd>${1 + longest}<small> T = ${longest} steps</small></dd></div><div><dt>contest boundaries</dt><dd>${atlas.splits.length}</dd></div>`}
      <div><dt>direct-run checks</dt><dd>${verified.length ? `${ok}<small>/${verified.length} match</small>` : '<small>none recorded</small>'}</dd></div>
    </dl>`;
}

// ------------------------------------------------------------------------ two bidders
function mountPair(lambda) {
  const { atlas } = state;
  state.leaves = [...atlas.leaves].sort((x, y) => x.lo - y.lo);
  state.keys = outcomeKeys(state.leaves);
  const problems = partitionProblems(state.leaves);
  const image = atlas.kind === 'image';
  const [A, B] = atlas.bidders;
  $('#view').style.setProperty('--frame-aspect', String(imageAspect(atlas.manifest || {})));
  $('#view').innerHTML = `
  ${problems.length ? `<div class="warning" role="alert"><strong>Atlas check failed.</strong> ${problems.map(esc).join(' ')}</div>` : ''}
  <section class="scrub panel">
    <div class="scrub-head">
      <h2>Drag the bid ratio</h2>
      <div class="bid-inputs">
        <label>λ <input id="lambda-input" type="number" min="0.0001" max="1" step="0.0001" inputmode="decimal"></label>
        <label>Total budget S <input id="budget-input" type="number" min="0.01" step="1" value="${state.S}" inputmode="decimal"><span class="unit">credits</span></label>
        <output id="bids-readout" class="bids-readout"></output>
      </div>
    </div>
    <div class="scrubber">
      <div class="track" id="track" role="slider" tabindex="0" aria-label="Bid ratio lambda = bid_A / (bid_A + bid_B)" aria-valuemin="0" aria-valuemax="1"></div>
      <div class="ticks" id="ticks"></div>
    </div>
    <div class="axis"><span>0⁺ · only ${esc(B)} bids</span><span>0.25</span><span>0.5 · equal bids</span><span>0.75</span><span>1 · only ${esc(A)} bids</span></div>
    <p class="hint">Drag, click, or focus the track and use <kbd>←</kbd> <kbd>→</kbd> to step answer by answer (<kbd>Shift</kbd> for fine steps). Click a tick to inspect that contest.</p>
  </section>
  <section class="stage-grid">
    <article class="stage panel" aria-live="polite">
      <header class="stage-head">
        <div><p class="eyebrow" id="leaf-eyebrow"></p><p class="interval" id="leaf-interval"></p></div>
        ${image ? '' : `<div class="segmented" role="group" aria-label="Highlight words that differ from a neighbouring answer">
          <button data-diff="lower">vs lower λ</button><button data-diff="upper">vs higher λ</button><button data-diff="none">no diff</button></div>`}
      </header>
      ${image ? '<div class="frame"><img id="img-a" alt=""><img id="img-b" alt=""></div><p class="caption" id="image-caption"></p>' : '<p class="answer" id="answer"></p><p class="caption" id="diff-caption"></p>'}
      <div class="here" id="here"></div>
    </article>
    <aside class="ledger panel" id="ledger"></aside>
  </section>
  <section class="panel lanes-panel">
    <div class="lanes" id="lanes">
      <div id="outcomes" class="outcomes"></div>
      <div class="chart-block"><div class="chart-head"><h3>Simulated charges across λ</h3><p class="muted small">Exact within each answer; jumps happen only at contest boundaries. Total budget S = <span id="chart-s"></span> credits.</p></div><div id="charge-chart" class="chart"></div></div>
      <div class="cursor" id="cursor" aria-hidden="true"><span id="cursor-label"></span></div>
    </div>
  </section>
  <section class="detail-grid">
    <article class="panel" id="tape-panel"><div class="chart-head"><h3>This answer's contest tape</h3><p class="muted small">Each dot is a contested step at height u. A wins when u &lt; λ. No contest uniform lies inside the shaded interval, so every λ inside it gives this exact answer.</p></div><div id="tape" class="chart"></div></article>
    <article class="panel split-panel" id="split-panel"></article>
  </section>
  <section class="all panel"><div class="chart-head"><h2>All answers, in λ order</h2><p class="muted small">Hover to locate an answer on the track; click to jump there.</p></div><div class="cards" id="cards"></div></section>`;

  renderTrack();
  renderOutcomes();
  renderCards();
  wirePair();
  const start = lambda ?? defaultLambda();
  setLambda(start, { force: true });
  observeResize(['#ticks', '#charge-chart', '#tape', '.num-plot'], () => { renderTicks(); renderChargeChart(); renderTape(); renderNumericPlots(); });
}

function defaultLambda() {
  const i = leafIndexAt(state.leaves, 0.5);
  return i >= 0 ? 0.5 : (state.leaves[0].lo + state.leaves[0].hi) / 2;
}

function leafColor(leaf) {
  let a = 0, b = 0;
  for (const s of leaf.steps) { if (s.branch === 'A') a += 1; else if (s.branch === 'B') b += 1; }
  const share = a + b ? a / (a + b) : 0.5;
  return `color-mix(in oklab, var(--a) ${Math.round(share * 100)}%, var(--b))`;
}

function renderTrack() {
  const track = $('#track'), ticks = $('#ticks');
  track.innerHTML = state.leaves.map((leaf, i) => `<div class="seg" data-leaf="${i}" style="left:${leaf.lo * 100}%;width:${(leaf.hi - leaf.lo) * 100}%;--seg:${leafColor(leaf)}"><span>${(leaf.hi - leaf.lo) > 0.045 ? i + 1 : ''}</span></div>`).join('')
    + '<div class="thumb" id="thumb"><span id="thumb-label"></span></div>';
  renderTicks();
}

/**
 * Greedy label rows for sorted tick positions (pixels): each label takes the first
 * row whose previous label is at least `gap` px to the left; -1 hides the label
 * (the tick mark stays). Dense real atlases have many nearby contest boundaries.
 */
export function tickRows(positions, gap, rows = 3) {
  const last = new Array(rows).fill(-Infinity);
  return positions.map(x => {
    const row = last.findIndex(end => x - end >= gap);
    if (row >= 0) last[row] = x;
    return row;
  });
}

function renderTicks() {
  const ticks = $('#ticks');
  if (!ticks) return;
  const width = ticks.clientWidth || 800;
  const splits = [...state.atlas.splits].sort((x, y) => x.u - y.u);
  const rows = tickRows(splits.map(s => s.u * width), 22);
  ticks.innerHTML = splits.map((s, k) => `<button class="tick row-${Math.max(0, rows[k])}${rows[k] < 0 ? ' nolabel' : ''}${state.split === state.atlas.splits.indexOf(s) ? ' is-selected' : ''}" data-split="${state.atlas.splits.indexOf(s)}" style="left:${s.u * 100}%" aria-label="Contest boundary at step ${s.step}, u = ${fmt(s.u, 6)}" title="step ${s.step} · u = ${fmt(s.u, 6)}"><i></i><span>${s.step}</span></button>`).join('');
}

function prettyKey(key) {
  const [kind, ...rest] = key.split(':'), name = rest.join(':');
  if (kind === 'mentions') return `Mentions “${name}”`;
  if (kind === 'clip') return `CLIP score · ${name}`;
  if (kind === 'clip_top') return `CLIP top label is ${name}`;
  return key;
}

function keyColor(key) {
  const i = state.atlas.bidders.findIndex(b => key.endsWith(`:${b}`));
  return i >= 0 ? `var(--${BIDDER_CLASSES[i]})` : 'var(--ink-2)';
}

function badgeFor(p) {
  if (p.type === 'number') {
    return p.monotone ? `<span class="pill ok">monotone ${p.direction === 'up' ? '↑' : p.direction === 'down' ? '↓' : '·'}</span>` : `<span class="pill warn">non-monotone · ${p.switches} turn${p.switches === 1 ? '' : 's'}</span>`;
  }
  if (p.switches === 0) return '<span class="pill">constant</span>';
  return p.monotone ? `<span class="pill ok">monotone ${p.direction === 'up' ? '↑ rises with λ' : '↓ falls with λ'}</span>` : `<span class="pill warn">non-monotone · ${p.switches} switches</span>`;
}

function renderOutcomes() {
  const wrap = $('#outcomes');
  if (!state.keys.length) { wrap.innerHTML = ''; return; }
  wrap.innerHTML = '<h3 class="lane-title">Outcomes along λ</h3>' + state.keys.map(key => {
    const p = outcomeProfile(state.leaves, key);
    if (p.type === 'number') {
      return `<div class="lane" data-key="${esc(key)}"><div class="lane-head"><strong>${esc(prettyKey(key))}</strong> <span class="muted small">${key.startsWith('clip') ? 'CLIP zero-shot score — noisy heuristic' : 'numeric outcome'}</span> <span class="muted small mono">range ${fmt(p.min, 3)}–${fmt(p.max, 3)}</span> ${badgeFor(p)} <span class="lane-now mono small" data-now></span></div><div class="num-plot" data-key="${esc(key)}"></div></div>`;
    }
    const color = keyColor(key);
    const segs = state.leaves.map((leaf, i) => `<div class="band-seg ${p.values[i] === true ? 'yes' : p.values[i] === false ? 'no' : 'unknown'}" data-leaf="${i}" style="left:${leaf.lo * 100}%;width:${(leaf.hi - leaf.lo) * 100}%"></div>`).join('');
    return `<div class="lane" data-key="${esc(key)}"><div class="lane-head"><strong>${esc(prettyKey(key))}</strong> <span class="mono small">true on ${pct(p.measure)} of λ</span> ${badgeFor(p)} <span class="lane-now small" data-now></span></div><div class="band" style="--yes:${color}" role="img" aria-label="${esc(prettyKey(key))}: true on ${pct(p.measure)} of the bid-ratio interval, ${p.switches} switches">${segs}</div></div>`;
  }).join('');
}

function renderNumericPlots() {
  for (const el of $$('.num-plot')) {
    const key = el.dataset.key, p = outcomeProfile(state.leaves, key);
    const w = el.clientWidth || 600, h = 64, pad = 6;
    const span = (p.max - p.min) || 1;
    const y = v => h - pad - ((v - p.min) / span) * (h - 2 * pad);
    let d = '';
    state.leaves.forEach((leaf, i) => { const yy = y(p.values[i]); d += `${i ? 'L' : 'M'}${leaf.lo * w},${yy}H${leaf.hi * w}`; });
    el.innerHTML = `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(prettyKey(key))} step plot"><line x1="0" x2="${w}" y1="${h - 1}" y2="${h - 1}" class="grid"/><path d="${d}" class="step-line" style="stroke:${keyColor(key)}"/></svg>`;
  }
}

function renderCards() {
  const image = state.atlas.kind === 'image';
  $('#cards').innerHTML = state.leaves.map((leaf, i) => {
    const dots = state.keys.filter(k => typeof leaf.outcomes[k] === 'boolean').map(k => `<span class="dot ${leaf.outcomes[k] ? 'on' : ''}" style="--yes:${keyColor(k)}" title="${esc(prettyKey(k))}: ${leaf.outcomes[k]}"></span>`).join('');
    const body = image && leaf.image ? `<img src="${esc(new URL(leaf.image, state.atlasUrl).pathname)}" alt="Answer ${i + 1} image">` : `<p>${esc(leaf.text || '(no text recorded)')}</p>`;
    return `<button class="card" data-leaf="${i}" style="--seg:${leafColor(leaf)}"><span class="card-head"><b>${i + 1}</b><span class="mono">(${fmt(leaf.lo, 3)}, ${fmt(leaf.hi, 3)}]</span><span class="mini"><i style="left:${leaf.lo * 100}%;width:${Math.max(1.5, (leaf.hi - leaf.lo) * 100)}%"></i></span></span>${body}<span class="card-foot">${dots}<span class="muted">${leaf.steps.filter(s => s.branch !== 'shared').length} contests</span></span></button>`;
  }).join('');
}

function wirePair() {
  const track = $('#track');
  const fromEvent = e => {
    const r = track.getBoundingClientRect();
    return Math.min(1, Math.max(1e-6, (e.clientX - r.left) / r.width));
  };
  track.addEventListener('pointerdown', e => {
    track.setPointerCapture(e.pointerId);
    track.classList.add('dragging');
    setLambda(fromEvent(e));
    track.focus({ preventScroll: true });
  });
  track.addEventListener('pointermove', e => { if (track.hasPointerCapture(e.pointerId)) setLambda(fromEvent(e)); });
  const end = e => { track.classList.remove('dragging'); if (track.hasPointerCapture?.(e.pointerId)) track.releasePointerCapture(e.pointerId); };
  track.addEventListener('pointerup', end);
  track.addEventListener('pointercancel', end);
  track.addEventListener('keydown', e => {
    const i = state.leaf, n = state.leaves.length, mid = k => (state.leaves[k].lo + state.leaves[k].hi) / 2;
    let next = null;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') next = e.shiftKey ? state.lambda + 0.001 : mid(Math.min(n - 1, i + 1));
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') next = e.shiftKey ? state.lambda - 0.001 : mid(Math.max(0, i - 1));
    else if (e.key === 'PageUp') next = state.lambda + 0.05;
    else if (e.key === 'PageDown') next = state.lambda - 0.05;
    else if (e.key === 'Home') next = mid(0);
    else if (e.key === 'End') next = 1;
    if (next == null) return;
    e.preventDefault();
    setLambda(Math.min(1, Math.max(1e-6, next)));
  });
  $('#ticks').addEventListener('click', e => {
    const tick = e.target.closest('.tick');
    if (tick) selectSplit(Number(tick.dataset.split), true);
  });
  $('#lambda-input').addEventListener('change', e => {
    const v = Number(e.target.value);
    if (v > 0 && v <= 1) setLambda(v); else e.target.value = state.lambda.toFixed(4);
  });
  $('#budget-input').addEventListener('input', e => {
    const v = Number(e.target.value);
    if (Number.isFinite(v) && v > 0) { state.S = v; updatePair(true); renderChargeChart(); }
  });
  $$('[data-diff]').forEach(b => b.addEventListener('click', () => { state.diff = b.dataset.diff; updateAnswer(); }));
  const cards = $('#cards');
  cards.addEventListener('pointerover', e => hoverLeaf(e.target.closest('.card')?.dataset.leaf));
  cards.addEventListener('pointerleave', () => hoverLeaf(null));
  cards.addEventListener('focusin', e => hoverLeaf(e.target.closest('.card')?.dataset.leaf));
  cards.addEventListener('focusout', () => hoverLeaf(null));
  cards.addEventListener('click', e => {
    const card = e.target.closest('.card');
    if (!card) return;
    const leaf = state.leaves[Number(card.dataset.leaf)];
    setLambda((leaf.lo + leaf.hi) / 2);
    $('#track').scrollIntoView({ behavior: state.reduced ? 'auto' : 'smooth', block: 'start' });
  });
  $('#split-panel').addEventListener('click', e => {
    const go = e.target.closest('[data-go]');
    if (go) setLambda(Number(go.dataset.go));
    const sel = e.target.closest('[data-select-split]');
    if (sel) selectSplit(Number(sel.dataset.selectSplit), false);
  });
  $('#ledger').addEventListener('click', e => {
    const sel = e.target.closest('[data-select-split]');
    if (sel) selectSplit(Number(sel.dataset.selectSplit), true);
  });
}

function hoverLeaf(index) {
  const i = index == null ? null : Number(index);
  if (state.hoverLeaf === i) return;
  state.hoverLeaf = i;
  $$('#track .seg, .band-seg').forEach(el => el.classList.toggle('is-hover', i != null && Number(el.dataset.leaf) === i));
  document.body.classList.toggle('has-hover', i != null);
}

function setLambda(value, { force = false } = {}) {
  const lambda = Math.min(1, Math.max(1e-9, value));
  const index = leafIndexAt(state.leaves, lambda);
  const changed = index !== state.leaf;
  state.lambda = lambda;
  state.leaf = index;
  updatePair(force || changed);
}

function updatePair(leafChanged) {
  const { lambda, S } = state, leaf = state.leaves[state.leaf];
  const [A, B] = state.atlas.bidders;
  const left = `${lambda * 100}%`;
  $('#thumb').style.left = left;
  $('#cursor').style.left = left;
  $('#thumb-label').textContent = `λ ${fmt(lambda, 4)}`;
  $('#cursor-label').textContent = fmt(lambda, 3);
  const track = $('#track');
  track.setAttribute('aria-valuenow', fmt(lambda, 4));
  track.setAttribute('aria-valuetext', `lambda ${fmt(lambda, 4)}, answer ${state.leaf + 1} of ${state.leaves.length}`);
  const input = $('#lambda-input');
  if (document.activeElement !== input) input.value = lambda.toFixed(4);
  $('#bids-readout').innerHTML = `<span class="who a"><i></i>${esc(A)} bids <b>${credits(S * lambda)}</b></span><span class="ratio mono">${fmt(lambda * 100, 1)} : ${fmt((1 - lambda) * 100, 1)}</span><span class="who b"><i></i>${esc(B)} bids <b>${credits(S * (1 - lambda))}</b></span>`;
  $('#chart-s').textContent = credits(S);
  $$('#track .seg').forEach(el => el.classList.toggle('is-current', Number(el.dataset.leaf) === state.leaf));
  $$('.band-seg').forEach(el => el.classList.toggle('is-current', Number(el.dataset.leaf) === state.leaf));
  $$('.card').forEach(el => { const on = Number(el.dataset.leaf) === state.leaf; el.classList.toggle('is-current', on); el.setAttribute('aria-current', on ? 'true' : 'false'); });
  if (!leaf) return;
  renderLedger(leaf);
  for (const lane of $$('.lane')) {
    const v = leaf.outcomes[lane.dataset.key];
    lane.querySelector('[data-now]').innerHTML = typeof v === 'number' ? `here ${fmt(v, 4)}` : v === true ? '<span class="truth yes">● true here</span>' : v === false ? '<span class="truth">○ false here</span>' : 'n/a here';
  }
  if (leafChanged) { updateAnswer(); renderTape(); renderSplitPanel(); }
  else updateTapeLambda();
  updateChargeCursor();
}

function renderLedger(leaf) {
  const { lambda, S } = state, [A, B] = state.atlas.bidders;
  const c = chargesAt(leaf, lambda, S);
  const { lower, upper } = boundarySplits(state.atlas, leaf);
  const bound = (s, side) => s
    ? `<button class="bound" data-select-split="${state.atlas.splits.indexOf(s)}"><span>${side === 'lower' ? '← lower edge' : 'upper edge →'}</span><b class="mono">u = ${fmt(s.u, 4)}</b><small>step ${s.step}</small></button>`
    : `<div class="bound end"><span>${side === 'lower' ? '← lower edge' : 'upper edge →'}</span><b class="mono">${side === 'lower' ? '0⁺' : '1'}</b><small>${side === 'lower' ? `only ${esc(B)} bids` : `only ${esc(A)} bids`}</small></div>`;
  $('#ledger').innerHTML = `
    <p class="eyebrow">THIS ANSWER AT λ = ${fmt(lambda, 4)}</p>
    <div class="charges">
      <div class="charge a"><span class="who a"><i></i>${esc(A)} pays</span><b class="mono">${credits(c.a)}</b><small>${c.wonA} contest${c.wonA === 1 ? '' : 's'} won · bid ${credits(S * lambda)}</small></div>
      <div class="charge b"><span class="who b"><i></i>${esc(B)} pays</span><b class="mono">${credits(c.b)}</b><small>${c.wonB} contest${c.wonB === 1 ? '' : 's'} won · bid ${credits(S * (1 - lambda))}</small></div>
    </div>
    <p class="formula small muted">Simulated credits, summed per contest: A pays S(1−λ)·u/(1−u); B pays S·λ·(1−u)/u.</p>
    <dl class="facts">
      <div><dt>Steps</dt><dd>${leaf.steps.length} <small>${c.shared} shared · ${c.contested} contested</small></dd></div>
      <div><dt>Interval width</dt><dd>${pct(leaf.hi - leaf.lo)} <small>of all bid ratios</small></dd></div>
      <div><dt>Stop</dt><dd>${esc(leaf.stop_reason || '—')}</dd></div>
    </dl>
    <p class="eyebrow">BOUNDED BY</p>
    <div class="bounds">${bound(lower, 'lower')}${bound(upper, 'upper')}</div>`;
}

function updateAnswer() {
  const leaf = state.leaves[state.leaf], i = state.leaf, n = state.leaves.length;
  $('#leaf-eyebrow').textContent = `ANSWER ${i + 1} OF ${n}`;
  $('#leaf-interval').innerHTML = `λ ∈ <span class="mono">(${fmt(leaf.lo, 4)}, ${fmt(leaf.hi, 4)}]</span>`;
  $$('[data-diff]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.diff === state.diff)));
  renderHere(leaf);
  if (state.atlas.kind === 'image') { showImage(leaf); return; }
  const neighbour = state.diff === 'lower' ? state.leaves[i - 1] : state.diff === 'upper' ? state.leaves[i + 1] : null;
  const answer = $('#answer');
  if (!neighbour) {
    answer.innerHTML = esc(leaf.text || '(no text recorded)');
    $('#diff-caption').textContent = state.diff === 'none' ? '' : `No ${state.diff === 'lower' ? 'lower' : 'higher'} neighbour: this is the ${state.diff === 'lower' ? 'first' : 'last'} answer.`;
  } else {
    const parts = wordDiff(leaf.text, neighbour.text);
    const changed = parts.filter(p => p.changed).length;
    answer.innerHTML = markRuns(parts);
    $('#diff-caption').innerHTML = `<mark>Highlighted</mark> ${changed} word${changed === 1 ? '' : 's'} not in answer ${i + (state.diff === 'lower' ? 0 : 2)} (${state.diff === 'lower' ? 'just below' : 'just above'} this interval).`;
  }
  if (!state.reduced) { answer.classList.remove('flash'); void answer.offsetWidth; answer.classList.add('flash'); }
}

/** Merge changed words, and the spaces between them, into single highlighted runs. */
function markRuns(parts) {
  let html = '', run = '';
  parts.forEach((p, k) => {
    const bridge = !p.changed && /^\s+$/.test(p.text) && run && parts[k + 1]?.changed;
    if (p.changed || bridge) { run += p.text; return; }
    if (run) { html += `<mark>${esc(run)}</mark>`; run = ''; }
    html += esc(p.text);
  });
  return html + (run ? `<mark>${esc(run)}</mark>` : '');
}

function renderHere(leaf) {
  const host = $('#here');
  if (!host) return;
  const chips = state.keys.map(k => {
    const v = leaf.outcomes[k];
    if (typeof v === 'number') return `<span class="here-chip"><b class="mono">${fmt(v, 3)}</b> ${esc(prettyKey(k))}</span>`;
    return `<span class="here-chip ${v ? 'on' : ''}" style="--yes:${keyColor(k)}"><i></i>${esc(prettyKey(k))}</span>`;
  }).join('');
  host.innerHTML = `<span class="small muted">Outcomes here</span>${chips}<span class="here-chip plain">stop: ${esc(leaf.stop_reason || '—')}</span><span class="here-chip plain">${leaf.token_ids?.length || leaf.steps.length} ${leaf.token_ids?.length ? 'tokens' : 'steps'}</span>`;
}

function showImage(leaf) {
  const a = $('#img-a'), b = $('#img-b');
  const front = a.classList.contains('front') ? a : b.classList.contains('front') ? b : null;
  const back = front === a ? b : a;
  const src = leaf.image ? new URL(leaf.image, state.atlasUrl).pathname : '';
  const what = { synthetic_fixture: 'Placeholder image', synthetic_model_test: 'Tiny-random-model image (noise by design)', live_model: 'Model image' }[state.atlas.mode] || 'Unlabeled image';
  const scores = state.keys.filter(k => typeof leaf.outcomes[k] === 'number').map(k => `<span class="mono">${esc(prettyKey(k))} = ${fmt(leaf.outcomes[k], 3)}</span>`);
  $('#image-caption').innerHTML = `${what} for answer ${state.leaf + 1}.${scores.length ? ` ${scores.join(' · ')} <span class="muted">(noisy heuristic)</span>` : ''}`;
  if (front && front.getAttribute('src') === src) return;
  back.alt = `Image for answer ${state.leaf + 1}, lambda in (${fmt(leaf.lo, 3)}, ${fmt(leaf.hi, 3)}]`;
  const swap = () => { back.classList.add('front'); front?.classList.remove('front'); };
  back.onload = swap;
  back.src = src;
  if (back.complete && back.naturalWidth) swap();
}

function selectSplit(index, jump) {
  state.split = index;
  const s = state.atlas.splits[index];
  $$('.tick').forEach(t => t.classList.toggle('is-selected', Number(t.dataset.split) === index));
  if (jump && s && !(state.lambda > s.u - 1e-12 && state.lambda <= s.u)) {
    // Stay on the current answer when it already touches this boundary; otherwise go to its lower side.
    const leaf = state.leaves[state.leaf];
    if (!(leaf && (leaf.lo === s.u || leaf.hi === s.u))) setLambda(s.u);
  }
  renderSplitPanel();
  renderTape();
  $('#split-panel').scrollIntoView({ behavior: state.reduced ? 'auto' : 'smooth', block: 'nearest' });
}

function renderSplitPanel() {
  const panel = $('#split-panel');
  const s = state.split == null ? null : state.atlas.splits[state.split];
  if (!s) {
    const { lower, upper } = boundarySplits(state.atlas, state.leaves[state.leaf]);
    const options = [lower, upper].filter(Boolean);
    panel.innerHTML = `<div class="chart-head"><h3>Contest boundaries</h3><p class="muted small">Every tick on the track is one contested step whose uniform u fell strictly inside a live interval. Choose a tick, or one of this answer's edges:</p></div><div class="bounds">${options.map(o => `<button class="bound" data-select-split="${state.atlas.splits.indexOf(o)}"><span>step ${o.step}</span><b class="mono">u = ${fmt(o.u, 4)}</b></button>`).join('') || '<p class="muted">This answer covers all of (0, 1].</p>'}</div>`;
    return;
  }
  const below = state.leaves[leafIndexAt(state.leaves, s.u)];
  const above = state.leaves[leafIndexAt(state.leaves, Math.min(1, s.u + 1e-12))];
  const [A, B] = state.atlas.bidders;
  const prefix = s.prefix_text;
  const tail = (leaf) => leaf && leaf.text && prefix != null && leaf.text.startsWith(prefix) ? leaf.text.slice(prefix.length, prefix.length + 90) : null;
  const tokenText = (text, id) => text != null ? `<code>${esc(JSON.stringify(text))}</code>` : id != null ? `<code>token #${esc(id)}</code>` : '<span class="muted">(no token: continuous step)</span>';
  const contBelow = tail(below), contAbove = tail(above);
  const order = [...state.atlas.splits].sort((x, y) => x.u - y.u);
  const rank = order.indexOf(s), prev = order[rank - 1], next = order[rank + 1];
  const nav = `<div class="split-nav"><button class="text-button" ${prev ? `data-select-split="${state.atlas.splits.indexOf(prev)}"` : 'disabled'}>← lower boundary</button><span class="small muted mono">${rank + 1} / ${order.length}</span><button class="text-button" ${next ? `data-select-split="${state.atlas.splits.indexOf(next)}"` : 'disabled'}>higher boundary →</button></div>`;
  panel.innerHTML = `${nav}
    <div class="chart-head"><p class="eyebrow">CONTEST BOUNDARY · STEP ${s.step}</p><h3>u = <span class="mono">${fmt(s.u, 6)}</span></h3>
    <p class="muted small">At step ${s.step} the live interval contained u, so it split: λ ≤ u keeps ${esc(B)}'s candidate, λ &gt; u takes ${esc(A)}'s. Parent leaf ${s.parent}, child leaf ${s.child}.</p></div>
    ${prefix != null ? `<p class="prefix"><span class="muted small">Shared prefix</span><br>${prefix.length > 220 ? '…' : ''}${esc(prefix.slice(-220))}<span class="caret">▍</span></p>` : '<p class="prefix muted small">Continuous (image) step: both sides share the state up to this step.</p>'}
    <div class="versus">
      <div class="side b"><span class="who b"><i></i>λ ≤ u · ${esc(B)}'s candidate</span>${tokenText(s.b_token_text, s.b_token)}${contBelow != null ? `<p class="cont">…${esc(contBelow)}…</p>` : ''}${below ? `<button class="text-button" data-go="${(below.lo + below.hi) / 2}">Go to answer ${state.leaves.indexOf(below) + 1} ←</button>` : ''}</div>
      <div class="side a"><span class="who a"><i></i>λ &gt; u · ${esc(A)}'s candidate</span>${tokenText(s.a_token_text, s.a_token)}${contAbove != null ? `<p class="cont">…${esc(contAbove)}…</p>` : ''}${above ? `<button class="text-button" data-go="${(above.lo + above.hi) / 2}">→ Go to answer ${state.leaves.indexOf(above) + 1}</button>` : ''}</div>
    </div>`;
}

function observeResize(selectors, callback) {
  if (state.resizeObserver) state.resizeObserver.disconnect();
  let frame = 0;
  const ro = new ResizeObserver(() => { cancelAnimationFrame(frame); frame = requestAnimationFrame(callback); });
  for (const sel of selectors) $$(sel).forEach(el => ro.observe(el));
  state.resizeObserver = ro;
}

function svg(tag, attrs = {}, text) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text != null) el.textContent = text;
  return el;
}

function renderChargeChart() {
  const host = $('#charge-chart');
  if (!host) return;
  const w = host.clientWidth || 800, h = 150, top = 10, bottom = 18;
  const series = { a: [], b: [] };
  let max = 0;
  for (const leaf of state.leaves) {
    const pts = { a: [], b: [] }, n = Math.max(2, Math.ceil((leaf.hi - leaf.lo) * 160));
    for (let k = 0; k <= n; k += 1) {
      const lam = k === 0 ? leaf.lo + (leaf.hi - leaf.lo) * 1e-9 : leaf.lo + (leaf.hi - leaf.lo) * (k / n);
      if (!(lam > leaf.lo)) continue;
      const c = chargesAt(leaf, lam, state.S);
      pts.a.push([lam, c.a]); pts.b.push([lam, c.b]);
      max = Math.max(max, c.a, c.b);
    }
    series.a.push(pts.a); series.b.push(pts.b);
  }
  max = max || 1;
  const x = l => l * w, y = v => top + (h - top - bottom) * (1 - v / max);
  const root = svg('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': 'Charges for both advertisers across the bid ratio' });
  for (const f of [0, 0.5, 1]) {
    root.append(svg('line', { x1: 0, x2: w, y1: y(max * f), y2: y(max * f), class: 'grid' }));
    root.append(svg('text', { x: 4, y: y(max * f) - 4, class: 'axis-label' }, `${credits(max * f)}`));
  }
  for (const key of ['b', 'a']) {
    for (const pts of series[key]) {
      if (pts.length) root.append(svg('path', { d: pts.map(([l, v], i) => `${i ? 'L' : 'M'}${x(l).toFixed(1)},${y(v).toFixed(1)}`).join(''), class: `charge-line ${key}` }));
    }
  }
  const [A, B] = state.atlas.bidders;
  root.append(svg('text', { x: w - 6, y: top + 12, class: 'series-label a', 'text-anchor': 'end' }, `${A} pays`));
  root.append(svg('text', { x: w - 6, y: top + 28, class: 'series-label b', 'text-anchor': 'end' }, `${B} pays`));
  root.append(svg('circle', { r: 5, class: 'charge-dot a', id: 'dot-a' }));
  root.append(svg('circle', { r: 5, class: 'charge-dot b', id: 'dot-b' }));
  host.replaceChildren(root);
  host.dataset.max = max;
  host.dataset.w = w;
  updateChargeCursor();
}

function updateChargeCursor() {
  const host = $('#charge-chart');
  const leaf = state.leaves[state.leaf];
  if (!host || !leaf || !host.dataset.max) return;
  const max = Number(host.dataset.max), w = Number(host.dataset.w), h = 150, top = 10, bottom = 18;
  const c = chargesAt(leaf, state.lambda, state.S);
  const y = v => top + (h - top - bottom) * (1 - v / max);
  for (const [k, v] of [['a', c.a], ['b', c.b]]) {
    const dot = host.querySelector(`#dot-${k}`);
    if (dot) { dot.setAttribute('cx', state.lambda * w); dot.setAttribute('cy', y(v)); }
  }
}

function renderTape() {
  const host = $('#tape');
  const leaf = state.leaves[state.leaf];
  if (!host || !leaf) return;
  const w = host.clientWidth || 600, h = 210, left = 34, right = 10, top = 10, bottom = 40;
  const n = leaf.steps.length;
  const x = k => left + (w - left - right) * ((k + 0.5) / Math.max(1, n));
  const y = u => top + (h - top - bottom) * (1 - u);
  const root = svg('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': `Contest uniforms for answer ${state.leaf + 1}` });
  root.append(svg('rect', { x: left, width: w - left - right, y: y(leaf.hi), height: Math.max(1, y(leaf.lo) - y(leaf.hi)), class: 'leaf-band' }));
  for (const u of [0, 0.5, 1]) {
    root.append(svg('line', { x1: left, x2: w - right, y1: y(u), y2: y(u), class: 'grid' }));
    root.append(svg('text', { x: left - 6, y: y(u) + 4, class: 'axis-label', 'text-anchor': 'end' }, String(u)));
  }
  root.append(svg('text', { x: left, y: h - 6, class: 'axis-label' }, `step 0 → ${n - 1}   ·   bars below: disagreement mass 1 − shared mass`));
  const bandTop = h - bottom + 6, bandH = 14;
  leaf.steps.forEach((s, k) => {
    const d = Math.max(0, Math.min(1, 1 - (s.shared_mass ?? 1)));
    root.append(svg('rect', { x: x(k) - 1.5, width: 3, y: bandTop + bandH * (1 - d), height: Math.max(0.5, bandH * d), class: `mass ${s.branch === 'shared' ? 'shared' : s.branch.toLowerCase()}` }));
  });
  const line = svg('g', { id: 'tape-lambda' });
  line.append(svg('line', { x1: left, x2: w - right, y1: y(state.lambda), y2: y(state.lambda), class: 'lambda-line' }));
  line.append(svg('text', { x: w - right, y: y(state.lambda) - 5, class: 'lambda-label', 'text-anchor': 'end' }, `λ = ${fmt(state.lambda, 3)}`));
  root.append(line);
  const selected = state.split == null ? null : state.atlas.splits[state.split];
  leaf.steps.forEach((s, k) => {
    if (s.branch !== 'A' && s.branch !== 'B') return;
    const bounding = s.u === leaf.lo || s.u === leaf.hi;
    const g = svg('g', { class: `tape-dot ${s.branch.toLowerCase()} ${bounding ? 'bounding' : ''} ${selected && selected.step === s.step && selected.u === s.u ? 'selected' : ''}` });
    g.append(svg('circle', { cx: x(k), cy: y(s.u), r: bounding ? 6 : 4.5 }));
    g.append(svg('title', {}, `step ${s.step} · ${s.branch} won · u = ${fmt(s.u, 6)} · shared mass ${fmt(s.shared_mass, 3)}`));
    root.append(g);
  });
  host.replaceChildren(root);
  host.dataset.h = h;
}

function updateTapeLambda() {
  const host = $('#tape'), g = host && host.querySelector('#tape-lambda');
  if (!g) return;
  const h = Number(host.dataset.h), top = 10, bottom = 40;
  const yy = top + (h - top - bottom) * (1 - state.lambda);
  const [line, text] = g.children;
  line.setAttribute('y1', yy); line.setAttribute('y2', yy);
  text.setAttribute('y', yy - 5);
  text.textContent = `λ = ${fmt(state.lambda, 3)}`;
}

// --------------------------------------------------------------------- three bidders
const TRI = { A: [300, 46], B: [40, 496], C: [560, 496] };
const toXY = p => [p[0] * TRI.A[0] + p[1] * TRI.B[0] + p[2] * TRI.C[0], p[0] * TRI.A[1] + p[1] * TRI.B[1] + p[2] * TRI.C[1]];
function toBary(x, y) {
  const [ax, ay] = TRI.A, [bx, by] = TRI.B, [cx, cy] = TRI.C;
  const det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
  let l0 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det;
  let l1 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det;
  let l2 = 1 - l0 - l1;
  [l0, l1, l2] = [l0, l1, l2].map(v => Math.max(0, v));
  const s = l0 + l1 + l2;
  return [l0 / s, l1 / s, l2 / s];
}

function dominant(leaf) {
  let best = -1, at = Infinity;
  state.atlas.bidders.forEach((b, i) => { const k = (leaf.text || '').indexOf(b); if (k >= 0 && k < at) { at = k; best = i; } });
  return best;
}

// A cell a time or cell budget cut off holds only a prefix of its answers.
const unresolved = leaf => leaf.stop_reason === 'truncated';
const unresolvedShare = leaves => leaves.reduce((s, l) => s + (unresolved(l) ? polygonArea(l.polygon) : 0), 0) / 0.5;

function cellFill(leaf, i) {
  if (unresolved(leaf)) return 'url(#unresolved)';
  if (state.colorBy === 'brand') {
    const d = dominant(leaf);
    const same = state.leaves.filter((l, j) => j < i && dominant(l) === d).length;
    const strength = [88, 64, 76, 52, 70, 58][same % 6];
    return d < 0 ? `color-mix(in oklab, var(--ink-2) ${strength - 30}%, var(--surface))` : `color-mix(in oklab, var(--${BIDDER_CLASSES[d]}) ${strength}%, var(--surface))`;
  }
  const v = leaf.outcomes[state.colorBy];
  return v === true ? keyColor(state.colorBy) : 'url(#hatch)';
}

function mountSimplex() {
  const { atlas } = state;
  state.leaves = atlas.leaves;
  state.keys = outcomeKeys(state.leaves);
  if (state.colorBy !== 'brand' && !state.keys.includes(state.colorBy)) state.colorBy = 'brand';
  const total = state.leaves.reduce((s, l) => s + polygonArea(l.polygon), 0);
  const problems = Math.abs(total - 0.5) > 1e-9 ? [`Cell areas sum to ${total.toFixed(6)}, not 0.5.`] : [];
  const open = unresolvedShare(state.leaves);
  $('#view').innerHTML = `
  ${problems.length ? `<div class="warning" role="alert"><strong>Atlas check failed.</strong> ${problems.map(esc).join(' ')}</div>` : ''}
  ${open > 0 ? `<div class="warning"><strong>Partly resolved.</strong> The generator's time or cell budget stopped ${pct(open)} of the triangle early (cross-hatched). Those cells hold only the shared prefix, not finished answers, and are not counted as outcomes.</div>` : ''}
  <section class="simplex-grid">
    <article class="panel tri-panel">
      <div class="chart-head"><h2>Drag the bid mix</h2><p class="muted small">Every point is a bid split λ = b / Σb. Each polygon is one cell: all bid mixes inside it produce exactly the same answer for this seed.</p></div>
      <div class="color-by" role="group" aria-label="Colour the triangle by"><span class="small muted">Colour by</span>
        <button data-color="brand">first brand mentioned</button>${state.keys.map(k => `<button data-color="${esc(k)}">${esc(prettyKey(k))}</button>`).join('')}</div>
      <div class="tri-wrap"><svg id="tri" viewBox="0 0 600 560" role="application" aria-label="Three-bidder simplex. Arrow keys move the bid point; Page Up and Page Down jump between cells." tabindex="0"></svg></div>
      <p class="hint">Drag the point, click anywhere, or focus the triangle and use the arrow keys (<kbd>Shift</kbd> for fine steps, <kbd>PgUp</kbd>/<kbd>PgDn</kbd> to jump cell to cell).</p>
      <p class="small muted" id="color-legend"></p>
    </article>
    <aside class="panel simplex-side" aria-live="polite">
      <div id="bary"></div>
      <label class="budget">Total budget S <input id="budget-input" type="number" min="0.01" step="1" value="${state.S}" inputmode="decimal"><span class="unit">credits</span></label>
      <p class="eyebrow" id="cell-eyebrow"></p>
      <p class="answer" id="answer"></p>
      <div id="cell-facts"></div>
    </aside>
  </section>
  <section class="all panel"><div class="chart-head"><h2>All cells</h2><p class="muted small">The 1 + T bound is for two bidders; with three, one step can cut several cells. Each cell is a convex polygon. Hover to locate, click to jump to its centroid.</p></div><div class="cards" id="cards"></div></section>`;
  renderTriangle();
  renderSimplexCards();
  wireSimplex();
  const centroid = poly => [0, 1, 2].map(k => poly.reduce((s, p) => s + p[k], 0) / poly.length);
  const start = cellIndexAt(state.leaves, state.point) >= 0 ? state.point : centroid(state.leaves[0].polygon);
  setPoint(start, true);
}

function renderTriangle() {
  const tri = $('#tri');
  const poly = p => p.map(q => toXY(q).map(v => v.toFixed(2)).join(',')).join(' ');
  const labels = state.atlas.bidders.map((b, i) => {
    const [x, y] = [TRI.A, TRI.B, TRI.C][i];
    const dy = i === 0 ? -16 : 34, anchor = i === 0 ? 'middle' : i === 1 ? 'start' : 'end';
    return `<g class="vertex ${BIDDER_CLASSES[i]}"><circle cx="${x}" cy="${y}" r="6"/><text x="${x + (i === 1 ? -26 : i === 2 ? 26 : 0)}" y="${y + dy}" text-anchor="${anchor}">${esc(b)}</text></g>`;
  }).join('');
  const apexes = state.atlas.splits.map(s => s.apex ? toXY(s.apex) : null).filter(Boolean).map(([x, y]) => `<circle class="apex" cx="${x}" cy="${y}" r="2.6"/>`).join('');
  tri.innerHTML = `<defs><pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="7" height="7" class="hatch-bg"/><line x1="0" y1="0" x2="0" y2="7" class="hatch-line"/></pattern><pattern id="unresolved" width="9" height="9" patternUnits="userSpaceOnUse"><rect width="9" height="9" class="unres-bg"/><path d="M0 0L9 9M9 0L0 9" class="unres-line"/></pattern></defs>
    <polygon class="tri-shadow" points="${poly([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}"/>
    <g id="cells">${state.leaves.map((leaf, i) => `<polygon class="cell" data-cell="${i}" points="${poly(leaf.polygon)}" style="fill:${cellFill(leaf, i)}"><title>Cell ${i + 1}</title></polygon>`).join('')}</g>
    <polygon id="cell-outline" class="cell-outline" points=""/>
    <g class="cell-labels">${state.leaves.map((leaf, i) => { const a = polygonArea(leaf.polygon); if (a < 0.006) return ''; const [x, y] = toXY([0, 1, 2].map(k => leaf.polygon.reduce((s, p) => s + p[k], 0) / leaf.polygon.length)); return `<text x="${x}" y="${y + 4}" text-anchor="middle">${i + 1}</text>`; }).join('')}</g>
    <g class="apexes" aria-hidden="true">${apexes}</g>
    ${labels}
    <g id="probe" class="probe"><line id="probe-a"/><line id="probe-b"/><line id="probe-c"/><circle r="15" class="probe-halo"/><circle r="7" class="probe-dot"/></g>`;
  updateColorLegend();
}

function updateColorLegend() {
  $$('[data-color]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.color === state.colorBy)));
  const legend = $('#color-legend');
  if (state.colorBy === 'brand') {
    legend.innerHTML = `Hue: the bidder whose brand the cell's answer names first (${state.atlas.bidders.map((b, i) => `<span class="who ${BIDDER_CLASSES[i]}"><i></i>${esc(b)}</span>`).join(' ')}; grey if none). Lightness only separates neighbouring cells. Small dots are race apexes λ ∝ E where three cones met.${unresolvedShare(state.leaves) > 0 ? ' Cross-hatched: unresolved (budget stopped it).' : ''}`;
  } else {
    const share = state.leaves.reduce((s, l) => s + (l.outcomes[state.colorBy] === true ? polygonArea(l.polygon) : 0), 0) / 0.5;
    legend.innerHTML = `Solid: ${esc(prettyKey(state.colorBy))} is true (${pct(share)} of the simplex by area). Hatched: false.`;
  }
}

function renderSimplexCards() {
  const poly = p => p.map(q => toXY(q).map(v => (v / 10).toFixed(1)).join(',')).join(' ');
  $('#cards').innerHTML = state.leaves.map((leaf, i) => `<button class="card" data-leaf="${i}" style="--seg:${dominant(leaf) < 0 ? 'var(--ink-2)' : `var(--${BIDDER_CLASSES[dominant(leaf)]})`}"><span class="card-head"><b>${i + 1}</b><svg class="mini-tri" viewBox="0 0 60 56" aria-hidden="true"><polygon points="${poly([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}" class="mini-bg"/><polygon points="${poly(leaf.polygon)}" style="fill:${cellFill(leaf, i)}" class="mini-cell"/></svg><span class="mono">${pct(polygonArea(leaf.polygon) / 0.5)} of area</span></span><p>${esc(leaf.text || '')}</p><span class="card-foot">${state.keys.map(k => `<span class="dot ${leaf.outcomes[k] ? 'on' : ''}" style="--yes:${keyColor(k)}" title="${esc(prettyKey(k))}: ${leaf.outcomes[k]}"></span>`).join('')}</span></button>`).join('');
}

function wireSimplex() {
  const tri = $('#tri');
  const fromEvent = e => {
    const pt = tri.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(tri.getScreenCTM().inverse());
    return toBary(p.x, p.y);
  };
  tri.addEventListener('pointerdown', e => { tri.setPointerCapture(e.pointerId); setPoint(fromEvent(e)); tri.focus({ preventScroll: true }); });
  tri.addEventListener('pointermove', e => { if (tri.hasPointerCapture(e.pointerId)) setPoint(fromEvent(e)); });
  tri.addEventListener('pointerup', e => tri.releasePointerCapture?.(e.pointerId));
  tri.addEventListener('keydown', e => {
    const step = e.shiftKey ? 2 : 10;
    const [x, y] = toXY(state.point);
    const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    if (moves[e.key]) { e.preventDefault(); setPoint(toBary(x + moves[e.key][0], y + moves[e.key][1])); return; }
    if (e.key === 'PageUp' || e.key === 'PageDown') {
      e.preventDefault();
      const n = state.leaves.length, next = (state.cell + (e.key === 'PageDown' ? 1 : n - 1)) % n;
      jumpToCell(next);
    }
  });
  $$('[data-color]').forEach(b => b.addEventListener('click', () => {
    state.colorBy = b.dataset.color;
    $$('#cells .cell').forEach(el => { const i = Number(el.dataset.cell); el.style.fill = cellFill(state.leaves[i], i); });
    $$('.mini-cell').forEach((el, i) => { el.style.fill = cellFill(state.leaves[i], i); });
    updateColorLegend();
  }));
  $('#budget-input').addEventListener('input', e => { const v = Number(e.target.value); if (Number.isFinite(v) && v > 0) { state.S = v; updateSimplex(); } });
  const cards = $('#cards');
  const hover = i => $$('#cells .cell').forEach(el => el.classList.toggle('is-hover', i != null && el.dataset.cell === String(i)));
  cards.addEventListener('pointerover', e => hover(e.target.closest('.card')?.dataset.leaf));
  cards.addEventListener('pointerleave', () => hover(null));
  cards.addEventListener('click', e => { const card = e.target.closest('.card'); if (card) { jumpToCell(Number(card.dataset.leaf)); tri.scrollIntoView({ behavior: state.reduced ? 'auto' : 'smooth', block: 'center' }); } });
}

function jumpToCell(i) {
  const poly = state.leaves[i].polygon;
  setPoint([0, 1, 2].map(k => poly.reduce((s, p) => s + p[k], 0) / poly.length));
}

function setPoint(point, force = false) {
  state.point = point;
  const cell = cellIndexAt(state.leaves, point);
  const changed = cell !== state.cell;
  state.cell = cell;
  const [x, y] = toXY(point);
  const probe = $('#probe');
  probe.querySelectorAll('circle').forEach(c => { c.setAttribute('cx', x); c.setAttribute('cy', y); });
  ['a', 'b', 'c'].forEach((k, i) => {
    const foot = toXY(point.map((v, j) => j === i ? 0 : v / (1 - point[i] || 1)));
    const line = $(`#probe-${k}`);
    line.setAttribute('x1', x); line.setAttribute('y1', y); line.setAttribute('x2', foot[0]); line.setAttribute('y2', foot[1]);
    line.setAttribute('class', `probe-line ${k}`);
  });
  $$('#cells .cell').forEach(el => el.classList.toggle('is-current', Number(el.dataset.cell) === cell));
  $('#cell-outline').setAttribute('points', cell < 0 ? '' : state.leaves[cell].polygon.map(q => toXY(q).map(v => v.toFixed(2)).join(',')).join(' '));
  $$('.card').forEach(el => el.classList.toggle('is-current', Number(el.dataset.leaf) === cell));
  $('#tri').setAttribute('aria-valuetext', `${state.atlas.bidders.map((b, i) => `${b} ${pct(point[i])}`).join(', ')}; cell ${cell + 1}`);
  updateSimplex(changed || force);
}

function updateSimplex(cellChanged = true) {
  const { point, S } = state, leaf = state.leaves[state.cell];
  $('#bary').innerHTML = `<p class="eyebrow">BID MIX · S = ${credits(S)} CREDITS</p>` + state.atlas.bidders.map((b, i) => `<div class="bary-row ${BIDDER_CLASSES[i]}"><span class="who ${BIDDER_CLASSES[i]}"><i></i>${esc(b)}</span><span class="bar"><i style="width:${point[i] * 100}%"></i></span><b class="mono">${pct(point[i])}</b><small class="mono">${credits(S * point[i])} cr</small></div>`).join('');
  if (!leaf) { $('#answer').textContent = 'This point lies on no recorded cell.'; return; }
  const c = raceCharges(leaf, point, S);
  $('#cell-eyebrow').textContent = `CELL ${state.cell + 1} OF ${state.leaves.length} · ${pct(polygonArea(leaf.polygon) / 0.5)} OF ALL BID MIXES${unresolved(leaf) ? ' · UNRESOLVED' : ''}`;
  if (cellChanged) {
    const answer = $('#answer');
    answer.textContent = unresolved(leaf)
      ? `${leaf.text || ''} … [unresolved: the ${leaf.truncation?.reason === 'max_cells' ? 'cell' : 'time'} budget stopped this region after ${leaf.steps.length} steps; the answers inside it were not computed]`
      : leaf.text || '(no text recorded)';
    if (!state.reduced) { answer.classList.remove('flash'); void answer.offsetWidth; answer.classList.add('flash'); }
  }
  $('#cell-facts').innerHTML = `
    <div class="charges three">${state.atlas.bidders.map((b, i) => `<div class="charge ${BIDDER_CLASSES[i]}"><span class="who ${BIDDER_CLASSES[i]}"><i></i>${esc(b)}</span><b class="mono">${credits(c.charges[i])}</b><small>${c.wins[i]} won</small></div>`).join('')}</div>
    <p class="formula small muted">Simulated credits. Each contest's winner i pays its critical bid E_i · max over j≠i of b_j / E_j (exponential race).</p>
    <dl class="facts"><div><dt>Steps</dt><dd>${leaf.steps.length} <small>${leaf.steps.length - c.contested} shared · ${c.contested} contested</small></dd></div><div><dt>Polygon</dt><dd>${leaf.polygon.length} vertices</dd></div><div><dt>Outcomes</dt><dd>${state.keys.map(k => `<span class="truth ${leaf.outcomes[k] ? 'yes' : ''}">${leaf.outcomes[k] ? '●' : '○'} ${esc(prettyKey(k))}</span>`).join('<br>')}</dd></div></dl>`;
}

if (typeof document !== 'undefined' && typeof window !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
}
