// Two non-LLM generators for the Atlas Playground. Both are synthetic,
// hand-written stochastic policies: they show that the auction and the atlas
// only need two next-step distributions, not a language model.
import { normalize, softmax } from './atlas_core.mjs';

// ---------- City walk: two sponsors pull a walker past their storefronts ----------
export const CITY = {
  width: 15,
  height: 10,
  start: [0, 7],
  goal: [14, 2],
  sponsors: [
    { name: 'Brindlewick Coffee', kind: 'café', at: [4, 1] },
    { name: 'Quillmere Books', kind: 'bookshop', at: [10, 8] },
  ],
  // Closed blocks (parks) that streets route around.
  parks: [[6, 3], [7, 3], [6, 4], [7, 4], [6, 5], [7, 5], [11, 4], [12, 4]],
};

const MOVES = [[0, -1], [1, 0], [0, 1], [-1, 0]]; // N E S W
export const MOVE_NAMES = ['north', 'east', 'south', 'west'];
const key = ([x, y]) => `${x},${y}`;
const manhattan = (p, q) => Math.abs(p[0] - q[0]) + Math.abs(p[1] - q[1]);

function blocked(world, p) {
  return p[0] < 0 || p[1] < 0 || p[0] >= world.width || p[1] >= world.height
    || world.parks.some(q => q[0] === p[0] && q[1] === p[1]);
}

// Shortest street distance from every intersection to a target (BFS).
function distanceField(world, target) {
  const dist = new Map([[key(target), 0]]);
  const queue = [target];
  while (queue.length) {
    const p = queue.shift();
    for (const [dx, dy] of MOVES) {
      const q = [p[0] + dx, p[1] + dy];
      if (!blocked(world, q) && !dist.has(key(q))) {
        dist.set(key(q), dist.get(key(p)) + 1);
        queue.push(q);
      }
    }
  }
  return dist;
}

export function cityWalk(world = CITY, { pull = 1.6, temperature = 0.55, revisit = 1.2 } = {}) {
  const toGoal = distanceField(world, world.goal);
  const toStore = world.sponsors.map(s => distanceField(world, s.at));
  const near = (p, store) => manhattan(p, store) <= 1;

  // Sponsor i's preferred walk: reach its storefront, then head for the goal.
  function policy(state, i) {
    const target = state.visited[i] ? toGoal : toStore[i];
    const seen = new Set(state.path.map(key));
    const logits = MOVES.map(([dx, dy]) => {
      const q = [state.at[0] + dx, state.at[1] + dy];
      if (blocked(world, q)) return -Infinity;
      return -pull * target.get(key(q)) - 0.25 * toGoal.get(key(q)) - (seen.has(key(q)) ? revisit : 0);
    });
    return softmax(logits, temperature);
  }

  return {
    world,
    init: () => ({ at: world.start, path: [world.start], visited: world.sponsors.map(s => near(world.start, s.at)) }),
    dists: state => world.sponsors.map((_, i) => policy(state, i)),
    apply(state, move) {
      const [dx, dy] = MOVES[move];
      const at = [state.at[0] + dx, state.at[1] + dy];
      return {
        at,
        path: [...state.path, at],
        visited: state.visited.map((v, i) => v || near(at, world.sponsors[i].at)),
      };
    },
    done: state => state.at[0] === world.goal[0] && state.at[1] === world.goal[1],
  };
}

// The same city with a third sponsor, for three-bidder atlases.
export const CITY3 = {
  ...CITY,
  sponsors: [...CITY.sponsors, { name: 'Tansy Market', kind: 'grocer', at: [12, 0] }],
};

// ---------- Melody: two composers share one keyboard, note by note ----------
// Options: MIDI pitches 57..81 (A3..A5) plus a rest.
export const PITCHES = Array.from({ length: 25 }, (_, i) => 57 + i);
export const REST = PITCHES.length;
const NAMES = ['C', 'C♯', 'D', 'E♭', 'E', 'F', 'F♯', 'G', 'A♭', 'A', 'B♭', 'B'];
export const noteName = option => (option === REST ? 'rest' : `${NAMES[PITCHES[option] % 12]}${Math.floor(PITCHES[option] / 12) - 1}`);

export const COMPOSERS = [
  {
    name: 'Halcyonne Records',
    mood: 'bright major, rising steps and arpeggios',
    // Pitch-class weights: C major with emphasis on the C-E-G triad.
    scale: { 0: 3.2, 2: 1.6, 4: 2.8, 5: 1.1, 7: 3.0, 9: 1.5, 11: 0.9 },
    leap: [0.25, 1.6, 1.4, 1.1, 1.3, 0.6, 0.3, 0.8], // by |interval| in scale-ish semitone bins
    drift: 0.35, // prefers to rise
    rest: 0.04,
  },
  {
    name: 'Velvetmoth Blues',
    mood: 'minor blues, blue notes and falling phrases',
    // C minor blues: C, E♭, F, F♯ (blue), G, B♭.
    scale: { 0: 3.0, 3: 2.8, 5: 1.8, 6: 1.3, 7: 2.4, 10: 2.2, 2: 0.4 },
    leap: [0.5, 1.8, 1.5, 1.4, 0.9, 0.7, 0.4, 0.5],
    drift: -0.3, // prefers to fall
    rest: 0.1,
  },
];

export function melody({ length = 32, composers = COMPOSERS } = {}) {
  function policy(state, c) {
    const spec = composers[c];
    const last = state.last;
    const weights = PITCHES.map(p => {
      const w = spec.scale[p % 12] ?? 0.02;
      if (last === null) return w * (p % 12 === 0 ? 4 : 1);
      const interval = p - PITCHES[last];
      const leap = spec.leap[Math.min(7, Math.abs(interval))] * Math.exp(-Math.max(0, Math.abs(interval) - 7) * 0.7);
      return w * leap * Math.exp(spec.drift * Math.sign(interval));
    });
    const total = weights.reduce((s, x) => s + x, 0);
    const restWeight = state.notes.length && state.notes.length % 8 === 7 ? 6 * spec.rest * total : spec.rest * total;
    return normalize([...weights, restWeight]);
  }
  return {
    init: () => ({ notes: [], last: null }),
    dists: state => [policy(state, 0), policy(state, 1)],
    apply: (state, option) => ({ notes: [...state.notes, option], last: option === REST ? state.last : option }),
    done: state => state.notes.length >= length,
  };
}

// A simple outcome for melodies: share of sounding notes that are blue notes.
export function blueShare(notes) {
  const sounding = notes.filter(n => n !== REST);
  if (!sounding.length) return 0;
  return sounding.filter(n => [3, 6, 10].includes(PITCHES[n] % 12)).length / sounding.length;
}
