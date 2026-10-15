// Query generation and preference estimation for the in-browser handover demo.
// A dependency-free implementation; the CMA-ES update follows the `cmaes`
// Python package (verified to match it to ~1e-12).
//
// Behaviours live in a fixed dataset of feature vectors. Points proposed in
// feature space are projected onto the dataset by nearest neighbour, as in the
// paper's real-robot experiments.

// ---------------------------------------------------------------------------
// Random numbers
// ---------------------------------------------------------------------------
export class RNG {
  constructor(seed = (Math.random() * 2 ** 32) >>> 0) {
    this.s = seed >>> 0;
    this.spare = null;
  }
  uniform() { // mulberry32
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  normal() {
    if (this.spare !== null) { const v = this.spare; this.spare = null; return v; }
    let u = 0;
    while (u === 0) u = this.uniform();
    const r = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * this.uniform();
    this.spare = r * Math.sin(th);
    return r * Math.cos(th);
  }
  int(n) { return Math.floor(this.uniform() * n); }
  choice(weights) { // index sampled proportionally to (normalised) weights
    let u = this.uniform(), acc = 0;
    for (let i = 0; i < weights.length; i++) { acc += weights[i]; if (u < acc) return i; }
    return weights.length - 1;
  }
}

// ---------------------------------------------------------------------------
// Small linear-algebra helpers (dense, row-major arrays of arrays)
// ---------------------------------------------------------------------------
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const norm = (a) => Math.sqrt(dot(a, a));
const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)));
const matVec = (M, v) => M.map((row) => dot(row, v));

// Cyclic Jacobi eigen-decomposition of a symmetric matrix: A = V diag(w) V^T.
export function eighSym(A) {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const V = eye(n);
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return { values: a.map((r, i) => r[i]), vectors: V }; // eigenvectors are columns of V
}

// ---------------------------------------------------------------------------
// CMA-ES, following cmaes.CMA (Hansen's tutorial, with active covariance update)
// ---------------------------------------------------------------------------
const EPS = 1e-8;

export class CMA {
  constructor(mean, sigma, popsize, rng) {
    const n = mean.length;
    this.n = n;
    this.popsize = popsize;
    this.rng = rng;
    const mu = Math.floor(popsize / 2);
    const wp = Array.from({ length: popsize }, (_, i) => Math.log((popsize + 1) / 2) - Math.log(i + 1));
    const sum = (xs) => xs.reduce((s, x) => s + x, 0);
    const muEff = sum(wp.slice(0, mu)) ** 2 / sum(wp.slice(0, mu).map((w) => w * w));
    const muEffMinus = sum(wp.slice(mu)) ** 2 / sum(wp.slice(mu).map((w) => w * w));
    const alphaCov = 2;
    const c1 = alphaCov / ((n + 1.3) ** 2 + muEff);
    const cmu = Math.min(1 - c1 - 1e-8, (alphaCov * (muEff - 2 + 1 / muEff)) / ((n + 2) ** 2 + (alphaCov * muEff) / 2));
    const minAlpha = Math.min(1 + c1 / cmu, 1 + (2 * muEffMinus) / (muEff + 2), (1 - c1 - cmu) / (n * cmu));
    const pos = sum(wp.filter((w) => w > 0));
    const neg = sum(wp.filter((w) => w < 0).map(Math.abs));
    this.weights = wp.map((w) => (w >= 0 ? w / pos : (minAlpha / neg) * w));
    this.mu = mu;
    this.muEff = muEff;
    this.c1 = c1;
    this.cmu = cmu;
    this.cSigma = (muEff + 2) / (n + muEff + 5);
    this.dSigma = 1 + 2 * Math.max(0, Math.sqrt((muEff - 1) / (n + 1)) - 1) + this.cSigma;
    this.cc = (4 + muEff / n) / (n + 4 + (2 * muEff) / n);
    this.chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
    this.pSigma = new Array(n).fill(0);
    this.pc = new Array(n).fill(0);
    this.mean = mean.slice();
    this.sigma = sigma;
    this.C = eye(n);
    this.g = 0;
    this._eig = null;
  }

  eig() {
    if (this._eig) return this._eig;
    const n = this.n;
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) this.C[i][j] = this.C[j][i] = (this.C[i][j] + this.C[j][i]) / 2;
    const { values, vectors: B } = eighSym(this.C);
    const D = values.map((v) => Math.sqrt(v < 0 ? EPS : v));
    this.C = B.map((_, i) => B.map((_, j) => { let s = 0; for (let k = 0; k < n; k++) s += B[i][k] * D[k] * D[k] * B[j][k]; return s; }));
    return (this._eig = { B, D });
  }

  ask() {
    const { B, D } = this.eig();
    const z = Array.from({ length: this.n }, () => this.rng.normal());
    return this.mean.map((m, i) => { let y = 0; for (let k = 0; k < this.n; k++) y += B[i][k] * D[k] * z[k]; return m + this.sigma * y; });
  }

  // solutions: [{x, f}] with lower f = better, length == popsize
  tell(solutions) {
    const n = this.n;
    this.g += 1;
    const sols = solutions.slice().sort((a, b) => a.f - b.f);
    const { B, D } = this.eig();
    this._eig = null;
    const yk = sols.map((s) => s.x.map((x, i) => (x - this.mean[i]) / this.sigma));
    const yw = new Array(n).fill(0);
    for (let k = 0; k < this.mu; k++) for (let i = 0; i < n; i++) yw[i] += this.weights[k] * yk[k][i];
    for (let i = 0; i < n; i++) this.mean[i] += this.sigma * yw[i];
    const C2 = B.map((_, i) => B.map((_, j) => { let s = 0; for (let k = 0; k < n; k++) s += (B[i][k] / D[k]) * B[j][k]; return s; }));
    const c2yw = matVec(C2, yw);
    const ks = Math.sqrt(this.cSigma * (2 - this.cSigma) * this.muEff);
    this.pSigma = this.pSigma.map((p, i) => (1 - this.cSigma) * p + ks * c2yw[i]);
    const nps = norm(this.pSigma);
    this.sigma = Math.min(this.sigma * Math.exp((this.cSigma / this.dSigma) * (nps / this.chiN - 1)), 1e32);
    const hLeft = nps / Math.sqrt(1 - (1 - this.cSigma) ** (2 * (this.g + 1)));
    const hSigma = hLeft < (1.4 + 2 / (n + 1)) * this.chiN ? 1 : 0;
    const kc = Math.sqrt(this.cc * (2 - this.cc) * this.muEff);
    this.pc = this.pc.map((p, i) => (1 - this.cc) * p + hSigma * kc * yw[i]);
    const wio = this.weights.map((w, k) => (w >= 0 ? w : (w * n) / (norm(matVec(C2, yk[k])) ** 2 + EPS)));
    const dh = (1 - hSigma) * this.cc * (2 - this.cc);
    const wsum = this.weights.reduce((s, w) => s + w, 0);
    const decay = 1 + this.c1 * dh - this.c1 - this.cmu * wsum;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let rankMu = 0;
        for (let k = 0; k < sols.length; k++) rankMu += wio[k] * yk[k][i] * yk[k][j];
        this.C[i][j] = decay * this.C[i][j] + this.c1 * this.pc[i] * this.pc[j] + this.cmu * rankMu;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers shared by the query generators
// ---------------------------------------------------------------------------
const clip = (x) => x.map((v) => Math.max(-1, Math.min(1, v)));

export function nearestUnique(points, features) {
  const chosen = [];
  for (const p of points) {
    let best = -1, bestD = Infinity;
    for (let i = 0; i < features.length; i++) {
      if (chosen.includes(i)) continue;
      let d = 0;
      for (let j = 0; j < p.length; j++) d += (features[i][j] - p[j]) ** 2;
      if (d < bestD) { bestD = d; best = i; }
    }
    chosen.push(best);
  }
  return chosen;
}

// k-means++ initialisation + Lloyd iterations, best of a few restarts.
export function kmeans(X, k, rng, nInit = 3, iters = 100) {
  const d2 = (a, b) => { let s = 0; for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2; return s; };
  let best = null;
  for (let r = 0; r < nInit; r++) {
    const centers = [X[rng.int(X.length)].slice()];
    while (centers.length < k) {
      const dist = X.map((x) => Math.min(...centers.map((c) => d2(x, c))));
      const tot = dist.reduce((s, v) => s + v, 0);
      centers.push(X[rng.choice(dist.map((v) => v / tot))].slice());
    }
    let labels = new Array(X.length).fill(0);
    for (let it = 0; it < iters; it++) {
      let changed = false;
      for (let i = 0; i < X.length; i++) {
        let bl = 0, bd = Infinity;
        for (let c = 0; c < k; c++) { const v = d2(X[i], centers[c]); if (v < bd) { bd = v; bl = c; } }
        if (labels[i] !== bl) { labels[i] = bl; changed = true; }
      }
      for (let c = 0; c < k; c++) {
        const members = X.filter((_, i) => labels[i] === c);
        if (!members.length) continue;
        centers[c] = members[0].map((_, j) => members.reduce((s, m) => s + m[j], 0) / members.length);
      }
      if (!changed && it > 0) break;
    }
    const inertia = X.reduce((s, x, i) => s + d2(x, centers[labels[i]]), 0);
    if (!best || inertia < best.inertia) best = { centers, inertia };
  }
  return best.centers;
}

// ---------------------------------------------------------------------------
// Particle belief over a linear reward weight ω in the unit ball
// ---------------------------------------------------------------------------
export class ParticleBelief {
  constructor(dim, rng, nParticles = 4000, rationality = 8) {
    this.dim = dim;
    this.n = nParticles;
    this.beta = rationality;
    this.rng = rng;
    this.particles = Array.from({ length: nParticles }, () => this.sampleBall());
    this.logW = new Float64Array(nParticles);
    this.history = []; // each entry: K feature rows, favourite first
  }

  sampleBall() {
    const x = Array.from({ length: this.dim }, () => this.rng.normal());
    const s = this.rng.uniform() ** (1 / this.dim) / norm(x);
    return x.map((v) => v * s);
  }

  // Plackett-Luce log-likelihood of a ranking (rows best-first) under ω.
  logLik(omega, ranked) {
    const r = ranked.map((row) => this.beta * dot(omega, row));
    let ll = 0;
    for (let i = 0; i < r.length - 1; i++) {
      let m = -Infinity;
      for (let j = i; j < r.length; j++) m = Math.max(m, r[j]);
      let s = 0;
      for (let j = i; j < r.length; j++) s += Math.exp(r[j] - m);
      ll += r[i] - (m + Math.log(s));
    }
    return ll;
  }

  logPosterior(omega) {
    if (norm(omega) > 1) return -Infinity;
    let lp = 0;
    for (const ranked of this.history) lp += this.logLik(omega, ranked);
    return lp;
  }

  get weights() {
    let m = -Infinity;
    for (const v of this.logW) m = Math.max(m, v);
    const w = Array.from(this.logW, (v) => Math.exp(v - m));
    const s = w.reduce((a, b) => a + b, 0);
    return w.map((v) => v / s);
  }

  update(ranked) {
    this.history.push(ranked);
    for (let i = 0; i < this.n; i++) this.logW[i] += this.logLik(this.particles[i], ranked);
    const w = this.weights;
    if (1 / w.reduce((s, v) => s + v * v, 0) < this.n / 2) this.resampleMove(w);
  }

  resampleMove(w, steps = 5, stepSize = 0.08) {
    // systematic resampling
    const x = [];
    const u0 = this.rng.uniform() / this.n;
    let acc = w[0], j = 0;
    for (let i = 0; i < this.n; i++) {
      const u = u0 + i / this.n;
      while (u > acc && j < this.n - 1) acc += w[++j];
      x.push(this.particles[j].slice());
    }
    const lp = x.map((p) => this.logPosterior(p));
    for (let s = 0; s < steps; s++) {
      for (let i = 0; i < this.n; i++) {
        const prop = x[i].map((v) => v + stepSize * this.rng.normal());
        const lpp = this.logPosterior(prop);
        if (Math.log(this.rng.uniform()) < lpp - lp[i]) { x[i] = prop; lp[i] = lpp; }
      }
    }
    this.particles = x;
    this.logW = new Float64Array(this.n);
  }

  mean() {
    const w = this.weights;
    const m = new Array(this.dim).fill(0);
    for (let i = 0; i < this.n; i++) for (let j = 0; j < this.dim; j++) m[j] += w[i] * this.particles[i][j];
    return m;
  }
}

// ---------------------------------------------------------------------------
// Query generators. query(belief) -> dataset indices; tell(queryIdx, ranking)
// where ranking lists positions in queryIdx from most to least preferred.
// ---------------------------------------------------------------------------
class CMABase {
  constructor(features, { k = 3, sigma = 0.5, rng = new RNG() } = {}) {
    this.features = features;
    this.k = k;
    this.rng = rng;
    this.optimizer = new CMA(new Array(features[0].length).fill(0), sigma, k, rng);
    this.lastSamples = [];
  }
  tell(queryIdx, ranking) {
    // CMA-ES minimises, so the favourite gets the lowest value.
    this.optimizer.tell(ranking.map((pos, place) => ({ x: this.features[queryIdx[pos]], f: place })));
  }
  searchDistribution() {
    const o = this.optimizer;
    o.eig();
    return { mean: o.mean.slice(), cov: o.C.map((r) => r.map((v) => v * o.sigma * o.sigma)) };
  }
}

export class CMAESIG extends CMABase {
  static label = "CMA-ES-IG";
  constructor(features, opts = {}) {
    super(features, opts);
    this.nSamples = opts.nSamples ?? 100;
  }
  query() {
    const samples = Array.from({ length: this.nSamples }, () => clip(this.optimizer.ask()));
    this.lastSamples = samples;
    return nearestUnique(kmeans(samples, this.k, this.rng), this.features);
  }
}

export class CMAES extends CMABase {
  static label = "CMA-ES";
  query() {
    const samples = Array.from({ length: this.k }, () => clip(this.optimizer.ask()));
    this.lastSamples = samples;
    return nearestUnique(samples, this.features);
  }
}

const PERMS3 = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];

export class InfoGain {
  static label = "Infogain";
  constructor(features, { k = 3, rng = new RNG(), nCandidates = 300, nBeliefSamples = 600 } = {}) {
    if (k !== 3) throw new Error("InfoGain demo implementation assumes K = 3");
    this.features = features;
    this.k = k;
    this.rng = rng;
    this.nCandidates = nCandidates;
    this.nBeliefSamples = nBeliefSamples;
    this.lastSamples = [];
  }
  query(belief) {
    const w = belief.weights;
    const omegas = Array.from({ length: this.nBeliefSamples }, () => belief.particles[this.rng.choice(w)]);
    let best = null, bestIG = -Infinity;
    for (let c = 0; c < this.nCandidates; c++) {
      const cand = [];
      while (cand.length < 3) { const i = this.rng.int(this.features.length); if (!cand.includes(i)) cand.push(i); }
      const rows = cand.map((i) => this.features[i]);
      const marginal = new Array(6).fill(0);
      let condEnt = 0;
      for (const om of omegas) {
        for (let p = 0; p < 6; p++) {
          const pr = Math.exp(belief.logLik(om, PERMS3[p].map((i) => rows[i])));
          marginal[p] += pr / omegas.length;
          condEnt -= (pr * Math.log(pr + 1e-12)) / omegas.length;
        }
      }
      const ig = -marginal.reduce((s, p) => s + p * Math.log(p + 1e-12), 0) - condEnt;
      if (ig > bestIG) { bestIG = ig; best = cand; }
    }
    return best;
  }
  tell() {} // driven entirely by the shared belief
  searchDistribution() { return null; }
}

export const ALGORITHMS = { "CMA-ES-IG": CMAESIG, "CMA-ES": CMAES, Infogain: InfoGain };

// Standardise features so the dataset roughly fills the [-1, 1] box.
export function standardise(rows) {
  const d = rows[0].length, n = rows.length;
  const mean = new Array(d).fill(0), std = new Array(d).fill(0);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j] / n;
  for (const r of rows) for (let j = 0; j < d; j++) std[j] += (r[j] - mean[j]) ** 2 / n;
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j]);
  return rows.map((r) => r.map((v, j) => (v - mean[j]) / (2 * std[j])));
}
