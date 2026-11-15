"""
Interactive preference search over a PCA latent space of MNIST.

    uv run prepare.py      # once: downloads MNIST, builds data/latent.npz
    uv run app.py          # then open http://127.0.0.1:5000

Each browser tab is its own session, so two tabs can run two algorithms side by side.
The query generators are the ones used in the paper's experiments:

    infogain       InfoGainQueryGenerator (irlpreference)
    cmaes          CMAESGenerator         (experiments/cmaes_generators.py)
    cmaesig        CMAESIGGenerator       -- k-means over 100 CMA-ES samples (deployed)
    cmaesig_exact  CMA-ES + exhaustive mutual-information search over 20 CMA-ES samples

and the belief is either the library's plain MonteCarloLinearReward or the resample-move
belief from rebuttal/resampling_belief.py.  By default each proposed point is replaced by the
nearest real MNIST digit in latent space, and the belief and CMA-ES learn from those real
digits (what the user actually saw); the decoded-latent view is available as an option.  A ranking updates the belief as K-1 sequential
Luce-Shepard choices, exactly as in the simulations.
"""

import base64
import io
import itertools
import os
import sys
import threading
import time
import uuid

import numpy as np
from cmaes import CMA
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(REPO, 'experiments'))
sys.path.insert(0, os.path.join(REPO, 'rebuttal'))

from irlpreference.input_models import LuceShepardChoice
from irlpreference.query_generation import InfoGainQueryGenerator
from irlpreference.reward_parameterizations import MonteCarloLinearReward
from cmaes_generators import CMAESGenerator, CMAESIGGenerator
from resampling_belief import ResamplingMonteCarloLinearReward

LATENT = np.load(os.path.join(HERE, 'data', 'latent.npz'))
N_SAMPLES = 10_000


def _load_digits():
    import gzip
    with gzip.open(os.path.join(HERE, 'data', 'train-images-idx3-ubyte.gz'), 'rb') as f:
        return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32) / 255


DIGITS = _load_digits()          # the real MNIST training images, aligned with LATENT['latents']


def nearest_digits(points, latents):
    """Index of the closest real digit (in latent space) to each point, all distinct."""
    chosen = []
    for p in points:
        order = np.argsort(np.sum((latents - p) ** 2, axis=1))
        chosen.append(next(int(i) for i in order if i not in chosen))
    return chosen

app = Flask(__name__, static_folder=os.path.join(HERE, 'static'))
SESSIONS = {}


# ------------------------------------------------------------------ images
def decode(x):
    """Latent point in [-1, 1]^d -> 28x28 image in [0, 1]."""
    d = len(x)
    img = LATENT['mean'] + (np.asarray(x) * LATENT['scale'][:d]) @ LATENT['components'][:d]
    return np.clip(img, 0, 1).reshape(28, 28)


def png(img, size=112):
    im = Image.fromarray((255 * (1 - img)).astype(np.uint8))      # dark digit on light paper
    im = im.resize((size, size), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ belief helpers
def weights(belief):
    lw = belief.hypothesis_log_probabilities
    pi = np.exp(lw - lw.max())
    return pi / pi.sum()


def mutual_information(queries, belief, beta):
    """Bits the user's first choice carries about omega, for each query in `queries`
    (array B x K x d), under the belief's posterior weights."""
    pi = weights(belief)
    keep = pi > 1e-10 * pi.max()
    omegas, pi = belief.hypothesis_samples[keep], (pi[keep] / pi[keep].sum()).astype(np.float32)
    r = beta * np.einsum('bkd,sd->bks', np.asarray(queries, np.float32), omegas.astype(np.float32))
    r -= r.max(axis=1, keepdims=True)
    p = np.exp(r)
    p /= p.sum(axis=1, keepdims=True)
    tiny = np.finfo(np.float32).tiny
    cond = np.sum(-p * np.log2(np.clip(p, tiny, None)), axis=1) @ pi
    pred = p @ pi
    return np.sum(-pred * np.log2(np.clip(pred, tiny, None)), axis=1) - cond


class InfogainGenerator(InfoGainQueryGenerator):
    """The library's Infogain generator; identical objective, bounds and random start, but
    calls L-BFGS-B without `iprint`, which SciPy >= 1.17 no longer accepts."""

    def _get_continuous_query(self, number_queries, reward_parameterization, input_model):
        import scipy.optimize as opt
        x0 = np.random.uniform(low=self.lower_bounds * number_queries,
                               high=self.upper_bounds * number_queries,
                               size=self.dimension * number_queries)

        def objective(x):
            return -self._info_gain(reward_parameterization, input_model,
                                    x.reshape(number_queries, self.dimension))

        res = opt.minimize(objective, x0, method='L-BFGS-B',
                           bounds=self.bounds * number_queries)
        return res.x.reshape(number_queries, self.dimension)


class ExactMIGenerator:
    """CMA-ES-IG with the k-means step replaced by exhaustive search: draw `pool` CMA-ES
    samples and show the K-subset with the highest mutual information."""

    def __init__(self, dim, k, beta, pool=20):
        self.optimizer = CMA(mean=np.zeros(dim), sigma=1.3, population_size=k)
        self.pool, self.beta = pool, beta

    def get_query(self, k, belief, _choice_model):
        pool = np.array([np.clip(self.optimizer.ask(), -1, 1) for _ in range(self.pool)])
        subsets = np.array(list(itertools.combinations(range(self.pool), k)))
        scores = np.concatenate([mutual_information(pool[subsets[i:i + 400]], belief, self.beta)
                                 for i in range(0, len(subsets), 400)])
        return pool[subsets[int(np.argmax(scores))]]

    def tell(self, solutions, rankings):
        self.optimizer.tell([(s, -rankings[i]) for i, s in enumerate(solutions)])


# ------------------------------------------------------------------ sessions
class Session:
    def __init__(self, method, belief, dim, k, beta, snap=True):
        self.method, self.belief_kind, self.dim, self.k, self.beta = method, belief, dim, k, beta
        # snap=True: show (and learn from) the real digit nearest to each proposed point
        self.snap = snap
        self.latents = LATENT['latents'][:, :dim]
        self.lock = threading.Lock()
        self.choice = LuceShepardChoice(rationality=beta)
        self.belief = (ResamplingMonteCarloLinearReward(dim, N_SAMPLES)
                       if belief == 'resample' else MonteCarloLinearReward(dim, number_samples=N_SAMPLES))
        limits = [(-1, 1)] * dim
        self.gen = {'infogain': lambda: InfogainGenerator(limits),
                    'cmaes': lambda: CMAESGenerator(dim, limits, k),
                    'cmaesig': lambda: CMAESIGGenerator(dim, limits, k),
                    'cmaesig_exact': lambda: ExactMIGenerator(dim, k, beta)}[method]()
        self.history = []          # dicts: images in ranked order, bits
        self.stats = []            # per answered query: bits, ess
        self.next_query()

    def next_query(self):
        t0 = time.perf_counter()
        proposed = np.array(self.gen.get_query(self.k, self.belief, self.choice), dtype=float)
        if self.snap:
            self.digit_idx = nearest_digits(proposed, self.latents)
            self.query = self.latents[self.digit_idx].astype(float)
            self.snap_dist = float(np.mean(np.linalg.norm(self.query - proposed, axis=1)))
        else:
            self.digit_idx, self.query, self.snap_dist = None, proposed, None
        self.gen_ms = 1000 * (time.perf_counter() - t0)
        self.bits = float(mutual_information(self.query[None], self.belief, self.beta)[0])

    def rank(self, order):
        """order: item indices from most to least preferred."""
        working = self.query.copy()
        remaining = list(range(self.k))
        for idx in order[:-1]:
            pos = remaining.index(idx)
            self.choice.tell_input(pos, working)
            self.belief.update(self.choice.get_probability_of_input)
            working = np.delete(working, pos, axis=0)
            remaining.pop(pos)
        if self.method != 'infogain':
            self.gen.tell([self.query[i] for i in order], list(range(self.k))[::-1])
        self.history.append({'images': [self.image(i, 56) for i in order], 'bits': self.bits})
        self.stats.append({'bits': self.bits, 'ess': ess(self.belief)})
        self.next_query()

    def image(self, i, size=112):
        """Item i of the current query: the real digit if snapping, else the decoded point."""
        if self.snap:
            return png(DIGITS[self.digit_idx[i]], size)
        return png(decode(self.query[i]), size)

    def state(self):
        w = self.belief.get_expectation()
        lat = LATENT['latents'][:, :self.dim]
        scores = lat @ w
        top = np.argsort(scores)[::-1]
        labels, counts = np.unique(LATENT['labels'][top[:200]], return_counts=True)
        return {
            'method': self.method, 'belief': self.belief_kind, 'dim': self.dim, 'k': self.k,
            'query_number': len(self.history) + 1,
            'query': [self.image(i) for i in range(self.k)],
            'snap': self.snap, 'snap_dist': self.snap_dist,
            'query_labels': ([int(LATENT['labels'][i]) for i in self.digit_idx]
                             if self.snap else None),
            'query_bits': self.bits, 'gen_ms': self.gen_ms,
            'ess': ess(self.belief), 'n_samples': N_SAMPLES,
            'resamples': getattr(self.belief, 'n_resamples', None),
            'top_digits': [png(DIGITS[i], 56) for i in top[:6]],
            'top_labels': {int(l): int(c) for l, c in zip(labels, counts)},
            'stats': self.stats,
            'history': self.history[::-1][:12],
        }


def ess(belief):
    return float(1.0 / np.sum(weights(belief) ** 2))


# ------------------------------------------------------------------ routes
@app.get('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.post('/api/new')
def new_session():
    cfg = request.get_json()
    sid = uuid.uuid4().hex
    SESSIONS[sid] = Session(cfg.get('method', 'cmaesig'), cfg.get('belief', 'resample'),
                            int(cfg.get('dim', 8)), int(cfg.get('k', 3)), float(cfg.get('beta', 10)),
                            snap=bool(cfg.get('snap', True)))
    return jsonify({'sid': sid, **SESSIONS[sid].state()})


@app.post('/api/rank')
def rank():
    body = request.get_json()
    s = SESSIONS.get(body['sid'])
    if s is None:
        return jsonify({'error': 'session expired; start a new one'}), 404
    with s.lock:
        s.rank([int(i) for i in body['order']])
        return jsonify({'sid': body['sid'], **s.state()})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 5000)), threaded=True)
