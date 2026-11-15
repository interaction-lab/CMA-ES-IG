"""
Ablation over how CMA-ES-IG approximates information gain when picking which of
its sampled candidates to show the user.

The deployed CMA-ES-IG (experiments/cmaes_generators.py) draws a pool of CMA-ES
samples and returns k-means cluster centres.  This module keeps the CMA-ES
machinery identical and swaps only the selection rule:

  'kmeans'    k-means cluster centres over the sampled pool  (the deployed rule)
  'kmedoids'  k-medoids (PAM, via scikit-learn-extra) -- like k-means but the
              returned items are actual sampled points rather than synthetic
              centroids
  'exact'     exhaustive maximisation of information gain over every size-N
              subset of the sampled pool
  'ellipsoid' continuous maximisation of information gain over the ellipsoid
              containing `ellipsoid_mass` of the CMA-ES sampling distribution,
              using finite-difference gradients
  'ellipsoid_grad'  the same search driven by the analytic information-gain
              gradient.  Finite differencing costs N*d+1 objective evaluations
              per gradient (129 at d=32); the analytic gradient costs one extra
              matrix product, so this is the same search roughly two orders of
              magnitude cheaper.  `omega_subsample` further trades estimator
              noise for speed by scoring against a fixed random subset of the
              hypothesis samples.

Note on 'exact': the deployed pool of 100 samples gives C(100,4) = 3,921,225
subsets per query, which is not tractable.  Runs therefore use a smaller pool
(`pool_size`), and k-means should be run at the *same* pool size as a control so
that the selection rule and the pool size are separable.

Information gain uses the same estimator as CMAESIGGenerator and
InfoGainQueryGenerator -- best entropy minus expected human entropy -- so the
numbers are directly comparable.  `_batch_info_gain` is a vectorised form of
that estimator; `_verify_batch_info_gain` checks it against the library.
"""

import itertools
import warnings

import numpy as np
from cmaes import CMA
from scipy.stats import chi2
import scipy.optimize as opt
from sklearn.cluster import KMeans

with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from sklearn_extra.cluster import KMedoids

METHODS = ('kmeans', 'kmedoids', 'exact', 'ellipsoid', 'ellipsoid_grad')


class CMAESIGAblation:
    def __init__(self, dim, limits, population_size=10, sigma=1.3,
                 method='kmeans', pool_size=100, ellipsoid_mass=0.9,
                 ellipsoid_restarts=1, eval_budget=None,
                 omega_subsample=None, grad_maxiter=40, rng=None,
                 ig_objective='legacy'):
        if method not in METHODS:
            raise ValueError(f'method must be one of {METHODS}, got {method!r}')
        self.optimizer = CMA(mean=np.zeros(dim), sigma=sigma, population_size=population_size)
        self.dimension = dim
        self.limits = limits
        self.lower_bounds = np.array([b[0] for b in limits], dtype=float)
        self.upper_bounds = np.array([b[1] for b in limits], dtype=float)
        self.population_size = population_size
        self.sigma = sigma
        self.method = method
        self.pool_size = pool_size
        self.ellipsoid_mass = ellipsoid_mass
        self.ellipsoid_restarts = ellipsoid_restarts
        # Number of information-gain evaluations the ellipsoid search may spend.
        # Defaults to C(pool_size, 4)-scale so that 'ellipsoid' and 'exact' are
        # compared at equal compute rather than equal wall-clock.
        self.eval_budget = eval_budget
        self.omega_subsample = omega_subsample
        self.grad_maxiter = grad_maxiter
        self.rng = np.random.default_rng() if rng is None else rng
        # 'legacy' reproduces the library estimator (entropy under the posterior
        # mean minus the *unweighted* average entropy over the hypothesis
        # samples).  'mi' is the mutual information between the user's first
        # choice and omega under the current posterior weights.
        if ig_objective not in ('legacy', 'mi'):
            raise ValueError(f'ig_objective must be legacy or mi, got {ig_objective!r}')
        self.ig_objective = ig_objective

    # ------------------------------------------------------------------
    # information gain
    # ------------------------------------------------------------------

    def _info_gain(self, reward_parameterization, input_model, query):
        '''
        Matches CMAESIGGenerator._info_gain exactly.
        '''
        if self.ig_objective == 'mi':
            return float(self._batch_mutual_information(
                np.asarray(query), np.arange(len(query))[None], reward_parameterization,
                input_model)[0])
        return reward_parameterization.get_best_entropy(query, input_model) - \
                reward_parameterization.get_human_entropy(query, input_model)

    @staticmethod
    def _entropy_from_rewards(scaled_rewards, axis):
        '''
        Shannon entropy (bits) of the Luce-Shepard choice distribution implied by
        `scaled_rewards` (already multiplied by the rationality), reduced along
        the option axis.  Computed in a shift-invariant way for stability.
        '''
        shifted = scaled_rewards - np.max(scaled_rewards, axis=axis, keepdims=True)
        expd = np.exp(shifted)
        p = expd / np.sum(expd, axis=axis, keepdims=True)
        tiny = np.finfo(p.dtype).tiny
        return np.sum(-p * np.log2(np.clip(p, tiny, None)), axis=axis)

    def _batch_info_gain(self, pool, subsets, reward_parameterization, input_model,
                         chunk=None):
        if self.ig_objective == 'mi':
            return self._batch_mutual_information(pool, subsets, reward_parameterization,
                                                  input_model, chunk)
        return self._batch_info_gain_legacy(pool, subsets, reward_parameterization,
                                            input_model, chunk)

    def _batch_mutual_information(self, pool, subsets, reward_parameterization,
                                  input_model, chunk=None):
        '''
        I(choice; omega) = H(sum_s pi_s p(.|omega_s)) - sum_s pi_s H(p(.|omega_s)),
        with pi the normalised posterior weights over the hypothesis samples.
        Samples whose weight is below 1e-10 of the largest are dropped; their
        total contribution is negligible and after a few updates this removes
        most of the set.
        '''
        lw = reward_parameterization.hypothesis_log_probabilities
        pi = np.exp(lw - lw.max())
        keep = pi > 1e-10
        omegas = reward_parameterization.hypothesis_samples[keep]
        pi = (pi[keep] / pi[keep].sum()).astype(np.float32)
        rationality = input_model.rationality

        pool_rewards = (rationality * (pool @ omegas.T)).astype(np.float32)   # (P, S)
        n_samples = len(omegas)
        if chunk is None:
            chunk = max(1, int(2 ** 21 // max(1, subsets.shape[1] * n_samples)))
        out = np.empty(len(subsets))
        tiny = np.finfo(np.float32).tiny
        for start in range(0, len(subsets), chunk):
            r = pool_rewards[subsets[start:start + chunk]]                    # (B, N, S)
            r = r - r.max(axis=1, keepdims=True)
            p = np.exp(r)
            p /= p.sum(axis=1, keepdims=True)
            cond = np.sum(-p * np.log2(np.clip(p, tiny, None)), axis=1) @ pi  # (B,)
            pred = p @ pi                                                     # (B, N)
            marg = np.sum(-pred * np.log2(np.clip(pred, tiny, None)), axis=1)
            out[start:start + chunk] = marg - cond
        return out

    def _batch_info_gain_legacy(self, pool, subsets, reward_parameterization, input_model,
                                chunk=None):
        '''
        Information gain for many candidate queries at once.

        Rewards against the hypothesis samples are computed once for the whole
        pool and then gathered per subset, which is what makes exhaustive search
        affordable: the per-subset cost is a softmax over N options rather than
        a fresh matrix product.
        '''
        omegas = reward_parameterization.hypothesis_samples          # (S, d)
        rationality = input_model.rationality

        # float32 halves the bandwidth of the exp/log pass, which dominates the
        # cost here; the entropies are only compared against each other, so the
        # reduced precision does not affect which subset wins.
        pool_rewards = (rationality * (pool @ omegas.T)).astype(np.float32)
        best_rewards = (rationality * (pool @ reward_parameterization.get_expectation())
                        ).astype(np.float32)

        n_samples = len(omegas)
        if chunk is None:
            # keep each gathered block near a few MB so it stays in cache
            chunk = max(1, int(2 ** 21 // max(1, subsets.shape[1] * n_samples)))
        out = np.empty(len(subsets))
        for start in range(0, len(subsets), chunk):
            idx = subsets[start:start + chunk]                       # (B, N)
            # human entropy: average over hypothesis samples
            human = self._entropy_from_rewards(pool_rewards[idx], axis=1).sum(axis=1) / n_samples
            # best entropy: single point estimate of the preference
            best = self._entropy_from_rewards(best_rewards[idx], axis=1)
            out[start:start + chunk] = best - human
        return out

    def _verify_batch_info_gain(self, pool, subsets, reward_parameterization, input_model):
        '''
        Cross-check of the vectorised estimator against the library one.
        Returns the largest absolute discrepancy.
        '''
        fast = self._batch_info_gain(pool, subsets, reward_parameterization, input_model)
        slow = np.array([self._info_gain(reward_parameterization, input_model, pool[s])
                         for s in subsets])
        return np.max(np.abs(fast - slow))


    # ------------------------------------------------------------------
    # analytic information-gain gradient
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy_and_coeff(scaled_rewards):
        """
        Choice entropy (bits) and dH/dR for a Luce-Shepard softmax taken over
        axis 0 of `scaled_rewards` (rewards already multiplied by rationality).

        For H = -sum_i p_i log2 p_i with p = softmax(R),
            dH/dR_k = -p_k * (log2 p_k + H),
        the 1/ln2 terms cancelling.  Returns (H, dH/dR).
        """
        shifted = scaled_rewards - np.max(scaled_rewards, axis=0, keepdims=True)
        expd = np.exp(shifted)
        p = expd / np.sum(expd, axis=0, keepdims=True)
        logp = np.log2(np.clip(p, np.finfo(p.dtype).tiny, None))
        H = -np.sum(p * logp, axis=0)
        return H, -p * (logp + H)

    def _info_gain_and_grad(self, query, reward_parameterization, input_model,
                            omegas=None):
        """
        Information gain and its gradient with respect to `query` (N, d).

        Matches _info_gain in value; the gradient is exact for that estimator.
        """
        beta = input_model.rationality
        if omegas is None:
            omegas = reward_parameterization.hypothesis_samples
        n_samples = len(omegas)
        best_omega = reward_parameterization.get_expectation()

        H_s, dH_s = self._entropy_and_coeff(beta * (query @ omegas.T))   # (S,), (N,S)
        human = H_s.sum() / n_samples
        grad_human = (beta / n_samples) * (dH_s @ omegas)                # (N, d)

        H_b, dH_b = self._entropy_and_coeff(beta * (query @ best_omega)[:, None])
        best = float(H_b[0])
        grad_best = beta * dH_b * best_omega[None, :]                    # (N, d)

        return best - human, grad_best - grad_human

    def _ellipsoid_grad(self, number_queries, reward_parameterization, input_model):
        """
        Same feasible region as _ellipsoid, driven by the analytic gradient.
        """
        mean = self.optimizer._mean
        B, Dg = self.optimizer._eigen_decomposition()
        scale = self.optimizer._sigma * (B * Dg)
        radius = np.sqrt(chi2.ppf(self.ellipsoid_mass, df=self.dimension))

        omegas = reward_parameterization.hypothesis_samples
        if self.omega_subsample and self.omega_subsample < len(omegas):
            idx = self.rng.choice(len(omegas), size=self.omega_subsample, replace=False)
            omegas = omegas[idx]

        def unpack(w):
            w = w.reshape(number_queries, self.dimension)
            n = np.sqrt(1.0 + np.sum(w ** 2, axis=1, keepdims=True))
            u = radius * w / n
            raw = mean + u @ scale.T
            pts = np.clip(raw, self.lower_bounds, self.upper_bounds)
            return w, n, pts, raw

        def objective(wflat):
            w, n, pts, raw = unpack(wflat)
            val, g_pts = self._info_gain_and_grad(pts, reward_parameterization,
                                                  input_model, omegas)
            # clipping is inactive where the point is interior; zero it elsewhere
            g_pts = np.where((raw > self.lower_bounds) & (raw < self.upper_bounds),
                             g_pts, 0.0)
            g_u = g_pts @ scale                                   # chain through xi = m + scale u
            dot = np.sum(g_u * w, axis=1, keepdims=True)
            g_w = radius * (g_u / n - w * dot / n ** 3)           # chain through the ball map
            return -val, -g_w.ravel()

        best_pts, best_val = None, np.inf
        for _ in range(max(1, self.ellipsoid_restarts)):
            w0 = self.rng.normal(size=number_queries * self.dimension)
            res = opt.fmin_l_bfgs_b(objective, x0=w0, iprint=-1,
                                    maxiter=self.grad_maxiter, maxfun=self.grad_maxiter * 2)
            if res[1] < best_val:
                best_val = res[1]
                best_pts = unpack(res[0])[2]
        return best_pts

    # ------------------------------------------------------------------
    # selection rules
    # ------------------------------------------------------------------

    def _sample_pool(self, size):
        pool = [np.clip(self.optimizer.ask(), self.lower_bounds, self.upper_bounds)
                for _ in range(size)]
        return np.array(pool)

    def _kmeans(self, pool, number_queries):
        return KMeans(n_clusters=number_queries, n_init='auto').fit(pool).cluster_centers_

    def _kmedoids(self, pool, number_queries):
        '''
        k-medoids via PAM (scikit-learn-extra) with k-medoids++ initialisation.
        Unlike k-means the returned items are actual sampled points, so every
        trajectory shown to the user is a real CMA-ES candidate rather than a
        synthetic centroid that may fall outside the sampled manifold.
        '''
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            km = KMedoids(n_clusters=number_queries, method='pam',
                          init='k-medoids++',
                          random_state=int(self.rng.integers(2 ** 31))).fit(pool)
        return km.cluster_centers_

    def _exact(self, pool, number_queries, reward_parameterization, input_model):
        subsets = np.array(list(itertools.combinations(range(len(pool)), number_queries)))
        gains = self._batch_info_gain(pool, subsets, reward_parameterization, input_model)
        return pool[subsets[int(np.argmax(gains))]]

    def _ellipsoid(self, number_queries, reward_parameterization, input_model):
        '''
        Maximise information gain over {m + sigma * B diag(D) u : ||u|| <= R},
        the ellipsoid holding `ellipsoid_mass` of the CMA-ES sampling
        distribution.  The unconstrained parameterisation
        u = R * w / sqrt(1 + ||w||^2) maps R^d onto the open ball, so the
        constraint is satisfied by construction and plain L-BFGS applies.
        '''
        mean = self.optimizer._mean
        B, Dg = self.optimizer._eigen_decomposition()
        scale = self.optimizer._sigma * (B * Dg)                    # (d, d)
        radius = np.sqrt(chi2.ppf(self.ellipsoid_mass, df=self.dimension))

        def to_points(w):
            w = w.reshape(number_queries, self.dimension)
            norms = np.sqrt(1.0 + np.sum(w ** 2, axis=1, keepdims=True))
            u = radius * w / norms
            pts = mean + u @ scale.T
            return np.clip(pts, self.lower_bounds, self.upper_bounds)

        def objective(w):
            return -self._info_gain(reward_parameterization, input_model, to_points(w))

        budget = self.eval_budget
        if budget is None:
            from math import comb
            budget = comb(max(self.pool_size, number_queries + 1), number_queries)
        budget = max(1, budget // self.ellipsoid_restarts)

        best_pts, best_val = None, np.inf
        for _ in range(self.ellipsoid_restarts):
            w0 = self.rng.normal(size=number_queries * self.dimension)
            res = opt.fmin_l_bfgs_b(objective, x0=w0, approx_grad=True, iprint=-1,
                                    maxfun=budget, maxiter=budget)
            if res[1] < best_val:
                best_val, best_pts = res[1], to_points(res[0])
        return best_pts

    # ------------------------------------------------------------------
    # generator interface, matching experiments/cmaes_generators.py
    # ------------------------------------------------------------------

    def get_query(self, number_queries, reward_parameterization=None, input_model=None):
        '''
        '''
        if self.method == 'ellipsoid':
            return np.array(self._ellipsoid(number_queries, reward_parameterization,
                                            input_model))
        if self.method == 'ellipsoid_grad':
            return np.array(self._ellipsoid_grad(number_queries, reward_parameterization,
                                                 input_model))

        pool = self._sample_pool(self.pool_size)

        if self.method == 'kmeans':
            query = self._kmeans(pool, number_queries)
        elif self.method == 'kmedoids':
            query = self._kmedoids(pool, number_queries)
        else:
            query = self._exact(pool, number_queries, reward_parameterization, input_model)

        return np.array(query)

    def tell(self, solutions, rankings):
        '''
        '''
        self.optimizer.tell([(s, -rankings[i]) for i, s in enumerate(solutions)])

    def reset(self):
        '''
        '''
        self.optimizer = CMA(mean=np.zeros(self.dimension), sigma=self.sigma,
                             population_size=self.population_size)
