"""
Monte Carlo belief over omega with resample-move rejuvenation.

`MonteCarloLinearReward` keeps a fixed set of hypothesis samples and only
reweights them.  Each ranking contributes several Luce-Shepard updates, so after
~10 queries the effective sample size (ESS) of a 10,000-sample belief is 1-4
even at d=8: the posterior mean is effectively one or two samples.

This subclass records every observation and, whenever the ESS falls below
`ess_threshold * N`, resamples in proportion to the weights and then applies
`n_moves` Metropolis-Hastings steps targeting prior x likelihood(history).  The
prior is uniform on the unit ball, matching the parent's initial samples, so
the MH chain leaves the posterior invariant and resampled duplicates spread
back out instead of collapsing.
"""

import numpy as np

from irlpreference.reward_parameterizations import MonteCarloLinearReward


class ResamplingMonteCarloLinearReward(MonteCarloLinearReward):
    def __init__(self, number_dimensions, number_samples=10_000, ess_threshold=0.5,
                 n_moves=5, rng=None):
        super().__init__(number_dimensions, number_samples=number_samples)
        self.ess_threshold = ess_threshold
        self.n_moves = n_moves
        self.rng = np.random.default_rng() if rng is None else rng
        self.history = []               # (items [m, d], chosen index, rationality)
        self.n_resamples = 0
        self.acceptance = []

    # ------------------------------------------------------------------
    def update(self, prob_input_given_omegas):
        # The choice model holds the observation being applied; record it so
        # moved samples can be re-scored against the full history.
        cm = prob_input_given_omegas.__self__
        self.history.append((np.array(cm.query, dtype=float), int(cm.choice),
                             float(cm.rationality)))
        super().update(prob_input_given_omegas)
        if self.ess() < self.ess_threshold * self.N:
            self._resample_move()

    def reset(self):
        super().reset()
        self.history = []

    def ess(self):
        pi = self._weights()
        return 1.0 / np.sum(pi ** 2)

    def _weights(self):
        lw = self.hypothesis_log_probabilities
        pi = np.exp(lw - lw.max())
        return pi / pi.sum()

    # ------------------------------------------------------------------
    def _log_likelihood(self, omegas):
        ll = np.zeros(len(omegas))
        for items, choice, beta in self.history:
            r = beta * (omegas @ items.T)                    # (S, m)
            m = r.max(axis=1, keepdims=True)
            ll += r[:, choice] - (m[:, 0] + np.log(np.exp(r - m).sum(axis=1)))
        return ll

    def _resample_move(self):
        pi = self._weights()
        idx = self.rng.choice(self.N, size=self.N, replace=True, p=pi)
        w = self.hypothesis_samples[idx].copy()
        ll = self._log_likelihood(w)

        # Random-walk proposal scaled to the current posterior spread.
        cov = np.cov(w.T) + 1e-9 * np.eye(self.D)
        chol = np.linalg.cholesky(cov) * (2.38 / np.sqrt(self.D))
        accepted = 0
        for _ in range(self.n_moves):
            prop = w + self.rng.standard_normal(w.shape) @ chol.T
            inside = np.linalg.norm(prop, axis=1) <= 1.0
            ll_prop = np.full(self.N, -np.inf)
            ll_prop[inside] = self._log_likelihood(prop[inside])
            accept = np.log(self.rng.random(self.N)) < ll_prop - ll
            w[accept] = prop[accept]
            ll[accept] = ll_prop[accept]
            accepted += accept.sum()

        self.hypothesis_samples = w
        self.hypothesis_log_probabilities = np.zeros(self.N)
        self.n_resamples += 1
        self.acceptance.append(accepted / (self.n_moves * self.N))
