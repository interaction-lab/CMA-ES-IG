"""
Informative + Revealing question generation.

Implementation of the algorithm from:

    Soheil Habibian, Ananth Jonnavittula, and Dylan P. Losey.
    "Here's What I've Learned: Asking Questions that Reveal Reward Learning."
    ACM Transactions on Human-Robot Interaction 11(4), Article 40, 2022.
    https://doi.org/10.1145/3526107   (rebuttal/3526107.pdf)

The generators here expose the same interface as the CMA-ES generators in
``experiments/cmaes_generators.py`` -- ``get_query``, ``tell``, ``reset`` -- so
they drop straight into ``experiments/simluate_preferences.py`` as another entry
in the ``generators`` list.

Mapping the paper onto this codebase
------------------------------------
This codebase represents a trajectory directly by its feature embedding, so the
paper's feature map is the identity, ``f(xi) = xi``, and reward is
``R(xi, theta) = theta . xi`` (Eq. 1).  The robot's belief ``b_R`` is the
``MonteCarloLinearReward`` object owned by the caller; like
``InfoGainQueryGenerator``, these generators read that belief but never update
it.

    Eq. 4      xi_theta = argmax_xi theta . xi         ``_optimal_trajectories``
    Eq. 5-7    z* = (mu_i, sigma_i) in R^2d            ``_belief_embedding``
    Eq. 12     (mu_Q, sigma_Q) over the shown xi       ``_question_embedding``
    Eq. 11     P(Q|z) prop. exp{-||(mu_Q,sigma_Q)-z||^2}
    Eq. 10     b_H(z) prop. P(z) * prod_{k recent} P(Q_tau|z)  ``_human_belief``
    Eq. 13     Q_reveal    = argmax_Q b_H(z*)          ``lam = inf``
    Eq. 14     Q_info      = argmax_Q I(theta*; q|Q)   ``lam = 0``
    Eq. 15     Q_spectrum  = argmax_Q I + lam * b_H(z*)

Information gain is computed with the identical expression used by
``CMAESIGGenerator`` and ``InfoGainQueryGenerator`` (best entropy minus expected
human entropy), so the two families of methods are scored on the same footing.

Two modelling choices the paper leaves to the implementer are made explicit and
configurable here: the discrete question set ``Q`` (Sec. 3, footnote 1) is drawn
by random sampling each round, and the human's prior ``P(z)`` (Eq. 10) is built
by kernel-reweighting the belief's own hypothesis samples so that it spans
confident beliefs (small sigma) through ignorant ones (large sigma).  See
``_sample_z_prior``.
"""

import numpy as np


class InformativeRevealingQueryGenerator:
    """
    Questions that trade off informing the robot against revealing the robot's
    learning to the human (Eq. 15).

    Args:
        dim: dimensionality of the trajectory embedding.
        limits: list of ``(low, high)`` per dimension, matching the convention
            used by the CMA-ES generators.
        lam: ``lambda >= 0`` from Eq. 15.  ``0`` recovers purely Informative
            questions (Eq. 14); ``np.inf`` recovers purely Revealing questions
            (Eq. 13).  The paper recommends starting at ``1``.
        memory: ``k`` from Eq. 9/10, the number of recent questions the human is
            assumed to remember.  The paper recommends ``k >= 3``.
        number_candidate_questions: size of the sampled question set ``Q``
            searched at each round.  Plays the same role as the 100 CMA-ES
            candidates drawn by ``CMAESIGGenerator.get_query``.
        trajectory_samples: optional ``(T, dim)`` array restricting trajectories
            to a discrete set, mirroring ``DiscreteCMAESIGGenerator``.  When
            ``None`` the trajectory space is the box given by ``limits``.
        number_z_samples: number of samples used to represent the human's prior
            over ``z``.  ``b_H(z*)`` is normalised over these plus ``z*`` itself.
        rng: optional ``np.random.Generator`` for reproducibility.

    Note on ``lam``: Eq. 15 adds a mutual information term (in bits, as computed
    by ``irlpreference``) to a probability in ``[0, 1]``.  The two are not on a
    common scale, and in this codebase's regime the mismatch is not a constant
    offset -- it grows.  Measured at ``dim=8`` with 4 items per query, the
    spread of the information gain term across candidate questions stays around
    ``1.8`` throughout a session, while the spread of ``b_H(z*)`` decays from
    ``4e-3`` at the first question to ``3e-6`` by the eighth: as the belief
    sharpens, ``z*`` acquires a small sigma that no randomly drawn question
    matches, so the softmax over the z prior puts vanishing mass on ``z*`` for
    every candidate.  With the paper's recommended ``lam=1`` the revealing term
    is therefore swamped and the generator degenerates into pure Informative,
    which is the opposite of Proposition 1.

    Setting ``score_log_belief=True`` adds ``lam * log b_H(z*)`` instead.  That
    is no longer literally Eq. 15, but it is scale stable and recovers the
    spectrum the paper describes.  Over 10 seeds at ``dim=8``, mean Human Error
    (distance between the question's ``(mu_Q, sigma_Q)`` and the robot's ``z*``)
    falls monotonically with ``lam`` while regret stays competitive with
    Informative:

        Informative (lam=0)     regret 0.772   human error 2.442
        Eq. 15 linear, lam=1    regret 0.633   human error 2.429   (degenerate)
        Eq. 15 log, lam=0.2     regret 0.269   human error 2.133
        Eq. 15 log, lam=1.0     regret 0.302   human error 1.956
        Revealing (lam=inf)     regret 0.496   human error 1.908

    The default is left at the paper's literal formulation.  For a like-for-like
    baseline comparison, prefer ``score_log_belief=True`` and tune ``lam``, or
    report the linear form with a ``lam`` large enough to matter (``1e3`` at
    this dimensionality).
    """

    def __init__(self,
                 dim,
                 limits,
                 lam=1.0,
                 memory=3,
                 number_candidate_questions=100,
                 trajectory_samples=None,
                 number_z_samples=100,
                 score_log_belief=False,
                 rng=None):
        self.dimension = dim
        self.limits = limits
        self.lower_bounds = np.array([b[0] for b in limits], dtype=float)
        self.upper_bounds = np.array([b[1] for b in limits], dtype=float)
        self.lam = lam
        self.memory = memory
        self.number_candidate_questions = number_candidate_questions
        self.trajectory_samples = (None if trajectory_samples is None
                                   else np.asarray(trajectory_samples, dtype=float))
        self.number_z_samples = number_z_samples
        self.score_log_belief = score_log_belief
        self.rng = np.random.default_rng() if rng is None else rng

        # (mu_Q, sigma_Q) for each question asked so far; Eq. 10 reads the tail.
        self.question_history = []
        self._z_prior = None

    # ------------------------------------------------------------------
    # robot perspective
    # ------------------------------------------------------------------

    def _info_gain(self, reward_parameterization, input_model, query):
        '''
        Eq. 14.  Identical to CMAESIGGenerator._info_gain so that the two
        methods are scored with the same estimator.
        '''
        return reward_parameterization.get_best_entropy(query, input_model) - \
                reward_parameterization.get_human_entropy(query, input_model)

    # ------------------------------------------------------------------
    # human perspective
    # ------------------------------------------------------------------

    @staticmethod
    def _posterior_weights(reward_parameterization):
        '''
        Normalised weights of the belief's hypothesis samples, computed in log
        space (``get_expectation`` exponentiates directly and can overflow once
        the belief sharpens).
        '''
        log_p = reward_parameterization.hypothesis_log_probabilities
        log_p = log_p - np.max(log_p)
        w = np.exp(log_p)
        return w / np.sum(w)

    def _optimal_trajectories(self, thetas):
        '''
        Eq. 4: xi_theta = argmax_xi R(xi, theta) for each row of ``thetas``.

        Over a box the maximiser is the corner picked out by the sign of theta;
        over a discrete trajectory set it is a nearest-neighbour style argmax,
        chunked to bound peak memory.
        '''
        thetas = np.atleast_2d(thetas)

        if self.trajectory_samples is None:
            return np.where(thetas > 0, self.upper_bounds, self.lower_bounds)

        best = np.empty((len(thetas), self.dimension))
        chunk = max(1, int(5e6 // max(1, len(self.trajectory_samples))))
        for start in range(0, len(thetas), chunk):
            block = thetas[start:start + chunk]
            rewards = block @ self.trajectory_samples.T
            best[start:start + chunk] = self.trajectory_samples[np.argmax(rewards, axis=1)]
        return best

    def _belief_embedding(self, reward_parameterization):
        '''
        Eqs. 5-7: z* = (mu_i, sigma_i), the mean and standard deviation of the
        features of the trajectory the robot would deploy, taken over its
        current belief about theta.
        '''
        thetas = reward_parameterization.hypothesis_samples
        weights = self._posterior_weights(reward_parameterization)

        optimal = self._optimal_trajectories(thetas)
        mu = np.average(optimal, axis=0, weights=weights)
        var = np.average((optimal - mu) ** 2, axis=0, weights=weights)
        return np.concatenate([mu, np.sqrt(var)])

    @staticmethod
    def _question_embedding(questions):
        '''
        Eq. 12: (mu_Q, sigma_Q), the mean and standard deviation of the features
        across the trajectories shown in each question.

        ``questions`` is ``(C, N, dim)`` for C candidate questions of N
        trajectories each; returns ``(C, 2 * dim)``.
        '''
        questions = np.asarray(questions, dtype=float)
        if questions.ndim == 2:
            questions = questions[None, ...]
        mu = questions.mean(axis=1)
        sigma = questions.std(axis=1)
        return np.concatenate([mu, sigma], axis=1)

    def _sample_z_prior(self, reward_parameterization):
        '''
        P(z) from Eq. 10 -- the human's prior over what the robot might have
        learned.  The paper does not commit to a form, so we build one that
        covers the reachable range of belief states: reweight the hypothesis
        samples with a Gaussian kernel of random centre and random bandwidth,
        then read off the induced (mu, sigma).  Narrow bandwidths give confident
        beliefs (sigma near zero), wide ones give near-ignorant beliefs.

        Cached, since it depends only on the fixed hypothesis samples.
        '''
        if self._z_prior is not None:
            return self._z_prior

        thetas = reward_parameterization.hypothesis_samples
        optimal = self._optimal_trajectories(thetas)

        centres = thetas[self.rng.integers(0, len(thetas), size=self.number_z_samples)]
        bandwidths = np.exp(self.rng.uniform(np.log(0.05), np.log(5.0),
                                             size=self.number_z_samples))

        z = np.empty((self.number_z_samples, 2 * self.dimension))
        for i, (centre, h) in enumerate(zip(centres, bandwidths)):
            log_w = -np.sum((thetas - centre) ** 2, axis=1) / (2 * h ** 2)
            w = np.exp(log_w - log_w.max())
            w /= w.sum()
            mu = np.average(optimal, axis=0, weights=w)
            var = np.average((optimal - mu) ** 2, axis=0, weights=w)
            z[i] = np.concatenate([mu, np.sqrt(var)])

        self._z_prior = z
        return z

    def _human_belief(self, candidate_embeddings, z_star, reward_parameterization):
        '''
        Eqs. 10-11.  Returns b_H(z*) for each candidate question, i.e. how
        strongly a human who saw the last k-1 questions plus this candidate
        would believe the robot has learned z*.

        The belief is normalised over the sampled z prior together with z*
        itself; without that normalisation the argmax would reduce to the
        nearest-embedding rule and the memory k would have no effect.
        '''
        z_prior = self._sample_z_prior(reward_parameterization)
        z_all = np.vstack([z_prior, z_star])          # (M + 1, 2d), z* last

        # log P(Q_tau | z) summed over the k-1 remembered questions (Eq. 10).
        history_log = np.zeros(len(z_all))
        for embedding in self.question_history[-(self.memory - 1):] if self.memory > 1 else []:
            history_log -= np.sum((z_all - embedding) ** 2, axis=1)

        # log P(Q_candidate | z) for every candidate against every z.
        candidate_log = -np.sum(
            (candidate_embeddings[:, None, :] - z_all[None, :, :]) ** 2, axis=2)

        logits = candidate_log + history_log[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        posterior = np.exp(logits)
        posterior /= posterior.sum(axis=1, keepdims=True)

        return posterior[:, -1]

    # ------------------------------------------------------------------
    # generator interface, matching experiments/cmaes_generators.py
    # ------------------------------------------------------------------

    def _sample_candidate_questions(self, number_queries):
        '''
        The discrete question set Q.  The paper pre-generates it offline by
        randomly pairing perturbed trajectories; we resample it each round,
        which keeps the search budget comparable to the 100 candidates drawn by
        CMAESIGGenerator.
        '''
        shape = (self.number_candidate_questions, number_queries, self.dimension)

        if self.trajectory_samples is not None:
            idx = self.rng.integers(0, len(self.trajectory_samples),
                                    size=(self.number_candidate_questions, number_queries))
            return self.trajectory_samples[idx]

        return self.rng.uniform(self.lower_bounds, self.upper_bounds, size=shape)

    def get_query(self, number_queries, reward_parameterization=None, input_model=None):
        '''
        Eq. 15: pick the candidate question maximising
        ``I(theta*; q | Q, b_R) + lam * b_H(z*)``.

        Returns a ``(number_queries, dim)`` array, as the CMA-ES generators do.
        '''
        candidates = self._sample_candidate_questions(number_queries)

        # Revealing-only and informative-only shortcuts avoid the cost of the
        # term that is about to be multiplied by zero.
        need_info = self.lam != np.inf and reward_parameterization is not None
        need_reveal = self.lam > 0 and reward_parameterization is not None

        score = np.zeros(len(candidates))

        if need_info:
            score += np.array([self._info_gain(reward_parameterization, input_model, q)
                               for q in candidates])

        if need_reveal:
            z_star = self._belief_embedding(reward_parameterization)
            embeddings = self._question_embedding(candidates)
            belief = self._human_belief(embeddings, z_star, reward_parameterization)
            if self.lam == np.inf:
                score = belief                                   # Eq. 13
            elif self.score_log_belief:
                score += self.lam * np.log(np.maximum(belief, 1e-300))
            else:
                score += self.lam * belief                       # Eq. 15

        query = candidates[int(np.argmax(score))]

        # Remember what was actually asked, for the human's bounded memory.
        self.question_history.append(self._question_embedding(query)[0])

        return np.array(query)

    def tell(self, solutions, rankings):
        '''
        Present for interface parity with the CMA-ES generators.  This method
        keeps no optimiser state of its own: the robot's belief b_R lives in the
        MonteCarloLinearReward object the caller updates, exactly as with
        InfoGainQueryGenerator.
        '''
        return None

    def reset(self):
        '''
        '''
        self.question_history = []
        self._z_prior = None


class RevealingQueryGenerator(InformativeRevealingQueryGenerator):
    '''
    Eq. 13, the purely Revealing extreme of the spectrum.  Used as a baseline in
    Sec. 7 of the paper.
    '''
    def __init__(self, dim, limits, **kwargs):
        kwargs.pop('lam', None)
        super().__init__(dim, limits, lam=np.inf, **kwargs)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', 'experiments',
                                    'preference-learning-from-selection'))

    from irlpreference.input_models import LuceShepardChoice
    from irlpreference.reward_parameterizations import MonteCarloLinearReward

    dim_embedding = 8
    items_per_query = 4
    number_of_queries = 15

    rng = np.random.default_rng(0)
    user_choice_model = LuceShepardChoice(rationality=10)
    user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=5_000)

    generators = {
        'Informative (lam=0)': InformativeRevealingQueryGenerator(
            dim_embedding, [(-1, 1)] * dim_embedding, lam=0.0, rng=rng),
        'Ours (lam=1)': InformativeRevealingQueryGenerator(
            dim_embedding, [(-1, 1)] * dim_embedding, lam=1.0, rng=rng),
        'Revealing': RevealingQueryGenerator(
            dim_embedding, [(-1, 1)] * dim_embedding, rng=rng),
    }

    all_trajectories = rng.standard_normal((5000, dim_embedding))
    true_preference = rng.uniform(-1, 1, size=dim_embedding)

    for name, generator in generators.items():
        user_estimate.reset()
        generator.reset()

        for _ in range(number_of_queries):
            query = generator.get_query(items_per_query, user_estimate, user_choice_model)
            probabilities = user_choice_model.get_choice_probabilities(
                query, np.array([true_preference])).flatten()
            choice = rng.choice(np.arange(items_per_query), p=probabilities)
            user_choice_model.tell_input(choice, query)
            user_estimate.update(user_choice_model.get_probability_of_input)

        estimated_omega = user_estimate.get_expectation()
        best = all_trajectories[np.argmax(all_trajectories @ estimated_omega)]
        regret = np.max(all_trajectories @ true_preference) - best @ true_preference
        alignment = (estimated_omega @ true_preference /
                     (np.linalg.norm(estimated_omega) * np.linalg.norm(true_preference)))

        print(f"{name:22s} regret={regret:7.4f}   alignment={alignment:6.3f}")
