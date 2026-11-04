"""Factorial hidden Markov model: one observed wire, several hidden on/off devices.

Each hidden chain is a device that switches on and off by its own Markov chain. The meter
reports only the sum:

    total(t) = base + sum over chains of  wattage[m] * state[m](t) + noise

Nobody tells the model how many devices exist or what they draw. EM learns each chain's
wattage and switching rates from the total alone.

Inference is exact: the M binary chains are combined into one chain over 2^M combinations,
whose transition matrix is the Kronecker product of the per-device matrices.
"""
import itertools

import numpy as np

from .hmm import backward_scaled, forward_scaled, log_gauss, viterbi


def kmeans_1d(x, k, iters=60):
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        lab = np.abs(x[:, None] - c).argmin(1)
        c = np.array([x[lab == i].mean() if (lab == i).any() else c[i] for i in range(k)])
    return np.sort(c)


def upward_steps(X, min_step=0.15):
    """Sizes of the upward jumps in the total. A device switching on leaves one of these."""
    s = np.diff(X, axis=1).ravel()
    return s[s > min_step]


class FactorialHMM:
    def __init__(self, n_chains, n_iter=40, wattage_floor=0.02, var_floor=0.03 ** 2):
        self.M, self.n_iter, self.wattage_floor, self.var_floor = n_chains, n_iter, wattage_floor, var_floor
        self.S = np.array(list(itertools.product([0, 1], repeat=n_chains)), float)
        self.K = 2 ** n_chains

    # ---- structure
    def transition(self):
        A = np.ones((1, 1))
        for a01, a10 in zip(self.p_on, self.p_off):
            A = np.kron(A, np.array([[1 - a01, a01], [a10, 1 - a10]]))
        return A

    def start_probs(self):
        pi = np.ones(1)
        for a01, a10 in zip(self.p_on, self.p_off):
            on = a01 / (a01 + a10)
            pi = np.kron(pi, np.array([1 - on, on]))
        return pi

    def means(self):
        return self.base + self.S @ self.watts

    def _b(self, X):
        lb = log_gauss(X, self.means(), self.var)
        m = lb.max(-1, keepdims=True)
        return np.exp(lb - m), m[..., 0]

    # ---- learning
    def init(self, X, watts=None):
        """Label-free start: k-means on the sizes of the upward jumps in the total."""
        self.watts = np.array(watts, float) if watts is not None else kmeans_1d(upward_steps(X), self.M)
        self.base = float(np.quantile(X, 0.05))
        self.p_on = np.full(self.M, 0.02)
        self.p_off = np.full(self.M, 0.10)
        self.var = np.full(self.K, 0.15 ** 2) + self.S.sum(1) * 0.02
        return self

    def fit(self, X, watts=None):
        self.init(X, watts)
        self.history = []
        for _ in range(self.n_iter):
            ll, stats = self._e_step(X)
            self.history.append(ll)
            self._m_step(stats)
        return self

    def _e_step(self, X):
        D, T = X.shape
        K = self.K
        A, pi = self.transition(), self.start_probs()
        b, m = self._b(X)
        al = np.zeros((D, T, K))
        c = np.zeros((D, T))
        a = pi * b[:, 0]
        c[:, 0] = a.sum(1)
        al[:, 0] = a / c[:, [0]]
        for t in range(1, T):
            a = (al[:, t - 1] @ A) * b[:, t]
            c[:, t] = a.sum(1)
            al[:, t] = a / c[:, [t]]
        ll = float(np.log(c).sum() + m.sum())
        be = np.ones((D, K))
        gs, sx, sxx, xi = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros((K, K))
        for t in range(T - 1, -1, -1):
            g = al[:, t] * be
            g /= g.sum(1, keepdims=True)
            gs += g.sum(0)
            sx += (g * X[:, [t]]).sum(0)
            sxx += (g * X[:, [t]] ** 2).sum(0)
            if t > 0:
                n = b[:, t] * be
                xi += (al[:, t - 1, :, None] * A * n[:, None, :] / c[:, t, None, None]).sum(0)
                be = (A @ n[..., None])[..., 0] / c[:, [t]]
        return ll, dict(gs=gs, sx=sx, sxx=sxx, xi=xi)

    def _m_step(self, s):
        S, xi, gs = self.S, s["xi"], s["gs"]
        # Switching rates: each device's expected on/off counts, summed out of the joint pairwise marginals.
        for k in range(self.M):
            off, on = S[:, k] == 0, S[:, k] == 1
            n00, n01 = xi[np.ix_(off, off)].sum(), xi[np.ix_(off, on)].sum()
            n10, n11 = xi[np.ix_(on, off)].sum(), xi[np.ix_(on, on)].sum()
            self.p_on[k] = (n01 + 1) / (n00 + n01 + 2)
            self.p_off[k] = (n10 + 1) / (n10 + n11 + 2)
        # Wattages: weighted least squares of each combination's mean on its device mix.
        # The weights are gamma / variance, which is what makes this a true M-step.
        D = np.c_[np.ones(self.K), S]
        W = np.diag((gs + 1e-6) / self.var)
        sol = np.linalg.solve(D.T @ W @ D + 1e-6 * np.eye(self.M + 1), D.T @ (s["sx"] / self.var))
        self.base = float(sol[0])
        self.watts = np.maximum(sol[1:], self.wattage_floor)
        mu = self.means()
        self.var = np.maximum((s["sxx"] - 2 * mu * s["sx"] + gs * mu ** 2) / (gs + 1e-6), self.var_floor)

    def sort_chains(self):
        """Renumber devices from smallest to largest. The model itself does not change."""
        order = np.argsort(self.watts)
        old_index = {tuple(s): i for i, s in enumerate(self.S)}
        self.var = np.array([self.var[old_index[tuple(np.array(s)[np.argsort(order)])]] for s in self.S])
        self.watts, self.p_on, self.p_off = self.watts[order], self.p_on[order], self.p_off[order]
        return self

    # ---- inference
    def loglik(self, X):
        A, pi = self.transition(), self.start_probs()
        b, m = self._b(X)
        a = pi * b[:, 0]
        c = a.sum(1)
        ll = np.log(c).sum()
        a = a / c[:, None]
        for t in range(1, X.shape[1]):
            a = (a @ A) * b[:, t]
            c = a.sum(1)
            ll += np.log(c).sum()
            a = a / c[:, None]
        return float(ll + m.sum())

    def smooth(self, x):
        """P(device on at t | whole day), one column per device. x: one day."""
        return self.joint_posterior(x) @ self.S

    def joint_posterior(self, x):
        A, pi = self.transition(), self.start_probs()
        b, _ = self._b(x)
        alpha, c = forward_scaled(pi, A, b)
        g = alpha * backward_scaled(A, b, c)
        return g / g.sum(1, keepdims=True)

    def filtered(self, x):
        """P(device on at t | readings up to t): what a live monitor could know."""
        b, _ = self._b(x)
        return forward_scaled(self.start_probs(), self.transition(), b)[0] @ self.S

    def viterbi(self, x):
        """Most likely single story for the day, one 0/1 column per device."""
        path, _ = viterbi(self.start_probs(), self.transition(), log_gauss(x, self.means(), self.var))
        return self.S[path]

    def smooth_many(self, X):
        """Smoothed per-device probabilities for days x minutes, in one batch."""
        D, T = X.shape
        A, pi = self.transition(), self.start_probs()
        b, _ = self._b(X)
        al = np.zeros((D, T, self.K))
        c = np.zeros((D, T))
        a = pi * b[:, 0]
        c[:, 0] = a.sum(1)
        al[:, 0] = a / c[:, [0]]
        for t in range(1, T):
            a = (al[:, t - 1] @ A) * b[:, t]
            c[:, t] = a.sum(1)
            al[:, t] = a / c[:, [t]]
        out = np.zeros((D, T, self.M), dtype=np.float32)
        be = np.ones((D, self.K))
        for t in range(T - 1, -1, -1):
            g = al[:, t] * be
            g /= g.sum(1, keepdims=True)
            out[:, t] = g @ self.S
            if t > 0:
                be = (A @ (b[:, t] * be)[..., None])[..., 0] / c[:, [t]]
        return out

    def filter_many(self, X):
        D, T = X.shape
        A, pi = self.transition(), self.start_probs()
        b, _ = self._b(X)
        out = np.zeros((D, T, self.M), dtype=np.float32)
        a = pi * b[:, 0]
        a = a / a.sum(1, keepdims=True)
        out[:, 0] = a @ self.S
        for t in range(1, T):
            a = (a @ A) * b[:, t]
            a = a / a.sum(1, keepdims=True)
            out[:, t] = a @ self.S
        return out

    # ---- plain-language summary of a device
    def run_minutes(self):
        return 1 / self.p_off

    def gap_minutes(self):
        return 1 / self.p_on

    def duty(self):
        return self.p_on / (self.p_on + self.p_off)

    def n_params(self):
        return 2 * self.M + self.M + 1 + self.K

    def state_dict(self):
        return {"n_chains": self.M, "watts_kW": self.watts.tolist(), "base_kW": self.base, "p_on": self.p_on.tolist(),
                "p_off": self.p_off.tolist(), "var": self.var.tolist()}

    @classmethod
    def from_state(cls, st):
        f = cls(st["n_chains"])
        f.watts = np.array(st["watts_kW"])
        f.base = st["base_kW"]
        f.p_on = np.array(st["p_on"])
        f.p_off = np.array(st["p_off"])
        f.var = np.array(st["var"])
        return f
