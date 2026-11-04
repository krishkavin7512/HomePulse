"""First-order Markov chain over discretised power levels, fitted by counting.

This is the baseline: the next reading depends only on the current one, and nothing is
hidden. To compare it fairly with Gaussian models, its likelihood is turned into a
density by dividing each bin's probability by the bin's width.
"""
import numpy as np


class MarkovChain:
    def __init__(self, n_bins=12, alpha=1.0):
        self.n_bins, self.alpha = n_bins, alpha

    def fit(self, X):
        """X: days x minutes. Bins are quantiles of the training readings."""
        edges = np.unique(np.quantile(X, np.linspace(0, 1, self.n_bins + 1)))
        edges[0], edges[-1] = 0.0, max(edges[-1], X.max()) + 1e-3
        self.edges = edges
        self.widths = np.diff(edges)
        K = len(edges) - 1
        s = self.discretise(X)
        C = np.zeros((K, K))
        np.add.at(C, (s[:, :-1].ravel(), s[:, 1:].ravel()), 1)
        self.counts = C
        self.A = (C + self.alpha) / (C + self.alpha).sum(1, keepdims=True)
        self.iid = (np.bincount(s.ravel(), minlength=K) + self.alpha) / (s.size + K * self.alpha)
        self.pi = self.stationary()
        return self

    def discretise(self, X):
        return np.clip(np.searchsorted(self.edges, X, side="right") - 1, 0, len(self.widths) - 1)

    def stationary(self, tol=1e-12):
        """Left eigenvector of A for eigenvalue 1."""
        w, v = np.linalg.eig(self.A.T)
        p = np.real(v[:, np.argmin(np.abs(w - 1))])
        return p / p.sum()

    def power_distance(self, ks):
        """Worst-case total-variation distance of each row of A^k from the stationary distribution."""
        return [float(0.5 * np.abs(np.linalg.matrix_power(self.A, k) - self.pi).sum(1).max()) for k in ks]

    def loglik(self, X):
        """Total log density of X under the chain."""
        s = self.discretise(X)
        lw = np.log(self.widths)
        ll = (np.log(self.pi[s[:, 0]]) - lw[s[:, 0]]).sum()
        return float(ll + (np.log(self.A[s[:, :-1], s[:, 1:]]) - lw[s[:, 1:]]).sum())

    def loglik_iid(self, X):
        """Same bins with the time order thrown away: what knowing the previous reading is worth."""
        s = self.discretise(X)
        return float((np.log(self.iid[s]) - np.log(self.widths)[s]).sum())

    def n_params(self):
        K = len(self.widths)
        return K * (K - 1)
