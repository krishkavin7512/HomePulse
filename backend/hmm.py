"""Gaussian hidden Markov model, written out in NumPy.

Includes the three classic jobs (evaluation, decoding, learning):
  * scaled forward-backward, plus unscaled and log-domain forward for comparison
  * Viterbi in the log domain and posterior (minute-by-minute) decoding
  * Baum-Welch (EM) over many independent day-long sequences
"""
import numpy as np


def log_gauss(x, mu, var):
    """log N(x | mu, var) for every minute and state. x: (..., T); returns (..., T, K)."""
    return -0.5 * (np.log(2 * np.pi * var) + (x[..., None] - mu) ** 2 / var)


def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else out.item()


def forward_scaled(pi, A, b):
    """Scaled forward pass on one sequence. b: T x K likelihoods (any common scale per minute)."""
    T, K = b.shape
    alpha, c = np.zeros((T, K)), np.zeros(T)
    a = pi * b[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, T):
        a = (alpha[t - 1] @ A) * b[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    return alpha, c


def backward_scaled(A, b, c):
    T, K = b.shape
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (b[t + 1] * beta[t + 1])) / c[t + 1]
    return beta


def forward_unscaled(pi, A, b):
    """The textbook recursion with no scaling. Returns log10 of the summed alpha per minute.
    The running product leaves the range of a float: towards 0 when densities are below 1,
    towards infinity when narrow Gaussians make them above 1. Stops at the first such minute."""
    a = pi * b[0]
    out = [a.sum()]
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        for t in range(1, len(b)):
            a = (a @ A) * b[t]
            s = a.sum()
            out.append(s)
            if not np.isfinite(s) or s == 0:
                break
        return np.log10(np.array(out))


def forward_log(pi, A, logb):
    """The same recursion in the log domain with log-sum-exp. Returns log p(x)."""
    la = np.log(pi + 1e-300) + logb[0]
    lA = np.log(A + 1e-300)
    for t in range(1, len(logb)):
        la = logsumexp(la[:, None] + lA, axis=0) + logb[t]
    return float(logsumexp(la))


def viterbi(pi, A, logb):
    """Most likely whole path, in the log domain. Returns (path, log joint probability)."""
    T, K = logb.shape
    lA = np.log(A + 1e-300)
    delta = np.log(pi + 1e-300) + logb[0]
    back = np.zeros((T, K), dtype=np.int32)
    for t in range(1, T):
        scores = delta[:, None] + lA
        back[t] = scores.argmax(0)
        delta = scores.max(0) + logb[t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = delta.argmax()
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path, float(delta.max())


class GaussianHMM:
    def __init__(self, n_states, n_iter=40, tol=1e-4, seed=0, var_floor=0.03 ** 2):
        self.K, self.n_iter, self.tol, self.seed, self.var_floor = n_states, n_iter, tol, seed, var_floor

    # ---- initial guess: 1-D k-means on the power readings
    def _init(self, X):
        K = self.K
        rng = np.random.default_rng(self.seed)
        sample = rng.choice(X.ravel(), min(200_000, X.size), replace=False)
        mu = np.quantile(sample, (np.arange(K) + 0.5) / K)
        for _ in range(30):
            lab = np.abs(sample[:, None] - mu).argmin(1)
            mu = np.array([sample[lab == k].mean() if (lab == k).any() else mu[k] for k in range(K)])
        self.mu = np.sort(mu)
        self.var = np.full(K, max(np.var(sample) / K, self.var_floor))
        self.A = np.full((K, K), 0.05 / max(K - 1, 1)) + np.eye(K) * (0.95 - 0.05 / max(K - 1, 1))
        self.pi = np.full(K, 1 / K)

    def _b(self, X):
        lb = log_gauss(X, self.mu, self.var)
        m = lb.max(-1, keepdims=True)
        return np.exp(lb - m), m[..., 0]

    def fit(self, X):
        """Baum-Welch over days x minutes. Records log-likelihood at each iteration."""
        self._init(X)
        self.history = []
        for _ in range(self.n_iter):
            ll, stats = self._e_step(X)
            self.history.append(ll)
            self._m_step(stats)
            if len(self.history) > 1 and abs(self.history[-1] - self.history[-2]) < self.tol * abs(self.history[-1]) * 1e-3:
                break
        return self

    def _e_step(self, X):
        D, T = X.shape
        K = self.K
        b, m = self._b(X)
        A = self.A
        al = np.zeros((D, T, K))
        c = np.zeros((D, T))
        a = self.pi * b[:, 0]
        c[:, 0] = a.sum(1)
        al[:, 0] = a / c[:, [0]]
        for t in range(1, T):
            a = (al[:, t - 1] @ A) * b[:, t]
            c[:, t] = a.sum(1)
            al[:, t] = a / c[:, [t]]
        ll = float(np.log(c).sum() + m.sum())
        be = np.ones((D, K))
        g0 = np.zeros(K)
        gs, sx, sxx, xi = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros((K, K))
        for t in range(T - 1, -1, -1):
            g = al[:, t] * be
            g /= g.sum(1, keepdims=True)
            gs += g.sum(0)
            sx += (g * X[:, [t]]).sum(0)
            sxx += (g * X[:, [t]] ** 2).sum(0)
            if t == 0:
                g0 = g.sum(0)
            else:
                n = b[:, t] * be
                xi += (al[:, t - 1, :, None] * A * n[:, None, :] / c[:, t, None, None]).sum(0)
                be = (A @ n[..., None])[..., 0] / c[:, [t]]
        return ll, dict(g0=g0, gs=gs, sx=sx, sxx=sxx, xi=xi, D=D)

    def _m_step(self, s):
        self.pi = (s["g0"] + 1e-6) / (s["g0"] + 1e-6).sum()
        self.A = (s["xi"] + 1e-6) / (s["xi"] + 1e-6).sum(1, keepdims=True)
        self.mu = s["sx"] / (s["gs"] + 1e-9)
        self.var = np.maximum(s["sxx"] / (s["gs"] + 1e-9) - self.mu ** 2, self.var_floor)

    def loglik(self, X):
        b, m = self._b(X)
        a = self.pi * b[:, 0]
        c = a.sum(1)
        ll = np.log(c).sum()
        a = a / c[:, None]
        for t in range(1, X.shape[1]):
            a = (a @ self.A) * b[:, t]
            c = a.sum(1)
            ll += np.log(c).sum()
            a = a / c[:, None]
        return float(ll + m.sum())

    def posterior(self, x):
        """gamma for one day: P(state at t | whole day)."""
        b, _ = self._b(x)
        alpha, c = forward_scaled(self.pi, self.A, b)
        beta = backward_scaled(self.A, b, c)
        g = alpha * beta
        return g / g.sum(1, keepdims=True)

    def filtered(self, x):
        """P(state at t | readings up to t): what a live system could know."""
        b, _ = self._b(x)
        return forward_scaled(self.pi, self.A, b)[0]

    def viterbi(self, x):
        return viterbi(self.pi, self.A, log_gauss(x, self.mu, self.var))

    def predict_next(self, x):
        """Predictive mean and variance of the next reading given everything so far."""
        p = self.filtered(x)[-1] @ self.A
        mean = float(p @ self.mu)
        return mean, float(p @ (self.var + self.mu ** 2) - mean ** 2)

    def n_params(self):
        K = self.K
        return (K - 1) + K * (K - 1) + 2 * K

    def bic(self, X):
        return -2 * self.loglik(X) + self.n_params() * np.log(X.size)
