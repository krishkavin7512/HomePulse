"""Linear dynamical system for the slow trend in daily consumption.

Hidden level:    z[t] = z[t-1] + process noise (variance q)
Observed value:  y[t] = z[t]   + reading noise (variance r)

A missing day (NaN) gets a predict step with no update, so the uncertainty band widens across
the gap on its own. q and r are learned by EM using the Rauch-Tung-Striebel smoother.
"""
import numpy as np


def kalman_filter(y, q, r, m0=None, p0=1.0):
    n = len(y)
    m0 = float(np.nanmean(y[:7])) if m0 is None else m0
    pred_m, pred_p, m, p, gain = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    mt, pt = m0, p0
    ll = 0.0
    for t in range(n):
        mp, pp = (mt, pt) if t == 0 else (mt, pt + q)     # z[0] ~ N(m0, p0); q only applies between days
        pred_m[t], pred_p[t] = mp, pp
        if np.isnan(y[t]):
            mt, pt = mp, pp
        else:
            s = pp + r
            k = pp / s
            e = y[t] - mp
            ll += -0.5 * (np.log(2 * np.pi * s) + e * e / s)
            mt, pt = mp + k * e, (1 - k) * pp
            gain[t] = k
        m[t], p[t] = mt, pt
    return dict(m=m, p=p, pred_m=pred_m, pred_p=pred_p, gain=gain, ll=float(ll))


def rts_smoother(f, q):
    """Backward pass: refines every estimate using the days that came after it."""
    m, p = f["m"].copy(), f["p"].copy()
    n = len(m)
    sm, sp, c = m.copy(), p.copy(), np.zeros(n)
    for t in range(n - 2, -1, -1):
        c[t] = p[t] / (p[t] + q)
        sm[t] = m[t] + c[t] * (sm[t + 1] - m[t])
        sp[t] = p[t] + c[t] ** 2 * (sp[t + 1] - (p[t] + q))
    return dict(m=sm, p=sp, c=c)


def em_fit(y, q=0.01, r=0.05, iters=2000, tol=1e-7):
    """Learn q and r from the data. Records the likelihood, which must not fall. EM is slow near the
    optimum on this model, so it runs until the likelihood stops moving instead of a fixed short count."""
    hist = []
    for _ in range(iters):
        f = kalman_filter(y, q, r)
        hist.append(f["ll"])
        if len(hist) > 1 and hist[-1] - hist[-2] < tol:
            break
        s = rts_smoother(f, q)
        m, p, c = s["m"], s["p"], s["c"]
        # lag-one cross covariance from the smoother: cov(z[t+1], z[t] | all data) = c[t] * p[t+1]
        cross = c[:-1] * p[1:]
        q = float(np.mean(p[1:] + p[:-1] - 2 * cross + (m[1:] - m[:-1]) ** 2))
        ok = ~np.isnan(y)
        r = float(np.mean((y[ok] - m[ok]) ** 2 + p[ok]))
    return q, r, hist


def batch_posterior(y, q, r, m0, p0=1.0):
    """The same answer by brute-force Gaussian conditioning. O(n^3); used only to check the filter."""
    n = len(y)
    v = p0 + q * np.arange(n)
    Z = np.minimum.outer(v, v)
    G = Z + r * np.eye(n)
    w = np.linalg.solve(G, y - m0)
    return m0 + Z @ w, np.diag(Z - Z @ np.linalg.solve(G, Z))
