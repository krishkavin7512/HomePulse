"""Message passing on a factor graph shaped like a chain.

Forward-backward and Viterbi are usually taught as special recursions. Here they come from
one general rule: send a message along each edge of the factor graph. Using "sum" gives
marginals (sum-product); using "max" gives the best path (max-product, in logs max-sum).
The tests check that the results match hmm.py to numerical precision.
"""
import numpy as np


def _factors(pi, A, b):
    """Pairwise factors psi[t](z_t, z_{t+1}) with the emission folded in, plus the first unary."""
    return pi * b[0], [A * b[t + 1][None, :] for t in range(len(b) - 1)]


def sum_product(pi, A, b):
    """Returns (marginals T x K, pairwise marginals list, log partition = log p(x))."""
    first, psi = _factors(pi, A, b)
    T, K = b.shape
    log_z = 0.0
    fwd = [first / first.sum()]
    log_z += np.log(first.sum())
    for f in psi:
        m = fwd[-1] @ f
        s = m.sum()
        log_z += np.log(s)
        fwd.append(m / s)
    bwd = [np.ones(K)] * T
    for t in range(T - 2, -1, -1):
        m = psi[t] @ bwd[t + 1]
        bwd[t] = m / m.sum()
    marg = np.array([f * g for f, g in zip(fwd, bwd)])
    marg /= marg.sum(1, keepdims=True)
    pair = []
    for t in range(T - 1):
        p = fwd[t][:, None] * psi[t] * bwd[t + 1][None, :]
        pair.append(p / p.sum())
    return marg, pair, float(log_z)


def max_product(pi, A, logb):
    """Max-sum message passing with back-pointers. Returns (best path, its log probability)."""
    T, K = logb.shape
    lA = np.log(A + 1e-300)
    msg = np.log(pi + 1e-300) + logb[0]
    back = []
    for t in range(1, T):
        scores = msg[:, None] + lA + logb[t][None, :]
        back.append(scores.argmax(0))
        msg = scores.max(0)
    path = [int(msg.argmax())]
    for bp in reversed(back):
        path.append(int(bp[path[-1]]))
    return np.array(path[::-1]), float(msg.max())
