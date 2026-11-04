"""Supervised rivals: a decision tree (CART) and AdaBoost, written out in NumPy.

These are given something the decoder never gets: real sub-meter labels from training-era days.
In real life those labels mean fitting a separate meter on the appliance. They answer the
question "how much would that extra hardware actually buy you?"
"""
import numpy as np


def window_features(X):
    """Features per minute from the total alone. X: days x minutes. Returns (N x F, names)."""
    D, T = X.shape
    out, names = [], []

    def add(a, n):
        out.append(a.ravel().astype(np.float32))
        names.append(n)

    add(X, "power now")
    prev = np.c_[X[:, :1], X[:, :-1]]
    nxt = np.c_[X[:, 1:], X[:, -1:]]
    add(X - prev, "step from last minute")
    add(nxt - X, "step to next minute")
    for w in (5, 15, 60):
        c = np.cumsum(np.c_[np.zeros((D, 1)), X], axis=1)
        lo = np.maximum(np.arange(T) - w, 0)
        hi = np.minimum(np.arange(T) + w + 1, T)
        mean = (c[:, hi] - c[:, lo]) / (hi - lo)
        add(mean, f"mean over +/-{w} min")
        add(X - mean, f"power minus +/-{w} min mean")
    pad = 5
    win = np.stack([np.roll(X, s, axis=1) for s in range(-pad, pad + 1)], 0)
    add(win.min(0), "min over +/-5 min")
    add(win.max(0), "max over +/-5 min")
    add(np.tile(np.arange(T) / 60, (D, 1)), "hour of day")
    return np.column_stack(out), names


class _Binner:
    """Quantile bins so each split search is a bincount, not a sort."""
    def __init__(self, n_bins=48):
        self.n_bins = n_bins

    def fit(self, X):
        q = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[:, j], q)) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        return np.column_stack([np.searchsorted(e, X[:, j], side="right") for j, e in enumerate(self.edges)]).astype(np.int16)


class CART:
    """Binary classification tree grown by Gini impurity, with sample weights."""
    def __init__(self, max_depth=6, min_leaf=20.0, n_bins=48, binner=None):
        self.max_depth, self.min_leaf, self.n_bins, self.binner = max_depth, min_leaf, n_bins, binner

    def fit(self, X, y, w=None, binned=None):
        """y in {0,1}. `binned` lets AdaBoost reuse one binning across rounds."""
        if self.binner is None:
            self.binner = _Binner(self.n_bins).fit(X)
        B = self.binner.transform(X) if binned is None else binned
        w = np.ones(len(y), float) if w is None else w
        self.importance = np.zeros(B.shape[1])
        self.nodes = []
        self._grow(B, y.astype(float), w, np.arange(len(y)), 0)
        total = self.importance.sum()
        self.importance = self.importance / total if total > 0 else self.importance
        return self

    def _leaf(self, y, w, idx):
        s = w[idx].sum()
        p = float((w[idx] * y[idx]).sum() / s) if s > 0 else 0.0
        self.nodes.append({"leaf": True, "p": p})
        return len(self.nodes) - 1

    def _grow(self, B, y, w, idx, depth):
        W = w[idx].sum()
        pos = (w[idx] * y[idx]).sum()
        if depth >= self.max_depth or W < 2 * self.min_leaf or pos <= 0 or pos >= W:
            return self._leaf(y, w, idx)
        parent = 1 - (pos / W) ** 2 - ((W - pos) / W) ** 2
        best = (1e-12, None, None)
        for j in range(B.shape[1]):
            n = len(self.binner.edges[j]) + 1
            b = B[idx, j]
            wt = np.bincount(b, w[idx], n)
            wp = np.bincount(b, w[idx] * y[idx], n)
            cl, pl = np.cumsum(wt)[:-1], np.cumsum(wp)[:-1]
            cr, pr = W - cl, pos - pl
            ok = (cl >= self.min_leaf) & (cr >= self.min_leaf)
            if not ok.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                gl = 1 - (pl / cl) ** 2 - ((cl - pl) / cl) ** 2
                gr = 1 - (pr / cr) ** 2 - ((cr - pr) / cr) ** 2
                gain = parent - (cl * gl + cr * gr) / W
            gain[~ok] = -1
            k = int(np.argmax(gain))
            if gain[k] > best[0]:
                best = (float(gain[k]), j, k)
        if best[1] is None:
            return self._leaf(y, w, idx)
        gain, j, k = best
        self.importance[j] += gain * W
        node = len(self.nodes)
        self.nodes.append({"leaf": False, "f": j, "bin": k})
        mask = B[idx, j] <= k
        self.nodes[node]["l"] = self._grow(B, y, w, idx[mask], depth + 1)
        self.nodes[node]["r"] = self._grow(B, y, w, idx[~mask], depth + 1)
        return node

    def proba(self, X, binned=None):
        B = self.binner.transform(X) if binned is None else binned
        out = np.zeros(len(B))
        stack = [(0, np.arange(len(B)))]
        while stack:
            n, idx = stack.pop()
            nd = self.nodes[n]
            if nd["leaf"]:
                out[idx] = nd["p"]
                continue
            mask = B[idx, nd["f"]] <= nd["bin"]
            stack.append((nd["l"], idx[mask]))
            stack.append((nd["r"], idx[~mask]))
        return out

    def predict(self, X, binned=None):
        return (self.proba(X, binned) >= 0.5).astype(int)

    def depth_leaves(self):
        return sum(n["leaf"] for n in self.nodes)


class AdaBoost:
    """Discrete AdaBoost with shallow CART trees as the weak learners."""
    def __init__(self, n_rounds=60, depth=2):
        self.n_rounds, self.depth = n_rounds, depth

    def fit(self, X, y, X_val=None, y_val=None):
        from .metrics import f1_score
        binner = _Binner().fit(X)
        B = binner.transform(X)
        BV = binner.transform(X_val) if X_val is not None else None
        n = len(y)
        w = np.full(n, 1 / n)
        ys = 2 * y.astype(int) - 1
        self.trees, self.alphas, self.errors, self.train_curve, self.val_curve = [], [], [], [], []
        Ftr = np.zeros(n)
        Fva = np.zeros(len(y_val)) if y_val is not None else None
        for _ in range(self.n_rounds):
            t = CART(self.depth, min_leaf=1e-4, binner=binner).fit(X, y, w, binned=B)
            h = 2 * t.predict(None, B) - 1
            err = float(np.clip(w[h != ys].sum(), 1e-9, 1 - 1e-9))
            if err >= 0.5:
                break
            a = 0.5 * np.log((1 - err) / err)
            w = w * np.exp(-a * ys * h)
            w /= w.sum()
            self.trees.append(t)
            self.alphas.append(float(a))
            self.errors.append(err)
            Ftr += a * h
            self.train_curve.append(f1_score(Ftr > 0, y == 1))
            if Fva is not None:
                Fva += a * (2 * t.predict(None, BV) - 1)
                self.val_curve.append(f1_score(Fva > 0, y_val == 1))
        self.binner = binner
        return self

    def score(self, X):
        B = self.binner.transform(X)
        F = np.zeros(len(X))
        for a, t in zip(self.alphas, self.trees):
            F += a * (2 * t.predict(None, B) - 1)
        return F

    def predict(self, X):
        return (self.score(X) > 0).astype(int)
