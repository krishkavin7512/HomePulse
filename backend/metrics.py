"""Scoring helpers. All written out so the numbers on screen can be traced by hand."""
import numpy as np


def confusion(pred, truth):
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    return dict(tp=int((pred & truth).sum()), fp=int((pred & ~truth).sum()),
                fn=int((~pred & truth).sum()), tn=int((~pred & ~truth).sum()))


def f1_score(pred, truth):
    c = confusion(pred, truth)
    return 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1)


def prf(pred, truth):
    c = confusion(pred, truth)
    precision = c["tp"] / max(c["tp"] + c["fp"], 1)
    recall = c["tp"] / max(c["tp"] + c["fn"], 1)
    return dict(precision=precision, recall=recall, f1=f1_score(pred, truth),
                accuracy=(c["tp"] + c["tn"]) / max(sum(c.values()), 1), **c)


def otsu(values, bins=256):
    """Threshold that best separates a histogram into two groups, with no labels."""
    h, e = np.histogram(values, bins=bins)
    mid = (e[:-1] + e[1:]) / 2
    w = h / h.sum()
    w0 = np.cumsum(w)
    m0 = np.cumsum(w * mid)
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m0) ** 2 / (w0 * (1 - w0))
    return float(e[np.nanargmax(between) + 1])     # the class boundary is the winning bin's upper edge


def flips_per_day(states):
    """Average number of on/off changes per day. states: days x minutes of 0/1."""
    return float(np.abs(np.diff(states.astype(np.int8), axis=1)).sum(1).mean())


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])
