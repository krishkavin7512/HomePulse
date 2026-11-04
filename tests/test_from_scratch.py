"""The from-scratch models, checked against brute force, against each other and against scikit-learn."""
import itertools

import numpy as np
import pytest

from backend import kalman
from backend.fhmm import FactorialHMM
from backend.hmm import GaussianHMM, backward_scaled, forward_log, forward_scaled, forward_unscaled, log_gauss, viterbi
from backend.markov import MarkovChain
from backend.metrics import f1_score, otsu, prf
from backend.rivals import CART, AdaBoost, window_features
from backend.sum_product import max_product, sum_product


@pytest.fixture(scope="module")
def tiny():
    """A small HMM, short enough to enumerate every possible state path."""
    pi = np.array([0.6, 0.3, 0.1])
    A = np.array([[0.8, 0.15, 0.05], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]])
    mu, var = np.array([0.0, 1.0, 3.0]), np.array([0.3, 0.2, 0.5])
    x = np.array([0.1, 0.2, 1.3, 2.9, 3.1, 0.8, 0.2])
    return pi, A, mu, var, x


def brute(pi, A, mu, var, x):
    T, K = len(x), len(pi)
    lb = log_gauss(x, mu, var)
    joint = {}
    for path in itertools.product(range(K), repeat=T):
        lp = np.log(pi[path[0]]) + lb[0, path[0]]
        for t in range(1, T):
            lp += np.log(A[path[t - 1], path[t]]) + lb[t, path[t]]
        joint[path] = lp
    vals = np.array(list(joint.values()))
    log_p = float(np.log(np.exp(vals).sum()))
    best = max(joint, key=joint.get)
    marg = np.zeros((T, K))
    for path, lp in joint.items():
        for t, s in enumerate(path):
            marg[t, s] += np.exp(lp)
    return log_p, np.array(best), joint[best], marg / marg.sum(1, keepdims=True)


def test_forward_backward_matches_enumeration(tiny):
    pi, A, mu, var, x = tiny
    log_p, _, _, marg = brute(pi, A, mu, var, x)
    b = np.exp(log_gauss(x, mu, var))
    alpha, c = forward_scaled(pi, A, b)
    gamma = alpha * backward_scaled(A, b, c)
    gamma /= gamma.sum(1, keepdims=True)
    assert np.log(c).sum() == pytest.approx(log_p, abs=1e-9)
    assert np.allclose(gamma, marg, atol=1e-9)


def test_viterbi_matches_enumeration(tiny):
    pi, A, mu, var, x = tiny
    _, best, best_lp, _ = brute(pi, A, mu, var, x)
    path, lp = viterbi(pi, A, log_gauss(x, mu, var))
    assert (path == best).all()
    assert lp == pytest.approx(best_lp, abs=1e-9)


def test_log_domain_and_scaled_agree(tiny):
    pi, A, mu, var, x = tiny
    b = np.exp(log_gauss(x, mu, var))
    assert forward_log(pi, A, log_gauss(x, mu, var)) == pytest.approx(np.log(forward_scaled(pi, A, b)[1]).sum(), abs=1e-9)


def test_unscaled_forward_leaves_float_range():
    pi, A = np.array([0.5, 0.5]), np.array([[0.9, 0.1], [0.1, 0.9]])
    low = forward_unscaled(pi, A, np.full((2000, 2), 0.1))
    high = forward_unscaled(pi, A, np.full((2000, 2), 10.0))
    assert np.isneginf(low[-1]) and np.isposinf(high[-1])
    assert len(low) < 2000 and len(high) < 2000


def test_sum_product_is_forward_backward(tiny):
    pi, A, mu, var, x = tiny
    log_p, _, _, marg = brute(pi, A, mu, var, x)
    m, pair, log_z = sum_product(pi, A, np.exp(log_gauss(x, mu, var)))
    assert log_z == pytest.approx(log_p, abs=1e-9)
    assert np.allclose(m, marg, atol=1e-9)
    assert np.allclose(pair[0].sum(1), m[0], atol=1e-9)


def test_max_product_is_viterbi(tiny):
    pi, A, mu, var, x = tiny
    lb = log_gauss(x, mu, var)
    p1, l1 = viterbi(pi, A, lb)
    p2, l2 = max_product(pi, A, lb)
    assert (p1 == p2).all() and l1 == pytest.approx(l2)


def synthetic_two_level(days=30, T=600, seed=1):
    rng = np.random.default_rng(seed)
    X = np.zeros((days, T))
    for d in range(days):
        s = 0
        for t in range(T):
            if rng.random() < (0.02 if s == 0 else 0.05):
                s = 1 - s
            X[d, t] = (0.3 if s == 0 else 1.8) + rng.normal(0, 0.05)
    return X


def test_baum_welch_rises_and_recovers_levels():
    X = synthetic_two_level()
    m = GaussianHMM(2, 25).fit(X)
    assert np.all(np.diff(m.history) > -1e-3)
    assert np.allclose(np.sort(m.mu), [0.3, 1.8], atol=0.03)


def test_markov_counts_and_stationary():
    X = np.array([[0.1, 0.1, 2.0, 2.0, 0.1, 2.0]])
    mk = MarkovChain(2, alpha=0.0).fit(X)
    s = mk.discretise(X)
    C = np.zeros((2, 2))
    for a, b in zip(s[0, :-1], s[0, 1:]):
        C[a, b] += 1
    assert np.allclose(mk.counts, C)
    assert np.allclose(mk.pi @ mk.A, mk.pi, atol=1e-9)
    assert mk.pi.sum() == pytest.approx(1)
    assert np.allclose(mk.A.sum(1), 1)


def two_devices(days=25, T=600, seed=2):
    rng = np.random.default_rng(seed)
    X = np.zeros((days, T))
    truth = np.zeros((days, T, 2), bool)
    for d in range(days):
        a = b = False
        for t in range(T):
            if rng.random() < (0.015 if not a else 0.04):
                a = not a
            if rng.random() < (0.01 if not b else 0.03):
                b = not b
            truth[d, t] = a, b
            X[d, t] = 0.2 + 0.5 * a + 1.5 * b + rng.normal(0, 0.03)
    return X, truth


@pytest.fixture(scope="module")
def fitted():
    X, truth = two_devices()
    return FactorialHMM(2, 30).fit(X), X, truth


def test_fhmm_finds_hidden_wattages(fitted):
    f, _, _ = fitted
    assert np.allclose(np.sort(f.watts), [0.5, 1.5], atol=0.05)
    assert f.base == pytest.approx(0.2, abs=0.05)


def test_fhmm_likelihood_never_falls(fitted):
    f, _, _ = fitted
    assert np.all(np.diff(f.history) > -1e-3)


def test_fhmm_decodes_each_device(fitted):
    f, X, truth = fitted
    order = np.argsort(f.watts)
    p = f.smooth_many(X)
    for k in (0, 1):
        assert f1_score(p[..., order[k]] > 0.5, truth[..., k]) > 0.9


def test_fhmm_transition_is_kronecker_of_devices(fitted):
    f, _, _ = fitted
    A = f.transition()
    assert A.shape == (4, 4) and np.allclose(A.sum(1), 1)
    assert np.allclose(A[0, 0], (1 - f.p_on[0]) * (1 - f.p_on[1]))
    assert np.allclose(f.start_probs() @ A, f.start_probs(), atol=1e-9)


def test_sort_chains_leaves_the_model_unchanged():
    X, _ = two_devices(days=6)
    f = FactorialHMM(2, 6).fit(X, [1.6, 0.4])
    before = f.loglik(X)
    p_before = f.smooth_many(X)[..., np.argsort(f.watts)]
    f.sort_chains()
    assert np.all(np.diff(f.watts) > 0)
    assert f.loglik(X) == pytest.approx(before, rel=1e-9)
    assert np.allclose(f.smooth_many(X), p_before, atol=1e-5)


def test_fhmm_batch_and_single_day_agree(fitted):
    f, X, _ = fitted
    assert np.allclose(f.smooth_many(X[:2])[1], f.smooth(X[1]), atol=1e-5)
    assert np.allclose(f.filter_many(X[:2])[1], f.filtered(X[1]), atol=1e-5)


def test_fhmm_state_roundtrip(fitted):
    f, X, _ = fitted
    g = FactorialHMM.from_state(f.state_dict())
    assert g.loglik(X[:3]) == pytest.approx(f.loglik(X[:3]))


def test_kalman_equals_batch_conditioning():
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(0, 0.3, 60)) + rng.normal(0, 1.0, 60)
    q, r, m0 = 0.09, 1.0, 0.0
    bm, bv = kalman.batch_posterior(y, q, r, m0, p0=2.0)
    f = kalman.kalman_filter(y, q, r, m0=m0, p0=2.0)
    s = kalman.rts_smoother(f, q)
    assert f["m"][-1] == pytest.approx(bm[-1], abs=1e-8)
    assert np.allclose(s["m"], bm, atol=1e-6)
    assert np.allclose(s["p"], bv, atol=1e-6)


def test_kalman_em_rises_and_recovers_noise():
    rng = np.random.default_rng(4)
    y = np.cumsum(rng.normal(0, np.sqrt(0.5), 3000)) + rng.normal(0, np.sqrt(4.0), 3000)
    q, r, hist = kalman.em_fit(y, q=0.1, r=1.0)
    assert np.all(np.diff(hist) > -1e-6)
    assert q == pytest.approx(0.5, rel=0.35) and r == pytest.approx(4.0, rel=0.2)


def test_kalman_em_reaches_the_same_answer_from_any_start():
    rng = np.random.default_rng(5)
    y = np.cumsum(rng.normal(0, 1.0, 1500)) + rng.normal(0, 3.0, 1500)
    y[200:215] = np.nan
    fits = [kalman.em_fit(y, q=a, r=b)[:2] for a, b in ((0.01, 0.05), (5.0, 50.0), (20.0, 1.0))]
    for q, r in fits[1:]:
        assert q == pytest.approx(fits[0][0], rel=0.01) and r == pytest.approx(fits[0][1], rel=0.01)


def test_kalman_band_widens_over_a_gap():
    y = np.r_[np.ones(30), np.full(10, np.nan), np.ones(5)]
    f = kalman.kalman_filter(y, 0.1, 1.0)
    assert np.all(np.diff(f["p"][30:40]) > 0)
    assert f["p"][40] < f["p"][39]


def test_cart_matches_sklearn():
    from sklearn.datasets import make_classification
    from sklearn.tree import DecisionTreeClassifier
    X, y = make_classification(4000, 8, n_informative=5, random_state=0)
    Xt, Xv, yt, yv = X[:3000], X[3000:], y[:3000], y[3000:]
    mine = (CART(6, n_bins=128).fit(Xt, yt).predict(Xv) == yv).mean()
    theirs = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=0).fit(Xt, yt).score(Xv, yv)
    assert mine == pytest.approx(theirs, abs=0.03)


def test_adaboost_matches_sklearn_and_beats_a_stump():
    from sklearn.datasets import make_classification
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    X, y = make_classification(4000, 8, n_informative=5, random_state=1)
    Xt, Xv, yt, yv = X[:3000], X[3000:], y[:3000], y[3000:]
    mine = (AdaBoost(50, 2).fit(Xt, yt).predict(Xv) == yv).mean()
    stump = (CART(1).fit(Xt, yt).predict(Xv) == yv).mean()
    theirs = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), n_estimators=50, random_state=0).fit(Xt, yt).score(Xv, yv)
    assert mine > stump
    assert mine == pytest.approx(theirs, abs=0.04)


def test_window_features_shape_and_no_nan():
    X = np.random.default_rng(0).random((3, 200))
    F, names = window_features(X)
    assert F.shape == (600, len(names)) and np.isfinite(F).all()


def test_metrics_by_hand():
    pred = np.array([1, 1, 0, 0, 1], bool)
    truth = np.array([1, 0, 0, 1, 1], bool)
    r = prf(pred, truth)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 1)
    assert r["f1"] == pytest.approx(2 / 3)


def test_otsu_splits_two_groups():
    rng = np.random.default_rng(0)
    low, high = rng.normal(0, 0.1, 2000), rng.normal(3, 0.1, 2000)
    th = otsu(np.r_[low, high])
    assert (low < th).all() and (high > th).all()
