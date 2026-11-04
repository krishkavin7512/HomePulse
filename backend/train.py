"""Train every model, score it on unseen days, write artifacts/.  python -m backend.train"""
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, kalman
from .data import chronological_split, load_days
from .fhmm import FactorialHMM, kmeans_1d, upward_steps
from .hmm import GaussianHMM, backward_scaled, forward_log, forward_scaled, forward_unscaled, log_gauss, viterbi
from .markov import MarkovChain
from .metrics import f1_score, flips_per_day, otsu, pearson, prf
from .rivals import CART, AdaBoost, window_features
from .sum_product import max_product, sum_product

HM = 60  # minutes per hour


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


def clean(o):
    """Make any nested result JSON-safe."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def truth(days, meter):
    return getattr(days, meter) >= config.ON_RULES[meter]


def auc(score, label):
    """Probability a positive outranks a negative (Mann-Whitney)."""
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = int(label.sum()), int((~label).sum())
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def empty_home_proxy(days):
    """No occupancy labels exist. Proxy from the answer key: nobody cooked, washed or used hot water."""
    return (days.K.sum(1) == 0) & (days.L.sum(1) < 1100) & (days.H.sum(1) < 4000)


# --------------------------------------------------------------------------- #
def describe(days, sp):
    P, K, L, H = days.P, days.K, days.L, days.H
    wh = P.sum() * 1000 / 60
    hval = H[H >= config.ON_RULES["H"]]
    mode_wh = float(pd.Series(hval.round()).mode().iloc[0])
    daily = P.sum(1) / 60
    return {
        "raw_minutes": days.raw_minutes, "raw_missing": days.raw_missing, "missing_pct": 100 * days.raw_missing / days.raw_minutes,
        "complete_days": len(days), "first": days.dates[0], "last": days.dates[-1],
        "train_days": len(sp.train), "val_days": len(sp.val), "test_days": len(sp.test), "era_days": len(sp.era),
        "train_minutes": len(sp.train) * config.MINUTES, "test_minutes": len(sp.test) * config.MINUTES,
        "shares": {"Water heater + AC": H.sum() / wh, "Laundry room": L.sum() / wh, "Kitchen": K.sum() / wh,
                   "Everything else": 1 - (H.sum() + L.sum() + K.sum()) / wh},
        "heater_mode_wh": mode_wh, "heater_mode_kw": mode_wh * 60 / 1000,
        "daily_kwh_mean": daily.mean(), "daily_kwh_sd": daily.std(),
        "power": {k: float(v) for k, v in zip(("p5", "p50", "p95", "max"), (*np.quantile(P, [.05, .5, .95]), P.max()))},
        "on_rates_test": {m: float(truth(days.take(sp.test), m).mean()) for m in config.ON_RULES},
    }


def stage_markov(Xtr, Xva):
    mk = MarkovChain(12).fit(Xtr)
    ks = [1, 2, 3, 5, 10, 20, 30, 60, 120, 240, 480]
    return mk, {"bins": len(mk.widths), "edges": mk.edges, "A": mk.A, "pi": mk.pi, "params": mk.n_params(),
                "ll_per_min": mk.loglik(Xva) / Xva.size, "ll_iid_per_min": mk.loglik_iid(Xva) / Xva.size,
                "power_k": ks, "power_tv": mk.power_distance(ks)}


def stage_hmm(Xtr, Xva):
    res, models = [], {}
    for K in (2, 3, 4, 6, 8, 12, 16):
        m = GaussianHMM(K, config.EM_ITERS, seed=config.SEED).fit(Xtr)
        models[K] = m
        res.append({"K": K, "val_ll_per_min": m.loglik(Xva) / Xva.size, "bic": m.bic(Xtr), "params": m.n_params(),
                    "iters": len(m.history), "monotone": bool(np.all(np.diff(m.history) > -1e-3)), "history": m.history})
        log(f"  HMM K={K}: val ll/min {res[-1]['val_ll_per_min']:.4f}, BIC {res[-1]['bic']:.0f}")
    best = min(res, key=lambda r: r["bic"])["K"]
    return models, best, res


def stage_fhmm(Xtr, Xva):
    res, models = [], {}
    for M in config.CHAIN_RANGE:
        t = time.time()
        f = FactorialHMM(M, config.EM_ITERS).fit(Xtr)
        models[M] = f
        res.append({"M": M, "states": 2 ** M, "params": f.n_params(), "watts": f.watts, "base": f.base,
                    "val_ll_per_min": f.loglik(Xva) / Xva.size, "fit_seconds": time.time() - t,
                    "monotone": bool(np.all(np.diff(f.history) > -1e-3)), "history": f.history})
        log(f"  FHMM M={M}: kW {np.round(f.watts, 3)}, val ll/min {res[-1]['val_ll_per_min']:.4f}, {res[-1]['fit_seconds']:.0f}s")
    gains = np.diff([r["val_ll_per_min"] for r in res])
    chosen = config.CHAIN_RANGE[-1]
    for i in range(1, len(gains)):
        if gains[i] < config.ELBOW_RULE * gains[i - 1]:
            chosen = config.CHAIN_RANGE[i]
            break
    return models, chosen, res, gains


def stage_restarts(Xtr, first, n=6):
    """EM only finds a local optimum, so the starting guess matters. Start from jittered wattages
    and keep whichever run explains the training days best. Labels play no part in this choice."""
    rng = np.random.default_rng(config.SEED)
    w0 = kmeans_1d(upward_steps(Xtr), first.M)
    runs = [{"model": first, "start": w0, "label": "From the jumps"}]
    for i in range(n):
        start = w0 * rng.uniform(0.65, 1.35, first.M)
        runs.append({"model": FactorialHMM(first.M, config.EM_ITERS).fit(Xtr, start), "start": np.sort(start), "label": f"Random start {i + 1}"})
    for r in runs:
        r["ll"] = r["model"].loglik(Xtr)
    best = max(runs, key=lambda r: r["ll"])
    table = [{"label": r["label"], "start": r["start"], "watts": np.sort(r["model"].watts), "ll": r["ll"], "kept": r is best} for r in runs]
    return best["model"], table


def pick_device(f, X, days, meter):
    """Which chain is this appliance? Decided on held-out training-era days, never on the test set."""
    post = f.smooth_many(X)
    t = truth(days, meter)
    return int(np.argmax([f1_score(post[..., m] > 0.5, t) for m in range(f.M)]))


def stage_fhmm_test(f, Dtest, Dval):
    Xte = Dtest.P
    post = f.smooth_many(Xte)
    filt = f.filter_many(Xte)
    vit = np.stack([f.viterbi(x) for x in Xte]).astype(np.int8)
    idx = pick_device(f, Dval.P, Dval, "H")
    t = truth(Dtest, "H")
    cross = {}
    for mname in config.ON_RULES:
        tr = truth(Dtest, mname)
        cross[mname] = [f1_score(post[..., m] > .5, tr) for m in range(f.M)]
    return post, filt, vit, idx, {
        "heater_chain": idx, "cross_f1": cross,
        "smoothed": prf(post[..., idx] > .5, t), "filtered": prf(filt[..., idx] > .5, t), "viterbi": prf(vit[..., idx] > 0, t),
        "flips": {"smoothed": flips_per_day(post[..., idx] > .5), "filtered": flips_per_day(filt[..., idx] > .5),
                  "viterbi": flips_per_day(vit[..., idx]), "truth": flips_per_day(t)},
    }


def energy_stats(est_daily, true_daily, name=""):
    return {"estimated_kwh": est_daily.sum(), "true_kwh": true_daily.sum(),
            "error_pct": 100 * (est_daily.sum() - true_daily.sum()) / true_daily.sum(),
            "daily_r": pearson(est_daily, true_daily), "daily_mae_kwh": float(np.abs(est_daily - true_daily).mean())}


def stage_rivals(Dtr, Dva, Dte, sp_rng, hmm_best, fhmm_row):
    """Who wins at 'is the heater on this minute?' on unseen days."""
    t_te = truth(Dte, "H")
    hkwh_te = Dte.H.sum(1) / 1000
    rows = []
    P_tr, P_te = Dtr.P, Dte.P
    on_power_kw = float(Dtr.H[truth(Dtr, "H")].mean() * 60 / 1000)

    def add(name, kind, labels, pred, energy_daily=None):
        pred = pred.reshape(t_te.shape)
        e = energy_daily if energy_daily is not None else pred.sum(1) * on_power_kw / HM
        rows.append({"name": name, "family": kind, "labels": labels, **prf(pred, t_te), "flips_per_day": flips_per_day(pred),
                     **energy_stats(e, hkwh_te)})

    # no labels at all: a threshold has to be guessed from the total
    centres = kmeans_1d(P_tr.ravel()[::20], 2)
    thr_km = float(centres.mean())
    thr_ot = otsu(P_tr.ravel())
    base = float(np.quantile(P_tr, 0.05))
    for nm, th in ((f"Threshold, 2-means ({thr_km:.2f} kW)", thr_km), (f"Threshold, Otsu ({thr_ot:.2f} kW)", thr_ot)):
        pred = P_te > th
        add(nm, "Rule", "none", pred, (np.maximum(P_te - base, 0) * pred).sum(1) / HM)
    # threshold tuned with the answer key on training days
    grid = np.arange(0.2, 3.01, 0.05)
    t_tr = truth(Dtr, "H")
    th = float(grid[int(np.argmax([f1_score(P_tr > g, t_tr) for g in grid]))])
    add(f"Threshold tuned on labels ({th:.2f} kW)", "Rule", "sub-meter", P_te > th)
    # plain HMM: whole-house levels, then labels decide which levels mean heater
    hm = hmm_best
    weight = np.zeros(hm.K)
    hit = np.zeros(hm.K)
    for x, y in zip(P_tr, t_tr):
        g = hm.posterior(x)
        weight += g.sum(0)
        hit += g[y].sum(0)
    state_map = hit / (weight + 1e-9)
    pred = np.stack([hm.posterior(x) @ state_map for x in P_te]) > 0.5
    add(f"Plain HMM, {hm.K} levels + labels", "HMM", "sub-meter", pred)
    # CART and AdaBoost on window features
    Ftr, names = window_features(P_tr)
    Ftr = Ftr.astype(np.float64)
    ytr = t_tr.ravel()
    Fva, _ = window_features(Dva.P)
    yva = truth(Dva, "H").ravel()
    Fte, _ = window_features(P_te)
    rng = np.random.default_rng(config.SEED)
    sub = rng.choice(len(ytr), 60_000, replace=False)
    depth_curve = []
    best_depth, best_val, cart = 1, -1, None
    for d in (1, 2, 3, 4, 6, 8, 10, 12, 16):
        c = CART(d).fit(Ftr[sub], ytr[sub].astype(int))
        tr_f, va_f = f1_score(c.predict(Ftr[sub]) > 0, ytr[sub]), f1_score(c.predict(Fva) > 0, yva)
        depth_curve.append({"depth": d, "train_f1": tr_f, "val_f1": va_f, "leaves": c.depth_leaves()})
        if va_f > best_val:
            best_depth, best_val, cart = d, va_f, c
    log(f"  CART best depth {best_depth}, val F1 {best_val:.3f}")
    add(f"CART, depth {best_depth}", "Tree", "sub-meter", cart.predict(Fte) > 0)
    ada = AdaBoost(60, 2).fit(Ftr[sub], ytr[sub].astype(int), Fva, yva)
    log(f"  AdaBoost rounds {len(ada.trees)}, val F1 {ada.val_curve[-1]:.3f}")
    add(f"AdaBoost, {len(ada.trees)} small trees", "Boosting", "sub-meter", ada.predict(Fte) > 0)
    # how many labelled days does the tree need to catch the decoder?
    budget = []
    per_day = np.arange(len(ytr)) // config.MINUTES
    for n in (1, 3, 7, 14, 30, 60, 120, 200):
        scores = []
        for rep in range(1 if n == 200 else 4):
            chosen = rng.choice(len(P_tr), n, replace=False)
            mask = np.where(np.isin(per_day, chosen))[0]
            mask = rng.choice(mask, min(len(mask), 60_000), replace=False)
            if ytr[mask].sum() in (0, len(mask)):
                scores.append(0.0)
                continue
            scores.append(f1_score(CART(best_depth).fit(Ftr[mask], ytr[mask].astype(int)).predict(Fte) > 0, t_te.ravel()))
        budget.append({"days": n, "f1": float(np.mean(scores)), "min": float(np.min(scores)), "max": float(np.max(scores))})
    log("  label budget: " + ", ".join(f"{b['days']}d={b['f1']:.3f}" for b in budget))
    rows.extend(fhmm_row)
    importance = sorted(zip(names, cart.importance), key=lambda kv: -kv[1])
    return rows, {"depth_curve": depth_curve, "best_depth": best_depth, "importance": importance, "budget": budget,
                  "ada_alphas": ada.alphas, "ada_train": ada.train_curve, "ada_val": ada.val_curve,
                  "thresholds_grid": grid, "thresholds_f1": [f1_score(P_te > g, t_te) for g in grid],
                  "threshold_values": {"two_means": thr_km, "otsu": thr_ot, "tuned": th}}


def stage_message_passing(hm, x):
    """Check on a real day: general message passing gives the same answers as the special recursions."""
    b, shift = hm._b(x)
    alpha, c = forward_scaled(hm.pi, hm.A, b)
    gamma = alpha * backward_scaled(hm.A, b, c)
    gamma /= gamma.sum(1, keepdims=True)
    marg, _, log_z = sum_product(hm.pi, hm.A, b)
    logb = log_gauss(x, hm.mu, hm.var)
    p1, l1 = viterbi(hm.pi, hm.A, logb)
    p2, l2 = max_product(hm.pi, hm.A, logb)
    ll_scaled = float(hm.loglik(x[None]))
    ll_log = forward_log(hm.pi, hm.A, logb)
    naive = forward_unscaled(hm.pi, hm.A, np.exp(logb))
    ok = np.isfinite(naive)
    breaks = None if ok.all() else int(np.argmin(ok))
    return {"max_gamma_diff": np.abs(gamma - marg).max(), "same_viterbi_path": bool((p1 == p2).all()), "viterbi_ll_diff": abs(l1 - l2),
            "gamma": gamma[::5].ravel(), "marg": marg[::5].ravel(), "ll_scaled": ll_scaled, "ll_log_domain": ll_log, "ll_diff": abs(ll_scaled - ll_log),
            "naive_breaks_at": breaks, "naive_direction": None if breaks is None else ("overflow" if naive[breaks] > 0 else "underflow"),
            "naive_log10": naive[ok], "scaled_log10": np.cumsum(np.log(c) + shift) / np.log(10),
            "viterbi_vs_posterior_agree": float((gamma.argmax(1) == p1).mean())}


def forecast_stage(hm, f, mk, Dte, Dtr, rng):
    """Predict the reading k minutes ahead from everything seen so far."""
    idx = rng.choice(len(Dte), min(120, len(Dte)), replace=False)
    X = Dte.P[idx]
    hour_mean = Dtr.P.mean(0)
    horizons = [1, 2, 5, 10, 15, 30, 60, 120, 240]
    out = {"horizons": horizons, "Persistence (last reading)": [], "Same minute, average day": [], "Markov chain": [],
           f"HMM, {hm.K} levels": [], f"Factorial HMM, {f.M} devices": []}

    def alphas(pi, A, b):
        a = pi * b[:, 0]
        a = a / a.sum(1, keepdims=True)
        al = [a]
        for t in range(1, b.shape[1]):
            a = (a @ A) * b[:, t]
            a = a / a.sum(1, keepdims=True)
            al.append(a)
        return np.stack(al, 1)
    al_h = alphas(hm.pi, hm.A, hm._b(X)[0])
    Af, muf = f.transition(), f.means()
    al_f = alphas(f.start_probs(), Af, f._b(X)[0])
    centers = np.array([X[mk.discretise(X) == i].mean() if (mk.discretise(X) == i).any() else (mk.edges[i] + mk.edges[i + 1]) / 2
                        for i in range(len(mk.widths))])
    s = mk.discretise(X)
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    for k in horizons:
        y = X[:, k:]
        out["Persistence (last reading)"].append(rmse(X[:, :-k], y))
        out["Same minute, average day"].append(rmse(np.tile(hour_mean[k:], (len(X), 1)), y))
        out["Markov chain"].append(rmse((np.linalg.matrix_power(mk.A, k) @ centers)[s[:, :-k]], y))
        out[f"HMM, {hm.K} levels"].append(rmse(al_h[:, :-k] @ (np.linalg.matrix_power(hm.A, k) @ hm.mu), y))
        out[f"Factorial HMM, {f.M} devices"].append(rmse(al_f[:, :-k] @ (np.linalg.matrix_power(Af, k) @ muf), y))
    return out


def stage_kalman(days, sp):
    """Slow trend in daily use, over the full calendar so missing days are real gaps."""
    dates = pd.to_datetime(days.dates)
    all_days = pd.date_range(dates[0], dates[-1], freq="D")
    y = pd.Series(days.P.sum(1) / HM, index=dates).reindex(all_days).to_numpy()
    n_era = int((all_days < config.SPLIT_DATE).sum())
    step_var = float(np.nanvar(np.diff(y[:n_era])))   # start EM from the data's own day-to-day scatter
    q, r, hist = kalman.em_fit(y[:n_era], q=step_var / 10, r=step_var / 2)
    f = kalman.kalman_filter(y, q, r)
    s = kalman.rts_smoother(f, q)
    # tomorrow's total, test period only
    te = np.where((all_days >= config.SPLIT_DATE) & ~np.isnan(y))[0]
    ser = pd.Series(y)
    ma7 = ser.rolling(7, min_periods=1).mean().to_numpy()
    last = ser.ffill().to_numpy()
    era_mean = float(np.nanmean(y[:n_era]))
    rmse = lambda p: float(np.sqrt(np.mean((p - y[te]) ** 2)))
    forecast = {"Kalman level": rmse(f["pred_m"][te]), "7-day average": rmse(ma7[te - 1]), "Yesterday": rmse(last[te - 1]),
                "Long-run average": rmse(np.full(len(te), era_mean))}
    # does it spot a slow decline sooner? 40% drop injected over 8 weeks, then checked against the previous 60 days
    yte = last[n_era:]

    def first_alarm(series, s0, k=3.0):
        h0 = max(0, s0 - 60)
        z = (series[s0:] - series[h0:s0].mean()) / (series[h0:s0].std() + 1e-6)
        hit = np.where(z < -k)[0]
        return int(hit[0]) if len(hit) else None
    res = {"Kalman level": [], "7-day average": [], "Raw daily total": []}
    for s0 in range(60, len(yte) - 70, 25):
        t = yte.copy()
        t[s0:] = t[s0:] * (1 - 0.4 * np.minimum(np.arange(len(t) - s0), 56) / 56)
        res["Kalman level"].append(first_alarm(kalman.kalman_filter(t, q, r)["m"], s0))
        res["7-day average"].append(first_alarm(pd.Series(t).rolling(7, min_periods=1).mean().to_numpy(), s0))
        res["Raw daily total"].append(first_alarm(t, s0))
    k_lvl, m7 = kalman.kalman_filter(yte, q, r)["m"], pd.Series(yte).rolling(7, min_periods=1).mean().to_numpy()
    series = {"Kalman level": k_lvl, "7-day average": m7, "Raw daily total": yte}
    false_alarm = {k: float(np.mean([first_alarm(v[:s0 + 56], s0) is not None for s0 in range(60, len(yte) - 56, 14)])) for k, v in series.items()}
    decline = {k: {"median_days": float(np.median([d for d in v if d is not None])) if any(d is not None for d in v) else None,
                   "missed": sum(d is None for d in v), "runs": len(v), "false_alarm_rate": false_alarm[k]} for k, v in res.items()}
    # the filter must equal brute-force Gaussian conditioning
    ys = y[:n_era][~np.isnan(y[:n_era])][:150]
    bm, _ = kalman.batch_posterior(ys, q, r, float(ys[:7].mean()))
    fm = kalman.kalman_filter(ys, q, r, m0=float(ys[:7].mean()))["m"]
    return {"q": q, "r": r, "gain_final": float(f["gain"][np.nonzero(f["gain"])[0][-1]]), "em_history": hist,
            "dates": all_days.strftime("%Y-%m-%d").tolist(), "y": y, "filter": f["m"], "filter_var": f["p"], "smooth": s["m"], "smooth_var": s["p"],
            "gain": f["gain"], "forecast": forecast, "decline": decline, "batch_last_diff": abs(bm[-1] - fm[-1]), "missing_days": int(np.isnan(y).sum())}


def run_lengths(states):
    """Lengths of consecutive on-runs in a days x minutes 0/1 array."""
    out = []
    for r in states.astype(np.int8):
        d = np.diff(np.r_[0, r, 0])
        out.extend((np.where(d == -1)[0] - np.where(d == 1)[0]).tolist())
    return np.array(out)


def first_after(states, start=5 * HM, hold=3):
    """First minute after `start` where a state stays on for `hold` minutes. NaN if never."""
    out = np.full(len(states), np.nan)
    kernel = np.ones(hold)
    for i, r in enumerate(states):
        c = np.convolve(r[start:].astype(float), kernel, "valid")
        hit = np.where(c >= hold)[0]
        if len(hit):
            out[i] = start + hit[0]
    return out


def choose_presets(rows, f1_stats):
    """Demo days, chosen by rule so the app cannot be accused of showing only its best days."""
    scored = [r for r in rows if r["f1"] is not None]
    nearest = lambda value: min(scored, key=lambda r: (abs(r["f1"] - value), r["date"]))
    winter = [r for r in rows if r["date"][5:7] in ("01", "02")]
    summer = sorted((r for r in rows if r["date"][5:7] in ("07", "08")), key=lambda r: r["kwh"])
    empty = [r for r in rows if r["empty"]]
    return [
        {"key": "typical", "label": "A typical day", "date": nearest(f1_stats["median"])["date"], "note": "Scored at the median across all unseen days."},
        {"key": "winter", "label": "Coldest-month peak", "date": max(winter, key=lambda r: r["kwh"])["date"], "note": "Highest total use in January or February."},
        {"key": "summer", "label": "Ordinary summer day", "date": summer[len(summer) // 2]["date"], "note": "Median total use in July or August."},
        {"key": "empty", "label": "Nobody home", "date": min(empty, key=lambda r: r["kwh"])["date"], "note": "No cooking, washing or hot water on the sub-meters."},
        {"key": "hard", "label": "A bad day for it", "date": nearest(f1_stats["p10"])["date"], "note": "Scored at the worst 10th percentile. Shown on purpose."},
    ]


def main():
    config.ARTIFACTS.mkdir(exist_ok=True)
    days = load_days()
    sp = chronological_split(days)
    Dtr, Dva, Dte = days.take(sp.train), days.take(sp.val), days.take(sp.test)
    Xtr, Xva, Xte = Dtr.P, Dva.P, Dte.P
    rng = np.random.default_rng(config.SEED)
    log(f"days {len(days)}: train {len(Dtr)}, val {len(Dva)}, test {len(Dte)}")
    rep = {"data": describe(days, sp), "split_date": config.SPLIT_DATE, "chain_rule": config.ELBOW_RULE}

    mk, rep["markov"] = stage_markov(Xtr, Xva)
    log(f"Markov val ll/min {rep['markov']['ll_per_min']:.3f} (iid {rep['markov']['ll_iid_per_min']:.3f})")

    hmm_models, hmm_k, rep["hmm_grid"] = stage_hmm(Xtr, Xva)
    hm = hmm_models[hmm_k]
    rep["hmm"] = {"K": hmm_k, "mu": hm.mu, "var": hm.var, "A": hm.A, "pi": hm.pi, "history": hm.history}
    log(f"HMM chosen by BIC: K={hmm_k}")

    fh_models, M, rep["fhmm_grid"], gains = stage_fhmm(Xtr, Xva)
    rep["fhmm_gains"] = gains
    f, rep["restarts"] = stage_restarts(Xtr, fh_models[M])
    f.sort_chains()
    rep["fhmm"] = {"M": M, **f.state_dict(), "run_min": f.run_minutes(), "gap_min": f.gap_minutes(), "duty": f.duty(), "history": f.history,
                   "train_ll_per_min": f.loglik(Xtr) / Xtr.size, "val_ll_per_min": f.loglik(Xva) / Xva.size}
    for r in rep["restarts"]:
        log(f"  {r['label']:16s} kW {np.round(r['watts'], 2)} ll {r['ll']:.0f}{'  <- kept' if r['kept'] else ''}")
    log(f"Chains kept: {M}, kW {np.round(f.watts, 3)}, base {f.base:.3f}")

    post, filt, vit, hidx, rep["decode"] = stage_fhmm_test(f, Dte, Dva)
    tH = truth(Dte, "H")
    log(f"Heater chain #{hidx} ({f.watts[hidx]:.3f} kW): smoothed {rep['decode']['smoothed']['f1']:.3f}, filtered {rep['decode']['filtered']['f1']:.3f}, viterbi {rep['decode']['viterbi']['f1']:.3f}")
    log(f"Cross F1 by chain: {rep['decode']['cross_f1']}")

    # energy per device on the unseen days
    dev_daily = (post * f.watts[None, None, :]).sum(1) / HM
    total_daily = Xte.sum(1) / HM
    base_daily = np.full(len(Dte), f.base * 24)
    est_h, true_h = dev_daily[:, hidx], Dte.H.sum(1) / 1000
    rep["energy"] = {**energy_stats(est_h, true_h), "device_kwh": dev_daily.sum(0), "base_kwh": base_daily.sum(), "total_kwh": total_daily.sum(),
                     "true_share": true_h.sum() / total_daily.sum(), "est_share": est_h.sum() / total_daily.sum(),
                     "residual_kwh": total_daily.sum() - dev_daily.sum() - base_daily.sum()}
    log(f"Energy: est {est_h.sum():.0f} vs true {true_h.sum():.0f} kWh ({rep['energy']['error_pct']:+.1f}%), daily r {rep['energy']['daily_r']:.3f}, share {rep['energy']['est_share']:.3f} vs {rep['energy']['true_share']:.3f}")
    mon = pd.DataFrame({"m": pd.to_datetime(Dte.dates).strftime("%Y-%m"), "est": est_h, "true": true_h, "n": 1}).groupby("m").sum()
    rep["monthly"] = {"month": mon.index.tolist(), "est_per_day": (mon.est / mon.n).tolist(), "true_per_day": (mon.true / mon.n).tolist(), "days": mon.n.tolist()}
    rep["monthly"]["r"] = pearson(mon.est / mon.n, mon.true / mon.n)
    # per-day accuracy, so demo days can be shown to be typical
    per_day = np.array([f1_score(p > .5, t) if t.any() or (p > .5).any() else np.nan for p, t in zip(post[..., hidx], tH)])
    rep["per_day_f1"] = {"p10": float(np.nanquantile(per_day, .1)), "median": float(np.nanmedian(per_day)), "p90": float(np.nanquantile(per_day, .9)), "values": per_day}

    # The small chains were checked against the fridge-level signal in the laundry sub-meter. They do not match it.
    fridge = (Dte.L >= 1) & (Dte.L <= 3)
    rep["fridge"] = {"f1_by_chain": [f1_score(post[..., m] > .5, fridge) for m in range(M)], "on_rate": float(fridge.mean())}
    log(f"Fridge-level L F1 by chain: {np.round(rep['fridge']['f1_by_chain'], 3)}")

    # run-length honesty check: HMM durations are geometric by construction
    runs_m = run_lengths(vit[..., hidx])
    runs_t = run_lengths(tH)
    rep["durations"] = {"decoded_mean": float(runs_m.mean()), "truth_mean": float(runs_t.mean()), "model_mean": float(f.run_minutes()[hidx]),
                        "p_off": float(f.p_off[hidx]), "decoded": np.bincount(np.minimum(runs_m, 200)).tolist(), "truth": np.bincount(np.minimum(runs_t, 200)).tolist()}

    # what the meter gives away: the largest device's weekly rhythm against the real kitchen + laundry rhythm
    big = int(np.argmax(f.watts)) if int(np.argmax(f.watts)) != hidx else int(np.argsort(f.watts)[-2])
    active = post[..., big] > .5
    weekday = pd.to_datetime(Dte.dates).dayofweek.to_numpy()
    real = truth(Dte, "K") | truth(Dte, "L")

    def week_hour(a):
        return np.array([a[weekday == d].mean(0).reshape(24, HM).mean(1) for d in range(7)])
    heat_dec, heat_true, heat_total = week_hour(active), week_hour(real), week_hour(Xte)
    rep["week"] = {"chain": big, "kw": float(f.watts[big]), "decoded": heat_dec, "truth": heat_true, "total_kw": heat_total,
                   "r_decoded": pearson(heat_dec.ravel(), heat_true.ravel()), "r_total": pearson(heat_total.ravel(), heat_true.ravel()),
                   "midday_weekday": float(heat_dec[:5, 11:14].mean()), "midday_weekend": float(heat_dec[5:, 11:14].mean()),
                   "midday_weekday_true": float(heat_true[:5, 11:14].mean()), "midday_weekend_true": float(heat_true[5:, 11:14].mean())}
    log(f"Week: r decoded {rep['week']['r_decoded']:.3f}, r total {rep['week']['r_total']:.3f}, midday {rep['week']['midday_weekday']:.3f}->{rep['week']['midday_weekend']:.3f}")

    # Tried and failed: can the first big load of the morning stand in for first kitchen use?
    first_dec, first_true = first_after(active), first_after(real, hold=1)
    both = ~np.isnan(first_dec) & ~np.isnan(first_true)
    rep["routine"] = {"days_compared": int(both.sum()), "median_abs_error_min": float(np.median(np.abs(first_dec[both] - first_true[both]))),
                      "r": pearson(first_dec[both], first_true[both])}
    log(f"Routine: {rep['routine']}")

    # empty-home check: honest comparison with the plain daily total
    empty = empty_home_proxy(Dte)
    active_minutes = active.sum(1)
    rep["empty_home"] = {"n_empty": int(empty.sum()), "n_days": len(empty),
                         "auc_daily_total": auc(-total_daily, empty), "auc_decoded_activity": auc(-active_minutes.astype(float), empty)}
    log(f"Empty home: {rep['empty_home']}")

    # rivals
    fhmm_rows = []
    on_e = dev_daily[:, hidx]
    for name, pred, e in ((f"Factorial HMM, whole day ({M} chains)", post[..., hidx] > .5, on_e),
                          (f"Factorial HMM, Viterbi path", vit[..., hidx] > 0, vit[..., hidx].sum(1) * f.watts[hidx] / HM),
                          (f"Factorial HMM, live (past only)", filt[..., hidx] > .5, (filt[..., hidx] * f.watts[hidx]).sum(1) / HM)):
        fhmm_rows.append({"name": name, "family": "Factorial HMM", "labels": "none", **prf(pred, tH), "flips_per_day": flips_per_day(pred), **energy_stats(e, true_h)})
    rows, extra = stage_rivals(Dtr, Dva, Dte, rng, hm, fhmm_rows)
    rep["rivals"] = rows
    rep["rival_detail"] = extra
    for r in rows:
        log(f"  {r['name']:44s} labels={r['labels']:9s} F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} flips={r['flips_per_day']:.1f} energy {r['error_pct']:+.1f}%")

    # same test for kitchen and laundry, on the honest-limits panel
    limits = {}
    for mname in ("K", "L"):
        i = pick_device(f, Xva, Dva, mname)
        t = truth(Dte, mname)
        grid = np.arange(0.2, 6.01, 0.1)
        th = grid[int(np.argmax([f1_score(Xtr > g, truth(Dtr, mname)) for g in grid]))]
        limits[mname] = {"chain": i, "kw": float(f.watts[i]), "f1": f1_score(post[..., i] > .5, t), "tuned_threshold_f1": f1_score(Xte > th, t), "on_rate": float(t.mean())}
    rep["limits"] = limits
    log(f"Limits: {limits}")

    rep["message_passing"] = stage_message_passing(hm, Xte[int(rng.integers(len(Xte)))])
    mp = rep["message_passing"]
    log(f"Message passing: gamma diff {mp['max_gamma_diff']:.2e}, viterbi same {mp['same_viterbi_path']}, unscaled forward {mp['naive_direction']} at minute {mp['naive_breaks_at']}, ll diff {mp['ll_diff']:.2e}")

    rep["forecast"] = forecast_stage(hm, f, mk, Dte, Dtr, rng)
    log(f"Forecast RMSE: {[(k, round(v[0], 3), round(v[4], 3)) for k, v in rep['forecast'].items() if k != 'horizons']}")

    rep["kalman"] = stage_kalman(days, sp)
    log(f"Kalman: q {rep['kalman']['q']:.3f} r {rep['kalman']['r']:.3f} gain {rep['kalman']['gain_final']:.3f}; forecast {rep['kalman']['forecast']}; decline {rep['kalman']['decline']}")

    rep["trained_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    per_day_rows = [{"date": d, "kwh": round(float(k), 2), "heater_true": round(float(t), 2), "heater_est": round(float(e), 2),
                     "f1": None if np.isnan(p) else round(float(p), 3), "empty": bool(em)}
                    for d, k, t, e, p, em in zip(Dte.dates.tolist(), total_daily, true_h, est_h, per_day, empty)]
    rep["days_index"] = per_day_rows
    rep["presets"] = choose_presets(per_day_rows, rep["per_day_f1"])
    model = {"fhmm": f.state_dict(), "heater_chain": hidx, "hmm": rep["hmm"], "markov": {k: rep["markov"][k] for k in ("edges", "A", "pi")}}
    (config.ARTIFACTS / "model.json").write_text(json.dumps(clean(model)))
    (config.ARTIFACTS / "report.json").write_text(json.dumps(clean(rep)))
    ctx = dict(days=days, sp=sp, Dtr=Dtr, Dva=Dva, Dte=Dte, f=f, hm=hm, mk=mk, post=post, filt=filt, vit=vit, hidx=hidx, rep=clean(rep))
    from .charts import build_graphs
    graphs = build_graphs(ctx)
    (config.ARTIFACTS / "graphs.json").write_text(json.dumps(clean(graphs)))
    log(f"Done. {len(graphs)} graphs.")


if __name__ == "__main__":
    main()
