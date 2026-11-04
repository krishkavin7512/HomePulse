"""Every graph from training, drawn from this run's own numbers, with a plain explanation."""
import numpy as np
import pandas as pd

from .metrics import confusion

BLUE, AQUA, ORANGE, VIOLET, AMBER, MAGENTA = "#3987e5", "#199e70", "#e0621f", "#9085e9", "#c98500", "#d55181"
DEV = [BLUE, AQUA, ORANGE, VIOLET, AMBER]     # palette validated against the #101418 chart surface, in this order
INK, INK2, MUTED, GRID, AXIS, REF, SURFACE = "#eef2f5", "#c4ccd4", "#8593a0", "#1f262d", "#36414b", "#a9b4be", "#101418"
SEQ = [[0, "#131a22"], [0.5, "#2c6cba"], [1, "#a8cdfb"]]
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"
HOUR_VALS = list(range(0, 1441, 180))
HOUR_TEXT = [f"{h:02d}:00" for h in range(0, 25, 3)]


def ax(title="", **k):
    return {"title": {"text": title, "font": {"color": MUTED, "size": 12}}, "gridcolor": GRID, "linecolor": AXIS, "zerolinecolor": AXIS,
            "tickcolor": AXIS, "ticks": "outside", "ticklen": 4, "automargin": True, **k}


def lay(x="", y="", xa=None, ya=None, **k):
    return {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "font": {"family": FONT, "color": INK2, "size": 12},
            "margin": {"l": 60, "r": 18, "t": 26, "b": 48}, "hoverlabel": {"bgcolor": "#1a2129", "bordercolor": "#44515d", "font": {"family": FONT, "color": INK, "size": 12}},
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "bgcolor": "rgba(0,0,0,0)"},
            "xaxis": {**ax(x), **(xa or {})}, "yaxis": {**ax(y), **(ya or {})}, **k}


def line(x, y, name="", color=BLUE, width=2, dash=None, **k):
    return {"type": "scatter", "mode": "lines", "x": x, "y": y, "name": name, "line": {"color": color, "width": width, **({"dash": dash} if dash else {})}, **k}


def dots(x, y, name="", color=BLUE, size=8, **k):
    return {"type": "scatter", "mode": "markers", "x": x, "y": y, "name": name, "marker": {"color": color, "size": size, "line": {"color": SURFACE, "width": 2}}, **k}


def bars(x, y, name="", color=BLUE, **k):
    return {"type": "bar", "x": x, "y": y, "name": name, "marker": {"color": color, "line": {"color": SURFACE, "width": 2}, "cornerradius": 4}, **k}


def vline(x, color=REF, dash="dash"):
    return {"type": "line", "x0": x, "x1": x, "yref": "paper", "y0": 0, "y1": 1, "line": {"color": color, "width": 2, "dash": dash}}


def hline(y, color=REF, dash="dash"):
    return {"type": "line", "y0": y, "y1": y, "xref": "paper", "x0": 0, "x1": 1, "line": {"color": color, "width": 2, "dash": dash}}


def note(x, y, text, **k):
    return {"x": x, "y": y, "text": text, "showarrow": False, "font": {"color": INK2, "size": 12}, "bgcolor": "rgba(16,20,24,.8)", **k}


def heat(z, x, y, hover, scale=SEQ, zmin=None, zmax=None, colorbar=True, **k):
    t = {"type": "heatmap", "z": z, "x": x, "y": y, "colorscale": scale, "hovertemplate": hover + "<extra></extra>", "xgap": 2, "ygap": 2,
         "colorbar": {"thickness": 10, "outlinewidth": 0, "tickfont": {"color": MUTED}}, "showscale": colorbar, **k}
    if zmin is not None:
        t["zmin"], t["zmax"] = zmin, zmax
    return t


def hour_axis(title="Time of day"):
    return {"tickmode": "array", "tickvals": HOUR_VALS, "ticktext": HOUR_TEXT, "title": {"text": title, "font": {"color": MUTED, "size": 12}}}


def G(gid, cat, title, insight, what, how, read, where, data, layout, height=380):
    return {"id": gid, "category": cat, "title": title, "insight": insight, "height": height,
            "explain": {"what": what, "how": how, "read": read, "where": where}, "figure": {"data": data, "layout": layout}}


def acf(X, lags):
    a = X - X.mean()
    den = (a * a).mean()
    return [float((a[:, :-l] * a[:, l:]).mean() / den) for l in lags]


def simulate(pi, A, mu, var, days, T, rng):
    K = len(pi)
    cum = A.cumsum(1)
    out = np.zeros((days, T))
    s = rng.choice(K, days, p=pi)
    for t in range(T):
        out[:, t] = mu[s] + np.sqrt(var[s]) * rng.standard_normal(days)
        s = (rng.random((days, 1)) > cum[s]).sum(1).clip(max=K - 1)
    return out


def build_graphs(ctx):
    rep, days, sp, Dtr, Dte = ctx["rep"], ctx["days"], ctx["sp"], ctx["Dtr"], ctx["Dte"]
    f, hm, mk, post, filt, vit, hx = ctx["f"], ctx["hm"], ctx["mk"], ctx["post"], ctx["filt"], ctx["vit"], ctx["hidx"]
    D, E, DEC = rep["data"], rep["energy"], rep["decode"]
    W = rep["fhmm"]["watts_kW"]
    M = f.M
    names = [f"Device {'ABCDE'[m]} · {W[m]:.2f} kW" for m in range(M)]
    rng = np.random.default_rng(11)
    dates = Dte.dates.tolist()
    ti = dates.index(next(p["date"] for p in rep["presets"] if p["key"] == "typical"))
    typ = dates[ti]
    mins = list(range(1440))
    out = []
    add = lambda *a, **k: out.append(G(*a, **k))

    # ------------------------------------------------------------------ DATA
    add("raw-day", "Data", "All the decoder ever sees",
        f"This one line, on {typ}, is the model's entire input. No labels, no appliance list.",
        "Total household power for one unseen day, one reading per minute.",
        "Taken straight from the meter column of the dataset. Nothing is smoothed.",
        "Flat stretches are the home idling. Sudden square jumps usually mean something switched on or off.",
        "Every smart meter records a line like this. Utilities use it for billing and for grid planning.",
        [line(mins, np.round(Dte.P[ti], 3).tolist(), "Total power", INK)], lay("", "Power (kW)", xa=hour_axis()))

    dd = pd.to_datetime(days.dates)
    daily = days.P.sum(1) / 60
    split = dd < rep["split_date"]
    add("daily-kwh", "Data", "Four years, split by time",
        f"{D['complete_days']:,} complete days. Everything from {rep['split_date'][:4]} on is unseen: {D['test_days']} test days.",
        "Total energy per day across the whole dataset.",
        "Sum of each day's minute readings divided by 60. Only complete days are kept, and the split is by date, never shuffled.",
        "The yearly wave is the seasons. The model learns from the left part and is scored only on the right part.",
        "Time-ordered splits are the standard defence against leakage in forecasting and sensor work.",
        [line(dd[split].strftime("%Y-%m-%d").tolist(), np.round(daily[split], 1).tolist(), "Before the split", BLUE, 1.5),
         line(dd[~split].strftime("%Y-%m-%d").tolist(), np.round(daily[~split], 1).tolist(), "Unseen days", ORANGE, 1.5)],
        lay("", "Energy per day (kWh)", xa={"type": "date"}))

    h, e = np.histogram(Dtr.P.ravel(), bins=80, range=(0, 8))
    add("power-hist", "Data", "Power readings pile up at certain levels",
        f"Half of all minutes are at or below {D['power']['p50']:.2f} kW, yet the top reading reaches {D['power']['max']:.1f} kW.",
        "How often each power level occurs across training minutes, on a log scale.",
        "A histogram with 0.1 kW bins. The vertical axis is logarithmic so rare high loads stay visible.",
        "Bumps are levels the house often sits at. Mixed peaks like these are why one Gaussian cannot describe the data.",
        "Multi-peak histograms are the classic sign that a hidden-state or mixture model is needed.",
        [bars(np.round((e[:-1] + e[1:]) / 2, 2).tolist(), h.tolist(), "Minutes", BLUE)], lay("Power (kW)", "Minutes (log scale)", ya={"type": "log", "dtick": 1}, bargap=0))

    months = dd.month.to_numpy()
    grid = np.array([days.P[months == m].mean(0).reshape(24, 60).mean(1) for m in range(1, 13)])
    add("hour-month", "Data", "When the house uses power, by month",
        "Evening peaks are strongest in winter. The summer dip shows up in almost every hour.",
        "Average power for each hour of the day in each calendar month.",
        "Group all days by month, then average by hour.",
        "Brighter means more power. Read across for the day's shape and down for the seasons.",
        "Utilities read the same pattern to plan peak capacity and time-of-day tariffs.",
        [heat(np.round(grid, 3).tolist(), list(range(24)), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], "%{y}, %{x}:00<br>%{z:.2f} kW")],
        lay("Hour of day", "", ya={"autorange": "reversed"}))

    sh = D["shares"]
    add("answer-shares", "Data", "The answer key: where the energy really went",
        f"Sub-meters say water heater + AC used {sh['Water heater + AC'] * 100:.1f}% of all energy. The models never see these meters.",
        "Share of total energy recorded by each sub-meter across all days.",
        "Sub-meter totals divided by total energy. 'Everything else' is whatever no sub-meter covered.",
        "These are the true values that the decoder's guesses are checked against after training.",
        "Sub-metering is the expensive way to get this breakdown. Decoding tries to avoid that hardware.",
        [bars(np.round([v * 100 for v in sh.values()], 1).tolist(), list(sh.keys()), "Share", BLUE, orientation="h")],
        lay("Share of energy (%)", "", ya={"autorange": "reversed"}), 300)

    st = np.diff(Dtr.P, axis=1).ravel()
    st = st[st > 0.15]
    h, e = np.histogram(st, bins=70, range=(0.15, 3.5))
    add("step-sizes", "Data", "Every switch-on leaves a jump",
        "The sizes of upward jumps give the training its label-free starting guess for each device's wattage.",
        "How large the minute-to-minute upward jumps in total power are.",
        "Differences of consecutive readings, keeping only rises above 0.15 kW, grouped by 1-D k-means.",
        "A bump near a size suggests a device of that size. The dashed line marks the true 1.08 kW heater step, added afterwards for checking.",
        "Step detection is the oldest idea in appliance monitoring and is still a common first stage.",
        [bars(np.round((e[:-1] + e[1:]) / 2, 3).tolist(), h.tolist(), "Jumps", BLUE)],
        lay("Size of upward jump (kW)", "Count", shapes=[vline(D["heater_mode_kw"])], bargap=0))

    # ------------------------------------------------------------------ MARKOV AND HMM
    MK = rep["markov"]
    lab = [f"{MK['edges'][i]:.2f}" for i in range(MK["bins"])]
    stay = np.mean(np.diag(MK["A"]))
    add("markov-matrix", "Markov & HMM", "Markov chain: next reading given this one",
        f"The bright diagonal says the house tends to stay put: average chance of staying in a level is {stay * 100:.0f}%.",
        "A first-order Markov chain over 12 power levels. Each row is a probability distribution over next minute's level.",
        "Count each move between levels in the training days, add one to every cell, then divide each row by its total.",
        "Row = this minute, column = next minute. Brighter cells are more likely moves.",
        "The same counting estimator is used for text prediction, web-page ranking and weather-state models.",
        [heat(np.round(MK["A"], 4).tolist(), lab, lab, "from %{y} kW to %{x} kW<br>probability %{z:.3f}")],
        lay("Next minute's level (kW, lower edge)", "This minute's level (kW, lower edge)", ya={"autorange": "reversed"}), 440)

    k_stat = next((k for k, v in zip(MK["power_k"], MK["power_tv"]) if v < 0.05), None)
    add("markov-power", "Markov & HMM", "How fast the chain forgets",
        f"Raising the matrix to the power k asks where you will be k minutes on. It is within 5% of the long-run mix by k = {k_stat} minutes." if k_stat else "The chain keeps some memory across the whole range tested.",
        "Worst-case distance between the k-step distribution and the chain's stationary distribution.",
        "Compute A to the power k. Compare each row with the stationary distribution, which is the left eigenvector of A with eigenvalue 1, using total variation.",
        "Falling towards zero means the starting level no longer matters.",
        "Mixing time is a core property behind PageRank and MCMC samplers.",
        [line(MK["power_k"], np.round(MK["power_tv"], 4).tolist(), "Distance", BLUE)], lay("Minutes ahead (k)", "Distance from long-run mix", xa={"type": "log", "tickvals": [1, 3, 10, 30, 60, 120, 240, 480]}))

    hg = rep["hmm_grid"]
    best_h = next(r for r in hg if r["K"] == rep["hmm"]["K"])
    fv = rep["fhmm"]["val_ll_per_min"]
    labels = ["Time order ignored", f"Markov chain ({MK['bins']} levels)", f"Factorial HMM ({M} devices)", f"HMM ({best_h['K']} levels)"]
    vals = [MK["ll_iid_per_min"], MK["ll_per_min"], fv, best_h["val_ll_per_min"]]
    add("model-likelihood", "Markov & HMM", "Which model explains unseen days best?",
        f"The plain HMM fits best but needs {best_h['params']} parameters. The factorial model uses {rep['fhmm_grid'][M - 1]['params']}, and its parameters mean something physical.",
        "Average log density per minute on held-out training-era days, for four models. Higher is better.",
        "Each model scores the same days. Discrete models are converted to densities by dividing by bin width so they can be compared.",
        "The jump from the first bar to the second is the value of knowing the previous reading.",
        "Held-out likelihood is the standard way to compare generative models when no labels exist.",
        [bars(np.round(vals, 3).tolist(), labels, "Log density per minute", BLUE, orientation="h")], lay("Log density per minute (higher is better)", "", ya={"autorange": "reversed"}), 320)

    lags = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240]
    sim_h = simulate(hm.pi, hm.A, hm.mu, hm.var, 80, 1440, rng)
    sim_f = simulate(f.start_probs(), f.transition(), f.means(), f.var, 80, 1440, rng)
    a_real, a_h, a_f = acf(Dte.P, lags), acf(sim_h, lags), acf(sim_f, lags)
    add("autocorrelation", "Markov & HMM", "Can the models imitate the real memory?",
        f"Real power stays correlated for hours: {a_real[-5]:.2f} at one hour. The fitted HMM forgets faster ({a_h[-5]:.2f}). It has no clock, so daily routine is invisible to it.",
        "How strongly a reading predicts one k minutes later, for real data and for data generated by each trained model.",
        "Sample 80 synthetic days from each model, then compute the autocorrelation at each lag for real and generated days.",
        "A model line below the real line has less memory than the house. The gap at long lags comes from daily routine, which these models do not represent.",
        "Sampling from a fitted model and comparing statistics is a standard model check.",
        [line(lags, np.round(a_real, 3).tolist(), "Real days", INK), line(lags, np.round(a_h, 3).tolist(), f"HMM, {hm.K} levels", BLUE), line(lags, np.round(a_f, 3).tolist(), f"Factorial HMM, {M} devices", ORANGE)],
        lay("Lag (minutes)", "Autocorrelation", xa={"type": "log", "tickvals": [1, 5, 15, 30, 60, 120, 240]}))

    tmin = Dtr.P.size
    curves = []
    for i, K in enumerate((2, 4, 8, 16)):
        hist = next(r["history"] for r in rep["hmm_grid"] if r["K"] == K)
        curves.append(line(list(range(1, len(hist) + 1)), np.round(np.array(hist) / tmin, 4).tolist(), f"HMM, {K} levels", [BLUE, AQUA, VIOLET, AMBER][i]))
    fh = rep["fhmm"]["history"]
    curves.append(line(list(range(1, len(fh) + 1)), np.round(np.array(fh) / tmin, 4).tolist(), f"Factorial HMM, {M} devices", ORANGE))
    all_mono = all(r["monotone"] for r in hg) and all(r["monotone"] for r in rep["fhmm_grid"])
    add("em-curves", "Markov & HMM", "Baum-Welch only climbs",
        "EM's guarantee, visible: log-likelihood never fell across any model in this run." if all_mono else "Tolerance check found a dip in at least one curve. See the tests.",
        "Training log-likelihood per minute after each EM iteration.",
        "E-step: forward-backward gives expected state counts. M-step: re-estimate parameters from those counts. Repeat.",
        "Each curve should rise and then flatten. A fall would mean a bug in the code.",
        "Baum-Welch trains HMMs in speech recognition, gene finding and activity recognition.",
        curves, lay("EM iteration", "Log-likelihood per minute"))

    Ks = [r["K"] for r in hg]
    add("heldout-states", "Markov & HMM", "Same number of states, far fewer parameters",
        f"A {2 ** M}-state factorial model has {rep['fhmm_grid'][M - 1]['params']} parameters. A {2 ** M}-state plain HMM has {next(r['params'] for r in hg if r['K'] == 2 ** M)}.",
        "Held-out fit against the number of hidden combinations, for the plain and factorial models.",
        "Each model is trained on the same days and scored on the same held-out days.",
        "The plain HMM's transition table grows with the square of its states. The factorial model only needs two rates per device, so its fit is a little lower.",
        "Factorising a large state space is the idea behind dynamic Bayesian networks.",
        [line(Ks, np.round([r["val_ll_per_min"] for r in hg], 3).tolist(), "Plain HMM", BLUE, mode="lines+markers"),
         line([r["states"] for r in rep["fhmm_grid"]], np.round([r["val_ll_per_min"] for r in rep["fhmm_grid"]], 3).tolist(), "Factorial HMM", ORANGE, mode="lines+markers")],
        lay("Hidden combinations", "Held-out log density per minute", xa={"type": "log", "tickvals": [2, 4, 8, 16, 32]}))

    add("bic", "Markov & HMM", "BIC keeps asking for more levels",
        f"BIC is lowest at the largest size tried, {rep['hmm']['K']} levels. With this much data its penalty barely bites.",
        "Bayesian Information Criterion for plain HMMs of different sizes.",
        "BIC = -2 x log-likelihood + parameters x log(number of readings). Lower is better.",
        "A curve that keeps falling means the criterion has not found a natural size.",
        "BIC and AIC are the default size selectors for mixture models and HMMs.",
        [line(Ks, np.round([r["bic"] for r in hg], 0).tolist(), "BIC", BLUE, mode="lines+markers")], lay("Levels in the plain HMM", "BIC (lower is better)"))

    xs = np.linspace(0, 6, 300)
    stat = np.linalg.matrix_power(hm.A, 2000)[0]
    comp = stat[:, None] * np.exp(-0.5 * (xs - hm.mu[:, None]) ** 2 / hm.var[:, None]) / np.sqrt(2 * np.pi * hm.var[:, None])
    hh, e = np.histogram(Dtr.P.ravel(), bins=120, range=(0, 6), density=True)
    tr = [bars(np.round((e[:-1] + e[1:]) / 2, 3).tolist(), np.round(hh, 4).tolist(), "Real readings", "#2a3641"),
          line(np.round(xs, 3).tolist(), np.round(comp.sum(0), 4).tolist(), "HMM mixture", ORANGE)]
    add("emissions", "Markov & HMM", "What the plain HMM's levels look like",
        f"{hm.K} Gaussians, weighted by how often each state occurs, reproduce the shape of the readings.",
        "The real distribution of power against the distribution implied by the trained HMM.",
        "Weights come from the stationary distribution of A. Each state contributes one Gaussian with its learned mean and variance.",
        "A close match means the emissions fit. Note that no state here means 'heater': each is a level of the whole house.",
        "Gaussian emissions are the textbook choice for continuous HMM observations.",
        tr, lay("Power (kW)", "Density", bargap=0))

    hlab = [f"{v:.2f}" for v in hm.mu]
    add("hmm-matrix", "Markov & HMM", "The plain HMM's learned transition table",
        f"A {hm.K} x {hm.K} table: {hm.K * (hm.K - 1)} free numbers, none of which tells you what any appliance is doing.",
        "Learned transition probabilities between the plain HMM's states, ordered by mean power.",
        "Baum-Welch re-estimates each cell as expected moves from i to j divided by expected visits to i.",
        "A bright diagonal means states last. Off-diagonal cells show which levels connect.",
        "Reading transition tables is how analysts interpret fitted HMMs.",
        [heat(np.round(hm.A[np.argsort(hm.mu)][:, np.argsort(hm.mu)], 4).tolist(), sorted(hlab, key=float), sorted(hlab, key=float), "from %{y} kW to %{x} kW<br>%{z:.3f}", zmin=0, zmax=0.5)],
        lay("To state (mean kW)", "From state (mean kW)", ya={"autorange": "reversed"}), 460)

    ton = np.where(Dte.H[ti] >= 5)[0]
    s0 = max(0, (int(ton[0]) if len(ton) else 360) - 30)
    s1 = min(1440, s0 + 360)
    wmin = list(range(s0, s1))
    add("filter-smooth", "Markov & HMM", "Live estimate against the whole-day estimate",
        f"Using the future changes F1 only from {DEC['filtered']['f1']:.3f} to {DEC['smoothed']['f1']:.3f}, but cuts on/off flicker from {DEC['flips']['filtered']:.0f} to {DEC['flips']['smoothed']:.0f} flips a day.",
        "The heater chain's probability of being on, from the forward pass alone and from forward-backward.",
        "Filtering uses readings up to now, alpha. Smoothing also uses later readings, gamma, which is proportional to alpha x beta.",
        "The whole-day line is steadier because later readings can overrule a brief blip.",
        "Live monitoring can only filter. Overnight billing analysis can smooth.",
        [line(wmin, np.round(filt[ti, s0:s1, hx], 3).tolist(), "Live (past only)", BLUE), line(wmin, np.round(post[ti, s0:s1, hx], 3).tolist(), "Whole day", ORANGE),
         line(wmin, (Dte.H[ti, s0:s1] >= 5).astype(int).tolist(), "Answer key", REF, 1, "dot")],
        lay("", "Probability heater on", xa={**hour_axis(), "tickvals": list(range(s0 - s0 % 60, s1, 60)), "ticktext": [f"{(v // 60) % 24:02d}:00" for v in range(s0 - s0 % 60, s1, 60)]}))

    rows = [(Dte.H[ti] >= 5), post[ti, :, hx] > .5, vit[ti, :, hx] > 0, filt[ti, :, hx] > .5]
    ylab = ["Answer key", "Whole-day probability", "Viterbi path", "Live estimate"]
    add("decoding-compare", "Markov & HMM", "Four ways to tell the same day's story",
        f"Across unseen days, Viterbi and minute-by-minute decoding score the same F1 ({DEC['viterbi']['f1']:.3f} and {DEC['smoothed']['f1']:.3f}). The real device flips {DEC['flips']['truth']:.0f} times a day.",
        "When the heater chain is judged on, by each method, against the sub-meter.",
        "Whole-day probability takes the most likely state each minute. Viterbi finds the single most likely path in the log domain. Live uses only past readings.",
        "Orange means on. Compare each row with the top row.",
        "Viterbi decodes phone signals, speech and DNA. Posterior decoding is preferred when per-position accuracy matters.",
        [heat(np.array(rows, int).tolist(), mins, ylab, "%{y}, minute %{x}<br>on = %{z}", [[0, "#161c23"], [1, ORANGE]], 0, 1, False, xgap=0)],
        lay("", "", xa=hour_axis(), ya={"autorange": "reversed"}), 300)

    mp = rep["message_passing"]
    nl = mp["naive_log10"]
    broke = mp["naive_breaks_at"]
    scaled = mp["scaled_log10"]
    add("numeric-range", "Markov & HMM", "Why the forward pass must be scaled",
        f"Left unscaled, the running product leaves the range of a floating-point number ({mp['naive_direction']}) at minute {broke}. Scaled and log-domain answers agree to {mp['ll_diff']:.1e}." if broke is not None else "On this day the unscaled product stayed in range.",
        "Size of the unscaled forward product over one real day, next to the log-likelihood from the scaled recursion.",
        "Unscaled: multiply as the textbook writes it. Scaled: normalise alpha each minute, keep the factor c, and use log p(x) = sum of log c.",
        "The line stops where a float gives up. Narrow Gaussians have densities above 1, so here it runs away upward, not down to zero.",
        "Every practical HMM library scales or works in logs for this reason.",
        [line(list(range(len(scaled))), np.round(scaled, 2).tolist(), "Scaled (log10 of p)", BLUE), line(list(range(len(nl))), np.round(nl, 2).tolist(), "Unscaled product (log10)", ORANGE)],
        lay("", "log10 of probability density", xa=hour_axis()))

    g_, m_ = np.array(mp["gamma"]), np.array(mp["marg"])
    pick = rng.choice(len(g_), min(1200, len(g_)), replace=False)
    add("sum-product", "Markov & HMM", "Sum-product is forward-backward",
        f"General message passing on the factor graph matches forward-backward to {mp['max_gamma_diff']:.1e}. Max-product returned the identical Viterbi path: {mp['same_viterbi_path']}.",
        "Each state marginal from the general algorithm against the same marginal from forward-backward, on one unseen day.",
        "Treat the HMM as a chain-shaped factor graph, then pass sum messages (marginals) or max messages (best path) along its edges.",
        "Every point on the diagonal means the two methods agree.",
        "The same rule powers error-correcting codes, Kalman filters and belief propagation.",
        [dots(np.round(g_[pick], 5).tolist(), np.round(m_[pick], 5).tolist(), "State marginals", BLUE, 6), line([0, 1], [0, 1], "Perfect agreement", REF, 1, "dash")],
        lay("Forward-backward", "Sum-product"))

    # ------------------------------------------------------------------ DECODER
    fg = rep["fhmm_grid"]
    add("chain-count", "Decoder", "How many devices should it look for?",
        f"It keeps {M}: the gain from one more chain drops by over {100 * (1 - rep['chain_rule']):.0f}% and fitting takes about {fg[-1]['fit_seconds'] / fg[M - 1]['fit_seconds']:.0f} times longer. Answer-key scores play no part in this choice.",
        "Held-out fit as chains are added.",
        "Train with 1 to 5 chains. Stop when the next chain adds under 20% of the previous chain's gain.",
        "Look for where the curve flattens. Likelihood keeps rising, so a stopping rule has to be chosen.",
        "Choosing latent structure by held-out likelihood is common wherever labels are unavailable.",
        [line([r["M"] for r in fg], np.round([r["val_ll_per_min"] for r in fg], 3).tolist(), "Held-out fit", BLUE, mode="lines+markers")],
        lay("Hidden on/off chains", "Held-out log density per minute", xa={"dtick": 1}, shapes=[vline(M)]))

    R = rep["restarts"]
    close = sum(any(abs(w - D["heater_mode_kw"]) / D["heater_mode_kw"] < .03 for w in r["watts"]) for r in R)
    kept = next(r for r in R if r["kept"])
    other = [r for r in R if not r["kept"]]
    add("restarts", "Decoder", "Different starting guesses, same answer?",
        f"{close} of {len(R)} runs put a device within 3% of the true {D['heater_mode_kw']:.2f} kW. Runs that did not landed on a much lower likelihood, and that is what rejects them.",
        "Wattages learned from different starting guesses.",
        "Start EM from the jump sizes, and from six random scalings of them. Keep the run with the highest training likelihood.",
        "Dots stacked in columns mean runs agree. The dashed line is the true heater, added only for checking.",
        "EM finds local optima, so random restarts are standard practice.",
        [dots([w for r in other for w in r["watts"]], [r["label"] for r in other for _ in r["watts"]], "Other runs", BLUE, 10),
         dots(kept["watts"], [kept["label"]] * len(kept["watts"]), "Kept run", ORANGE, 12)],
        lay("Learned device power (kW)", "", ya={"autorange": "reversed"}, shapes=[vline(D["heater_mode_kw"])]), 360)

    stack = [{"type": "scatter", "mode": "none", "x": mins, "y": [round(f.base, 3)] * 1440, "stackgroup": "a", "name": f"Always on · {f.base:.2f} kW", "fillcolor": "#2a3641", "line": {"width": 0}}]
    for m in range(M):
        stack.append({"type": "scatter", "mode": "none", "x": mins, "y": np.round(post[ti, :, m] * W[m], 3).tolist(), "stackgroup": "a", "name": names[m], "fillcolor": DEV[m], "line": {"width": 0}})
    stack.append(line(mins, np.round(Dte.P[ti], 3).tolist(), "Real total", INK, 1))
    add("stacked-day", "Decoder", "One line, unmixed",
        f"The coloured layers are the model's explanation of {typ}. The white line is the total it was given.",
        "Each discovered device's expected power through a typical unseen day.",
        "P(device on) from forward-backward, times that device's learned wattage, stacked on the always-on level.",
        "Where layers follow the white line, the story explains the reading. Short spikes it does not cover are devices it has no chain for.",
        "This is the output of non-intrusive load monitoring. Companies sell it as an energy report.",
        stack, lay("", "Power (kW)", xa=hour_axis()), 420)

    add("answer-key-day", "Decoder", "Opening the answer key",
        "The filled area is the real sub-meter, which the model never saw. The line is the chain it found alone.",
        "The heater sub-meter against the matching chain for one typical day.",
        "Sub-meter Wh per minute x 60 / 1000 gives kW. The chain's line is P(on) x learned wattage.",
        "Matching blocks mean correct detection. Gaps between the two are real mistakes, left visible.",
        "Blind validation like this separates a real discovery from a model that merely fits.",
        [{"type": "scatter", "mode": "lines", "x": mins, "y": np.round(Dte.H[ti] * 60 / 1000, 3).tolist(), "name": "Sub-meter (answer key)", "fill": "tozeroy", "fillcolor": "rgba(169,180,190,.25)", "line": {"color": REF, "width": 1}},
         line(mins, np.round(post[ti, :, hx] * W[hx], 3).tolist(), names[hx], ORANGE)], lay("", "Power (kW)", xa=hour_axis()))

    hv = Dte.H[Dte.H >= 5]
    uv, uc = np.unique(np.round(hv), return_counts=True)
    add("found-vs-true", "Decoder", f"It found {W[hx]:.3f} kW. The real one is {D['heater_mode_kw']:.2f} kW.",
        f"Learned from the total alone: {W[hx]:.3f} kW. The sub-meter's most common on-power: {D['heater_mode_kw']:.2f} kW.",
        "The true distribution of heater-meter power when on, with the model's learned wattage marked.",
        "Sub-meter readings come in steps of 60 W, so the bars are spaced. The dashed line is the chain's wattage from EM.",
        "The line should sit on the tallest bar, and it does. This is the project's central check.",
        "Recovering a physical constant from indirect data is the strongest evidence a latent model is real.",
        [bars(np.round(uv * 60 / 1000, 2).tolist(), uc.tolist(), "Minutes at this power", BLUE)], lay("True sub-meter power (kW)", "Minutes", shapes=[vline(W[hx], ORANGE)], bargap=.2), 340)

    cross = DEC["cross_f1"]
    add("cross-f1", "Decoder", "Does each chain match a real meter?",
        f"Only the heater gets a clean match, F1 {cross['H'][hx]:.2f}. Kitchen and laundry peak at {max(cross['K']):.2f} and {max(cross['L']):.2f}: those appliances stay mixed together.",
        "F1 score of each chain against each sub-meter on unseen days.",
        "Mark a chain 'on' when P(on) > 0.5 and compare with the sub-meter's on/off state. Chains are matched to meters on validation days, not test days.",
        "One bright cell per row would mean clean separation. The bottom two rows show where this fails.",
        "Blind source separation studies report the same kind of confusion table.",
        [heat([np.round(cross[k], 3).tolist() for k in ("H", "K", "L")], [f"Device {'ABCDE'[m]}" for m in range(M)], ["Water heater + AC", "Kitchen", "Laundry room"], "%{y} vs %{x}<br>F1 %{z:.2f}", zmin=0, zmax=1)],
        lay("", "", ya={"autorange": "reversed"}), 320)

    Mo = rep["monthly"]
    add("monthly", "Decoder", "Heater energy by month: guess against truth",
        f"Over {D['test_days']} unseen days the model estimated {E['estimated_kwh']:,.0f} kWh against {E['true_kwh']:,.0f} kWh real ({E['error_pct']:+.1f}%). Monthly correlation r = {Mo['r']:.2f}.",
        "Average heater energy per day for each unseen month.",
        "Estimated = sum of P(on) x wattage / 60. True = sub-meter total / 1000.",
        "The two lines should rise and fall together. The gap between them is the bias.",
        "Monthly appliance breakdowns are what utility energy reports show customers.",
        [line(Mo["month"], np.round(Mo["true_per_day"], 2).tolist(), "Sub-meter (answer key)", REF), line(Mo["month"], np.round(Mo["est_per_day"], 2).tolist(), "Decoded with no labels", ORANGE)],
        lay("", "kWh per day"))

    est_d = (post[..., hx] * W[hx]).sum(1) / 60
    tru_d = Dte.H.sum(1) / 1000
    add("daily-scatter", "Decoder", "Heater energy, one dot per day",
        f"Day-level correlation r = {E['daily_r']:.2f}, with an average miss of {E['daily_mae_kwh']:.1f} kWh a day.",
        "Each unseen day's estimated heater energy against its true value.",
        "Same formulas as the monthly chart, without averaging.",
        "Dots on the dashed line are perfect. A cloud below it means the model under-counts.",
        "Parity plots are the standard way to inspect a regression-type estimate.",
        [dots(np.round(tru_d, 2).tolist(), np.round(est_d, 2).tolist(), "Unseen days", ORANGE, 6, opacity=.7), line([0, float(tru_d.max())], [0, float(tru_d.max())], "Perfect estimate", REF, 1, "dash")],
        lay("True heater energy (kWh)", "Estimated (kWh)"))

    dk = E["device_kwh"]
    bl = [f"Always on · {f.base:.2f} kW"] + names + ["Unexplained"]
    bv = [E["base_kwh"]] + dk + [E["residual_kwh"]]
    add("bill", "Decoder", "The bill, split by discovered device",
        f"The device matched to the heater takes {E['est_share'] * 100:.1f}% of the total. The sub-meter says {E['true_share'] * 100:.1f}%.",
        "Share of total energy attributed to each discovered device on unseen days.",
        "Device kWh = sum of P(on) x wattage / 60. Unexplained = total minus everything the model accounts for.",
        "This is what the app turns into rupees. The model does not name devices; the answer key names one of them after training.",
        "The appliance-level bill is the main consumer use of load monitoring.",
        [bars(np.round(100 * np.array(bv) / E["total_kwh"], 1).tolist(), bl, "Share of total", BLUE, orientation="h")],
        lay("Share of total energy (%)", "", ya={"autorange": "reversed"}), 340)

    Dd = rep["durations"]

    def ccdf(hist):
        c = np.array(hist, float)
        return (c[::-1].cumsum()[::-1] / c.sum())
    n = list(range(1, 181))
    add("durations", "Decoder", "Where the HMM's assumption breaks",
        f"An HMM can only produce a straight line here. Real on-runs average {Dd['truth_mean']:.0f} minutes and the chain believes {Dd['model_mean']:.0f}.",
        "Chance that a heater on-run lasts at least n minutes: real, decoded, and implied by the model's switching rate.",
        "Measure run lengths from the on/off sequences. The model's line is (1 - p_off) to the power n.",
        "On a log axis a geometric distribution is straight. Bends away from it are real behaviour the model cannot express.",
        "Hidden semi-Markov models exist to fix exactly this.",
        [line(n, np.round(ccdf(Dd["truth"])[1:181], 5).tolist(), "Sub-meter runs", REF), line(n, np.round(ccdf(Dd["decoded"])[1:181], 5).tolist(), "Decoded runs", ORANGE),
         line(n, np.round((1 - Dd["p_off"]) ** (np.array(n) - 1), 5).tolist(), "What the model assumes", BLUE, 2, "dash")],
        lay("Run length n (minutes)", "Chance of lasting at least n", ya={"type": "log", "range": [-2, 0], "dtick": 1}))

    pf = rep["per_day_f1"]
    h, e = np.histogram([v for v in pf["values"] if v is not None], bins=25, range=(0, 1))
    add("f1-spread", "Decoder", "Not every day is a good day",
        f"Median day F1 {pf['median']:.2f}, worst tenth {pf['p10']:.2f}, best tenth {pf['p90']:.2f}. The app's bad-day button shows that low end.",
        "Distribution of the heater F1 score, computed one day at a time.",
        "Score each unseen day separately. Days when neither the model nor the meter sees the heater on are left out.",
        "A long left tail means some days are poorly explained.",
        "Reporting a spread stops a single average from hiding failures.",
        [bars(np.round((e[:-1] + e[1:]) / 2, 3).tolist(), h.tolist(), "Days", BLUE)], lay("F1 for that day", "Days", bargap=.05, shapes=[vline(pf["median"])]), 320)

    Wk = rep["week"]
    wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hrs = list(range(24))
    add("week-decoded", "Decoder", "A week, seen only through the meter",
        f"The {Wk['kw']:.2f} kW chain's midday use rises from {Wk['midday_weekday'] * 100:.0f}% of minutes on weekdays to {Wk['midday_weekend'] * 100:.0f}% at weekends.",
        "How often the largest non-heater chain is on, per hour and weekday.",
        "Average P(on) > 0.5 over all unseen days, grouped by weekday and hour.",
        "Bright cells are when the home is busy. Nobody told the model what a weekend is.",
        "This is the privacy concern raised in smart-meter research.",
        [heat(np.round(Wk["decoded"], 3).tolist(), hrs, wd, "%{y} %{x}:00<br>on %{z:.0%} of minutes")], lay("Hour of day", "", ya={"autorange": "reversed"}), 330)
    add("week-truth", "Decoder", "The same week, from the real kitchen and laundry meters",
        f"Pattern correlation with the decoded chain is r = {Wk['r_decoded']:.2f}. Raw total power scores r = {Wk['r_total']:.2f}, so the leak is in the meter data itself, not in this model.",
        "How often the kitchen or laundry sub-meter is on, per hour and weekday.",
        "Same grouping as the previous chart, using the sub-meters.",
        "Compare the shapes, not the brightness. The evening peak and weekend midday show in both.",
        "Comparisons like this are why meter data is treated as personal data in several countries.",
        [heat(np.round(Wk["truth"], 3).tolist(), hrs, wd, "%{y} %{x}:00<br>on %{z:.0%} of minutes")], lay("Hour of day", "", ya={"autorange": "reversed"}), 330)

    Fc = rep["forecast"]
    pal = [REF, MAGENTA, BLUE, AQUA, ORANGE]
    tr = [line(Fc["horizons"], np.round(v, 3).tolist(), k, pal[i], mode="lines+markers") for i, (k, v) in enumerate((k, v) for k, v in Fc.items() if k != "horizons")]
    add("forecast", "Decoder", "Predicting the next reading",
        f"One minute ahead, nothing beats copying the last reading (error {Fc['Persistence (last reading)'][0]:.3f} kW). By 15 minutes the HMM leads ({Fc[f'HMM, {hm.K} levels'][4]:.3f} against {Fc['Persistence (last reading)'][4]:.3f}).",
        "Prediction error against how far ahead the model looks.",
        "From the filtered state alpha at time t, the k-step prediction is alpha x A^k x means. Compare with what actually happened.",
        "Lower is better. Lines meeting the 'average day' line at long range means the current state has been forgotten.",
        "Short-term load forecasting uses the same logic for grid balancing.",
        tr, lay("Minutes ahead", "Error (RMSE, kW)", xa={"type": "log", "tickvals": [1, 5, 15, 30, 60, 120, 240]}))

    # ------------------------------------------------------------------ KALMAN
    Kf = rep["kalman"]
    y, m_, v_ = np.array(Kf["y"], float), np.array(Kf["filter"]), np.array(Kf["filter_var"])
    y_list = [None if not np.isfinite(a) else round(float(a), 2) for a in y]
    add("kalman-level", "Kalman", "The slow trend under the daily noise",
        f"Learned noise: process q = {Kf['q']:.2f}, reading r = {Kf['r']:.1f}. Reading noise dominates, so the filter moves only {Kf['gain_final'] * 100:.0f}% of the way towards each new day.",
        "Daily energy with the Kalman filter's level and the backward-smoothed level.",
        "Local-level model: level follows a random walk, and each day is level plus noise. EM with the RTS smoother learns q and r.",
        "Grey is one noisy day at a time. The filter line lags slightly. The smoother uses later days, so it does not lag.",
        "Kalman filters run in navigation, tracking, economics and sensor fusion.",
        [line(Kf["dates"], y_list, "Daily energy", "#3b4752", 1), line(Kf["dates"], np.round(m_, 2).tolist(), "Live level (filter)", BLUE), line(Kf["dates"], np.round(Kf["smooth"], 2).tolist(), "Hindsight level (smoother)", ORANGE)],
        lay("", "Energy per day (kWh)", xa={"type": "date"}), 420)

    nan = np.isnan(y).astype(int)
    d = np.diff(np.r_[0, nan, 0])
    st_, en_ = np.where(d == 1)[0], np.where(d == -1)[0]
    i = int(np.argmax(en_ - st_))
    a, b = max(0, int(st_[i]) - 40), min(len(y), int(en_[i]) + 40)
    x_ = Kf["dates"][a:b]
    up, lo = (m_[a:b] + 2 * np.sqrt(v_[a:b])), (m_[a:b] - 2 * np.sqrt(v_[a:b]))
    add("kalman-gap", "Kalman", "Missing days widen the band on their own",
        f"A {int(en_[i] - st_[i])}-day outage: with no readings the model only predicts, so its uncertainty grows until data returns. {Kf['missing_days']} days are missing or incomplete in total.",
        "Filter level with a two-standard-deviation band around the longest gap.",
        "On a missing day, run the predict step (variance + q) and skip the update.",
        "A wider band means less certainty. It snaps back within days of data returning.",
        "Handling dropouts without special cases is a major practical advantage of state-space models.",
        [line(x_, np.round(up, 2).tolist(), "Upper band", "#2c6cba", 1), line(x_, np.round(lo, 2).tolist(), "Lower band", "#2c6cba", 1, fill="tonexty", fillcolor="rgba(57,135,229,.18)"),
         line(x_, np.round(m_[a:b], 2).tolist(), "Live level", BLUE), dots(x_, y_list[a:b], "Daily energy", INK2, 6)],
        lay("", "Energy per day (kWh)", xa={"type": "date"}))

    g = np.array(Kf["gain"])[:90]
    g_list = [None if v == 0 else round(float(v), 4) for v in g]
    add("kalman-gain", "Kalman", "The gain settles quickly",
        f"It settles at {Kf['gain_final']:.3f}. In steady state this filter behaves like a moving average with a learned weight, which is why it shows no speed advantage in the slow-decline test.",
        "Kalman gain over the first 90 days.",
        "Gain = predicted variance / (predicted variance + r). Breaks in the line are missing days, which have no update.",
        "A large gain trusts new data while the level is uncertain. It flattens quickly, and jumps briefly after a gap because uncertainty grew.",
        "Steady-state gains make Kalman filters cheap enough for embedded hardware.",
        [line(list(range(len(g))), g_list, "Gain", BLUE)], lay("Day", "Kalman gain"), 320)

    add("kalman-em", "Kalman", "EM learns the two noise levels",
        "The likelihood rises at every step, which is checked by a test in the repository.",
        "Training log-likelihood after each EM iteration for the linear dynamical system.",
        "E-step: filter and RTS smoother. M-step: closed-form updates for q and r.",
        "Rises and flattens. It climbs slowly near the top, which is why training runs to convergence instead of a fixed short count. A dip would indicate a bug.",
        "EM for state-space models is used whenever noise levels cannot be measured directly.",
        [line(list(range(1, len(Kf["em_history"]) + 1)), np.round(Kf["em_history"], 2).tolist(), "Log-likelihood", BLUE)], lay("EM iteration", "Log-likelihood"), 320)

    kf = Kf["forecast"]
    add("kalman-forecast", "Kalman", "Predicting tomorrow's total",
        f"Kalman {kf['Kalman level']:.2f} kWh against a plain 7-day average {kf['7-day average']:.2f} kWh: effectively level. Both clearly beat 'same as yesterday' at {kf['Yesterday']:.2f}.",
        "One-day-ahead error on unseen days.",
        "Kalman uses the predicted level. The others use the rule named on each bar.",
        "Lower is better.",
        "These simple rules are the benchmarks any forecasting model should have to beat.",
        [bars(list(kf.keys()), np.round(list(kf.values()), 2).tolist(), "RMSE", BLUE)], lay("", "Error (RMSE, kWh)"), 320)

    dc = Kf["decline"]
    add("decline", "Kalman", "Tried and not better: spotting a slow decline",
        f"Injected 40% decline over 8 weeks: Kalman alarms at {dc['Kalman level']['median_days']:.0f} days, 7-day average at {dc['7-day average']['median_days']:.0f}. False alarms are high for both ({dc['Kalman level']['false_alarm_rate'] * 100:.0f}% and {dc['7-day average']['false_alarm_rate'] * 100:.0f}%) because seasons also shift use.",
        "Median days until each method raises an alarm for a simulated slow decline.",
        "Reduce consumption gradually from a chosen day. Raise an alarm when the signal falls 3 standard deviations below the previous 60 days. The decline is simulated, not real.",
        "Shorter is better. The bars are close, so the Kalman filter gives no real detection advantage here.",
        "Change-point monitoring would need a seasonal model first. This is reported as a negative result.",
        [bars(list(dc.keys()), [dc[k]["median_days"] for k in dc], "Median days to alarm", BLUE)], lay("", "Median days to alarm"), 320)

    # ------------------------------------------------------------------ RIVALS
    Rv = rep["rivals"]
    Rd = rep["rival_detail"]
    srt = sorted(Rv, key=lambda r: r["f1"])
    order = [r["name"] for r in srt]
    noL = [r for r in srt if r["labels"] == "none"]
    yesL = [r for r in srt if r["labels"] != "none"]
    best = max(Rv, key=lambda r: r["f1"])
    bestn = max(noL, key=lambda r: r["f1"])
    base = max((r for r in noL if r["family"] == "Rule"), key=lambda r: r["f1"])
    hl = {"margin": {"l": 240, "r": 18, "t": 26, "b": 48}}
    two = lambda key, mul=1, nd=3: [bars([round(r[key] * mul, nd) for r in noL], [r["name"] for r in noL], "No labels", ORANGE, orientation="h"),
                                    bars([round(r[key] * mul, nd) for r in yesL], [r["name"] for r in yesL], "Needs a sub-meter", BLUE, orientation="h")]
    cat = {"categoryorder": "array", "categoryarray": order}
    add("rivals-f1", "Rivals", "Is the heater on this minute? Everyone's score",
        f"With labels, {best['name']} wins at {best['f1']:.3f}. With no labels, the decoder scores {bestn['f1']:.3f} against {base['f1']:.3f} for the best threshold. Labels help, and they require extra hardware.",
        f"F1 on {D['test_days']} unseen days for every method.",
        "All methods predict on/off each minute on the same days. Thresholds and trees learn on training-era days only.",
        "Orange methods see only the total. Blue methods also learned from the real sub-meter.",
        "Benchmarks against simple baselines stop complex models from being adopted without reason.",
        two("f1"), lay("F1 score", "", ya=cat, barmode="overlay", xa={"range": [0, 1]}, **hl), 460)

    add("rivals-pr", "Rivals", "Precision and recall",
        "The no-label thresholds are precise but miss most of the heater's minutes. The decoder keeps the precision and finds far more of them.",
        "Each method's precision against its recall on unseen days.",
        "Precision = correct 'on' calls / all 'on' calls. Recall = correct 'on' calls / all real 'on' minutes.",
        "Top right is best. Hover a dot for its method.",
        "Precision and recall matter when 'on' is the minority, as it is here.",
        [dots([round(r["recall"], 3) for r in noL], [round(r["precision"], 3) for r in noL], "No labels", ORANGE, 11, text=[r["name"] for r in noL], hovertemplate="%{text}<br>recall %{x}, precision %{y}<extra></extra>"),
         dots([round(r["recall"], 3) for r in yesL], [round(r["precision"], 3) for r in yesL], "Needs a sub-meter", BLUE, 11, text=[r["name"] for r in yesL], hovertemplate="%{text}<br>recall %{x}, precision %{y}<extra></extra>")],
        lay("Recall", "Precision", xa={"range": [0, 1.02]}, ya={"range": [0.5, 1.02]}))

    add("rivals-flips", "Rivals", "Sequence models do not flicker",
        f"The real heater switches {DEC['flips']['truth']:.0f} times a day. CART, looking at each minute on its own, reports {next(r['flips_per_day'] for r in Rv if r['family'] == 'Tree'):.0f}. The whole-day decoder reports {DEC['flips']['smoothed']:.0f}.",
        "On/off changes per day for each method.",
        "Count how often the predicted state changes, averaged across days.",
        "Closer to the dashed line is more realistic. Too many means phantom switching.",
        "Hidden Markov smoothing is often added after per-frame classifiers for this reason.",
        two("flips_per_day", 1, 1), lay("Changes per day", "", ya=cat, barmode="overlay", shapes=[vline(DEC["flips"]["truth"])], **hl), 460)

    add("rivals-energy", "Rivals", "Energy total, per method",
        f"The decoder under-counts by {abs(bestn['error_pct']):.1f}%. Rule-based energy estimates can look close while being wrong minute by minute.",
        "Percentage error in total heater energy on unseen days.",
        "Estimated total against the sub-meter total. Rivals using on/off multiply on-minutes by the average on-power seen in training.",
        "Closer to zero is better, but errors can cancel, so read this next to the F1 chart.",
        "Billing estimates care about the total, while control systems care about timing.",
        two("error_pct", 1, 1), lay("Energy error (%)", "", ya=cat, barmode="overlay", **hl), 460)

    bd = Rd["budget"]
    xn = [b["days"] for b in bd]
    overtake = next((b["days"] for b in bd if b["f1"] > bestn["f1"]), None)
    add("label-budget", "Rivals", "How many labelled days until a tree wins?",
        f"A decision tree passes the decoder with about {overtake} {'day' if overtake == 1 else 'days'} of real sub-meter labels. That is the honest price of 'no labels': zero hardware, some accuracy." if overtake else "The tree did not pass the decoder in the tested range.",
        "CART's F1 on unseen days against how many days of sub-meter labels it trained on.",
        "Random subsets of training days, four repeats each, averaged. The dashed line is the decoder, which uses none.",
        "Where the line crosses the dashes is where labelling starts to pay off. It would need redoing for every home.",
        "Learning curves like this guide decisions on how much labelling to pay for.",
        [line(xn, np.round([b["f1"] for b in bd], 3).tolist(), "CART, with labels", BLUE, mode="lines+markers")],
        lay("Labelled days given to CART", "F1 on unseen days", xa={"type": "log", "tickvals": xn}, shapes=[hline(bestn["f1"], ORANGE)], annotations=[note(0.3, bestn["f1"], "Decoder, no labels", yshift=12, xanchor="left")]))

    dep = Rd["depth_curve"]
    add("cart-depth", "Rivals", "Tree depth and overfitting",
        f"Validation peaks at depth {Rd['best_depth']}, which is what gets used on the unseen days.",
        "Training and validation F1 as the tree is allowed to grow deeper.",
        "Grow by Gini impurity, choosing the best split over quantile-binned features.",
        "A widening gap between the lines means the tree is memorising training minutes.",
        "Depth is CART's main complexity control.",
        [line([r["depth"] for r in dep], np.round([r["train_f1"] for r in dep], 3).tolist(), "Training", BLUE, mode="lines+markers"), line([r["depth"] for r in dep], np.round([r["val_f1"] for r in dep], 3).tolist(), "Validation", ORANGE, mode="lines+markers")],
        lay("Maximum depth", "F1"), 340)

    rd = list(range(1, len(Rd["ada_val"]) + 1))
    add("ada-rounds", "Rivals", "AdaBoost, round by round",
        f"Reaches F1 {Rd['ada_val'][-1]:.3f} on validation after {len(rd)} small trees.",
        "Boosted ensemble F1 as weak learners are added.",
        "Each round reweights the minutes the ensemble got wrong, fits a depth-2 tree, and adds it with weight alpha = 0.5 ln((1-err)/err).",
        "Rising lines mean each tree adds something. A falling validation line would mean overfitting.",
        "Boosting and its descendants lead on many tabular problems.",
        [line(rd, np.round(Rd["ada_train"], 4).tolist(), "Training", BLUE), line(rd, np.round(Rd["ada_val"], 4).tolist(), "Validation", ORANGE)], lay("Boosting round", "F1"), 340)

    add("ada-alpha", "Rivals", "How much each small tree gets to vote",
        "Early trees get the biggest votes. Later ones focus on the hard minutes and count for less.",
        "The weight alpha assigned to each weak learner.",
        "Alpha is computed from each tree's weighted error, so a more accurate tree gets a larger vote.",
        "A falling curve is normal.",
        "This additive weighting is AdaBoost's defining step.",
        [line(rd, np.round(Rd["ada_alphas"], 4).tolist(), "Alpha", BLUE)], lay("Boosting round", "Tree weight (alpha)"), 300)

    imp = Rd["importance"][:9]
    add("importance", "Rivals", "What the tree looks at",
        f"Its most-used features are '{imp[0][0]}' ({imp[0][1] * 100:.0f}%) and '{imp[1][0]}' ({imp[1][1] * 100:.0f}%).",
        "Share of the tree's total impurity reduction attributed to each feature.",
        "Add up the weighted Gini gain of every split on each feature, then normalise.",
        "A long bar means the tree relies on that feature.",
        "Impurity importance is a quick way to inspect tree models, though it can favour high-cardinality features.",
        [bars([round(v, 4) for _, v in imp], [k for k, _ in imp], "Importance", BLUE, orientation="h")], lay("Share of impurity reduction", "", ya={"autorange": "reversed"}, margin={"l": 190, "r": 18, "t": 26, "b": 48}), 340)

    tv = Rd["threshold_values"]
    add("threshold-sweep", "Rivals", "A threshold needs to know the answer",
        f"The good threshold is around {tv['tuned']:.2f} kW. Without labels, 2-means and Otsu land near {tv['two_means']:.2f} kW and score poorly.",
        "F1 on unseen days for every possible threshold on total power.",
        "Sweep thresholds and score each one. Thresholds used by the rivals were chosen on training days, not read from this curve.",
        "The curve shows how much depends on picking the right cut-off, which needs labels.",
        "Threshold rules are the simplest possible baseline for any detector.",
        [line(np.round(Rd["thresholds_grid"], 2).tolist(), np.round(Rd["thresholds_f1"], 3).tolist(), "F1 at this threshold", BLUE)],
        lay("Threshold on total power (kW)", "F1", shapes=[vline(tv["tuned"], BLUE), vline(tv["two_means"], ORANGE)],
            annotations=[note(tv["tuned"], 1.02, "Tuned with labels", yref="paper"), note(tv["two_means"], 1.02, "2-means, no labels", yref="paper")]), 360)

    c = confusion(post[..., hx] > .5, Dte.H >= 5)
    add("confusion", "Rivals", "Decoder's mistakes, counted",
        f"Misses {c['fn']:,} heater minutes and raises {c['fp']:,} false ones, out of {sum(c.values()):,} minutes.",
        "Confusion matrix for the heater chain on unseen days.",
        "Compare P(on) > 0.5 with the sub-meter for every minute.",
        "The diagonal is correct. Misses (bottom left) outnumber false alarms (top right), which matches the energy under-count.",
        "The basic summary behind precision, recall and F1.",
        [heat([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], ["Decoded off", "Decoded on"], ["Really off", "Really on"], "%{y}, %{x}<br>%{z:,} minutes", [[0, "#131a22"], [1, "#2c6cba"]], colorbar=False, text=[[f"{c['tn']:,}", f"{c['fp']:,}"], [f"{c['fn']:,}", f"{c['tp']:,}"]], texttemplate="%{text}", textfont={"color": INK, "size": 14})],
        lay("", "", ya={"autorange": "reversed"}), 300)
    return out
