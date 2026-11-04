import { capturing, reduced, cursor, initReveals, magnetic, marquee, setRangeFill, slidingPill, splitWords, whenVisible, tween, fmt } from "./fx.js";
import { render } from "./charts.js";
import { DEV, initDecoder } from "./decoder.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const params = new URLSearchParams(location.search);
const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
const CAP = capturing ? params.get("capture") : null;
if (capturing) document.documentElement.classList.add("capture");
if (CAP) document.getElementById(CAP)?.classList.add("cap-target");

const graphs = {};
const graph = async (id) => (graphs[id] ??= fetch(`/api/graphs/${id}`).then((r) => r.json()));
async function plotGraph(el, id, { height, layout = {} } = {}) {
  const g = await graph(id);
  await render(el, { data: g.figure.data, layout: { ...g.figure.layout, autosize: true, height: height ?? el.clientHeight, ...layout } });
}
const lazy = (el, fn) => whenVisible(el, fn, 0.1);

async function boot() {
  const [ov, list] = await Promise.all([fetch("/api/overview").then((r) => r.json()), fetch("/api/days").then((r) => r.json())]);
  const decoder = initDecoder(ov);
  const heater = ov.devices.find((d) => d.is_heater);
  const heaterIdx = ov.devices.indexOf(heater);
  const days = list.days;

  buildHero(ov, heater);
  buildPicker(list.presets, days, decoder);
  buildFound(ov, heater);
  buildBill(ov, heater, heaterIdx);
  buildPrivacy(ov);
  buildRivals(ov);
  buildLimits(ov, heater);
  buildInside(ov, heater);
  $("#nGraphs").textContent = ov.n_graphs; $("#nGraphs2").textContent = ov.n_graphs;

  splitWords($("#heroTitle"), capturing ? 0 : 1250);
  cursor();
  initReveals();
  $$("[data-magnetic]").forEach((b) => magnetic(b));
  initNav();
  finishLoader(heater.kw.toFixed(3));
  $("#heroTitle").classList.add("in");

  const preset = list.presets.find((p) => p.key === (params.get("preset") || "typical"));
  await decoder.load(params.get("date") || preset.date, preset.note);
  startHeroCanvas(decoder.data);
  if (capturing && CAP === "decoder") {
    if (params.has("decoded")) await decoder.decode();
    if (params.has("key")) await decoder.toggleKey();
  }
}

function finishLoader(found) {
  const l = $("#loader");
  if (capturing) { l.remove(); document.body.classList.remove("is-loading"); return; }
  const el = $("#loaderKw");
  const t0 = performance.now();
  const tick = (now) => {
    if (now - t0 > 1250) { el.textContent = found; return; }
    el.textContent = (Math.random() * 4).toFixed(3);
    requestAnimationFrame(tick);
  };
  reduced ? (el.textContent = found) : requestAnimationFrame(tick);
  setTimeout(() => { l.classList.add("done"); document.body.classList.remove("is-loading"); setTimeout(() => l.remove(), 1100); }, reduced ? 0 : 1350);
}

function buildHero(ov, heater) {
  const D = ov.data;
  $("#heroStats").innerHTML = `
    <div class="stat"><div class="stat__v">${heater.kw.toFixed(3)}<small>kW</small></div><div class="stat__l">device it found, with no labels</div></div>
    <div class="stat"><div class="stat__v">${D.heater_mode_kw.toFixed(2)}<small>kW</small></div><div class="stat__l">what the hidden sub-meter says</div></div>
    <div class="stat"><div class="stat__v">0</div><div class="stat__l">labels and extra sensors used in training</div></div>
    <div class="stat"><div class="stat__v">${D.test_days}</div><div class="stat__l">unseen days it is scored on</div></div>`;
  const words = ["One wire", "No labels", "Hidden states", "No extra sensors", "Forward and backward", "Answer key opened last"];
  const track = $("#marquee");
  track.innerHTML = [...words, ...words].map((w) => `<span>${w}</span>`).join("");
  if (!reduced) marquee(track);
}

/** The hero trace: one white line that splits into coloured strands. Real data from an unseen day. */
function startHeroCanvas(day) {
  const c = $("#heroCanvas"), ctx = c.getContext("2d");
  let W, H;
  const size = () => { const dpr = Math.min(devicePixelRatio || 1, 2); W = c.clientWidth; H = c.clientHeight; c.width = W * dpr; c.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); };
  size(); addEventListener("resize", size);
  const span = 600;
  let visible = true;
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; if (visible && !reduced) requestAnimationFrame(frame); }).observe(c);
  const frame = (t) => {
    if (!visible) return;
    ctx.clearRect(0, 0, W, H);
    const off = reduced ? 400 : (t / 90) % 1440;
    const at = (arr, x) => arr[Math.floor(off + (x / W) * span) % 1440];
    const yBase = H * 0.72, scale = H * 0.13;
    const fan = (x) => { const s = Math.max(0, (x - W * 0.36) / (W * 0.3)); return Math.min(1, s * s * (3 - 2 * s)); };
    ctx.lineWidth = 2;
    day.devices.forEach((d, m) => {
      ctx.beginPath();
      for (let x = W * 0.3; x <= W; x += 3) {
        const f = fan(x);
        const merged = yBase - at(day.total, x) * scale;
        const strand = H * (0.3 + 0.19 * m) - at(d.p, x) * d.kw * scale * 0.85 - 6;
        const y = merged + (strand - merged) * f;
        x <= W * 0.3 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = DEV[m]; ctx.stroke();
    });
    ctx.beginPath();
    for (let x = 0; x <= W * 0.5; x += 3) {
      const y = yBase - at(day.total, x) * scale;
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    const g = ctx.createLinearGradient(W * 0.3, 0, W * 0.5, 0);
    g.addColorStop(0, "#eef2f5"); g.addColorStop(1, "rgba(238,242,245,0)");
    ctx.strokeStyle = g; ctx.stroke();
    if (!reduced) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

function buildPicker(presets, days, decoder) {
  const box = $("#presetChips");
  const activate = (b) => $$(".chip-btn", box).forEach((x) => x.classList.toggle("active", x === b));
  box.innerHTML = presets.map((p) => `<button class="chip-btn" type="button" data-date="${p.date}" data-note="${p.note}">${p.label}</button>`).join("");
  $$(".chip-btn", box).forEach((b, i) => {
    if (i === 0) b.classList.add("active");
    b.addEventListener("click", () => { activate(b); decoder.load(b.dataset.date, b.dataset.note); });
  });
  $("#randomDay").addEventListener("click", () => {
    activate(null);
    decoder.load(days[Math.floor(Math.random() * days.length)].date, "A random unseen day. No hand-picking.");
  });
  const input = $("#dayInput");
  input.min = days[0].date; input.max = days[days.length - 1].date;
  input.addEventListener("change", () => { if (input.value) { activate(null); decoder.load(input.value, "A day you picked."); } });
}

function buildFound(ov, heater) {
  const D = ov.data, R = ov.restarts;
  $("#mLearned").textContent = heater.kw.toFixed(3);
  $("#mTrue").textContent = D.heater_mode_kw.toFixed(2);
  const near = R.filter((r) => r.watts.some((w) => Math.abs(w - D.heater_mode_kw) / D.heater_mode_kw < 0.03)).length;
  $("#foundText").textContent = `A home is modelled as a few devices, each flipping between on and off by its own hidden Markov chain. The meter only ever reports the sum. Training is told the form of that equation and nothing else: no appliance list, no wattages, no labels.`;
  $("#foundText2").textContent = `Expectation-maximisation then has to invent wattages and switching rates that make the observed line probable. The ${ov.devices.length} chains and the always-on level are its answer. The sub-meter stayed locked away until training ended, so the match above is a test, not a fit.`;
  $("#restartNote").textContent = `${near} of ${R.length} runs placed a device within 3% of the real ${D.heater_mode_kw.toFixed(2)} kW. EM can stall in a poorer solution, so the run with the highest likelihood on training days is kept. No labels are involved in that choice.`;
  lazy($("#plotRestarts"), () => plotGraph($("#plotRestarts"), "restarts", { height: 340 }));
}

function buildBill(ov, heater, heaterIdx) {
  const E = ov.energy, n = ov.data.test_days, perMonth = (kwh) => (kwh / n) * 30;
  const items = [...ov.devices.map((d, i) => ({ label: d.is_heater ? `Device ${d.id} · ${d.kw.toFixed(2)} kW (water heater + AC)` : `Device ${d.id} · ${d.kw.toFixed(2)} kW`, kwh: perMonth(d.kwh), col: DEV[i] })),
    { label: `Always on · ${ov.base_kw.toFixed(2)} kW`, kwh: perMonth(E.base_kwh), col: "#55636f" }];
  if (E.residual_kwh > 0) items.push({ label: "Not explained by any device", kwh: perMonth(E.residual_kwh), col: "#34404a" });
  const max = Math.max(...items.map((i) => i.kwh));
  $("#billLead").textContent = `On ${n} unseen days the device matched to the water heater + AC takes ${pct(E.est_share)} of all energy. The real meter says ${pct(E.true_share)}. Total heater energy came out ${Math.abs(E.error_pct).toFixed(1)}% ${E.error_pct < 0 ? "low" : "high"}.`;
  $("#billNote").textContent = `Monthly figures scale the average unseen day to 30 days. The rupee rate is yours to set, and the data is from a house in France, so treat the rupees as illustrative.`;
  $("#monthlyNote").textContent = `Month-by-month correlation r = ${ov.monthly.r.toFixed(2)}. Day-by-day r = ${E.daily_r.toFixed(2)}, with an average miss of ${E.daily_mae_kwh.toFixed(1)} kWh a day.`;
  $("#cutName").textContent = `Device ${heater.id}`;
  $("#billBars").innerHTML = items.map((i) => `<div><div class="bar__top"><span>${i.label}</span><b class="bv"></b></div><div class="bar__track"><i style="--col:${i.col};--w:${(100 * i.kwh / max).toFixed(1)}%"></i></div></div>`).join("");
  const tariff = $("#tariff"), cut = $("#cut");
  tariff.value = ov.tariff;
  const update = () => {
    const t = +tariff.value;
    $("#tariffV").textContent = `₹${t.toFixed(1)}/kWh`;
    $("#cutV").textContent = `${cut.value}%`;
    $$("#billBars .bv").forEach((b, j) => { b.textContent = `₹${fmt(items[j].kwh * t)} /mo`; });
    $("#saveV").textContent = `₹${fmt(items[heaterIdx].kwh * t * (+cut.value / 100))}`;
    setRangeFill(tariff); setRangeFill(cut);
  };
  tariff.addEventListener("input", update); cut.addEventListener("input", update);
  update();
  lazy($("#plotMonthly"), () => plotGraph($("#plotMonthly"), "monthly", { height: 420 }));
}

function buildPrivacy(ov) {
  const W = ov.week, S = ov.sources;
  $("#privLead").textContent = `Nobody told the model what a weekend is. But the ${W.kw.toFixed(2)} kW chain is on ${pct(W.midday_weekday, 0)} of weekday midday minutes and ${pct(W.midday_weekend, 0)} at weekends. The real kitchen and laundry meters say ${pct(W.midday_weekday_true, 0)} and ${pct(W.midday_weekend_true, 0)}. Plain total power matches the same pattern (r = ${W.r_total.toFixed(2)}, against r = ${W.r_decoded.toFixed(2)} for the chain), so the leak is in the meter data, not in this model.`;
  $("#privFacts").innerHTML = `
    <div class="fact"><b>7.24 crore</b><p>smart meters installed in India, with 20.33 crore sanctioned under RDSS, as of 30 June 2026.</p><a href="${S.india_meters}" target="_blank" rel="noopener">T&amp;D India, citing Parliament</a></div>
    <div class="fact"><b>Already sold</b><p>Japanese utilities offer elder-care check-ins built from 15 to 30 minute meter data.</p><a href="${S.japan_services}" target="_blank" rel="noopener">Published review, PMC</a></div>
    <div class="fact"><b>0.5 seconds</b><p>Readings this fine have been shown to reveal which TV channel a home is watching.</p><a href="${S.greveler}" target="_blank" rel="noopener">Greveler et al., 2012</a></div>`;
  lazy($("#plotWeekDec"), () => plotGraph($("#plotWeekDec"), "week-decoded", { height: 320 }));
  lazy($("#plotWeekTrue"), () => plotGraph($("#plotWeekTrue"), "week-truth", { height: 320 }));
}

function buildRivals(ov) {
  const rows = [...ov.rivals].sort((a, b) => b.f1 - a.f1);
  const main = ov.rivals.find((r) => r.family === "Factorial HMM");
  const rules = ov.rivals.filter((r) => r.labels === "none" && r.family === "Rule").sort((a, b) => b.f1 - a.f1)[0];
  const best = rows[0];
  const passed = ov.budget.find((b) => b.f1 > main.f1);
  const flips = ov.decode.flips;
  $("#rivalLead").textContent = `${best.name} wins at F1 ${best.f1.toFixed(3)}, but it learned from a real sub-meter. With only the total to go on, the decoder scores ${main.f1.toFixed(3)} and the best threshold ${rules.f1.toFixed(3)}.`;
  const table = $("#rivalTable");
  table.innerHTML = `<thead><tr><th>Method</th><th>Extra hardware?</th><th>F1 on unseen days</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">Changes/day</th><th class="num">Energy error</th></tr></thead><tbody>` +
    rows.map((r, i) => `<tr class="${r.family === "Factorial HMM" ? "me" : ""}"><td>${r.name}</td><td><span class="pill ${r.labels === "none" ? "pill--no" : ""}">${r.labels === "none" ? "None" : "Sub-meter"}</span></td>
      <td class="fbar"><div class="fbar__wrap"><b>${r.f1.toFixed(3)}</b><div class="fbar__t"><i style="--w:${(r.f1 * 100).toFixed(1)}%;--r:${i};--col:${r.labels === "none" ? DEV[2] : DEV[0]}"></i></div></div></td>
      <td class="num">${r.precision.toFixed(3)}</td><td class="num">${r.recall.toFixed(3)}</td><td class="num">${r.flips_per_day.toFixed(1)}</td><td class="num">${r.error_pct >= 0 ? "+" : ""}${r.error_pct.toFixed(1)}%</td></tr>`).join("") + `</tbody>`;
  whenVisible(table, () => table.classList.add("in"));
  $("#rivalNote").textContent = `The task: is the water heater + AC on during this minute? The real device changes state ${flips.truth.toFixed(0)} times a day. Passing the decoder took about ${passed ? `${passed.days} ${passed.days === 1 ? "day" : "days"}` : "more days than tested"} of sub-meter labels in this one home, and every other home would need its own.`;
  $("#flipNote").textContent = `Dashed line: the real device, ${flips.truth.toFixed(0)} changes a day. Orange methods use no labels, blue methods learned from a sub-meter. Far above the line means switching that never happened.`;
  lazy($("#plotBudget"), () => plotGraph($("#plotBudget"), "label-budget", { height: 330 }));
  lazy($("#plotFlips"), () => plotGraph($("#plotFlips"), "rivals-flips", { height: 330, layout: { margin: { l: 210, r: 12, t: 26, b: 44 }, showlegend: false } }));
}

function buildLimits(ov, heater) {
  const L = ov.limits, K = ov.kalman, E = ov.empty_home, W = ov.week, R = ov.routine;
  const lm = (t, p) => `<article class="limit" data-reveal><h3>${t}</h3><p>${p}</p></article>`;
  $("#limitsList").innerHTML = [
    lm("Only one appliance separates cleanly", `Device ${heater.id} is a real match. Kitchen scores only F1 ${L.K.f1.toFixed(2)} and laundry ${L.L.f1.toFixed(2)}, below a threshold tuned with labels (${L.K.tuned_threshold_f1.toFixed(2)} and ${L.L.tuned_threshold_f1.toFixed(2)}). Their power overlaps. The smallest chains match no meter.`),
    lm("Quiet-morning alarms: a simple rule wins", `An HMM likelihood alarm was the original idea. In testing, plain daily energy detected empty-home days better (AUC ${E.auc_daily_total.toFixed(2)} against ${E.auc_decoded_activity.toFixed(2)} for decoded activity). Empty days are a proxy from the sub-meters, since no occupancy labels exist.`),
    lm("It cannot tell you when someone woke up", `The first big load of the morning has no useful link to first kitchen use: r = ${R.r.toFixed(2)}, typical miss ${(R.median_abs_error_min / 60).toFixed(1)} hours over ${R.days_compared} days. That claim was dropped.`),
    lm("The Kalman filter is not a better alarm", `For a simulated 40% decline it alarms at ${K.decline["Kalman level"].median_days.toFixed(0)} days against ${K.decline["7-day average"].median_days.toFixed(0)} for a 7-day average, with false alarms around ${pct(K.decline["Kalman level"].false_alarm_rate, 0)} because seasons also shift use. It earns its place by handling ${K.missing_days} missing days and giving honest uncertainty.`),
    lm("An HMM's clock is geometric", `Every chain assumes the same chance of switching off each minute, however long it has been on. Real runs do not behave that way. See the durations graph.`),
    lm("One house", `This is one home in France, 2006 to 2010. The method learns per home, which is the point, but nothing here proves it works across homes, climates or Indian appliances.`),
  ].join("");
}

function buildInside(ov, heater) {
  const M = ov.message_passing, K = ov.kalman, D = ov.decode, n = ov.devices.length, states = 2 ** n;
  const c = (t, p, eq) => `<article class="concept" data-reveal><h3>${t}</h3>${eq ? `<pre>${eq}</pre>` : ""}<p>${p}</p></article>`;
  $("#insideGrid").innerHTML = [
    c("Markov chain, the baseline", "The next reading depends only on this one. Fitted by counting moves between 12 levels. Powers of A show how fast it forgets.", "P(next | now) = count(now→next) / count(now)"),
    c("Factorial HMM, the decoder", `${n} on/off chains give ${states} combinations. Their joint transition matrix is the Kronecker product of the device matrices, which keeps inference exact.`, "A = A<sub>1</sub> ⊗ A<sub>2</sub> ⊗ … ⊗ A<sub>M</sub>"),
    c("Forward-backward with scaling", `Left unscaled, the running product leaves float range (${M.naive_direction}) at minute ${M.naive_breaks_at} of a real day. Normalising each step fixes it and matches the log-domain answer to ${M.ll_diff.toExponential(0)}.`, "γ<sub>t</sub> ∝ α<sub>t</sub> β<sub>t</sub>      log p(x) = Σ log c<sub>t</sub>"),
    c("Viterbi and posterior decoding", `Most likely path against most likely state per minute. Same F1 here (${D.viterbi.f1.toFixed(3)} and ${D.smoothed.f1.toFixed(3)}). Live estimates flip ${D.flips.filtered.toFixed(0)} times a day and whole-day ones ${D.flips.smoothed.toFixed(0)}.`, "δ<sub>t</sub>(j) = max<sub>i</sub> δ<sub>t-1</sub>(i) + log A<sub>ij</sub> + log b<sub>j</sub>(x<sub>t</sub>)"),
    c("Sum-product on a factor graph", `One message-passing rule gives both. With sum it matches forward-backward to ${M.max_gamma_diff.toExponential(0)}. With max it returned the same path as Viterbi: ${M.same_viterbi_path ? "yes" : "no"}.`, "μ(z<sub>t+1</sub>) = Σ<sub>z</sub> ψ(z<sub>t</sub>, z<sub>t+1</sub>) μ(z<sub>t</sub>)"),
    c("Baum-Welch (EM)", "Learns wattages and switching rates from the total alone. Wattages are solved by least squares weighted by γ/variance, and tests confirm the likelihood never falls.", "E: γ, ξ      M: p<sub>on</sub>, p<sub>off</sub>, wattage, variance"),
    c("Linear dynamical system", `A Kalman filter and RTS smoother track the slow trend in daily use. EM learned q = ${K.q.toFixed(2)} and r = ${K.r.toFixed(1)}, so each day moves the estimate about ${(K.gain_final * 100).toFixed(0)}%.`, "z<sub>t</sub> = z<sub>t-1</sub> + w      y<sub>t</sub> = z<sub>t</sub> + v"),
    c("CART and AdaBoost, the rivals", "Written from scratch and given what the decoder never gets: sub-meter labels. They exist to answer how much that hardware buys.", "α<sub>t</sub> = ½ ln((1 − err<sub>t</sub>) / err<sub>t</sub>)"),
  ].join("");
}

function initNav() {
  const links = $$("#navLinks a"), pill = $(".nav__pill"), box = $("#navLinks");
  const nav = $("#nav"), progress = $("#progress");
  const setActive = (a) => { links.forEach((l) => l.classList.toggle("active", l === a)); pill.style.opacity = a ? 1 : 0; if (a) { pill.style.width = a.offsetWidth + "px"; pill.style.transform = `translateX(${a.offsetLeft}px)`; } };
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) setActive(links.find((l) => l.getAttribute("href") === `#${e.target.id}`)); }), { rootMargin: "-45% 0px -50% 0px" });
  $$("main > section[id]").forEach((s) => io.observe(s));
  let last = 0;
  addEventListener("scroll", () => {
    const y = scrollY, max = document.documentElement.scrollHeight - innerHeight;
    progress.style.transform = `scaleX(${max > 0 ? y / max : 0})`;
    nav.classList.toggle("hidden", y > last && y > 500);
    last = y;
  }, { passive: true });
}

boot();
