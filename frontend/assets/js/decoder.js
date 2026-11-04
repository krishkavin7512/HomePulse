// The decoder stage: draws the day, runs the two passes of the algorithm on screen, plays the day, opens the answer key.
import { reduced, tween, ease, countTo, capturing } from "./fx.js";

export const DEV = ["#3987e5", "#199e70", "#e0621f", "#9085e9", "#c98500"];
const BASE_COL = "#2a3641";
const $ = (s) => document.querySelector(s);
const clock = (m) => `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;

export function initDecoder(ov) {
  const canvas = $("#decCanvas"), plotBox = canvas.parentElement, ctx = canvas.getContext("2d"), tip = $("#decTip"), stage = $(".dec");
  const btnDecode = $("#btnDecode"), btnPlay = $("#btnPlay"), btnKey = $("#btnKey"), status = $("#decStatus");
  let data = null, W = 0, H = 0, runId = 0;
  const S = { decoded: false, head: -1, back: 1440, key: 0, cursor: null, playing: false, drawn: 1 };
  const names = ov.devices.map((d) => `Device ${d.id}`);
  buildDevices();

  function buildDevices() {
    $("#devs").innerHTML = ov.devices.map((d, i) => `
      <div class="dev" id="dev${i}" style="--col:${DEV[i]}">
        <div class="dev__top"><span class="dev__name">${names[i]}</span><span class="dev__lamp"></span></div>
        <div class="dev__kw">${d.kw.toFixed(2)}<small>kW</small></div>
        <div class="dev__meta"><span class="dev__today">Not decoded yet</span></div>
      </div>`).join("") + `
      <div class="dev" id="devBase" style="--col:${BASE_COL}">
        <div class="dev__top"><span class="dev__name">Always on</span></div>
        <div class="dev__kw">${ov.base_kw.toFixed(2)}<small>kW</small></div>
        <div class="dev__meta">The level the house never drops below</div>
      </div>`;
  }

  function resize() {
    const r = plotBox.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    W = r.width; H = r.height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  const stripsH = () => S.key * 132;
  const area = () => ({ x0: 44, x1: W - 12, y0: 20, y1: H - 26 - stripsH() });

  function draw() {
    if (!data || !W) return;
    const { x0, x1, y0, y1 } = area();
    const ymax = Math.max(1.5, Math.ceil(Math.max(...data.total) * 1.05 * 2) / 2);
    const X = (i) => x0 + (i / 1439) * (x1 - x0), Y = (v) => y1 - (v / ymax) * (y1 - y0);
    ctx.clearRect(0, 0, W, H);
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.lineWidth = 1;
    const stepY = ymax > 6 ? 2 : 1;
    for (let v = 0; v <= ymax; v += stepY) {
      ctx.strokeStyle = "#1f262d"; ctx.beginPath(); ctx.moveTo(x0, Y(v)); ctx.lineTo(x1, Y(v)); ctx.stroke();
      ctx.fillStyle = "#8593a0"; ctx.textAlign = "right"; ctx.fillText(`${v}`, x0 - 8, Y(v) + 4);
    }
    ctx.textAlign = "left"; ctx.fillText("kW", x0, 12);
    ctx.textAlign = "center";
    for (let h = 0; h <= 24; h += W < 560 ? 6 : 3) ctx.fillText(`${String(h).padStart(2, "0")}:00`, Math.min(X(h * 60), x1 - 18), H - 8);
    // decoded layers
    if (S.head >= 0) {
      const last = Math.floor(S.head);
      let lo = new Float32Array(1440).fill(0);
      const band = (hi, col) => {
        ctx.beginPath();
        ctx.moveTo(X(0), Y(hi[0]));
        for (let i = 1; i <= last; i++) ctx.lineTo(X(i), Y(hi[i]));
        for (let i = last; i >= 0; i--) ctx.lineTo(X(i), Y(lo[i]));
        ctx.closePath(); ctx.fillStyle = col; ctx.fill();
        ctx.beginPath(); ctx.moveTo(X(0), Y(hi[0]));
        for (let i = 1; i <= last; i++) ctx.lineTo(X(i), Y(hi[i]));
        ctx.strokeStyle = "#101418"; ctx.lineWidth = 1.5; ctx.stroke();
        lo = hi;
      };
      band(new Float32Array(1440).fill(data.base_kw), BASE_COL);
      data.devices.forEach((d, m) => {
        const hi = new Float32Array(1440);
        for (let i = 0; i < 1440; i++) hi[i] = lo[i] + (i >= S.back ? d.p[i] : d.live[i]) * d.kw;
        band(hi, DEV[m]);
      });
    }
    // the only thing the model sees
    const draws = Math.floor(S.drawn * 1439);
    ctx.beginPath(); ctx.moveTo(X(0), Y(data.total[0]));
    for (let i = 1; i <= draws; i++) ctx.lineTo(X(i), Y(data.total[i]));
    ctx.strokeStyle = "#eef2f5"; ctx.lineWidth = S.head >= 0 ? 1.1 : 1.6; ctx.stroke();
    // pass heads
    const passHead = (i, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(i), y0); ctx.lineTo(X(i), y1); ctx.stroke(); };
    if (S.head >= 0 && S.head < 1439) passHead(S.head, "#d7ff3e");
    if (S.back > 0 && S.back < 1440) passHead(S.back, "#d7ff3e");
    if (S.key > 0) drawStrips(X, x0, x1, y1);
    if (S.cursor != null) {
      ctx.strokeStyle = "rgba(238,242,245,.65)"; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(X(S.cursor), y0); ctx.lineTo(X(S.cursor), H - 26); ctx.stroke(); ctx.setLineDash([]);
    }
  }

  function drawStrips(X, x0, x1, y1) {
    const ans = data.answer, heater = data.devices.find((d) => d.is_heater), hi = data.devices.indexOf(heater);
    const rows = [
      [`Device ${heater.id}, decoded`, (i) => heater.p[i] > 0.5, DEV[hi]],
      ["Water heater + AC, real meter", (i) => ans.heater_kw[i] >= 0.3, "#a9b4be"],
      ["Kitchen, real meter", (i) => ans.kitchen_kw[i] >= 0.3, "#5d6a76"],
      ["Laundry room, real meter", (i) => ans.laundry_kw[i] >= 0.6, "#5d6a76"],
    ];
    const top = y1 + 10, rh = 13, wipe = Math.floor(S.key * 1439);
    ctx.globalAlpha = S.key;
    rows.forEach(([label, on, col], r) => {
      const y = top + r * (rh + 18) + 15;
      ctx.font = "600 11px Inter, sans-serif"; ctx.textAlign = "left";
      ctx.fillStyle = "#c4ccd4"; ctx.fillText(label, x0, y - 4);
      ctx.fillStyle = "#161c23"; ctx.fillRect(x0, y, x1 - x0, rh);
      ctx.fillStyle = col;
      let start = -1;
      for (let i = 0; i <= wipe; i++) {
        const v = on(i);
        if (v && start < 0) start = i;
        if ((!v || i === wipe) && start >= 0) { ctx.fillRect(X(start), y, Math.max(1, X(i) - X(start)), rh); start = -1; }
      }
    });

    ctx.globalAlpha = 1;
    ctx.font = "11px 'JetBrains Mono', monospace";
  }

  function readout(m) {
    if (!data) return;
    $("#decClock").textContent = m == null ? "--:--" : clock(m);
    $("#decKw").textContent = m == null ? "–" : data.total[m].toFixed(2);
    data.devices.forEach((d, i) => document.getElementById(`dev${i}`).classList.toggle("lit", S.decoded && m != null && d.path[m] === 1));
  }

  function showTip(m) {
    if (!S.decoded || m == null) { tip.hidden = true; return; }
    const { x0, x1 } = area(), x = x0 + (m / 1439) * (x1 - x0);
    tip.innerHTML = `<b>${clock(m)}</b> · <b>${data.total[m].toFixed(2)}</b> kW` + data.devices.map((d, i) =>
      `<div class="row"><i style="background:${DEV[i]}"></i><span>Device ${d.id}</span><b>${Math.round(d.p[m] * 100)}%</b></div>`).join("");
    tip.hidden = false;
    tip.style.left = (x > W * 0.62 ? x - tip.offsetWidth - 14 : x + 14) + "px";
  }

  function setCursor(m) { S.cursor = m; readout(m); showTip(m); draw(); }

  canvas.addEventListener("pointermove", (e) => {
    if (S.playing || !data) return;
    const r = canvas.getBoundingClientRect(), { x0, x1 } = area();
    const m = Math.round(((e.clientX - r.left - x0) / (x1 - x0)) * 1439);
    setCursor(m >= 0 && m < 1440 ? m : null);
  });
  canvas.addEventListener("pointerleave", () => { if (!S.playing) setCursor(null); });

  function reset() {
    Object.assign(S, { decoded: false, head: -1, back: 1440, key: 0, cursor: null, playing: false });
    stage.classList.remove("decoded");
    plotBox.classList.remove("keyed");
    btnPlay.disabled = true; btnKey.disabled = true; btnDecode.disabled = false;
    btnPlay.textContent = "▶ Play the day"; btnKey.textContent = "Open the answer key"; btnKey.classList.remove("on");
    $("#verdict").hidden = true;
    tip.hidden = true;
    status.textContent = "Model input: one line. Nothing else.";
    ov.devices.forEach((d, i) => {
      const card = document.getElementById(`dev${i}`);
      card.classList.remove("lit");
      card.querySelector(".dev__today").textContent = "Not decoded yet";
      card.querySelector(".dev__who")?.remove();
    });
    readout(null);
  }

  async function load(date, note) {
    const id = ++runId;
    const res = await fetch(`/api/decode?date=${date}`);
    if (id !== runId) return;
    if (!res.ok) { status.textContent = `No unseen data for ${date}. Pick another day.`; return; }
    data = await res.json();
    reset();
    const d = new Date(date + "T00:00:00");
    $("#decDate").textContent = d.toLocaleDateString("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric" });
    $("#decKwh").textContent = data.total_kwh.toFixed(1);
    $("#decNote").textContent = note || "";
    $("#dayInput").value = date;
    S.drawn = 0;
    resize();
    await tween(0, 1, 900, (v) => { S.drawn = v; draw(); }, ease.inOutCubic);
  }

  async function decode() {
    if (!data || S.decoded) return;
    const id = runId;
    btnDecode.disabled = true;
    status.textContent = "Forward pass: only what could be known at that minute.";
    await tween(0, 1439, 1800, (v) => { if (id === runId) { S.head = v; draw(); } }, ease.inOutCubic);
    if (id !== runId) return;
    status.textContent = "Backward pass: later readings correct the earlier guesses.";
    await tween(1439, 0, 1500, (v) => { if (id === runId) { S.back = v; draw(); } }, ease.inOutCubic);
    if (id !== runId) return;
    S.back = 0; S.head = 1439; S.decoded = true;
    stage.classList.add("decoded");
    status.textContent = `Done. ${data.devices.length} devices explain this day. Nobody told it they exist.`;
    const money = (kwh) => `₹${(kwh * ov.tariff).toFixed(0)}`;
    data.devices.forEach((d, i) => {
      document.getElementById(`dev${i}`).querySelector(".dev__today").textContent = `${d.minutes_on} min on · ${d.kwh.toFixed(1)} kWh · ${money(d.kwh)} today`;
    });
    btnPlay.disabled = false; btnKey.disabled = false;
    draw();
  }

  async function play() {
    if (S.playing) { S.playing = false; btnPlay.textContent = "▶ Play the day"; return; }
    S.playing = true;
    btnPlay.textContent = "❚❚ Pause";
    const start = S.cursor ?? 0, t0 = performance.now(), dur = 16000 * ((1439 - start) / 1439);
    const id = runId;
    const step = (now) => {
      if (!S.playing || id !== runId) return;
      const p = Math.min(1, (now - t0) / dur);
      const m = Math.round(start + (1439 - start) * p);
      S.cursor = m; readout(m); draw();
      if (p < 1) requestAnimationFrame(step);
      else { S.playing = false; btnPlay.textContent = "▶ Play the day"; setCursor(null); }
    };
    requestAnimationFrame(step);
  }

  async function toggleKey() {
    const opening = S.key === 0;
    plotBox.classList.toggle("keyed", opening);
    btnKey.classList.toggle("on", opening);
    btnKey.textContent = opening ? "Close the answer key" : "Open the answer key";
    if (opening) showVerdict(); else { $("#verdict").hidden = true; ov.devices.forEach((_, i) => document.getElementById(`dev${i}`).querySelector(".dev__who")?.remove()); }
    const from = S.key, to = opening ? 1 : 0;
    const follow = setInterval(resize, 60);
    await tween(from, to, 900, (v) => { S.key = v; draw(); }, ease.inOutCubic);
    clearInterval(follow);
    S.key = to;
    resize();
  }

  function showVerdict() {
    const heater = data.devices.find((d) => d.is_heater);
    const cross = ov.decode.cross_f1, meters = [["H", "water heater + AC"], ["K", "kitchen"], ["L", "laundry"]];
    ov.devices.forEach((d, i) => {
      const card = document.getElementById(`dev${i}`);
      card.querySelector(".dev__who")?.remove();
      const best = meters.map(([k, n]) => [cross[k][i], n]).sort((a, b) => b[0] - a[0])[0];
      const who = document.createElement("div");
      who.className = "dev__who";
      who.textContent = d.is_heater ? `This is the water heater + AC (F1 ${best[0].toFixed(2)} on unseen days)` : `No clean match to any meter (best F1 ${best[0].toFixed(2)})`;
      card.appendChild(who);
    });
    const err = 100 * (data.heater_est_kwh - data.answer.heater_kwh) / Math.max(data.answer.heater_kwh, 0.01);
    const v = $("#verdict");
    v.innerHTML = `
      <div><b>${data.heater_est_kwh.toFixed(1)} kWh</b><span>Device ${heater.id}'s energy today, from the total alone</span></div>
      <div><b>${data.answer.heater_kwh.toFixed(1)} kWh</b><span>The real sub-meter${data.answer.heater_kwh > 0.5 ? `, so the model is ${Math.abs(err).toFixed(0)}% ${err >= 0 ? "high" : "low"} today` : ""}</span></div>
      <div><b>${data.f1 == null ? "n/a" : data.f1.toFixed(2)}</b><span>${data.f1 == null ? "No heater use today, so nothing to score" : "F1 for this one day: minute-by-minute on/off against the meter"}</span></div>`;
    v.hidden = false;
  }

  btnDecode.addEventListener("click", decode);
  btnPlay.addEventListener("click", play);
  btnKey.addEventListener("click", toggleKey);
  addEventListener("resize", resize);
  return { load, decode, toggleKey, resize, get data() { return data; } };
}
