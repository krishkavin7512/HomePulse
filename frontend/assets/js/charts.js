// Plotly helpers shared by the landing page and the graphs page.
import { reduced } from "./fx.js";

export const CONFIG = { displayModeBar: false, responsive: true };
export const CONFIG_TOOLS = { displayModeBar: "hover", displaylogo: false, responsive: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"] };

const clone = (o) => JSON.parse(JSON.stringify(o));

/** Render a figure, then play an entrance: lines draw themselves, bars grow, markers fade in. */
export async function render(el, fig, { animate = true, config = CONFIG } = {}) {
  await Plotly.newPlot(el, clone(fig.data), clone(fig.layout), config);
  if (animate && !reduced) entrance(el);
  return el;
}

const OUT = "cubic-bezier(.16,1,.3,1)";

export function entrance(el) {
  el.querySelectorAll(".scatterlayer .trace").forEach((tr, ti) => {
    tr.querySelectorAll("path.js-line").forEach((p) => {
      const dashed = p.style.strokeDasharray && p.style.strokeDasharray !== "none";
      const len = p.getTotalLength?.() || 0;
      if (dashed || !len) { p.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 900, delay: 200 + ti * 120, fill: "backwards" }); return; }
      p.animate([{ strokeDasharray: `${len} ${len}`, strokeDashoffset: len }, { strokeDasharray: `${len} ${len}`, strokeDashoffset: 0 }],
        { duration: 1500, delay: ti * 150, easing: OUT, fill: "backwards" });
    });
    tr.querySelectorAll("path.js-fill").forEach((p) => p.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 1200, delay: 300, fill: "backwards" }));
    tr.querySelectorAll(".points path").forEach((p, i) => p.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500, delay: 250 + ti * 140 + Math.min(i, 120) * 6, fill: "backwards" }));
  });
  el.querySelectorAll(".barlayer .trace").forEach((tr, ti) => {
    const horizontal = tr.__data__?.[0]?.trace?.orientation === "h";
    tr.querySelectorAll(".point").forEach((pt, i) => {
      const p = pt.querySelector("path");
      if (!p) return;
      const neg = (pt.__data__?.s ?? 0) < 0;
      p.style.transformBox = "fill-box";
      p.style.transformOrigin = horizontal ? (neg ? "right center" : "left center") : (neg ? "center top" : "center bottom");
      p.animate([{ transform: horizontal ? "scaleX(0)" : "scaleY(0)" }, { transform: "none" }], { duration: 950, delay: ti * 110 + Math.min(i, 60) * 14, easing: OUT, fill: "backwards" });
    });
  });
  el.querySelectorAll(".heatmaplayer").forEach((l) => l.animate([{ opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0)" }], { duration: 1000, easing: OUT, fill: "backwards" }));
  el.querySelectorAll(".annotation, .shapelayer path").forEach((a, i) => a.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 600, delay: 900 + i * 80, fill: "backwards" }));
}

/** Plain HTML table of every trace, the accessible twin of a chart. */
export function tableView(fig) {
  const wrap = document.createElement("div");
  wrap.className = "table-wrap";
  const table = document.createElement("table");
  table.className = "data";
  const thead = table.createTHead().insertRow();
  const tbody = table.createTBody();
  const traces = fig.data.filter((t) => t.x || t.y || t.z);
  const heat = traces.find((t) => t.type === "heatmap");
  const cell = (v) => (typeof v === "number" ? (Math.abs(v) >= 1000 ? v.toLocaleString() : +v.toFixed(5)) : v ?? "");
  if (heat) {
    thead.insertCell().textContent = "";
    const step = Math.max(1, Math.ceil(heat.x.length / 40));
    const xs = heat.x.filter((_, i) => i % step === 0);
    xs.forEach((x) => { const th = document.createElement("th"); th.textContent = x; thead.appendChild(th); });
    heat.y.forEach((y, i) => {
      const r = tbody.insertRow();
      r.insertCell().textContent = y;
      heat.z[i].filter((_, j) => j % step === 0).forEach((v) => { r.insertCell().textContent = cell(v); });
    });
  } else {
    ["Series", "x", "y"].forEach((h) => { const th = document.createElement("th"); th.textContent = h; thead.appendChild(th); });
    for (const t of traces) {
      const xs = t.x || [], ys = t.y || [];
      const n = Math.max(xs.length, ys.length);
      const step = Math.max(1, Math.ceil(n / 60));
      for (let i = 0; i < n; i += step) {
        const r = tbody.insertRow();
        r.insertCell().textContent = t.name || t.type || "trace";
        r.insertCell().textContent = cell(xs[i]);
        r.insertCell().textContent = cell(ys[i]);
      }
    }
  }
  wrap.appendChild(table);
  return wrap;
}
