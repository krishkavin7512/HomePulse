import { initReveals, splitWords, cursor, spotlight, countTo, slidingPill, reduced, capturing } from "./fx.js";
import { render, tableView, CONFIG_TOOLS } from "./charts.js";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const TABS = [["what", "What it shows"], ["how", "How it's computed"], ["read", "How to read it"], ["where", "Where it's used"]];

cursor();
splitWords($("#gTitle"), 100);
initProgress();

fetch("/api/graphs").then((r) => r.json()).then(({ graphs }) => {
  countTo($("#gCount"), graphs.length, { duration: 1600 });
  buildFilters(graphs);
  graphs.forEach((g, i) => $("#grid").appendChild(card(g, i)));
  initReveals();
  $("#gTitle").classList.add("in");
  lazyPlots();
});

function card(g, i) {
  const el = document.createElement("article");
  el.className = "card g-card";
  el.dataset.reveal = "";
  el.style.setProperty("--d", (i % 2) * 110);
  el.dataset.cat = g.category;
  el.dataset.search = (g.title + " " + g.category + " " + Object.values(g.explain).join(" ")).toLowerCase();
  el.innerHTML = `
    <div class="g-card__top"><div><span class="g-card__num">${String(i + 1).padStart(2, "0")} / ${g.id}</span><h3></h3></div><span class="g-card__cat"></span></div>
    <div class="g-plot loading" style="min-height:${g.height}px"></div>
    <div class="g-insight"></div>
    <div class="g-tabs" role="tablist">${TABS.map(([k, l], j) => `<button role="tab" data-k="${k}" class="${j === 0 ? "active" : ""}">${l}</button>`).join("")}<i class="ul"></i></div>
    <div class="g-text"><p></p></div>
    <div class="g-actions"><button class="btn" data-act="expand">Expand ↗</button><button class="btn" data-act="table">Table view</button></div>`;
  $("h3", el).textContent = g.title;
  $(".g-card__cat", el).textContent = g.category;
  $(".g-insight", el).textContent = g.insight;
  $(".g-text p", el).textContent = g.explain.what;
  const tabs = $(".g-tabs", el), ul = $(".ul", tabs);
  const moveUl = () => { const a = $("button.active", tabs); ul.style.width = a.offsetWidth + "px"; ul.style.transform = `translateX(${a.offsetLeft}px)`; };
  requestAnimationFrame(moveUl);
  addEventListener("resize", moveUl);
  $$("button", tabs).forEach((b) => b.addEventListener("click", () => {
    $$("button", tabs).forEach((x) => x.classList.toggle("active", x === b));
    moveUl();
    const p = document.createElement("p");
    p.textContent = g.explain[b.dataset.k];
    $(".g-text", el).replaceChildren(p);
  }));
  const plot = $(".g-plot", el);
  plot._graph = g;
  $('[data-act="expand"]', el).addEventListener("click", () => openModal(g));
  $('[data-act="table"]', el).addEventListener("click", (e) => {
    if (el._table) { el._table.remove(); el._table = null; plot.hidden = false; e.target.textContent = "Table view"; return; }
    const t = tableView(g.figure);
    t.classList.add("g-table");
    plot.after(t);
    plot.hidden = true;
    el._table = t;
    e.target.textContent = "Chart view";
  });
  spotlight(el);
  return el;
}

function lazyPlots() {
  const draw = (el) => { const g = el._graph; el.classList.remove("loading");
    return render(el, { data: g.figure.data, layout: { ...g.figure.layout, height: g.height, autosize: true } }, { config: CONFIG_TOOLS }); };
  if (capturing) { $$(".g-plot").forEach(draw); return; }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      io.unobserve(e.target);
      setTimeout(() => draw(e.target), reduced ? 0 : 200);
    }
  }, { rootMargin: "200px 0px" });
  $$(".g-plot").forEach((p) => io.observe(p));
}

function buildFilters(graphs) {
  const f = $("#filters");
  ["All", ...new Set(graphs.map((g) => g.category))].forEach((c, i) => {
    const b = document.createElement("button");
    b.innerHTML = `${c}<sup>${c === "All" ? graphs.length : graphs.filter((g) => g.category === c).length}</sup>`;
    b.dataset.cat = c;
    if (i === 0) b.classList.add("active");
    f.appendChild(b);
  });
  const move = slidingPill(f, $(".tabs__pill", f));
  let cat = "All";
  const apply = () => {
    const q = $("#search").value.trim().toLowerCase();
    let shown = 0;
    $$(".g-card").forEach((c) => {
      const ok = (cat === "All" || c.dataset.cat === cat) && (!q || c.dataset.search.includes(q));
      c.classList.toggle("hide", !ok);
      if (ok) { shown++; c.classList.add("in"); }
    });
    $("#empty").style.display = shown ? "none" : "block";
    window.dispatchEvent(new Event("resize"));
  };
  $$("button", f).forEach((b) => b.addEventListener("click", () => {
    $$("button", f).forEach((x) => x.classList.toggle("active", x === b));
    cat = b.dataset.cat;
    move();
    apply();
  }));
  $("#search").addEventListener("input", apply);
}

function openModal(g) {
  const m = $("#modal");
  $("#mCat").textContent = g.category;
  $("#mTitle").textContent = g.title;
  const ex = $("#mExplain");
  ex.innerHTML = "";
  [["Insight", g.insight], ...TABS.map(([k, l]) => [l, g.explain[k]])].forEach(([h, t]) => {
    const d = document.createElement("div");
    const hh = document.createElement("h4"); hh.textContent = h;
    const p = document.createElement("p"); p.textContent = t;
    d.append(hh, p);
    ex.appendChild(d);
  });
  m.classList.add("open");
  m.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  const plot = $("#mPlot");
  Plotly.purge(plot);
  setTimeout(() => render(plot, { data: g.figure.data, layout: { ...g.figure.layout, height: 520, autosize: true } }, { config: CONFIG_TOOLS }), 250);
  $(".modal__close", m).focus();
}
function closeModal() {
  const m = $("#modal");
  m.classList.remove("open");
  m.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}
$$("[data-close]").forEach((b) => b.addEventListener("click", closeModal));
addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

function initProgress() {
  const p = $("#progress");
  addEventListener("scroll", () => {
    const max = document.documentElement.scrollHeight - innerHeight;
    p.style.transform = `scaleX(${max > 0 ? scrollY / max : 0})`;
  }, { passive: true });
}
