// Motion toolkit: reveals, split text, count-ups, magnetic buttons, tilt, cursor, marquee.
// Capture mode (?capture=...) renders every animation in its final state, for screenshots.
export const capturing = new URLSearchParams(location.search).has("capture");
export const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches || capturing;
export const fine = window.matchMedia("(hover: hover) and (pointer: fine)").matches;

export const ease = {
  outExpo: (t) => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t)),
  outCubic: (t) => 1 - Math.pow(1 - t, 3),
  inOutCubic: (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
};

export function tween(from, to, dur, fn, easing = ease.outExpo) {
  if (reduced) { fn(to); return Promise.resolve(); }
  return new Promise((resolve) => {
    const t0 = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - t0) / dur);
      fn(from + (to - from) * easing(t));
      if (t < 1) requestAnimationFrame(step); else resolve();
    };
    requestAnimationFrame(step);
  });
}

const fmtCache = {};
export function fmt(v, { decimals = 0, pct = false, compact = false } = {}) {
  if (pct) return (v * 100).toFixed(decimals) + "%";
  if (compact && Math.abs(v) >= 1e4) return Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(v);
  const key = decimals;
  fmtCache[key] ??= new Intl.NumberFormat("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  return fmtCache[key].format(v);
}

/** Animate the number in an element from its current value to `to`. */
export function countTo(el, to, opts = {}) {
  const from = el._val ?? opts.from ?? 0;
  el._val = to;
  const suffix = opts.suffix ?? "";
  return tween(from, to, opts.duration ?? 1400, (v) => { el.textContent = fmt(v, opts) + suffix; });
}

/** Reveal elements when they scroll into view; run count-ups inside them. */
export function initReveals(root = document) {
  if (capturing) {
    root.querySelectorAll("[data-reveal], .split, [data-count], [data-observe]").forEach((el) => {
      el.classList.add("in");
      if (el.hasAttribute("data-count")) startCount(el);
    });
    return null;
  }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const el = e.target;
      el.classList.add("in");
      el.querySelectorAll?.("[data-count]").forEach(startCount);
      if (el.hasAttribute("data-count")) startCount(el);
      io.unobserve(el);
    }
  }, { threshold: 0.15, rootMargin: "0px 0px -8% 0px" });
  root.querySelectorAll("[data-reveal], .split, [data-count], [data-observe]").forEach((el) => io.observe(el));
  return io;
}

function startCount(el) {
  if (el._counted) return;
  el._counted = true;
  const to = parseFloat(el.dataset.count);
  countTo(el, to, { decimals: +(el.dataset.decimals || 0), pct: el.dataset.pct === "1", suffix: el.dataset.suffix || "", duration: 1800 });
}

/** Wrap each word of an element in masked spans for a staggered rise. */
export function splitWords(el, base = 0) {
  const walk = (node, out) => {
    for (const child of [...node.childNodes]) {
      if (child.nodeType === 3) {
        const parts = child.textContent.split(/(\s+)/);
        const frag = document.createDocumentFragment();
        for (const p of parts) {
          if (!p) continue;
          if (/^\s+$/.test(p)) { frag.appendChild(document.createTextNode(p)); continue; }
          const w = document.createElement("span");
          w.className = "w";
          const inner = document.createElement("span");
          inner.textContent = p;
          inner.style.setProperty("--i", out.i++);
          w.appendChild(inner);
          frag.appendChild(w);
        }
        child.replaceWith(frag);
      } else if (child.nodeType === 1) {
        walk(child, out);
      }
    }
  };
  walk(el, { i: 0 });
  el.style.setProperty("--base", base + "ms");
  el.classList.add("split");
}

/** Buttons drift towards the cursor and snap back with a spring. */
export function magnetic(el, strength = 0.35) {
  if (!fine || reduced) return;
  el.addEventListener("pointermove", (e) => {
    const r = el.getBoundingClientRect();
    const x = e.clientX - r.left - r.width / 2;
    const y = e.clientY - r.top - r.height / 2;
    el.style.transform = `translate(${x * strength}px, ${y * strength}px)`;
    el.style.transition = "transform .15s ease-out";
  });
  el.addEventListener("pointerleave", () => {
    el.style.transform = "";
    el.style.transition = "transform .8s cubic-bezier(.34,1.56,.64,1)";
  });
}

/** 3D tilt plus a spotlight that follows the pointer on cards. */
export function tilt(el, max = 6) {
  if (!fine || reduced) return;
  el.style.transformStyle = "preserve-3d";
  el.addEventListener("pointermove", (e) => {
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width;
    const py = (e.clientY - r.top) / r.height;
    el.style.setProperty("--mx", `${px * 100}%`);
    el.style.setProperty("--my", `${py * 100}%`);
    if (max) el.style.transform = `perspective(1000px) rotateX(${(0.5 - py) * max}deg) rotateY(${(px - 0.5) * max}deg)`;
    el.style.transition = "transform .1s linear";
  });
  el.addEventListener("pointerleave", () => {
    el.style.transform = "";
    el.style.transition = "transform .9s cubic-bezier(.16,1,.3,1)";
  });
}

export function spotlight(el) {
  if (!fine) return;
  el.addEventListener("pointermove", (e) => {
    const r = el.getBoundingClientRect();
    el.style.setProperty("--mx", `${((e.clientX - r.left) / r.width) * 100}%`);
    el.style.setProperty("--my", `${((e.clientY - r.top) / r.height) * 100}%`);
  });
}

/** Custom cursor: a dot and a lagging ring that grows over anything clickable. */
export function cursor() {
  if (!fine) return;
  const dot = document.createElement("div");
  dot.className = "cursor";
  const ring = document.createElement("div");
  ring.className = "cursor-ring";
  const label = document.createElement("span");
  ring.appendChild(label);
  document.body.append(dot, ring);
  let x = innerWidth / 2, y = innerHeight / 2, rx = x, ry = y;
  addEventListener("pointermove", (e) => { x = e.clientX; y = e.clientY; }, { passive: true });
  const loop = () => {
    rx += (x - rx) * 0.16;
    ry += (y - ry) * 0.16;
    dot.style.transform = `translate(${x}px, ${y}px)`;
    ring.style.transform = `translate(${rx}px, ${ry}px)`;
    requestAnimationFrame(loop);
  };
  loop();
  document.addEventListener("pointerover", (e) => {
    const t = e.target.closest("a, button, [data-cursor], input, select, .toggle, label.q-click");
    ring.classList.toggle("hover", !!t);
    const text = t?.dataset?.cursor || "";
    label.textContent = text;
    ring.classList.toggle("has-label", !!text);
  });
}

/** Infinite marquee whose speed reacts to scroll velocity. */
export function marquee(track, base = 0.6) {
  const w = track.scrollWidth / 2;
  let x = 0, vel = 0, lastY = scrollY;
  const loop = () => {
    const dy = scrollY - lastY;
    lastY = scrollY;
    vel += (dy * 0.25 - vel) * 0.1;
    if (!reduced) x -= base + Math.abs(vel);
    if (x <= -w) x += w;
    track.style.transform = `translate3d(${x}px,0,0) skewX(${Math.max(-12, Math.min(12, -vel * 0.4))}deg)`;
    requestAnimationFrame(loop);
  };
  loop();
}

/** Text scramble for labels that change value. */
export function scramble(el, text, dur = 700) {
  if (reduced) { el.textContent = text; return; }
  const chars = "!<>-_\\/[]{}=+*^?#01";
  const t0 = performance.now();
  const step = (now) => {
    const p = Math.min(1, (now - t0) / dur);
    const n = Math.floor(p * text.length);
    let out = text.slice(0, n);
    for (let i = n; i < text.length; i++) out += text[i] === " " ? " " : chars[(Math.random() * chars.length) | 0];
    el.textContent = out;
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

/** Sliding pill indicator behind the active button of a tab/segment group. */
export function slidingPill(container, pill, activeSel = ".active") {
  const move = () => {
    const a = container.querySelector(activeSel);
    if (!a) return;
    pill.style.width = a.offsetWidth + "px";
    pill.style.transform = `translateX(${a.offsetLeft}px)`;
  };
  move();
  addEventListener("resize", move);
  return move;
}

export function toast(msg) {
  let t = document.querySelector(".toast");
  if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.remove("show"), 2600);
}

export function setRangeFill(input) {
  const p = ((input.value - input.min) / (input.max - input.min)) * 100;
  input.style.setProperty("--fill", p + "%");
}

export function debounce(fn, ms = 120) {
  let h;
  return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); };
}

/** Run fn once the element scrolls into view (immediately in capture mode). */
export function whenVisible(el, fn, threshold = 0.2) {
  if (capturing) { setTimeout(fn, 0); return; }
  const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { io.disconnect(); fn(); } }, { threshold });
  io.observe(el);
}
