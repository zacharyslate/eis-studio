// Visual equivalent-circuit editor for EIS Studio (Streamlit components v2).
//
// data   = { tree, presets: [{label, text, tree}], kinds: [...] }
// state  = "tree"  (sent back to Python on every edit; Python converts it to an
//                    impedance.py string, assigns element names and re-renders)
//
// Interaction: drag an element from the palette (or an existing element) onto the canvas.
// Where you drop it decides the topology: left/right of a target = in series with it,
// above/below a target = in parallel with it. A target is the innermost element or group
// under the pointer; each group is outlined, and its padding belongs to the group, so
// pointing just outside a group's elements targets the whole group.
// Click selects; Delete/Backspace removes; clicking a palette chip adds the element next to
// the selection (or at the end) using the "Click adds" mode. Everything also works by touch.

const NS = "http://www.w3.org/2000/svg";
const PALETTE = [
  ["R", "Resistor"], ["C", "Capacitor"], ["CPE", "Constant phase element"], ["L", "Inductor"],
  ["W", "Warburg (semi-infinite)"], ["Wo", "Warburg, open (finite space)"], ["Ws", "Warburg, short (finite length)"],
  ["G", "Gerischer"], ["Gs", "Gerischer, finite length"], ["La", "Modified inductance"],
  ["TLMQ", "Transmission line (simplified)"], ["Zarc", "ZARC (R parallel CPE)"], ["K", "Voigt RC (R, τ)"],
];

// geometry
const EW = 64, EH = 34, LAB = 18;        // element box (incl. lead wires) and label space
const GAP = 14, PAD = 12, BGAP = 16;    // series gap, group padding, parallel branch gap

let uid = 0;
const newId = () => "n" + (++uid);

function withIds(node) {
  if (!node) return null;
  const out = Object.assign({}, node, { id: newId() });
  if (node.c) out.c = node.c.map(withIds);
  return out;
}
function strip(node) {
  if (!node) return null;
  if (node.t === "e") return { t: "e", k: node.k, n: node.n || null };
  return { t: node.t, c: node.c.map(strip) };
}
function clone(node) { return node ? JSON.parse(JSON.stringify(node)) : null; }

function normalize(node) {
  if (!node) return null;
  if (node.t === "e") return node;
  let kids = [];
  for (const c of node.c.map(normalize)) {
    if (!c) continue;
    if (c.t === node.t) kids = kids.concat(c.c); else kids.push(c);
  }
  if (!kids.length) return null;
  if (kids.length === 1) return kids[0];
  return Object.assign({}, node, { c: kids });
}
function findParent(root, id, parent = null) {
  if (!root) return null;
  if (root.id === id) return { node: root, parent };
  if (root.c) for (const c of root.c) { const r = findParent(c, id, root); if (r) return r; }
  return null;
}
function remove(root, id) {
  if (!root || root.id === id) return null;
  if (root.c) root.c = root.c.map((c) => remove(c, id)).filter(Boolean);
  return normalize(root);
}
// place `item` relative to target node `tid` (zone: left|right|top|bottom)
function insert(root, tid, zone, item) {
  if (!root) return item;
  const found = findParent(root, tid);
  if (!found) return root;
  const { node, parent } = found;
  const want = zone === "left" || zone === "right" ? "s" : "p";
  const before = zone === "left" || zone === "top";
  if (parent && parent.t === want) {
    const i = parent.c.indexOf(node);
    parent.c.splice(before ? i : i + 1, 0, item);
    return normalize(root);
  }
  const group = { t: want, id: newId(), c: before ? [item, node] : [node, item] };
  if (!parent) return normalize(group);
  parent.c[parent.c.indexOf(node)] = group;
  return normalize(root);
}

// ------------------------------------------------------------------ layout
function measure(n) {
  if (n.t === "e") { n.w = EW; n.h = EH + 2 * LAB; return; }
  n.c.forEach(measure);
  if (n.t === "s") {
    n.w = n.c.reduce((a, c) => a + c.w, 0) + GAP * (n.c.length - 1) + 2 * PAD;
    n.h = Math.max(...n.c.map((c) => c.h)) + 2 * PAD;
  } else {
    n.w = Math.max(...n.c.map((c) => c.w)) + 2 * PAD + 2 * 18;
    n.h = n.c.reduce((a, c) => a + c.h, 0) + BGAP * (n.c.length - 1) + 2 * PAD;
  }
}
function place(n, x, y) {               // x, y = top-left; n.cy = y of the signal line
  n.x = x; n.y = y; n.cy = y + n.h / 2;
  if (n.t === "e") return;
  if (n.t === "s") {
    let cx = x + PAD;
    for (const c of n.c) { place(c, cx, n.cy - c.h / 2); cx += c.w + GAP; }
  } else {
    let cy = y + PAD;
    for (const c of n.c) { place(c, x + PAD + 18 + (n.w - 2 * PAD - 36 - c.w) / 2, cy); cy += c.h + BGAP; }
  }
}

// ------------------------------------------------------------------ drawing
function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function line(g, x1, y1, x2, y2, cls = "wire") { return el("line", { x1, y1, x2, y2, class: cls }, g); }

function symbol(g, kind, x, y) {          // element body centred at (x, y), lead wires to ±EW/2
  const L = x - EW / 2, R = x + EW / 2;
  if (kind === "C") {
    line(g, L, y, x - 5, y); line(g, x + 5, y, R, y);
    line(g, x - 5, y - 13, x - 5, y + 13, "plate"); line(g, x + 5, y - 13, x + 5, y + 13, "plate");
  } else if (kind === "CPE") {
    line(g, L, y, x - 2, y); line(g, x + 7, y, R, y);
    el("path", { d: `M${x - 8},${y - 13} L${x - 2},${y} L${x - 8},${y + 13}`, class: "plate" }, g);
    el("path", { d: `M${x + 1},${y - 13} L${x + 7},${y} L${x + 1},${y + 13}`, class: "plate" }, g);
  } else if (kind === "L") {
    line(g, L, y, x - 18, y); line(g, x + 18, y, R, y);
    let d = `M${x - 18},${y}`;
    for (let i = 0; i < 4; i++) d += ` a4.5,6 0 0 1 9,0`;
    el("path", { d, class: "wire" }, g);
  } else {
    const bw = kind === "R" ? 34 : 42;
    line(g, L, y, x - bw / 2, y); line(g, x + bw / 2, y, R, y);
    el("rect", { x: x - bw / 2, y: y - 9, width: bw, height: 18, class: "part" }, g);
    if (kind !== "R") {
      const t = el("text", { x, y: y + 4.5, class: "kindtext", "text-anchor": "middle" }, g);
      t.textContent = kind;
    }
  }
}

function draw(svg, n, sel, depth = 0) {
  const g = el("g", { "data-id": n.id }, svg);
  if (n.t === "e") {
    const cx = n.x + n.w / 2;
    el("rect", { x: n.x, y: n.cy - EH / 2 - LAB, width: n.w, height: EH + 2 * LAB, class: "hit" + (n.id === sel ? " sel" : ""), rx: 6 }, g);
    symbol(g, n.k, cx, n.cy);
    const label = el("text", { x: cx, y: n.cy - EH / 2 - 4, class: "name", "text-anchor": "middle" }, g);
    const m = /^([A-Za-z]+)(\d+)$/.exec(n.n || "");
    if (m) {
      label.appendChild(document.createTextNode(m[1]));
      const s = el("tspan", { "baseline-shift": "-0.25em", "font-size": "75%" }, label);
      s.textContent = m[2];
    } else { label.textContent = n.k; label.setAttribute("class", "name pending"); }
    return;
  }
  el("rect", { x: n.x + 2, y: n.y + 2, width: n.w - 4, height: n.h - 4, rx: 10, class: "group " + n.t + (n.id === sel ? " sel" : "") + (depth === 0 ? " root" : "") }, g);
  if (n.t === "s") {
    for (let i = 0; i < n.c.length; i++) {
      const c = n.c[i];
      if (i === 0) line(g, n.x, n.cy, c.x, n.cy);
      if (i < n.c.length - 1) line(g, c.x + c.w, n.cy, n.c[i + 1].x, n.cy);
      else line(g, c.x + c.w, n.cy, n.x + n.w, n.cy);
    }
  } else {
    const lx = n.x + PAD + 9, rx = n.x + n.w - PAD - 9;
    const top = n.c[0].cy, bot = n.c[n.c.length - 1].cy;
    line(g, n.x, n.cy, lx, n.cy); line(g, rx, n.cy, n.x + n.w, n.cy);
    line(g, lx, top, lx, bot); line(g, rx, top, rx, bot);
    for (const c of n.c) { line(g, lx, c.cy, c.x, c.cy); line(g, c.x + c.w, c.cy, rx, c.cy); }
  }
  for (const c of n.c) draw(svg, c, sel, depth + 1);
}

function render(svg, tree, sel, scale = 1) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!tree) { svg.setAttribute("viewBox", "0 0 10 10"); svg.setAttribute("width", 0); svg.setAttribute("height", 0); return; }
  measure(tree);
  const M = 20;
  place(tree, M + 14, M);
  const W = tree.w + 2 * M + 28, H = tree.h + 2 * M;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", W * scale);
  svg.setAttribute("height", H * scale);
  line(svg, M, tree.cy, M + 14, tree.cy);
  line(svg, M + 14 + tree.w, tree.cy, W - M, tree.cy);
  el("circle", { cx: M, cy: tree.cy, r: 4, class: "node" }, svg);
  el("circle", { cx: W - M, cy: tree.cy, r: 4, class: "node" }, svg);
  draw(svg, tree, sel);
  return { W, H };
}

// hit test: innermost node whose box contains (x, y)
function hit(n, x, y) {
  if (!n) return null;
  const inside = x >= n.x && x <= n.x + n.w && y >= n.y && y <= n.y + n.h;
  if (!inside) return null;
  if (n.c) for (const c of n.c) { const h = hit(c, x, y); if (h) return h; }
  return n;
}
function zoneOf(n, x, y) {
  const fx = (x - n.x) / n.w, fy = (y - n.y) / n.h;
  const dx = Math.min(fx, 1 - fx), dy = Math.min(fy, 1 - fy);
  if (dx < dy * (n.w / n.h) * 0.9) return fx < 0.5 ? "left" : "right";
  return fy < 0.5 ? "top" : "bottom";
}

// ------------------------------------------------------------------ component
export default function (component) {
  const { data, setStateValue, parentElement } = component;
  // Streamlit may call this again (new data) without running the previous cleanup:
  // tear down any earlier instance in this mount point first.
  if (parentElement.__ceCleanup) { try { parentElement.__ceCleanup(); } catch (e) { /* already gone */ } }
  parentElement.querySelectorAll(":scope > .ce").forEach((n) => n.remove());
  const root = document.createElement("div");
  root.className = "ce";
  parentElement.appendChild(root);

  let tree = withIds(data && data.tree);
  let sel = null;
  let mode = "s";                               // what a palette click adds: s(eries) or p(arallel)
  const send = () => setStateValue("tree", { tree: strip(tree), nonce: Date.now() });

  root.innerHTML = `
    <div class="presets" role="list" aria-label="Preset circuits"></div>
    <div class="bar">
      <div class="palette" role="toolbar" aria-label="Circuit elements: drag onto the canvas, or click to add"></div>
      <div class="tools">
        <label class="mode">Click adds
          <select aria-label="Where a clicked element is added">
            <option value="s">in series</option><option value="p">in parallel</option>
          </select></label>
        <button type="button" class="del" disabled>Delete</button>
        <button type="button" class="clear">Clear</button>
        <button type="button" class="svgdl">Download SVG</button>
      </div>
    </div>
    <div class="canvas" tabindex="0" aria-label="Circuit canvas">
      <svg class="circuit" xmlns="${NS}"></svg>
      <p class="empty">Drag an element here to start, or pick a preset above.</p>
    </div>
    <p class="hint">Drop to the <b>left or right</b> of an element or group to connect it <b>in series</b>, <b>above or below</b> to connect it <b>in parallel</b>. Point just outside a group's elements (inside its outline) to target the whole group. Drag elements to move them; select and press Delete to remove.</p>
    <div class="ghost" aria-hidden="true"></div>`;
  const svg = root.querySelector("svg.circuit");
  const canvas = root.querySelector(".canvas");
  const empty = root.querySelector(".empty");
  const ghost = root.querySelector(".ghost");
  const delBtn = root.querySelector(".del");
  root.querySelector(".mode select").addEventListener("change", (e) => { mode = e.target.value; });

  function refresh() {
    render(svg, tree, sel);
    empty.style.display = tree ? "none" : "block";
    delBtn.disabled = !sel;
  }

  // presets as mini schematics
  const presetBox = root.querySelector(".presets");
  for (const p of (data && data.presets) || []) {
    const b = document.createElement("button");
    b.type = "button"; b.className = "preset"; b.setAttribute("role", "listitem");
    b.title = p.text;
    const mini = document.createElementNS(NS, "svg");
    mini.setAttribute("class", "circuit mini");
    b.appendChild(mini);
    const cap = document.createElement("span"); cap.textContent = p.label; b.appendChild(cap);
    presetBox.appendChild(b);
    const t = withIds(p.tree);
    const dims = render(mini, t, null);
    if (dims) { mini.setAttribute("width", "100%"); mini.setAttribute("height", "78"); mini.setAttribute("preserveAspectRatio", "xMidYMid meet"); }
    if (data.current && p.text === data.current) { b.classList.add("active"); b.setAttribute("aria-current", "true"); }
    b.addEventListener("click", () => { tree = withIds(p.tree); sel = null; refresh(); send(); });
  }

  // palette
  const pal = root.querySelector(".palette");
  for (const [k, title] of PALETTE) {
    if (data && data.kinds && !data.kinds.includes(k)) continue;
    const chip = document.createElement("button");
    chip.type = "button"; chip.className = "chip"; chip.title = title + " (drag onto the circuit, or click to add)";
    chip.setAttribute("aria-label", title);
    const s = document.createElementNS(NS, "svg");
    s.setAttribute("viewBox", `0 0 ${EW} 30`); s.setAttribute("class", "circuit chipsym");
    symbol(s, k, EW / 2, 15);
    chip.appendChild(s);
    const lab = document.createElement("span"); lab.textContent = k; chip.appendChild(lab);
    chip.dataset.kind = k;
    pal.appendChild(chip);
  }

  // ---------------------------------------------------------------- dragging
  let drag = null;                         // {kind | id, startX, startY, moved}
  function toSvg(ev) {
    const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
    const m = svg.getScreenCTM();
    return m ? pt.matrixTransform(m.inverse()) : { x: -1, y: -1 };
  }
  function clearMarks() { svg.querySelectorAll(".dropmark, .target").forEach((e) => e.remove()); }
  function targetAt(ev) {
    if (!tree) {
      const r = canvas.getBoundingClientRect();
      const inside = ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom;
      return inside ? { empty: true } : null;
    }
    const p = toSvg(ev);
    const r = canvas.getBoundingClientRect();
    if (!(ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom)) return null;
    let n = hit(tree, p.x, p.y) || tree;    // outside everything: the whole circuit
    if (drag && drag.id && n.id === drag.id) return null;
    // dropping an element next to itself inside its own group is a no-op
    if (drag && drag.id) { const f = findParent(tree, drag.id); if (f && isAncestor(f.node, n.id)) return null; }
    return { node: n, zone: zoneOf(n, Math.min(Math.max(p.x, n.x), n.x + n.w), Math.min(Math.max(p.y, n.y), n.y + n.h)) };
  }
  function isAncestor(a, id) { if (a.id === id) return true; return !!(a.c && a.c.some((c) => isAncestor(c, id))); }
  function showTarget(t) {
    clearMarks();
    if (!t || t.empty) { canvas.classList.toggle("dropok", !!(t && t.empty)); return; }
    canvas.classList.remove("dropok");
    const n = t.node;
    el("rect", { x: n.x, y: n.y, width: n.w, height: n.h, rx: 10, class: "target" }, svg);
    const T = 5;
    const r = { left: [n.x - T, n.y, T * 2, n.h], right: [n.x + n.w - T, n.y, T * 2, n.h],
                top: [n.x, n.y - T, n.w, T * 2], bottom: [n.x, n.y + n.h - T, n.w, T * 2] }[t.zone];
    el("rect", { x: r[0], y: r[1], width: r[2], height: r[3], rx: T, class: "dropmark" }, svg);
    ghost.dataset.hint = t.zone === "left" || t.zone === "right" ? "in series" : "in parallel";
  }
  function startDrag(ev, payload, label) {
    drag = Object.assign({ startX: ev.clientX, startY: ev.clientY, moved: false, pointer: ev.pointerId }, payload);
    ghost.textContent = label;
  }
  function onMove(ev) {
    if (!drag || ev.pointerId !== drag.pointer) return;
    if (!drag.moved && Math.hypot(ev.clientX - drag.startX, ev.clientY - drag.startY) < 5) return;
    drag.moved = true;
    ev.preventDefault();
    ghost.style.display = "block";
    const host = root.getBoundingClientRect();
    ghost.style.left = (ev.clientX - host.left + 12) + "px";
    ghost.style.top = (ev.clientY - host.top + 12) + "px";
    const t = targetAt(ev);
    showTarget(t);
    ghost.classList.toggle("hinted", !!(t && !t.empty));
    drag.target = t;
  }
  function onUp(ev) {
    if (!drag || ev.pointerId !== drag.pointer) return;
    const d = drag; drag = null;
    ghost.style.display = "none"; ghost.classList.remove("hinted"); clearMarks(); canvas.classList.remove("dropok");
    if (!d.moved) { if (d.kind) clickAdd(d.kind); else if (d.id) { sel = d.id; refresh(); } return; }
    const t = d.target;
    if (!t) return;
    let item;
    if (d.kind) item = { t: "e", k: d.kind, n: null, id: newId() };
    else { const f = findParent(tree, d.id); if (!f) return; item = f.node; tree = remove(tree, d.id); }
    tree = t.empty || !tree ? item : insert(tree, findParent(tree, t.node.id) ? t.node.id : tree.id, t.zone, item);
    sel = item.id;
    refresh(); send();
  }
  function clickAdd(kind) {
    const item = { t: "e", k: kind, n: null, id: newId() };
    if (!tree) tree = item;
    else tree = insert(tree, sel && findParent(tree, sel) ? sel : tree.id, mode === "s" ? "right" : "bottom", item);
    sel = item.id;
    refresh(); send();
  }

  pal.addEventListener("pointerdown", (ev) => {
    const chip = ev.target.closest(".chip");
    if (!chip) return;
    startDrag(ev, { kind: chip.dataset.kind }, chip.dataset.kind);
  });
  pal.addEventListener("keydown", (ev) => {
    const chip = ev.target.closest(".chip");
    if (chip && (ev.key === "Enter" || ev.key === " ")) { ev.preventDefault(); clickAdd(chip.dataset.kind); }
  });
  svg.addEventListener("pointerdown", (ev) => {
    const g = ev.target.closest("g[data-id]");
    if (!g) { sel = null; refresh(); return; }
    const f = findParent(tree, g.dataset.id);
    if (!f) return;
    startDrag(ev, { id: f.node.id }, f.node.n || f.node.k || (f.node.t === "s" ? "series group" : "parallel group"));
    ev.stopPropagation();
  });
  root.ownerDocument.addEventListener("pointermove", onMove, { passive: false });
  root.ownerDocument.addEventListener("pointerup", onUp);
  root.ownerDocument.addEventListener("pointercancel", onUp);

  function deleteSel() {
    if (!sel) return;
    tree = remove(tree, sel); sel = null; refresh(); send();
  }
  delBtn.addEventListener("click", deleteSel);
  canvas.addEventListener("keydown", (ev) => {
    if ((ev.key === "Delete" || ev.key === "Backspace") && sel) { ev.preventDefault(); deleteSel(); }
    if (ev.key === "Escape") { sel = null; refresh(); }
  });
  root.querySelector(".clear").addEventListener("click", () => { tree = null; sel = null; refresh(); send(); });
  root.querySelector(".svgdl").addEventListener("click", () => {
    if (!tree) return;
    const out = document.createElementNS(NS, "svg");
    out.setAttribute("xmlns", NS);
    render(out, withIds(strip(tree)), null);
    const style = document.createElementNS(NS, "style");
    style.textContent = ".wire,.plate{stroke:#15212B;fill:none;stroke-width:1.8}.plate{stroke-width:2.6}" +
      ".part{fill:#fff;stroke:#15212B;stroke-width:1.8}.node{fill:#15212B}.hit,.group{fill:none;stroke:none}" +
      ".name{font:italic 15px 'Times New Roman',serif;fill:#15212B}.kindtext{font:11px sans-serif;fill:#15212B}";
    out.insertBefore(style, out.firstChild);
    const blob = new Blob([new XMLSerializer().serializeToString(out)], { type: "image/svg+xml" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "equivalent-circuit.svg";
    root.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });

  refresh();
  const cleanup = () => {
    root.ownerDocument.removeEventListener("pointermove", onMove);
    root.ownerDocument.removeEventListener("pointerup", onUp);
    root.ownerDocument.removeEventListener("pointercancel", onUp);
    root.remove();
    if (parentElement.__ceCleanup === cleanup) parentElement.__ceCleanup = null;
  };
  parentElement.__ceCleanup = cleanup;
  return cleanup;
}
