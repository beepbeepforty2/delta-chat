"""Visual diff viewer: a human-in-the-loop debug tool, NOT the delta engine.

Renders both PDF pages side by side and overlays color-coded boxes from a
*naive* greedy content matcher (exact match, then rapidfuzz best-match,
then leftovers are add/remove). This exists to eyeball what the native-PDF
adapter extracted and how a simple matcher pairs it up -- useful right now,
before README Plan step 3 (the real deterministic bipartite-matching delta
engine) exists, and useful afterward too, to sanity-check the real engine's
output against a quick independent baseline.

Usage:
    python -m tools.visual_diff --a path/A.pdf --b path/B.pdf --out diff.html
"""
from __future__ import annotations

import argparse
import base64
import json

import fitz
from rapidfuzz import fuzz

from src.ingest.pdf_native import PdfNativeAdapter

FUZZY_THRESHOLD = 55
MOVE_THRESHOLD = 0.04  # normalized bbox-center distance beyond which an
                        # exact-content match is flagged "moved" not "unchanged"


def _center(bbox):
    return ((bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2)


def _dist(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def _record(el):
    return {
        "id": el.id, "type": el.type, "content": el.content,
        "zone": el.zone, "bbox": [el.bbox.x0, el.bbox.y0, el.bbox.x1, el.bbox.y1],
        "confidence": el.attrs.get("type_confidence", 1.0),
    }


def naive_match(elements_a, elements_b):
    """Greedy, per-type, exact-then-fuzzy matcher. NOT the real delta
    engine's deterministic bipartite matching (README Plan step 3) --
    a quick debug baseline, explicitly not claiming optimality."""
    by_type_a, by_type_b = {}, {}
    for e in elements_a:
        by_type_a.setdefault(e.type, []).append(e)
    for e in elements_b:
        by_type_b.setdefault(e.type, []).append(e)

    matches = []
    for etype in set(by_type_a) | set(by_type_b):
        pool_a = list(by_type_a.get(etype, []))
        pool_b = list(by_type_b.get(etype, []))

        # exact content match, greedy
        by_content_b = {}
        for eb in pool_b:
            by_content_b.setdefault(eb.content, []).append(eb)
        remaining_a = []
        for ea in pool_a:
            cands = by_content_b.get(ea.content)
            if cands:
                eb = cands.pop(0)
                moved = _dist(_center(ea.bbox), _center(eb.bbox)) > MOVE_THRESHOLD
                matches.append({
                    "status": "moved" if moved else "unchanged",
                    "a": _record(ea), "b": _record(eb), "score": 100,
                })
            else:
                remaining_a.append(ea)
        remaining_b = [eb for group in by_content_b.values() for eb in group]

        # fuzzy match, greedy by descending score
        scored = []
        for ea in remaining_a:
            for eb in remaining_b:
                s = fuzz.ratio(ea.content, eb.content)
                if s >= FUZZY_THRESHOLD:
                    scored.append((s, ea, eb))
        scored.sort(key=lambda t: -t[0])
        used_a, used_b = set(), set()
        for s, ea, eb in scored:
            if ea.id in used_a or eb.id in used_b:
                continue
            used_a.add(ea.id)
            used_b.add(eb.id)
            matches.append({"status": "modified", "a": _record(ea), "b": _record(eb), "score": s})

        for ea in remaining_a:
            if ea.id not in used_a:
                matches.append({"status": "removed", "a": _record(ea), "b": None, "score": None})
        for eb in remaining_b:
            if eb.id not in used_b:
                matches.append({"status": "added", "a": None, "b": _record(eb), "score": None})

    return matches


def render_page_png_b64(path: str, dpi: int = 130):
    doc = fitz.open(path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
    w, h = pix.width, pix.height
    doc.close()
    return b64, w, h


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>P&ID Visual Diff</title>
<style>
:root {{
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --surface-2: #f2f1ee;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --border: rgba(11,11,11,0.10);
  --accent: #2a78d6; --accent-wash: rgba(42,120,214,0.10);
  --added: #0ca30c; --modified: #d98a00; --removed: #d03b3b; --moved: #4a3aa7;
  --added-wash: rgba(12,163,12,0.10); --modified-wash: rgba(217,138,0,0.12);
  --removed-wash: rgba(208,59,59,0.10); --moved-wash: rgba(74,58,167,0.10);
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --surface-2: #222220;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --border: rgba(255,255,255,0.10);
    --accent: #3987e5; --accent-wash: rgba(57,135,229,0.14);
    --added: #0ca30c; --modified: #c98500; --removed: #e66767; --moved: #9085e9;
    --added-wash: rgba(12,163,12,0.16); --modified-wash: rgba(201,133,0,0.16);
    --removed-wash: rgba(230,103,103,0.14); --moved-wash: rgba(144,133,233,0.16);
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --surface-2: #222220;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
  --border: rgba(255,255,255,0.10);
  --accent: #3987e5; --accent-wash: rgba(57,135,229,0.14);
  --added: #0ca30c; --modified: #c98500; --removed: #e66767; --moved: #9085e9;
  --added-wash: rgba(12,163,12,0.16); --modified-wash: rgba(201,133,0,0.16);
  --removed-wash: rgba(230,103,103,0.14); --moved-wash: rgba(144,133,233,0.16);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--page); color: var(--text-primary);
  font: 14px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif;
}}
.mono {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
header {{
  padding: 14px 20px; background: var(--surface); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
}}
header h1 {{ font-size: 15px; font-weight: 600; margin: 0; letter-spacing: 0.01em; }}
header .subtitle {{ font-size: 12px; color: var(--text-muted); margin-top: 2px; }}
.titlewrap {{ margin-right: auto; }}
.legend {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.chip {{
  display: inline-flex; align-items: center; gap: 6px; padding: 5px 10px;
  border-radius: 6px; border: 1px solid var(--border); background: var(--surface-2);
  font-size: 12px; cursor: pointer; user-select: none; color: var(--text-secondary);
}}
.chip input {{ accent-color: var(--accent); margin: 0; }}
.chip .dot {{ width: 9px; height: 9px; border-radius: 2px; flex: none; }}
.chip.off {{ opacity: 0.45; }}
.dot.added {{ background: var(--added); }}
.dot.modified {{ background: var(--modified); }}
.dot.removed {{ background: var(--removed); }}
.dot.moved {{ background: var(--moved); }}
.dot.unchanged {{ background: var(--text-muted); }}
.count {{ font-variant-numeric: tabular-nums; color: var(--text-muted); }}
main {{ display: grid; grid-template-columns: 1fr 1fr 320px; gap: 1px; background: var(--border); height: calc(100vh - 60px); }}
.pane {{ background: var(--page); overflow: auto; position: relative; }}
.pane-label {{
  position: sticky; top: 0; z-index: 3; padding: 6px 12px; font-size: 11px;
  font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase;
  color: var(--text-muted); background: var(--surface); border-bottom: 1px solid var(--border);
}}
.imgwrap {{ position: relative; width: 100%; }}
.imgwrap img {{ width: 100%; display: block; }}
.box {{
  position: absolute; border: 1.75px solid; border-radius: 2px; cursor: pointer;
  background: transparent; transition: background 0.1s, box-shadow 0.1s;
}}
.box.status-added {{ border-color: var(--added); }}
.box.status-modified {{ border-color: var(--modified); }}
.box.status-removed {{ border-color: var(--removed); }}
.box.status-moved {{ border-color: var(--moved); }}
.box.status-unchanged {{ border-color: var(--text-muted); border-style: dashed; opacity: 0.6; }}
.box.hidden {{ display: none; }}
.box.active {{
  z-index: 2; box-shadow: 0 0 0 3px var(--accent-wash); background: var(--accent-wash);
  border-color: var(--accent);
}}
.hide-unchanged .box.status-unchanged {{ display: none; }}
aside {{ background: var(--surface); display: flex; flex-direction: column; overflow: hidden; }}
.toolbar {{ padding: 10px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 8px; }}
.toolbar input, .toolbar select {{
  width: 100%; padding: 6px 8px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--surface-2); color: var(--text-primary); font-size: 12px;
}}
.rows {{ overflow: auto; flex: 1; }}
.row {{
  padding: 7px 10px; border-bottom: 1px solid var(--border); cursor: pointer; font-size: 12px;
}}
.row:hover {{ background: var(--surface-2); }}
.row.active {{ background: var(--accent-wash); }}
.row .top {{ display: flex; align-items: center; gap: 6px; margin-bottom: 2px; }}
.row .status-dot {{ width: 8px; height: 8px; border-radius: 2px; flex: none; }}
.row .type {{ color: var(--text-muted); font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.03em; }}
.row .content {{ color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
.footnote {{ padding: 8px 10px; font-size: 11px; color: var(--text-muted); border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<header>
  <div class="titlewrap">
    <h1>Visual diff — {name_a} vs {name_b}</h1>
    <div class="subtitle">naive greedy matcher (debug tool, not the delta engine) &middot; {n_geom_a} + {n_geom_b} geometry elements omitted from overlay</div>
  </div>
  <div class="legend" id="legend"></div>
</header>
<main id="main">
  <div class="pane" id="pane-a">
    <div class="pane-label">{name_a}</div>
    <div class="imgwrap"><img src="data:image/png;base64,{img_a}"><div class="boxlayer" id="boxes-a"></div></div>
  </div>
  <div class="pane" id="pane-b">
    <div class="pane-label">{name_b}</div>
    <div class="imgwrap"><img src="data:image/png;base64,{img_b}"><div class="boxlayer" id="boxes-b"></div></div>
  </div>
  <aside>
    <div class="toolbar">
      <input id="search" type="text" placeholder="Filter by content...">
      <select id="typeFilter"><option value="">All types</option></select>
    </div>
    <div class="rows" id="rows"></div>
    <div class="footnote" id="footnote"></div>
  </aside>
</main>
<script>
const DATA = {data_json};
const STATUSES = ["added", "modified", "removed", "moved", "unchanged"];
const DEFAULT_ON = {{added: true, modified: true, removed: true, moved: true, unchanged: false}};
let activeOn = {{...DEFAULT_ON}};

function boxDiv(rec, status, matchId) {{
  const d = document.createElement("div");
  d.className = "box status-" + status;
  d.style.left = (rec.bbox[0] * 100) + "%";
  d.style.top = (rec.bbox[1] * 100) + "%";
  d.style.width = ((rec.bbox[2] - rec.bbox[0]) * 100) + "%";
  d.style.height = ((rec.bbox[3] - rec.bbox[1]) * 100) + "%";
  d.dataset.matchId = matchId;
  d.title = rec.type + ": " + rec.content.slice(0, 80);
  d.addEventListener("click", () => selectMatch(matchId));
  return d;
}}

const boxesA = document.getElementById("boxes-a");
const boxesB = document.getElementById("boxes-b");
const rowsEl = document.getElementById("rows");
const typeSet = new Set();

DATA.matches.forEach((m, i) => {{
  if (m.a) {{ boxesA.appendChild(boxDiv(m.a, m.status, i)); typeSet.add(m.a.type); }}
  if (m.b) {{ boxesB.appendChild(boxDiv(m.b, m.status, i)); typeSet.add(m.b.type); }}
}});

const typeFilter = document.getElementById("typeFilter");
[...typeSet].sort().forEach(t => {{
  const o = document.createElement("option"); o.value = t; o.textContent = t;
  typeFilter.appendChild(o);
}});

function renderRows() {{
  const q = document.getElementById("search").value.toLowerCase();
  const tf = typeFilter.value;
  rowsEl.innerHTML = "";
  DATA.matches.forEach((m, i) => {{
    if (!activeOn[m.status]) return;
    const rec = m.b || m.a;
    if (tf && rec.type !== tf) return;
    if (q && !rec.content.toLowerCase().includes(q)) return;
    const row = document.createElement("div");
    row.className = "row"; row.dataset.matchId = i;
    row.innerHTML = `<div class="top"><span class="status-dot dot ${{m.status}}"></span>` +
      `<span class="type">${{rec.type}}</span></div>` +
      `<div class="content mono">${{rec.content.replace(/</g,"&lt;").slice(0,90) || "(empty)"}}</div>`;
    row.addEventListener("click", () => selectMatch(i));
    rowsEl.appendChild(row);
  }});
  document.getElementById("footnote").textContent = rowsEl.children.length + " of " + DATA.matches.length + " elements shown";
}}

let selected = null;
function selectMatch(i) {{
  if (selected !== null) {{
    document.querySelectorAll(`[data-match-id="${{selected}}"]`).forEach(el => el.classList.remove("active"));
  }}
  selected = i;
  const els = document.querySelectorAll(`[data-match-id="${{i}}"]`);
  els.forEach(el => el.classList.add("active"));
  els.forEach(el => {{ if (el.classList.contains("box")) el.scrollIntoView({{block: "center", inline: "center", behavior: "smooth"}}); }});
  const row = rowsEl.querySelector(`.row[data-match-id="${{i}}"]`);
  if (row) {{ row.scrollIntoView({{block: "nearest"}}); row.classList.add("active"); setTimeout(() => row.classList.remove("active"), 900); }}
}}

function updateVisibility() {{
  const main = document.getElementById("main");
  main.classList.toggle("hide-unchanged", !activeOn.unchanged);
  document.querySelectorAll(".box").forEach(el => {{
    const status = [...el.classList].find(c => c.startsWith("status-")).slice(7);
    el.classList.toggle("hidden", !activeOn[status]);
  }});
  renderRows();
}}

const counts = {{}};
STATUSES.forEach(s => counts[s] = 0);
DATA.matches.forEach(m => counts[m.status]++);

const legend = document.getElementById("legend");
STATUSES.forEach(s => {{
  const chip = document.createElement("label");
  chip.className = "chip" + (activeOn[s] ? "" : " off");
  chip.innerHTML = `<input type="checkbox" ${{activeOn[s] ? "checked" : ""}}>` +
    `<span class="dot ${{s}}"></span>${{s}} <span class="count">${{counts[s]}}</span>`;
  chip.querySelector("input").addEventListener("change", e => {{
    activeOn[s] = e.target.checked;
    chip.classList.toggle("off", !e.target.checked);
    updateVisibility();
  }});
  legend.appendChild(chip);
}});

document.getElementById("search").addEventListener("input", renderRows);
typeFilter.addEventListener("change", renderRows);

updateVisibility();
</script>
</body>
</html>
"""


def build_html(name_a, name_b, img_a, wa, ha, img_b, wb, hb, matches, n_geom_a, n_geom_b) -> str:
    return HTML_TEMPLATE.format(
        name_a=name_a, name_b=name_b, img_a=img_a, img_b=img_b,
        n_geom_a=n_geom_a, n_geom_b=n_geom_b,
        data_json=json.dumps({"matches": matches}),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="visual_diff.html")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--name-a", default=None, help="display label for A (default: derived from --a path)")
    ap.add_argument("--name-b", default=None, help="display label for B (default: derived from --b path)")
    args = ap.parse_args()

    adapter = PdfNativeAdapter()
    doc_a = adapter.ingest("a", args.a)
    doc_b = adapter.ingest("b", args.b)
    els_a = [e for e in doc_a.sheets[0].elements if e.type != "geometry"]
    els_b = [e for e in doc_b.sheets[0].elements if e.type != "geometry"]
    n_geom_a = len(doc_a.sheets[0].elements) - len(els_a)
    n_geom_b = len(doc_b.sheets[0].elements) - len(els_b)

    matches = naive_match(els_a, els_b)
    img_a, wa, ha = render_page_png_b64(args.a, args.dpi)
    img_b, wb, hb = render_page_png_b64(args.b, args.dpi)

    def default_name(path):
        # e.g. eval/datasets/v0/pairs/edited_000/a/L0.pdf -> edited_000/a/L0.pdf
        parts = path.replace("\\", "/").split("/")
        return "/".join(parts[-3:]) if len(parts) >= 3 else path

    name_a = args.name_a or default_name(args.a)
    name_b = args.name_b or default_name(args.b)
    html = build_html(name_a, name_b, img_a, wa, ha, img_b, wb, hb, matches, n_geom_a, n_geom_b)
    with open(args.out, "w") as f:
        f.write(html)
    counts = {}
    for m in matches:
        counts[m["status"]] = counts.get(m["status"], 0) + 1
    print(f"wrote {args.out}  ({len(matches)} elements: {counts})")


if __name__ == "__main__":
    main()
