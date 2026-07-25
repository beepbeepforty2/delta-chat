"""End-user delta report: a single self-contained HTML file, opt-in
(`python -m src.cli run ... --html`, off by default so a normal run stays
report.json/.md only).

This is the UI layer for the person who requested the diff, not a
debugging tool for the person building the engine -- that distinction
matters for what feeds it: every box, badge, and row here comes from the
REAL deterministic pipeline (src/delta/'s register -> align -> classify ->
severity -> semantic_null -> raster_diff -> raster_join), the same `Delta` objects
`report.py` renders to markdown. It borrows its two-pane-plus-sidebar
layout and click-to-navigate interaction from `tools/visual_diff.py`,
which is a deliberately independent, naive matcher built as a sanity check
against the real engine -- reusing its UI shape here is fine (a good
layout is a good layout), reusing its *matching logic* would defeat the
point of this report, so nothing is imported from it.

Boxes are positioned client-side as absolute-percentage divs against each
sheet's L0 raster (`CanonicalDocument.raster_paths`, already on disk --
just base64-inlined, not re-rendered), using `CanonicalElement.bbox`
directly: bbox is normalized [0,1] against that same raster for both
adapters (see `overlay.py`'s own docstring), so plain percentages are
correct with no pixel-size lookup needed -- simpler than `overlay.py`'s
PIL path, which denormalizes to pixels because PIL draws server-side.

Reuses `overlay.py::_index_elements`/`_collect_boxes` for delta -> element
-> bbox resolution, so this, the PNG overlay, and the PDF annotator all
agree on WHAT gets a box; this module only adds WHERE the sidebar's
metadata (severity, confidence, semantic_null, cascade) comes from, which
neither of those two markup paths needed.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from src.canonical.model import CanonicalDocument
from src.delta.model import Delta
from src.delta.severity import SEVERITY_ORDER
from src.markup.overlay import COLORS, _collect_boxes, _index_elements

_KIND_LABELS = {
    "add": "Added",
    "remove": "Removed",
    "modify": "Modified",
    "move": "Moved",
    "unclassified_visual_change": "Unclassified visual change",
}


def _bbox_list(el) -> list[float]:
    b = el.bbox
    return [b.x0, b.y0, b.x1, b.y1]


def _bbox_obj_list(b) -> list[float]:
    return [b.x0, b.y0, b.x1, b.y1]


def _build_delta_records(deltas: list[Delta], boxes_a: dict, boxes_b: dict) -> list[dict]:
    """One record per Delta (primary AND cascade -- the sidebar filters,
    it doesn't pre-drop). `unclassified_visual_change` deltas have no
    id_a/id_b by construction (raster_join.py finds pixels, not
    elements), so they never resolve through the element-lookup path
    (boxes_a/boxes_b) -- but raster_join.py DOES set bbox_a/bbox_b
    directly on the Delta itself (the region it found), so that's used
    as a fallback location. Only a Delta with genuinely neither (older
    callers, or a future producer that doesn't set them) falls through
    to the sidebar's "no exact location" note."""
    box_a_by_did = {d.did: _bbox_list(el) for entries in boxes_a.values() for el, d in entries}
    box_b_by_did = {d.did: _bbox_list(el) for entries in boxes_b.values() for el, d in entries}
    records = []
    for d in deltas:
        box_a = box_a_by_did.get(d.did) or (_bbox_obj_list(d.bbox_a) if d.bbox_a is not None else None)
        box_b = box_b_by_did.get(d.did) or (_bbox_obj_list(d.bbox_b) if d.bbox_b is not None else None)
        records.append({
            "did": d.did,
            "kind": d.kind,
            "element_type": d.element_type,
            "sheet": d.sheet,
            "zone": d.zone_b or d.zone_a or "?",
            "severity": d.severity or "low",
            "confidence": round(d.confidence, 2),
            "description": d.description or "",
            "is_cascade": d.is_cascade,
            "primary_did": d.primary_did,
            "semantic_null": d.semantic_null,
            "semantic_null_reason": d.semantic_null_reason,
            "visual_change_kind": d.visual_change_kind,
            "box_a": box_a,
            "box_b": box_b,
        })
    return records


def _b64_png(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _build_sheets(doc_a: CanonicalDocument, doc_b: CanonicalDocument) -> list[dict]:
    sheet_nos = sorted(set(doc_a.raster_paths) | set(doc_b.raster_paths))
    sheets = []
    for n in sheet_nos:
        sheets.append({
            "number": n,
            "img_a": _b64_png(doc_a.raster_paths[n]) if n in doc_a.raster_paths else None,
            "img_b": _b64_png(doc_b.raster_paths[n]) if n in doc_b.raster_paths else None,
        })
    return sheets


def _kind_hex(kind: str) -> str:
    r, g, b = COLORS[kind]
    return f"#{r:02x}{g:02x}{b:02x}"


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Delta report — __TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  color-scheme: light;
  --bg: #eef1f6; --surface: #ffffff; --surface-2: #f4f6fa; --surface-3: #e9edf4;
  --ink: #12151f; --ink-2: #4b5567; --muted: #8892a6;
  --border: rgba(18,21,31,0.11);
  --accent: #0e8a7a; --accent-wash: rgba(14,138,122,0.12);
  --sev-critical: #a5222d; --sev-high: #c2610c; --sev-medium: #9a7b0a; --sev-low: #6b7280;
  --sev-critical-wash: rgba(165,34,45,0.10); --sev-high-wash: rgba(194,97,12,0.10);
  --sev-medium-wash: rgba(154,123,10,0.10); --sev-low-wash: rgba(107,114,128,0.10);
  --shadow: 0 1px 2px rgba(18,21,31,0.06), 0 4px 16px rgba(18,21,31,0.05);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --bg: #0d1017; --surface: #161a24; --surface-2: #1c212d; --surface-3: #232937;
    --ink: #eef1f6; --ink-2: #b9c1d3; --muted: #7c8598;
    --border: rgba(238,241,246,0.11);
    --accent: #2dd4bf; --accent-wash: rgba(45,212,191,0.14);
    --sev-critical: #f0605f; --sev-high: #f0975a; --sev-medium: #e0c358; --sev-low: #98a2b3;
    --sev-critical-wash: rgba(240,96,95,0.14); --sev-high-wash: rgba(240,151,90,0.14);
    --sev-medium-wash: rgba(224,195,88,0.14); --sev-low-wash: rgba(152,162,179,0.14);
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 20px rgba(0,0,0,0.35);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0d1017; --surface: #161a24; --surface-2: #1c212d; --surface-3: #232937;
  --ink: #eef1f6; --ink-2: #b9c1d3; --muted: #7c8598;
  --border: rgba(238,241,246,0.11);
  --accent: #2dd4bf; --accent-wash: rgba(45,212,191,0.14);
  --sev-critical: #f0605f; --sev-high: #f0975a; --sev-medium: #e0c358; --sev-low: #98a2b3;
  --sev-critical-wash: rgba(240,96,95,0.14); --sev-high-wash: rgba(240,151,90,0.14);
  --sev-medium-wash: rgba(224,195,88,0.14); --sev-low-wash: rgba(152,162,179,0.14);
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 20px rgba(0,0,0,0.35);
}
:root[data-theme="light"] {
  color-scheme: light;
  --bg: #eef1f6; --surface: #ffffff; --surface-2: #f4f6fa; --surface-3: #e9edf4;
  --ink: #12151f; --ink-2: #4b5567; --muted: #8892a6;
  --border: rgba(18,21,31,0.11);
  --accent: #0e8a7a; --accent-wash: rgba(14,138,122,0.12);
  --sev-critical: #a5222d; --sev-high: #c2610c; --sev-medium: #9a7b0a; --sev-low: #6b7280;
  --sev-critical-wash: rgba(165,34,45,0.10); --sev-high-wash: rgba(194,97,12,0.10);
  --sev-medium-wash: rgba(154,123,10,0.10); --sev-low-wash: rgba(107,114,128,0.10);
  --shadow: 0 1px 2px rgba(18,21,31,0.06), 0 4px 16px rgba(18,21,31,0.05);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
.mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
header {
  padding: 18px 22px; background: var(--surface); border-bottom: 1px solid var(--border);
  display: flex; align-items: flex-start; gap: 22px; flex-wrap: wrap; box-shadow: var(--shadow);
}
header .titlewrap { margin-right: auto; min-width: 260px; }
.eyebrow {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 11px;
  letter-spacing: 0.09em; text-transform: uppercase; color: var(--accent); font-weight: 600;
}
header h1 { font-size: 17px; font-weight: 650; margin: 4px 0 3px; letter-spacing: -0.01em; }
header .subtitle { font-size: 12px; color: var(--muted); }
.stats { display: flex; gap: 10px; flex-wrap: wrap; }
.stat {
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px;
  padding: 8px 14px; min-width: 84px; text-align: center;
}
.stat .n { font-size: 19px; font-weight: 700; font-variant-numeric: tabular-nums; display: block; }
.stat .label { font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
.stat.critical .n { color: var(--sev-critical); }
.stat.high .n { color: var(--sev-high); }
.stat.medium .n { color: var(--sev-medium); }
.stat.low .n { color: var(--sev-low); }

.sheettabs { display: flex; gap: 6px; padding: 10px 22px 0; background: var(--surface); }
.sheettabs button {
  border: 1px solid var(--border); background: var(--surface-2); color: var(--ink-2);
  padding: 6px 14px; border-radius: 8px 8px 0 0; font-size: 12.5px; cursor: pointer;
  border-bottom: none; position: relative; top: 1px;
}
.sheettabs button.active { background: var(--bg); color: var(--ink); font-weight: 600; border-bottom: 1px solid var(--bg); }

main { display: grid; grid-template-columns: 1fr 1fr 340px; gap: 1px; background: var(--border); height: calc(100vh - 138px); }
.sheetview { display: none; }
.sheetview.active { display: contents; }
.pane { background: var(--bg); overflow: auto; position: relative; }
.pane-label {
  position: sticky; top: 0; z-index: 3; padding: 7px 14px; font-size: 11px;
  font-weight: 650; letter-spacing: 0.03em; text-transform: uppercase;
  color: var(--muted); background: var(--surface); border-bottom: 1px solid var(--border);
}
.imgwrap { position: relative; width: 100%; }
.imgwrap img { width: 100%; display: block; }
.imgwrap .empty { padding: 40px; color: var(--muted); text-align: center; font-size: 13px; }
.box {
  position: absolute; border-radius: 2px; cursor: pointer;
  background: transparent; transition: box-shadow 0.12s, background 0.12s;
}
.box.hidden { display: none; }
.box.active { z-index: 4; box-shadow: 0 0 0 3px var(--accent-wash); background: var(--accent-wash); }

aside { background: var(--surface); display: flex; flex-direction: column; overflow: hidden; }
.toolbar { padding: 12px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 8px; }
.toolbar input {
  width: 100%; padding: 7px 9px; border-radius: 7px; border: 1px solid var(--border);
  background: var(--surface-2); color: var(--ink); font-size: 12.5px;
}
.chiprow { display: flex; gap: 6px; flex-wrap: wrap; }
.chip {
  display: inline-flex; align-items: center; gap: 5px; padding: 4px 9px;
  border-radius: 6px; border: 1px solid var(--border); background: var(--surface-2);
  font-size: 11.5px; cursor: pointer; user-select: none; color: var(--ink-2);
}
.chip input { accent-color: var(--accent); margin: 0; }
.chip .dot { width: 8px; height: 8px; border-radius: 2px; flex: none; }
.chip.off { opacity: 0.45; }
.chip .count { font-variant-numeric: tabular-nums; color: var(--muted); }

.rows { overflow: auto; flex: 1; }
.row { padding: 8px 12px; border-bottom: 1px solid var(--border); cursor: pointer; font-size: 12px; }
.row:hover { background: var(--surface-2); }
.row.active { background: var(--accent-wash); }
.row.cascade { opacity: 0.72; }
.row .top { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; flex-wrap: wrap; }
.row .kind-dot { width: 8px; height: 8px; border-radius: 2px; flex: none; }
.row .sev-badge {
  font-size: 9.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;
  padding: 1.5px 6px; border-radius: 4px;
}
.row .sev-badge.critical { color: var(--sev-critical); background: var(--sev-critical-wash); }
.row .sev-badge.high { color: var(--sev-high); background: var(--sev-high-wash); }
.row .sev-badge.medium { color: var(--sev-medium); background: var(--sev-medium-wash); }
.row .sev-badge.low { color: var(--sev-low); background: var(--sev-low-wash); }
.row .zone { color: var(--muted); font-size: 10.5px; }
.row .nullbadge {
  font-size: 9.5px; color: var(--muted); border: 1px dashed var(--border);
  border-radius: 4px; padding: 1px 5px;
}
.row .desc { color: var(--ink-2); font-size: 12px; line-height: 1.4; }
.row .noloc { color: var(--muted); font-size: 10.5px; font-style: italic; margin-top: 2px; }
.footnote { padding: 9px 12px; font-size: 11px; color: var(--muted); border-top: 1px solid var(--border); }
</style>
</head>
<body>
<header>
  <div class="titlewrap">
    <div class="eyebrow">delta-chat report</div>
    <h1 id="h1title"></h1>
    <div class="subtitle">Deterministic delta engine output (register → align → classify → severity) — not a re-derived comparison.</div>
  </div>
  <div class="stats" id="stats"></div>
</header>
<div class="sheettabs" id="sheettabs"></div>
<main id="main"></main>
<script>
const DATA = __DATA_JSON__;
const KIND_COLOR = __KIND_COLOR_JSON__;
const KIND_LABEL = __KIND_LABEL_JSON__;
const KINDS = Object.keys(KIND_COLOR);
const SEVS = ["critical", "high", "medium", "low"];
let activeKind = {}; KINDS.forEach(k => activeKind[k] = true);
let showCascade = false;
let currentSheet = DATA.sheets[0] ? DATA.sheets[0].number : null;
let selected = null;

document.getElementById("h1title").textContent = DATA.pid_a + "  →  " + DATA.pid_b;

const statsEl = document.getElementById("stats");
function stat(label, n, cls) {
  const d = document.createElement("div");
  d.className = "stat" + (cls ? " " + cls : "");
  d.innerHTML = `<span class="n">${n}</span><span class="label">${label}</span>`;
  return d;
}
statsEl.appendChild(stat("primary", DATA.summary.n_primary));
statsEl.appendChild(stat("cascade", DATA.summary.n_cascade));
SEVS.forEach(s => {
  const n = DATA.summary.severity_counts[s] || 0;
  if (n) statsEl.appendChild(stat(s, n, s));
});
if (DATA.summary.n_semantic_null) statsEl.appendChild(stat("flagged no-op", DATA.summary.n_semantic_null));

const tabsEl = document.getElementById("sheettabs");
const mainEl = document.getElementById("main");

function boxDiv(did, box, kind) {
  const d = document.createElement("div");
  d.className = "box";
  d.dataset.did = did;
  d.style.borderColor = KIND_COLOR[kind];
  d.style.borderWidth = "2.5px";
  d.style.borderStyle = "solid";
  d.style.left = (box[0] * 100) + "%";
  d.style.top = (box[1] * 100) + "%";
  d.style.width = ((box[2] - box[0]) * 100) + "%";
  d.style.height = ((box[3] - box[1]) * 100) + "%";
  d.addEventListener("click", () => selectDelta(did));
  return d;
}

DATA.sheets.forEach(sh => {
  const tab = document.createElement("button");
  tab.textContent = "Sheet " + sh.number;
  tab.className = sh.number === currentSheet ? "active" : "";
  tab.addEventListener("click", () => switchSheet(sh.number));
  tabsEl.appendChild(tab);

  const view = document.createElement("div");
  view.className = "sheetview" + (sh.number === currentSheet ? " active" : "");
  view.dataset.sheet = sh.number;

  const paneA = document.createElement("div");
  paneA.className = "pane";
  paneA.innerHTML = `<div class="pane-label">${DATA.pid_a} — sheet ${sh.number}</div>`;
  const wrapA = document.createElement("div");
  wrapA.className = "imgwrap";
  wrapA.innerHTML = sh.img_a ? `<img src="data:image/png;base64,${sh.img_a}">` : `<div class="empty">no page ${sh.number} in ${DATA.pid_a}</div>`;
  paneA.appendChild(wrapA);
  view.appendChild(paneA);

  const paneB = document.createElement("div");
  paneB.className = "pane";
  paneB.innerHTML = `<div class="pane-label">${DATA.pid_b} — sheet ${sh.number}</div>`;
  const wrapB = document.createElement("div");
  wrapB.className = "imgwrap";
  wrapB.innerHTML = sh.img_b ? `<img src="data:image/png;base64,${sh.img_b}">` : `<div class="empty">no page ${sh.number} in ${DATA.pid_b}</div>`;
  paneB.appendChild(wrapB);
  view.appendChild(paneB);

  DATA.deltas.filter(d => d.sheet === sh.number).forEach(d => {
    if (d.box_a) wrapA.appendChild(boxDiv(d.did, d.box_a, d.kind));
    if (d.box_b) wrapB.appendChild(boxDiv(d.did, d.box_b, d.kind));
  });

  mainEl.appendChild(view);
});

const asideEl = document.createElement("aside");
asideEl.innerHTML = `
  <div class="toolbar">
    <input id="search" type="text" placeholder="Filter by description or zone...">
    <div class="chiprow" id="kindChips"></div>
    <label class="chip" id="cascadeChip"><input type="checkbox" id="cascadeToggle"><span>show cascade changes</span></label>
  </div>
  <div class="rows" id="rows"></div>
  <div class="footnote" id="footnote"></div>
`;
mainEl.appendChild(asideEl);

const kindChips = asideEl.querySelector("#kindChips");
const counts = {};
KINDS.forEach(k => counts[k] = 0);
DATA.deltas.forEach(d => { if (!d.is_cascade) counts[d.kind] = (counts[d.kind] || 0) + 1; });
KINDS.forEach(k => {
  if (!counts[k]) return;
  const chip = document.createElement("label");
  chip.className = "chip";
  chip.innerHTML = `<input type="checkbox" checked><span class="dot" style="background:${KIND_COLOR[k]}"></span>${KIND_LABEL[k]} <span class="count">${counts[k]}</span>`;
  chip.querySelector("input").addEventListener("change", e => {
    activeKind[k] = e.target.checked;
    chip.classList.toggle("off", !e.target.checked);
    renderRows();
  });
  kindChips.appendChild(chip);
});
asideEl.querySelector("#cascadeToggle").addEventListener("change", e => {
  showCascade = e.target.checked;
  renderRows();
});

const rowsEl = asideEl.querySelector("#rows");
function renderRows() {
  const q = asideEl.querySelector("#search").value.toLowerCase();
  rowsEl.innerHTML = "";
  let shown = 0;
  DATA.deltas.forEach(d => {
    if (!activeKind[d.kind]) return;
    if (d.is_cascade && !showCascade) return;
    if (q && !((d.description || "").toLowerCase().includes(q) || (d.zone || "").toLowerCase().includes(q))) return;
    shown++;
    const row = document.createElement("div");
    row.className = "row" + (d.is_cascade ? " cascade" : "");
    row.dataset.did = d.did;
    const nullBadge = d.semantic_null ? `<span class="nullbadge" title="${(d.semantic_null_reason || '').replace(/"/g, '&quot;')}">≈ no-op</span>` : "";
    const noloc = (!d.box_a && !d.box_b) ? `<div class="noloc">no exact location — zone only (raster recall)</div>` : "";
    row.innerHTML = `<div class="top">
        <span class="kind-dot" style="background:${KIND_COLOR[d.kind]}"></span>
        <span class="sev-badge ${d.severity}">${d.severity}</span>
        <span class="zone">Sheet ${d.sheet}, ${d.zone}</span>
        ${nullBadge}
      </div>
      <div class="desc">${(d.description || "(no description)").replace(/</g, "&lt;")}</div>
      ${noloc}`;
    row.addEventListener("click", () => selectDelta(d.did));
    rowsEl.appendChild(row);
  });
  asideEl.querySelector("#footnote").textContent = shown + " of " + DATA.deltas.length + " deltas shown";
}
asideEl.querySelector("#search").addEventListener("input", renderRows);

function switchSheet(n) {
  currentSheet = n;
  document.querySelectorAll(".sheettabs button").forEach(b => b.classList.toggle("active", b.textContent === "Sheet " + n));
  document.querySelectorAll(".sheetview").forEach(v => v.classList.toggle("active", Number(v.dataset.sheet) === n));
}

function selectDelta(did) {
  if (selected !== null) {
    document.querySelectorAll(`[data-did="${selected}"]`).forEach(el => el.classList.remove("active"));
  }
  selected = did;
  const d = DATA.deltas.find(x => x.did === did);
  if (d && d.sheet !== currentSheet) switchSheet(d.sheet);
  requestAnimationFrame(() => {
    const els = document.querySelectorAll(`[data-did="${did}"]`);
    els.forEach(el => {
      el.classList.add("active");
      if (el.classList.contains("box")) el.scrollIntoView({block: "center", inline: "center", behavior: "smooth"});
    });
    const row = rowsEl.querySelector(`.row[data-did="${did}"]`);
    if (row) row.scrollIntoView({block: "nearest"});
  });
}

renderRows();
</script>
</body>
</html>
"""


def render_html_report(doc_a: CanonicalDocument, doc_b: CanonicalDocument, deltas: list[Delta],
                        pid_a_label: str, pid_b_label: str, out_path: str) -> str:
    """Writes a single self-contained HTML file to out_path (base64-inlined
    rasters, no external assets, opens directly from disk). pid_a_label/
    pid_b_label are display names only (e.g. the --a/--b paths passed to
    the CLI) -- everything else comes straight off doc_a/doc_b/deltas."""
    els_a, els_b = _index_elements(doc_a), _index_elements(doc_b)
    boxes_a, boxes_b = _collect_boxes(deltas, els_a, els_b)
    records = _build_delta_records(deltas, boxes_a, boxes_b)
    sheets = _build_sheets(doc_a, doc_b)

    primary = [d for d in deltas if not d.is_cascade]
    severity_counts = {sev: sum(1 for d in primary if d.severity == sev) for sev in SEVERITY_ORDER}

    data = {
        "pid_a": pid_a_label,
        "pid_b": pid_b_label,
        "sheets": sheets,
        "deltas": records,
        "summary": {
            "n_primary": len(primary),
            "n_cascade": sum(1 for d in deltas if d.is_cascade),
            "severity_counts": severity_counts,
            "n_semantic_null": sum(1 for d in primary if d.semantic_null),
        },
    }
    kind_color = {kind: _kind_hex(kind) for kind in COLORS}
    kind_label = dict(_KIND_LABELS)

    html = (HTML_TEMPLATE
            .replace("__TITLE__", f"{pid_a_label} vs {pid_b_label}")
            .replace("__DATA_JSON__", json.dumps(data))
            .replace("__KIND_COLOR_JSON__", json.dumps(kind_color))
            .replace("__KIND_LABEL_JSON__", json.dumps(kind_label)))

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return str(path)
