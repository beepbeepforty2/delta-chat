/* Browser-level verification of the web UI (src/web/static/).
 *
 * OPTIONAL and deliberately outside `make test`. The application itself
 * needs no Node and no build step -- that is DESIGN.md decision 11 and it
 * stays true; this is a test harness, not a dependency, and the Python
 * suite is complete without it. It drives whatever Chrome is already
 * installed, so it downloads no browser.
 *
 *   make web                      # in another shell
 *   npm install playwright-core   # once; installs a driver, not a browser
 *   node tools/uicheck.mjs
 *
 * It exists because `tests/test_web_app.py` can prove the API serves the
 * right numbers but cannot prove a human can see them. Everything below is
 * a claim only a real renderer can settle:
 *
 *   - the pdf.js canvas actually paints ink, not just exists
 *   - overlay boxes land ON the drawing and stay registered through zoom
 *   - clicking a change highlights it in both panes AND the sidebar
 *   - locked panes really do pan together
 *   - no console errors
 *
 * It earned its place on first run by catching three bugs the Python
 * tests and a careful code read both missed: a CSS grid item without
 * `min-height: 0` silently clipped the bottom two-thirds of both drawings
 * with no scrollbar; `escapeHtml` turned every `"` into `&quot;` so the
 * JSON view's syntax highlighter matched nothing; and a missing favicon
 * logged a console error on every load.
 *
 * Screenshots land in tools/uicheck-shots/ (gitignored) -- useful for
 * eyeballing a change, and the only record of what a failure looked like.
 */
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { renderMarkdown } from "../src/web/static/md.js";

// Override with CHROME_PATH= for a non-default install or another platform.
const CHROME = process.env.CHROME_PATH ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const BASE = process.env.BASE_URL || "http://127.0.0.1:8000";
const OUT = join(dirname(fileURLToPath(import.meta.url)), "uicheck-shots");
mkdirSync(OUT, { recursive: true });

const results = [];
const check = (name, pass, detail = "") => {
  results.push({ name, pass, detail });
  console.log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
};

// ── markdown renderer: deterministic, no browser and no LLM needed ──────
// Chat answers are Markdown, so these run first and fail fast: a bug here
// would otherwise only show up as an oddly-formatted live answer.
{
  const md = renderMarkdown;
  check("md: bold", md("a **b** c") === "<p>a <strong>b</strong> c</p>");
  check("md: inline code", md("use `AC21` here") === "<p>use <code>AC21</code> here</p>");
  check("md: ordered list",
        md("1. one\n2. two") === "<ol><li>one</li><li>two</li></ol>");
  check("md: bullet list", md("- one\n- two") === "<ul><li>one</li><li>two</li></ul>");
  check("md: blank-line-separated items keep their numbering",
        md("2. two\n\n3. three") === '<ol start="2"><li>two</li></ol><ol start="3"><li>three</li></ol>');
  check("md: wrapped list item stays one item",
        md("1. a long\n   continuation") === "<ol><li>a long continuation</li></ol>");
  check("md: paragraphs split on blank lines",
        md("one\n\ntwo") === "<p>one</p><p>two</p>");
  check("md: single newline is a line break",
        md("one\ntwo") === "<p>one<br>two</p>");
  // The domain-safety rules. Underscores are everywhere in P&ID field names
  // and must never be read as emphasis.
  check("md: underscores in field names survive",
        md("note_deleted note_no changed") === "<p>note_deleted note_no changed</p>");
  check("md: __double underscore__ is not bold",
        md("__x__") === "<p>__x__</p>");
  check("md: lone asterisk is left alone", md('3/4"* mark').includes('3/4"* mark'));
  // Citation markers must pass through untouched for chip substitution.
  check("md: citation marker survives",
        md("see [delta:1:F-8:delta0009].").includes("[delta:1:F-8:delta0009]"));
  check("md: escaped entities are not re-mangled",
        md("a &amp; b") === "<p>a &amp; b</p>");
}

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 });

const consoleErrors = [];
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text()); });
page.on("pageerror", (e) => consoleErrors.push("pageerror: " + e.message));

await page.goto(BASE, { waitUntil: "networkidle" });
await page.screenshot({ path: `${OUT}/01-landing.png` });
check("landing renders", await page.locator("#landing").isVisible());
check("both dropzones present", (await page.locator(".dropzone").count()) === 2);
check("bundled samples offered", (await page.locator("#samples-row .btn").count()) >= 1,
      `${await page.locator("#samples-row .btn").count()} sample button(s)`);

// ── run the bundled pair ────────────────────────────────────────────────
await page.locator("#samples-row .btn").first().click();
await page.waitForSelector("#progress:not([hidden])", { timeout: 5000 }).catch(() => {});
await page.screenshot({ path: `${OUT}/02-progress.png` });
await page.waitForSelector("#workspace:not([hidden])", { timeout: 120000 });
await page.waitForTimeout(2500);                       // let pdf.js finish painting
await page.screenshot({ path: `${OUT}/03-workspace.png` });
check("workspace opened", await page.locator("#workspace").isVisible());

// ── the canvases actually painted ───────────────────────────────────────
for (const side of ["a", "b"]) {
  const info = await page.evaluate((s) => {
    const c = document.querySelector(`.pane[data-side="${s}"] canvas`);
    if (!c) return null;
    const ctx = c.getContext("2d");
    const { data } = ctx.getImageData(0, 0, c.width, c.height);
    let nonWhite = 0;
    for (let i = 0; i < data.length; i += 4 * 97) {     // sparse sample
      if (data[i] < 240 || data[i + 1] < 240 || data[i + 2] < 240) nonWhite++;
    }
    return { w: c.width, h: c.height, cssW: c.clientWidth, nonWhite };
  }, side);
  check(`pane ${side.toUpperCase()} canvas painted`, info && info.nonWhite > 50,
        info ? `${info.w}x${info.h} backing, ${info.cssW}px css, ${info.nonWhite} ink samples` : "no canvas");
}

// ── overlay boxes are on the drawing ────────────────────────────────────
const boxes = await page.evaluate(() => {
  const out = { a: 0, b: 0, offCanvas: [] };
  for (const side of ["a", "b"]) {
    const pane = document.querySelector(`.pane[data-side="${side}"]`);
    const page_ = pane.querySelector(".page").getBoundingClientRect();
    for (const b of pane.querySelectorAll(".box")) {
      out[side]++;
      const r = b.getBoundingClientRect();
      if (r.left < page_.left - 1 || r.top < page_.top - 1 ||
          r.right > page_.right + 1 || r.bottom > page_.bottom + 1) {
        out.offCanvas.push(`${side}:${b.dataset.did}`);
      }
    }
  }
  return out;
});
check("boxes drawn in both panes", boxes.a > 0 && boxes.b > 0, `A=${boxes.a} B=${boxes.b}`);
check("no box escapes the page bounds", boxes.offCanvas.length === 0,
      boxes.offCanvas.length ? boxes.offCanvas.join(", ") : "all inside");

// ── sidebar ─────────────────────────────────────────────────────────────
const nRows = await page.locator("#rows .row").count();
check("change rows listed", nRows > 0, `${nRows} rows`);
const firstSev = await page.locator("#rows .row .sev-badge").first().textContent();
check("rows severity-sorted (highest first)", firstSev.trim() === "high",
      `first badge = ${firstSev.trim()}`);

// ── click a change: both panes + row highlight ──────────────────────────
const did = await page.locator("#rows .row").first().getAttribute("data-did");
await page.locator(`#rows .row[data-did="${did}"]`).click();
await page.waitForTimeout(900);
await page.screenshot({ path: `${OUT}/04-selected.png` });
const sel = await page.evaluate((d) => ({
  activeBoxes: document.querySelectorAll(`.box.active[data-did="${d}"]`).length,
  activeRow: document.querySelectorAll(`.row.active[data-did="${d}"]`).length,
}), did);
check("clicking a change highlights both panes", sel.activeBoxes === 2, `${sel.activeBoxes} active boxes`);
check("clicking a change highlights its row", sel.activeRow === 1);

// ── click a box, sidebar follows ────────────────────────────────────────
const otherDid = await page.locator("#rows .row").nth(2).getAttribute("data-did");
await page.locator(`.pane[data-side="a"] .box[data-did="${otherDid}"]`).click({ force: true });
await page.waitForTimeout(600);
check("clicking a box selects its row",
      await page.locator(`#rows .row.active[data-did="${otherDid}"]`).count() === 1);

// ── filters ─────────────────────────────────────────────────────────────
const before = await page.locator("#rows .row").count();
await page.locator("#kindchips .chip").first().click();
await page.waitForTimeout(400);
const after = await page.locator("#rows .row").count();
check("kind filter changes the list", after < before, `${before} -> ${after}`);
await page.locator("#kindchips .chip").first().click();
await page.waitForTimeout(300);

await page.locator("#search").fill("pipe_class");
await page.waitForTimeout(400);
const searched = await page.locator("#rows .row").count();
check("search filters", searched > 0 && searched < before, `${searched} match "pipe_class"`);
await page.locator("#search").fill("");
await page.waitForTimeout(300);

// ── zoom keeps boxes registered ─────────────────────────────────────────
const geomAt = async () => page.evaluate(() => {
  const pane = document.querySelector('.pane[data-side="a"]');
  const p = pane.querySelector(".page").getBoundingClientRect();
  const b = pane.querySelector(".box").getBoundingClientRect();
  return { rel: (b.left - p.left) / p.width, relY: (b.top - p.top) / p.height, pageW: p.width };
});
const g1 = await geomAt();
await page.locator("#zoom-in").click();
await page.locator("#zoom-in").click();
await page.waitForTimeout(1600);
const g2 = await geomAt();
await page.screenshot({ path: `${OUT}/05-zoomed.png` });
check("zoom actually scales the page", g2.pageW > g1.pageW * 1.4,
      `${Math.round(g1.pageW)}px -> ${Math.round(g2.pageW)}px`);
check("box stays registered through zoom",
      Math.abs(g1.rel - g2.rel) < 0.002 && Math.abs(g1.relY - g2.relY) < 0.002,
      `rel x ${g1.rel.toFixed(5)} -> ${g2.rel.toFixed(5)}`);

// ── locked panes pan together ───────────────────────────────────────────
await page.evaluate(() => {
  document.querySelector('.pane[data-side="a"] .pane-viewport').scrollTop = 400;
  document.querySelector('.pane[data-side="a"] .pane-viewport').dispatchEvent(new Event("scroll"));
});
await page.waitForTimeout(600);
const scrolls = await page.evaluate(() => ({
  a: document.querySelector('.pane[data-side="a"] .pane-viewport').scrollTop,
  b: document.querySelector('.pane[data-side="b"] .pane-viewport').scrollTop,
}));
check("locked panes scroll together", scrolls.b > 100, `A=${scrolls.a} B=${scrolls.b}`);

await page.locator("#zoom-fit").click();
await page.waitForTimeout(1200);

// ── data tab ────────────────────────────────────────────────────────────
await page.locator('.tabbtn[data-tab="json"]').click();
await page.waitForTimeout(700);
await page.screenshot({ path: `${OUT}/06-json.png` });
const jsonLen = (await page.locator("#jsonview").textContent()).length;
check("raw JSON rendered", jsonLen > 1000, `${jsonLen} chars`);
check("JSON syntax-highlighted", (await page.locator("#jsonview .j-key").count()) > 10);
check("all five downloads offered", (await page.locator("#downloads a").count()) === 5);

// ── ask tab ─────────────────────────────────────────────────────────────
await page.locator('.tabbtn[data-tab="ask"]').click();
await page.waitForTimeout(400);
check("chat suggestions built from findings", (await page.locator(".suggestion").count()) >= 2);
await page.screenshot({ path: `${OUT}/07-ask.png` });

await page.locator(".suggestion").first().click();
await page.waitForSelector(".msg.a:not(.thinking)", { timeout: 90000 });
await page.waitForTimeout(600);
await page.screenshot({ path: `${OUT}/08-answer.png` });
const chips = await page.locator(".cite").count();
check("chat answered", (await page.locator(".msg.a").count()) >= 1);
check("citation chips rendered", chips > 0, `${chips} chip(s)`);

// The answer must not arrive as one run-on paragraph of literal asterisks.
const answer = await page.locator(".msg.a").last().innerHTML();
check("answer has no literal markdown left", !answer.includes("**"),
      answer.includes("**") ? "found ** in rendered output" : "clean");
const blocks = await page.locator(".msg.a").last()
  .locator("p, li, strong, code").count();
check("answer is formatted, not a run-on", blocks > 0, `${blocks} block/inline element(s)`);
if (chips > 0) {
  const dead = await page.locator(".cite.dead").count();
  check("citation chips are live (not dead)", dead === 0, `${dead} dead of ${chips}`);
  await page.locator(".cite").first().click();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${OUT}/09-citation-jump.png` });
  check("clicking a citation highlights the drawing",
        (await page.locator(".box.active").count()) > 0);
}

// ── theme toggle ────────────────────────────────────────────────────────
await page.locator("#btn-theme").click();
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/10-dark.png` });
check("theme toggle applies", (await page.evaluate(() =>
  document.documentElement.dataset.theme)) !== undefined);

// ── console hygiene ─────────────────────────────────────────────────────
check("no console errors", consoleErrors.length === 0,
      consoleErrors.slice(0, 3).join(" | ") || "clean");

await browser.close();
const failed = results.filter((r) => !r.pass);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length) { console.log("FAILED: " + failed.map((f) => f.name).join(", ")); process.exit(1); }
