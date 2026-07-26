/* A deliberately small Markdown subset, for chat answers only.
 *
 * The model writes Markdown whether or not you ask it to -- `**bold**`
 * labels and numbered lists, mostly. Rendered into a div as-is, the
 * asterisks show literally and every newline collapses to a space, so a
 * tidy three-point answer arrives as one run-on paragraph.
 *
 * Hand-rolled rather than vendored: a full CommonMark parser is ~50x the
 * code for a feature that needs bold, code and lists, and this project
 * already prefers a small correct thing it owns (see the BM25 index) to a
 * dependency it does not.
 *
 * INPUT MUST ALREADY BE HTML-ESCAPED. This function inserts tags; it does
 * not sanitize. Callers escape first, then render, then substitute
 * citation chips -- in that order, so a chip's markup is never re-escaped
 * and a `[A:1:F-7:el_…]` marker survives untouched (no link syntax is
 * implemented, precisely so those brackets are left alone).
 *
 * NOT supported, on purpose:
 *   - underscore emphasis (`_x_`, `__x__`). P&ID vocabulary is full of
 *     underscores -- `line_tag`, `pipe_class`, `note_deleted`,
 *     `element_type` -- and treating them as emphasis would mangle the
 *     exact identifiers an engineer is reading the answer for.
 *   - single-asterisk italics. Same risk for less payoff; drawings carry
 *     stray asterisks (footnote marks, `3/4"*`) and a false match would
 *     silently eat text.
 *   - links, images, tables, blockquotes, headings. A grounded two-document
 *     answer has no use for them, and `[...]` must stay literal.
 */

const OL = /^\s*(\d+)[.)]\s+(.*)$/;
const UL = /^\s*[-*+]\s+(.*)$/;

/** Bold and inline code only -- see the header for what is excluded. */
function inline(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(?=\S)([^*]*[^*\s])\*\*/g, "<strong>$1</strong>");
}

function listBlock(tag, lines, re, textGroup, start) {
  const items = [];
  for (const line of lines) {
    const m = line.match(re);
    if (m) items.push(m[textGroup]);
    // A wrapped continuation line belongs to the item above it, not to a
    // new one -- models hard-wrap long list items.
    else if (items.length) items[items.length - 1] += " " + line.trim();
    else items.push(line.trim());
  }
  const open = tag === "ol" && start > 1 ? `<ol start="${start}">` : `<${tag}>`;
  return open + items.map((i) => `<li>${inline(i)}</li>`).join("") + `</${tag}>`;
}

function block(chunk) {
  const lines = chunk.split("\n").filter((l) => l.trim() !== "");
  if (!lines.length) return "";

  const ol = lines[0].match(OL);
  // `start` preserves numbering when a model separates items with blank
  // lines: each becomes its own block, and without this every one would
  // restart at 1.
  if (ol) return listBlock("ol", lines, OL, 2, parseInt(ol[1], 10));
  if (UL.test(lines[0])) return listBlock("ul", lines, UL, 1, 1);

  return `<p>${lines.map(inline).join("<br>")}</p>`;
}

export function renderMarkdown(escaped) {
  return String(escaped ?? "").trim().split(/\n\s*\n/).map(block).join("");
}
