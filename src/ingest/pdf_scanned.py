"""Scanned (raster) PDF adapter: rasterize -> pytesseract OCR -> canonical.

Spike findings that drove this design (see CLAUDE.md Steps #4 for the
write-up): tesseract's own word-level detections (level=5) are reliable,
but its own line/block/par grouping is NOT -- it merged an entire row of
widely-spaced zone-grid digits (~540px apart) into one polluted string on
a real degraded page, while correctly joining normal 7-9px-spaced prose
words on the same page. Same failure class as pdf_native.py's fitz-line
problem, same fix: ignore the OCR engine's own line grouping and re-cluster
words ourselves by (same baseline, small x-gap relative to word height).

Deliberately does NOT attempt geometry extraction (no line/circle
detection from the raster) -- that's a real computer-vision problem
(Hough transforms etc.), out of scope for a pytesseract-based adapter and
a bigger lift than the OCR text path for the same time budget. Documented
cut, not a silent gap.

extraction_confidence is real per-element OCR confidence here (unlike
pdf_native's always-1.0), per CLAUDE.md decision #1 ("1.0 native; OCR
conf for scans") and src/canonical/model.py's own field comment.
"""
from __future__ import annotations

import os
import re
from typing import Optional

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from src.canonical.model import BBox, CanonicalDocument, CanonicalElement, CanonicalSheet
from src.canonical.classify import classify
from src.canonical.zones import compute_zone
from src.ingest.base import FormatAdapter, element_id
from src.ingest.pdf_native import MIN_TEXT_WORDS as NATIVE_MIN_TEXT_WORDS

OCR_DPI = int(os.environ.get("PDF_SCANNED_OCR_DPI", "200"))
GAP_MULTIPLIER = float(os.environ.get("PDF_SCANNED_GAP_MULTIPLIER", "3.0"))
Y_TOL_RATIO = float(os.environ.get("PDF_SCANNED_Y_TOL_RATIO", "0.5"))
RASTER_CACHE_DIR = os.environ.get("PDF_SCANNED_RASTER_CACHE_DIR",
                                   os.environ.get("PDF_NATIVE_RASTER_CACHE_DIR", "raster_cache"))
MIN_OCR_CONF = int(os.environ.get("PDF_SCANNED_MIN_OCR_CONF", "0"))  # tesseract's -1 sentinel is always dropped

_WS_RE = re.compile(r"\s+")


def _page_image(page: "fitz.Page", dpi: int = OCR_DPI) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _ocr_words(img: Image.Image) -> list[dict]:
    """Non-empty, non-negative-confidence word-level OCR detections,
    normalized to [0,1] page-fraction coordinates (top-left/y-down --
    PIL/tesseract's native convention, no flip needed, matches BBox)."""
    w, h = img.size
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < MIN_OCR_CONF:
            continue
        left, top = data["left"][i], data["top"][i]
        width, height = data["width"][i], data["height"][i]
        words.append({
            "text": text, "conf": conf,
            "x0": left / w, "y0": top / h,
            "x1": (left + width) / w, "y1": (top + height) / h,
            "height_norm": height / h,
        })
    return words


def _cluster_words(words: list[dict]) -> list[list[dict]]:
    """Two passes, not one sorted sweep -- unlike fitz's exact vector
    coordinates (same drawString call = identical origin y for every word
    on a line), OCR word boxes jitter a pixel or two vertically even on
    one physical line. Sorting by (round(y0), x0) as a single key put
    jittered words from the SAME line into different sort buckets,
    scrambling left-to-right order and corrupting the gap merge --
    observed on a real degraded page: "OIL CHANGE BY USING..." split into
    "4. OIL [gap] BY USING" and "CHANGE [gap] TEMPORARY ARRANGEMENT..."
    as two separate, wrongly-ordered fragments. Band first (y-proximity,
    order-independent), then sweep each band left-to-right by x.
    """
    bands: list[list[dict]] = []
    for w in sorted(words, key=lambda w: w["y0"]):
        if bands and abs(w["y0"] - bands[-1][0]["y0"]) <= Y_TOL_RATIO * bands[-1][0]["height_norm"]:
            bands[-1].append(w)
        else:
            bands.append([w])

    clusters: list[list[dict]] = []
    for band in bands:
        cur: list[dict] = []
        for w in sorted(band, key=lambda w: w["x0"]):
            if not cur:
                cur = [w]
                continue
            prev = cur[-1]
            gap = w["x0"] - prev["x1"]
            # gap>=0 guard matters here too -- without it two words merely
            # close in y but far apart in x (like the zone-digit row) can
            # wrongly chain together.
            if 0 <= gap <= GAP_MULTIPLIER * prev["height_norm"]:
                cur.append(w)
            else:
                clusters.append(cur)
                cur = [w]
        if cur:
            clusters.append(cur)
    return clusters


def _cluster_text(cluster: list[dict]) -> str:
    return _WS_RE.sub(" ", " ".join(w["text"] for w in cluster)).strip()


def _cluster_bbox(cluster: list[dict]) -> BBox:
    return BBox(
        min(w["x0"] for w in cluster), min(w["y0"] for w in cluster),
        max(w["x1"] for w in cluster), max(w["y1"] for w in cluster),
    )


def _cluster_confidence(cluster: list[dict]) -> float:
    return round(sum(w["conf"] for w in cluster) / len(cluster) / 100.0, 4)


def _text_elements(img: Image.Image, sheet_no: int) -> list[CanonicalElement]:
    out = []
    for cluster in _cluster_words(_ocr_words(img)):
        text = _cluster_text(cluster)
        if not text:
            continue
        bbox = _cluster_bbox(cluster)
        # anchor = first word's top-left corner, the same role fitz span
        # origin plays in pdf_native.py -- zone/type computed from this
        # point, not the bbox centroid.
        ax_norm, ay_norm = cluster[0]["x0"], cluster[0]["y0"]
        etype, attrs = classify(text, ax_norm, ay_norm)
        zone = compute_zone(ax_norm, ay_norm)
        confidence = _cluster_confidence(cluster)
        eid = element_id(sheet_no, etype, text, bbox)
        out.append(CanonicalElement(
            id=eid, type=etype, content=text, bbox=bbox, sheet=sheet_no,
            zone=zone, extraction_confidence=confidence, attrs=attrs,
        ))
    return out


def _revision_label(elements: list[CanonicalElement]) -> Optional[str]:
    for el in elements:
        if el.type == "title_field" and el.attrs.get("field") == "rev":
            return el.attrs.get("value")
    return None


class PdfScannedAdapter(FormatAdapter):
    format_name = "pdf_scanned"

    def detect(self, path: str) -> bool:
        """The complement of PdfNativeAdapter.detect(): claims PDFs whose
        native text layer is sparse or absent (a real scan, or one of the
        eval dataset's degraded-raster L1-L3 renders), sharing the exact
        same threshold so there is no gap or overlap between the two
        adapters' claims."""
        if not path.lower().endswith(".pdf"):
            return False
        try:
            doc = fitz.open(path)
            if doc.page_count == 0:
                return False
            n_words = len(doc[0].get_text("text").split())
            doc.close()
            return n_words < NATIVE_MIN_TEXT_WORDS
        except Exception:
            return False

    def ingest(self, pid: str, path: str) -> CanonicalDocument:
        doc = fitz.open(path)
        sheets = []
        raster_paths: dict[int, str] = {}
        os.makedirs(RASTER_CACHE_DIR, exist_ok=True)
        for i, page in enumerate(doc):
            sheet_no = i + 1
            img = _page_image(page)
            out_path = os.path.join(RASTER_CACHE_DIR, f"{pid}_sheet{sheet_no}.png")
            img.save(out_path)
            raster_paths[sheet_no] = out_path

            elements = _text_elements(img, sheet_no)
            sheets.append(CanonicalSheet(
                number=sheet_no, width=float(img.width), height=float(img.height),
                elements=elements,
            ))
        revision_label = _revision_label(sheets[0].elements) if sheets else None
        doc.close()
        return CanonicalDocument(
            pid=pid, source_format="pdf_scanned", revision_label=revision_label,
            sheets=sheets, raster_paths=raster_paths,
        )
