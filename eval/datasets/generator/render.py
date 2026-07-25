"""Render a Sheet model to vector PDF (L0), DXF, and a degradation ladder
of raster PDFs (L1..L3). Degradation records its transform so L1 bbox
labels can be mapped into degraded space.

Two PDF producer variants ('standard', 'alt') differ in font and text
emission granularity — used for producer-variation null pairs.
"""
from __future__ import annotations

import io
import json
import math
import random

import ezdxf
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageFilter
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas

from .model import Sheet

MM2PT = mm  # reportlab unit


def render_pdf(sheet: Sheet, path: str, producer: str = "standard") -> None:
    c = rl_canvas.Canvas(path, pagesize=(sheet.width * MM2PT, sheet.height * MM2PT))
    font = "Helvetica" if producer == "standard" else "Courier"
    jitter = 0.0 if producer == "standard" else 0.05  # sub-point coordinate noise
    rng = random.Random(0)
    # border + zone grid lines
    c.setLineWidth(0.6)
    c.rect(6 * MM2PT, 6 * MM2PT, (sheet.width - 12) * MM2PT, (sheet.height - 12) * MM2PT)
    for el in sorted(sheet.elements.values(), key=lambda e: e.eid):
        x, y = el.anchor
        if jitter:
            x += rng.uniform(-jitter, jitter); y += rng.uniform(-jitter, jitter)
        if el.role == "geom_line":
            c.setLineWidth(0.9)
            c.line(x * MM2PT, y * MM2PT, (x + el.attrs["dx"]) * MM2PT, (y + el.attrs["dy"]) * MM2PT)
        elif el.role == "geom_circle":
            c.setLineWidth(0.9)
            c.circle(x * MM2PT, y * MM2PT, el.attrs["r"] * MM2PT)
        elif el.text:
            c.setFont(font, el.font_mm * MM2PT)
            if producer == "standard":
                c.drawString(x * MM2PT, y * MM2PT, el.text)
            else:  # alt producer fragments text runs word-by-word
                cx = x
                for w in el.text.split(" "):
                    c.drawString(cx * MM2PT, y * MM2PT, w)
                    cx += (len(w) + 1) * el.font_mm * 0.62
    c.showPage()
    c.save()


def render_dxf(sheet: Sheet, path: str) -> None:
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    layers = {el.layer for el in sheet.elements.values()}
    for ly in layers:
        if ly not in doc.layers:
            doc.layers.add(ly)
    for el in sorted(sheet.elements.values(), key=lambda e: e.eid):
        x, y = el.anchor
        if el.role == "geom_line":
            msp.add_line((x, y), (x + el.attrs["dx"], y + el.attrs["dy"]),
                         dxfattribs={"layer": el.layer})
        elif el.role == "geom_circle":
            msp.add_circle((x, y), el.attrs["r"], dxfattribs={"layer": el.layer})
        elif el.text:
            msp.add_text(el.text, dxfattribs={"layer": el.layer, "height": el.font_mm,
                                              "insert": (x, y)})
    doc.saveas(path)


# ---- degradation ladder ---------------------------------------------------

def degrade(pdf_path: str, out_pdf: str, level: int, seed: int) -> dict:
    """L1: 300dpi clean raster. L2: 200dpi + JPEG q70. L3: + skew/noise/blur.
    Returns the transform record (must be stored next to GT labels)."""
    rng = random.Random(seed)
    # A1 sheet: 200dpi ≈ 6600x4700 px. 300dpi (~70MP) is realistic for archive
    # scans but too slow for iterating; bump via DPI_LADDER if needed.
    dpi = {1: 200, 2: 150, 3: 120}[level]
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    transform = {"level": level, "dpi": dpi, "skew_deg": 0.0, "jpeg_q": None,
                 "noise_sigma": 0.0, "blur_px": 0.0}
    if level >= 2:
        transform["jpeg_q"] = 70
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=70)
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if level >= 3:
        skew = rng.uniform(0.3, 1.2) * rng.choice([-1, 1])
        transform["skew_deg"] = round(skew, 3)
        img = img.rotate(skew, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))
        arr = np.asarray(img).astype(np.float32)
        sigma = rng.uniform(4, 9)
        transform["noise_sigma"] = round(sigma, 2)
        arr += np.random.default_rng(seed).normal(0, sigma, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        blur = rng.uniform(0.4, 0.9)
        transform["blur_px"] = round(blur, 2)
        img = img.filter(ImageFilter.GaussianBlur(blur))
    out = fitz.open()
    w_pt, h_pt = page.rect.width, page.rect.height
    p = out.new_page(width=w_pt, height=h_pt)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    p.insert_image(p.rect, stream=buf.getvalue())
    out.save(out_pdf)
    return transform


def label_transform_matrix(transform: dict, width_mm: float, height_mm: float):
    """Homogeneous matrix mapping model-space mm -> degraded-image px,
    so L1 element anchors/bboxes remain valid ground truth on rasters."""
    s = transform["dpi"] / 25.4
    th = math.radians(transform["skew_deg"])
    cx, cy = width_mm * s / 2, height_mm * s / 2
    # scale, y-flip, then rotate about image centre (PIL rotates about centre)
    def to_px(x_mm, y_mm):
        px, py = x_mm * s, (height_mm - y_mm) * s
        dx, dy = px - cx, py - cy
        return (cx + dx * math.cos(th) - dy * math.sin(th),
                cy + dx * math.sin(th) + dy * math.cos(th))
    return to_px
