"""Registration: align page coordinate frames between A and B before
spatial matching. DESIGN.md decision #3: "page-level registration
(similarity transform from high-confidence anchors)".

For native-native pairs there is genuinely nothing to correct: both
documents are extracted at exact page dimensions with no skew, so their
anchor positions already agree almost to the pixel and estimation
correctly converges on (near-)identity. The real payoff is scanned pairs,
where DPI drift and page skew mean the two normalized coordinate frames
are NOT the same transform -- match_elements' spatial cost term would
silently compare misaligned coordinates without this step.

Anchors are elements whose type is part of the drawing's fixed scaffolding
-- title block fields, the equipment tag, border-grid zone labels -- which
DESIGN.md's own operator set (per-family renumbering, note rewording,
DELETED-placeholder collapse) never touches. An anchor is only trusted
when its content is IDENTICAL between A and B on the same sheet number AND
unique within that sheet (ambiguous content, e.g. two identical zone
labels, is dropped rather than guessed at). Two or more such anchors give
a similarity transform (scale + rotation + translation) via the Umeyama/
Kabsch least-squares solution; fewer than two anchors falls back to
identity, since a single point pair can't constrain rotation or scale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.canonical.model import CanonicalDocument, CanonicalSheet

ANCHOR_TYPES = {"title_field", "equipment_tag", "zone_label"}
MIN_ANCHORS = 2


@dataclass
class Transform:
    """Maps a normalized (x, y) point in document B's frame into document
    A's frame. Identity by default."""
    scale: float = 1.0
    rotation: float = 0.0  # radians
    tx: float = 0.0
    ty: float = 0.0

    def apply(self, x: float, y: float) -> tuple[float, float]:
        cos_r, sin_r = math.cos(self.rotation), math.sin(self.rotation)
        xs, ys = x * self.scale, y * self.scale
        return (xs * cos_r - ys * sin_r + self.tx, xs * sin_r + ys * cos_r + self.ty)


def _center(el) -> tuple[float, float]:
    return ((el.bbox.x0 + el.bbox.x1) / 2, (el.bbox.y0 + el.bbox.y1) / 2)


def _sheet_anchors(sheet_a: CanonicalSheet, sheet_b: CanonicalSheet) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """(point_in_b, point_in_a) pairs for this sheet number, one per
    content string that appears exactly once on each side."""
    def by_content(sheet):
        counts: dict[str, list] = {}
        for el in sheet.elements:
            if el.type in ANCHOR_TYPES and el.content.strip():
                counts.setdefault(el.content, []).append(el)
        return {content: els[0] for content, els in counts.items() if len(els) == 1}

    a_by_content = by_content(sheet_a)
    b_by_content = by_content(sheet_b)
    shared = set(a_by_content) & set(b_by_content)
    return [(_center(b_by_content[c]), _center(a_by_content[c])) for c in shared]


def _collect_anchors(doc_a: CanonicalDocument, doc_b: CanonicalDocument) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    by_number_a = {s.number: s for s in doc_a.sheets}
    by_number_b = {s.number: s for s in doc_b.sheets}
    anchors = []
    for number in set(by_number_a) & set(by_number_b):
        anchors.extend(_sheet_anchors(by_number_a[number], by_number_b[number]))
    return anchors


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Transform:
    """Least-squares similarity transform (uniform scale + rotation +
    translation, no reflection) mapping src points onto dst points."""
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean

    var_src = (src_c ** 2).sum(axis=1).mean()
    if var_src < 1e-12:
        return Transform()  # degenerate: all anchors coincide in B

    cov = (dst_c.T @ src_c) / len(src)
    u, s, vt = np.linalg.svd(cov)
    # Reflection correction (Umeyama/Kabsch): if det(U @ V^T) < 0 the SVD
    # found a reflected solution, so flip the sign of the last singular
    # vector. The degenerate det == 0 case is reflection-ambiguous; pick
    # d = +1 (no flip) as a stable arbitrary choice. Written explicitly
    # rather than as ``np.sign(...) or 1.0`` to avoid relying on numpy
    # scalar falsiness, which is easy to misread and brittle if this is
    # ever vectorized.
    det = float(np.linalg.det(u @ vt))
    d = 1.0 if det >= 0 else -1.0
    sign_fix = np.diag([1.0, d])
    r = u @ sign_fix @ vt
    scale = float(np.trace(np.diag(s) @ sign_fix) / var_src)
    t = dst_mean - scale * (r @ src_mean)

    rotation = math.atan2(r[1, 0], r[0, 0])
    return Transform(scale=scale, rotation=rotation, tx=float(t[0]), ty=float(t[1]))


def register(doc_a: CanonicalDocument, doc_b: CanonicalDocument) -> Transform:
    """Estimate the transform mapping doc_b's page-normalized coordinates
    into doc_a's frame, from high-confidence anchor correspondences.
    Falls back to identity when fewer than MIN_ANCHORS are found (native
    pairs correctly converge on near-identity anyway; a genuinely
    unregisterable scan pair is safer left unmoved than force-fit to a
    spurious 2-point transform)."""
    anchors = _collect_anchors(doc_a, doc_b)
    if len(anchors) < MIN_ANCHORS:
        return Transform()

    src = np.array([p for p, _ in anchors], dtype=float)   # B's frame
    dst = np.array([p for _, p in anchors], dtype=float)   # A's frame
    return _umeyama(src, dst)
