"""Raster diff: registers B onto A's frame (reusing register.py's own
Transform -- never re-registers), structural-diffs at the pixel level,
cleans up morphologically, and returns candidate change regions in A's
normalized coordinate space.

This module PROPOSES; it never adjudicates. A region here is a candidate,
not a delta -- see raster_join.py for the step that decides whether the
symbolic pipeline already explains it, and only then emits a Delta. This
split is the "raster localizes, symbolic classifies" principle: the raw
diff mask must never be emitted as deltas directly.

Every cleanup step below exists to kill one specific false-positive
source and is independently toggleable via RasterCfg, so an ablation run
can measure each step's actual contribution rather than trusting the
pipeline as one opaque block:
  - grayscale normalize + optional Otsu: near-binary P&ID content, not a
    photograph -- flattens producer-to-producer brightness/contrast drift.
  - SSIM (not raw |A-B|): a raw pixel diff lights up the 1px anti-alias
    fringe on every unchanged glyph/line; SSIM compares local structure
    and tolerates sub-pixel misalignment and producer AA differences.
  - morphological open then dilate: open kills single-pixel edge fringe
    that survives thresholding; dilate merges a cluster of changed
    strokes into one region instead of fifty specks.
  - min/max area filtering: drops single-pixel noise and, at the other
    end, whole-page-shift-sized "regions" that indicate a registration
    failure, not a real content change.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

from src.canonical.model import BBox
from src.delta.register import Transform


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return default if val is None else val == "1"


@dataclass
class RasterCfg:
    enable_raster: bool = True
    dpi: int = 150  # documentation/sanity-check only -- see module note below
    diff_threshold: float = 0.25
    otsu_binarize: bool = False
    use_ssim: bool = True
    ssim_win_size: int = 7
    blur_sigma: float = 1.0
    enable_open: bool = True
    open_kernel: int = 3
    enable_merge_dilate: bool = True
    merge_kernel: int = 9
    min_area_px: int = 150
    max_area_frac: float = 0.25
    explain_iou: float = 0.10
    tag_proximity_norm: float = 0.02
    conf_base: float = 0.15
    conf_scale: float = 0.35
    conf_cap: float = 0.45

    @classmethod
    def from_env(cls) -> RasterCfg:
        return cls(
            enable_raster=_env_bool("DELTA_RASTER_DIFF", False),
            dpi=int(os.environ.get("RASTER_DIFF_DPI", "150")),
            diff_threshold=float(os.environ.get("RASTER_DIFF_THRESHOLD", "0.25")),
            otsu_binarize=_env_bool("RASTER_DIFF_OTSU_BINARIZE", False),
            use_ssim=_env_bool("RASTER_DIFF_USE_SSIM", True),
            ssim_win_size=int(os.environ.get("RASTER_DIFF_SSIM_WIN_SIZE", "7")),
            blur_sigma=float(os.environ.get("RASTER_DIFF_BLUR_SIGMA", "1.0")),
            enable_open=_env_bool("RASTER_DIFF_ENABLE_OPEN", True),
            open_kernel=int(os.environ.get("RASTER_DIFF_OPEN_KERNEL", "3")),
            enable_merge_dilate=_env_bool("RASTER_DIFF_ENABLE_MERGE", True),
            merge_kernel=int(os.environ.get("RASTER_DIFF_MERGE_KERNEL", "9")),
            min_area_px=int(os.environ.get("RASTER_DIFF_MIN_AREA_PX", "150")),
            max_area_frac=float(os.environ.get("RASTER_DIFF_MAX_AREA_FRAC", "0.25")),
            explain_iou=float(os.environ.get("RASTER_JOIN_EXPLAIN_IOU", "0.10")),
            tag_proximity_norm=float(os.environ.get("RASTER_JOIN_TAG_PROXIMITY_NORM", "0.02")),
            conf_base=float(os.environ.get("RASTER_JOIN_CONF_BASE", "0.15")),
            conf_scale=float(os.environ.get("RASTER_JOIN_CONF_SCALE", "0.35")),
            conf_cap=float(os.environ.get("RASTER_JOIN_CONF_CAP", "0.45")),
        )


@dataclass
class ChangeRegion:
    sheet: int
    bbox: BBox  # normalized [0,1], A's frame, top-left/y-down (matches BBox convention)
    area_px: int
    mean_diff_magnitude: float


def load_raster_gray(path: str) -> np.ndarray:
    """Loads an L0 raster PNG (CanonicalDocument.raster_paths) as uint8
    grayscale. This does NOT re-rasterize from the source PDF -- the
    ingest adapters already wrote these PNGs; this stage only reads them."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read raster: {path}")
    return img


def _warp_matrix_b_to_a_px(transform: Transform, size_a: tuple[int, int],
                            size_b: tuple[int, int]) -> np.ndarray:
    """2x3 matrix mapping B-PIXEL coords -> A-PIXEL coords, derived by
    composing: B-px -> B-normalized -> Transform.apply (B-norm -> A-norm,
    register.py's own documented direction) -> A-px. Passed to
    cv2.warpAffine WITHOUT WARP_INVERSE_MAP: OpenCV's default semantics
    treat M as the forward source(B)->destination(A) map and invert it
    internally to backward-sample -- exactly the convention this matrix
    is built in. Do not trust this by inspection alone; see the
    correctness tests in tests/test_delta_raster_diff.py that validate it
    against Transform.apply()'s own predictions at sample points."""
    w_a, h_a = size_a
    w_b, h_b = size_b
    s, r, tx, ty = transform.scale, transform.rotation, transform.tx, transform.ty
    cos_r, sin_r = math.cos(r), math.sin(r)
    return np.array([
        [s * cos_r * w_a / w_b, -s * sin_r * w_a / h_b, tx * w_a],
        [s * sin_r * h_a / w_b, s * cos_r * h_a / h_b, ty * h_a],
    ], dtype=np.float64)


def warp_b_into_a_frame(raster_b: np.ndarray, transform: Transform,
                         size_a: tuple[int, int]) -> np.ndarray:
    h_b, w_b = raster_b.shape[:2]
    m = _warp_matrix_b_to_a_px(transform, size_a, (w_b, h_b))
    return cv2.warpAffine(raster_b, m, size_a, flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _normalize_gray(img: np.ndarray, cfg: RasterCfg) -> np.ndarray:
    """Deliberately does NOT do a global cv2.normalize min-max stretch:
    that rescales each image independently to its OWN observed range,
    which is fine for a photograph but actively wrong here -- a blank (or
    near-blank) P&ID page is close to constant, and stretching a
    near-constant image to [0,255] amplifies noise into a spurious
    full-page "difference" against the other, real-content image (caught
    by a synthetic test: a blank page vs. one real rectangle produced a
    whole-page diff, not a one-region diff, until this was removed).
    SSIM's own luminance/contrast terms already normalize locally within
    each comparison window, which is the correct place for this kind of
    normalization to happen, not as a separate global pre-pass. Otsu
    binarization, if requested, is a well-defined per-image thresholding
    step and is kept."""
    if cfg.otsu_binarize:
        _, out = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out
    return img


def _structural_diff(gray_a: np.ndarray, gray_b: np.ndarray, cfg: RasterCfg) -> tuple[np.ndarray, str]:
    """Returns (diff_magnitude in [0,1], method actually used) -- the
    method string is surfaced so a run/trace can tell which path
    executed (SSIM vs the blur+absdiff fallback)."""
    if cfg.use_ssim:
        try:
            from skimage.metrics import structural_similarity
            win = min(cfg.ssim_win_size, gray_a.shape[0], gray_a.shape[1])
            if win % 2 == 0:
                win -= 1
            win = max(3, win)
            _, ssim_map = structural_similarity(gray_a, gray_b, full=True, win_size=win, data_range=255)
            return (1.0 - ssim_map).astype(np.float32), "ssim"
        except ImportError:
            pass
    k = max(1, round(cfg.blur_sigma * 3) | 1)
    blur_a = cv2.GaussianBlur(gray_a, (k, k), cfg.blur_sigma)
    blur_b = cv2.GaussianBlur(gray_b, (k, k), cfg.blur_sigma)
    absdiff = cv2.absdiff(blur_a, blur_b).astype(np.float32) / 255.0
    return absdiff, "blur_absdiff"


def propose_change_regions(raster_a: np.ndarray, raster_b: np.ndarray,
                            transform: Transform, cfg: RasterCfg, *,
                            sheet: int = 0) -> list[ChangeRegion]:
    """Register B onto A's frame, structural-diff, clean, connected-
    components. Returns change regions in A's normalized coordinate
    space. `sheet` is keyword-only with a default: the literal call shape
    diffs a single sheet's pair of rasters, and the caller (one per sheet
    in a multi-sheet document) is what actually knows the sheet number."""
    h_a, w_a = raster_a.shape[:2]
    warped_b = warp_b_into_a_frame(raster_b, transform, (w_a, h_a))

    gray_a = _normalize_gray(raster_a, cfg)
    gray_b = _normalize_gray(warped_b, cfg)

    diff_mag, _method = _structural_diff(gray_a, gray_b, cfg)
    mask = (diff_mag > cfg.diff_threshold).astype(np.uint8) * 255

    if cfg.enable_open:
        k = np.ones((cfg.open_kernel, cfg.open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if cfg.enable_merge_dilate:
        k = np.ones((cfg.merge_kernel, cfg.merge_kernel), np.uint8)
        mask = cv2.dilate(mask, k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sheet_area = w_a * h_a
    regions = []
    for label_id in range(1, n_labels):
        x, y, w, h, area = stats[label_id]
        if area < cfg.min_area_px or area > cfg.max_area_frac * sheet_area:
            continue
        region_mask = labels[y:y + h, x:x + w] == label_id
        mean_mag = float(diff_mag[y:y + h, x:x + w][region_mask].mean())
        bbox = BBox(x / w_a, y / h_a, (x + w) / w_a, (y + h) / h_a)
        regions.append(ChangeRegion(sheet=sheet, bbox=bbox, area_px=int(area),
                                     mean_diff_magnitude=mean_mag))
    return regions
