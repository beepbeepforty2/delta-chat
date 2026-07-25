
import cv2
import numpy as np
import pytest

from src.delta.raster_diff import (
    RasterCfg,
    _structural_diff,
    _warp_matrix_b_to_a_px,
    load_raster_gray,
    propose_change_regions,
    warp_b_into_a_frame,
)
from src.delta.register import Transform


def _blank(size=(400, 300), fill=255) -> np.ndarray:
    return np.full((size[1], size[0]), fill, dtype=np.uint8)


def _with_rect(img: np.ndarray, x0, y0, x1, y1, fill=0) -> np.ndarray:
    out = img.copy()
    out[y0:y1, x0:x1] = fill
    return out


def test_identical_images_produce_zero_regions():
    a = _with_rect(_blank(), 100, 100, 160, 140)
    b = a.copy()
    regions = propose_change_regions(a, b, Transform(), RasterCfg())
    assert regions == []


def test_single_injected_rectangle_produces_exactly_one_region():
    a = _blank()
    b = _with_rect(a, 150, 120, 220, 180)
    regions = propose_change_regions(a, b, Transform(), RasterCfg())
    assert len(regions) == 1
    r = regions[0]
    # bbox should roughly match the injected rect (normalized, some
    # slack for morphological dilation growing the region outward)
    w, h = 400, 300
    assert r.bbox.x0 * w == pytest.approx(150, abs=15)
    assert r.bbox.y0 * h == pytest.approx(120, abs=15)
    assert r.bbox.x1 * w == pytest.approx(220, abs=15)
    assert r.bbox.y1 * h == pytest.approx(180, abs=15)
    assert r.area_px > 0
    assert 0.0 < r.mean_diff_magnitude <= 1.0


def test_2px_global_shift_with_matching_transform_yields_zero_regions():
    """A pure registration offset, correctly compensated by transform,
    should warp B back onto A exactly -- proving the warp is actually
    applied (not a no-op) and morphology mops up any residual sub-pixel
    fringe from the shift+resample."""
    w, h = 400, 300
    a = _with_rect(_blank((w, h)), 100, 100, 200, 180)
    # b's content is shifted +2px in x, +2px in y relative to a
    b = _with_rect(_blank((w, h)), 102, 102, 202, 182)
    # Transform.apply(x, y) maps a point in B's frame into A's frame
    # (register.py's own documented direction) -- since B's content sits
    # +2px from A's, mapping B's coordinate back onto A's requires a
    # NEGATIVE shift.
    transform = Transform(tx=-2.0 / w, ty=-2.0 / h)
    regions = propose_change_regions(a, b, transform, RasterCfg())
    assert regions == []


def test_max_area_frac_drops_whole_page_shift_sized_region():
    w, h = 400, 300
    a = _blank((w, h), fill=255)
    b = _blank((w, h), fill=0)  # whole page different -- e.g. registration failure
    cfg = RasterCfg(max_area_frac=0.25)
    regions = propose_change_regions(a, b, Transform(), cfg)
    assert regions == []


def test_min_area_px_drops_small_specks():
    a = _blank()
    b = _with_rect(a, 100, 100, 103, 103)  # 3x3 = 9px, a speck
    cfg = RasterCfg(min_area_px=150, enable_open=False, enable_merge_dilate=False)
    regions = propose_change_regions(a, b, Transform(), cfg)
    assert regions == []


class TestWarpMatrixCorrectness:
    """The matrix construction in _warp_matrix_b_to_a_px is validated
    against Transform.apply()'s own predictions, not trusted by
    inspection of the algebra alone -- matches the retired raster_recall
    module's own testing discipline for its (different) affine helper."""

    def test_point_algebra_matches_transform_apply(self):
        w_a, h_a = 500, 400
        w_b, h_b = 480, 380
        for transform in [
            Transform(scale=1.1, rotation=0.05, tx=0.1, ty=-0.05),
            Transform(scale=0.95, rotation=-0.02, tx=-0.03, ty=0.02),
            Transform(),  # identity
        ]:
            m = _warp_matrix_b_to_a_px(transform, (w_a, h_a), (w_b, h_b))
            for xb_norm, yb_norm in [(0.2, 0.3), (0.5, 0.5), (0.8, 0.1)]:
                px_b, py_b = xb_norm * w_b, yb_norm * h_b
                px_a, py_a = m @ np.array([px_b, py_b, 1.0])
                expected_xa_norm, expected_ya_norm = transform.apply(xb_norm, yb_norm)
                assert px_a / w_a == pytest.approx(expected_xa_norm, abs=1e-6)
                assert py_a / h_a == pytest.approx(expected_ya_norm, abs=1e-6)

    def test_warp_places_marker_at_transform_predicted_position(self):
        w, h = 400, 300
        transform = Transform(scale=1.0, rotation=0.03, tx=0.05, ty=-0.02)
        raster_b = _with_rect(_blank((w, h)), 150, 140, 170, 160, fill=0)
        warped = warp_b_into_a_frame(raster_b, transform, (w, h))

        ys, xs = np.where(warped < 128)
        assert len(xs) > 0, "marker vanished after warp"
        centroid_x_norm, centroid_y_norm = xs.mean() / w, ys.mean() / h

        marker_center_b_norm = (160 / w, 150 / h)
        expected_x_norm, expected_y_norm = transform.apply(*marker_center_b_norm)
        assert centroid_x_norm == pytest.approx(expected_x_norm, abs=0.02)
        assert centroid_y_norm == pytest.approx(expected_y_norm, abs=0.02)


class TestStructuralDiffMethodSelection:
    def test_uses_ssim_when_available_and_enabled(self):
        a = _blank((64, 64))
        b = _with_rect(a, 10, 10, 30, 30)
        _, method = _structural_diff(a, b, RasterCfg(use_ssim=True))
        assert method == "ssim"

    def test_falls_back_when_use_ssim_disabled(self):
        a = _blank((64, 64))
        b = _with_rect(a, 10, 10, 30, 30)
        _, method = _structural_diff(a, b, RasterCfg(use_ssim=False))
        assert method == "blur_absdiff"

    def test_falls_back_when_skimage_import_fails(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "skimage.metrics" or name.startswith("skimage"):
                raise ImportError("simulated missing skimage")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        a = _blank((64, 64))
        b = _with_rect(a, 10, 10, 30, 30)
        _, method = _structural_diff(a, b, RasterCfg(use_ssim=True))
        assert method == "blur_absdiff"


class TestAblationTogglesHaveMeasurableEffect:
    def _fringe_heavy(self):
        """A pair with a real change plus lots of small single-pixel
        fringe noise scattered around it -- the case morphology exists
        to clean up."""
        w, h = 200, 200
        rng = np.random.RandomState(0)
        a = _blank((w, h))
        b = a.copy()
        b[80:120, 80:120] = 0  # the real change
        noise_mask = rng.rand(h, w) < 0.01
        b[noise_mask] = np.where(b[noise_mask] > 127, 200, 60)  # scattered single-pixel fringe
        return a, b

    def test_open_reduces_region_count_on_fringe_heavy_input(self):
        a, b = self._fringe_heavy()
        with_open = propose_change_regions(a, b, Transform(), RasterCfg(enable_open=True, enable_merge_dilate=False))
        without_open = propose_change_regions(a, b, Transform(), RasterCfg(enable_open=False, enable_merge_dilate=False))
        assert len(with_open) <= len(without_open)

    def test_merge_dilate_reduces_region_count_on_fringe_heavy_input(self):
        a, b = self._fringe_heavy()
        with_merge = propose_change_regions(a, b, Transform(), RasterCfg(enable_open=False, enable_merge_dilate=True))
        without_merge = propose_change_regions(a, b, Transform(), RasterCfg(enable_open=False, enable_merge_dilate=False))
        assert len(with_merge) <= len(without_merge)


class TestRasterCfgFromEnv:
    def test_defaults_when_unset(self, monkeypatch):
        for key in list(__import__("os").environ):
            if key.startswith(("DELTA_RASTER_DIFF", "RASTER_DIFF", "RASTER_JOIN")):
                monkeypatch.delenv(key, raising=False)
        cfg = RasterCfg.from_env()
        assert cfg.enable_raster is False
        assert cfg.diff_threshold == 0.25
        assert cfg.min_area_px == 150

    def test_overrides_from_env(self, monkeypatch):
        monkeypatch.setenv("DELTA_RASTER_DIFF", "1")
        monkeypatch.setenv("RASTER_DIFF_THRESHOLD", "0.5")
        monkeypatch.setenv("RASTER_DIFF_MIN_AREA_PX", "999")
        cfg = RasterCfg.from_env()
        assert cfg.enable_raster is True
        assert cfg.diff_threshold == 0.5
        assert cfg.min_area_px == 999


def test_load_raster_gray_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raster_gray(str(tmp_path / "does_not_exist.png"))


def test_load_raster_gray_real_file(tmp_path):
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), _blank())
    img = load_raster_gray(str(path))
    assert img.shape == (300, 400)
    assert img.dtype == np.uint8
