"""폴리곤 → bbox·크기 산출 테스트. 스펙 §7-2."""

from __future__ import annotations

import math

import numpy as np
import pytest

from data.convert.geometry import (
    FLAG_MULTIPART,
    FLAG_NEGATIVE_COORD,
    FLAG_OUT_OF_BOUNDS,
    FLAG_SELF_INTERSECT,
    FLAG_TOO_FEW_POINTS,
    polygon_metrics,
    px_to_mm,
)

W, H = 1280, 720


def test_bbox_from_axis_aligned_rect() -> None:
    poly = [(100, 200), (300, 200), (300, 260), (100, 260)]
    g = polygon_metrics(poly, W, H)
    assert g.valid and g.flags == ()
    assert g.bbox_px == (100, 200, 300, 260)
    assert g.area_px == pytest.approx(200 * 60)
    assert g.major_axis_px == pytest.approx(200, abs=0.5)
    assert g.minor_axis_px == pytest.approx(60, abs=0.5)


def test_bbox_clip_to_image_bounds() -> None:
    poly = [(-50, -20), (100, -20), (100, 80), (-50, 80)]
    g = polygon_metrics(poly, W, H)
    assert g.valid
    assert FLAG_NEGATIVE_COORD in g.flags
    assert g.bbox_px == (0, 0, 100, 80)


def test_out_of_bounds_flagged() -> None:
    poly = [(W - 10, H - 10), (W + 200, H - 10), (W + 200, H + 200), (W - 10, H + 200)]
    g = polygon_metrics(poly, W, H)
    assert g.valid and FLAG_OUT_OF_BOUNDS in g.flags
    _, _, x2, y2 = g.bbox_px
    assert x2 <= W and y2 <= H


def test_major_axis_rotated_rect() -> None:
    """45° 회전 사각형의 장축은 bbox 대각이 아니라 최소외접사각형의 긴 변이다."""
    long_side, short_side = 200.0, 40.0
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    base = np.array(
        [(-long_side / 2, -short_side / 2), (long_side / 2, -short_side / 2),
         (long_side / 2, short_side / 2), (-long_side / 2, short_side / 2)]
    )
    rot = base @ np.array([[c, s], [-s, c]])
    poly = rot + np.array([400.0, 300.0])
    g = polygon_metrics(poly, W, H)
    assert g.valid
    assert g.major_axis_px == pytest.approx(long_side, rel=0.02)
    assert g.minor_axis_px == pytest.approx(short_side, rel=0.05)
    # bbox 는 축정렬이라 회전된 사각형보다 넓다 — 장축과 혼동하면 안 된다
    x1, _, x2, _ = g.bbox_px
    assert (x2 - x1) > g.major_axis_px * 0.7


def test_equiv_diameter_circle() -> None:
    r = 30.0
    t = np.linspace(0, 2 * math.pi, 256, endpoint=False)
    poly = np.stack([500 + r * np.cos(t), 400 + r * np.sin(t)], axis=1)
    g = polygon_metrics(poly, W, H)
    assert g.valid
    assert g.equiv_diameter_px == pytest.approx(2 * r, rel=0.01)


def test_selfintersect_bowtie() -> None:
    poly = [(0, 0), (100, 100), (100, 0), (0, 100)]
    g = polygon_metrics(poly, W, H)
    assert g.valid
    assert FLAG_SELF_INTERSECT in g.flags
    assert FLAG_MULTIPART in g.flags
    # 최대 면적 조각만 채택 — 두 삼각형 중 하나
    assert g.area_px == pytest.approx(2500, rel=0.02)


def test_degenerate_rejected() -> None:
    g = polygon_metrics([(10, 10), (20, 20)], W, H)
    assert not g.valid and FLAG_TOO_FEW_POINTS in g.flags
    assert g.bbox_px is None and g.area_px is None


def test_zero_area_line_rejected() -> None:
    g = polygon_metrics([(10, 10), (20, 10), (30, 10)], W, H)
    assert not g.valid


def test_thin_defect_bbox_has_min_extent() -> None:
    """얇은 결함이 정수 격자에서 0 폭으로 접히지 않아야 한다 (IV10 위반 방지)."""
    poly = [(100.2, 200.1), (100.6, 200.1), (100.6, 260.0), (100.2, 260.0)]
    g = polygon_metrics(poly, W, H)
    assert g.valid
    x1, y1, x2, y2 = g.bbox_px
    assert x2 > x1 and y2 > y1


def test_mm_null_when_no_scale() -> None:
    assert px_to_mm(100.0, None) is None
    assert px_to_mm(None, 5.0) is None
    assert px_to_mm(100.0, 5.0) == pytest.approx(20.0)
    with pytest.raises(ValueError):
        px_to_mm(100.0, 0.0)


def test_polygon_json_not_mutated() -> None:
    poly = [(-5.0, 10.0), (100.0, 10.0), (100.0, 50.0)]
    original = [tuple(p) for p in poly]
    arr = np.array(poly)
    snapshot = arr.copy()
    polygon_metrics(poly, W, H)
    polygon_metrics(arr, W, H)
    assert [tuple(p) for p in poly] == original
    assert np.array_equal(arr, snapshot)
