"""폴리곤 → bbox·크기 산출. 스펙 §5.

전부 순수함수다. 원본 폴리곤을 수정하지 않는다 — 파생값만 새로 만들어 반환한다.
mm 환산은 `px_per_mm` 이 있을 때만 하고, 없으면 None 이다(0 으로 채우지 않는다).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid

Point = tuple[float, float]

FLAG_NEGATIVE_COORD = "negative_coord"
FLAG_OUT_OF_BOUNDS = "out_of_bounds"
FLAG_SELF_INTERSECT = "self_intersect"
FLAG_MULTIPART = "multipart_largest_kept"
FLAG_TOO_FEW_POINTS = "too_few_points"
FLAG_ZERO_AREA = "zero_area"


@dataclass(frozen=True)
class GeomResult:
    """폴리곤 하나에서 나온 파생 기하값."""

    bbox_px: tuple[int, int, int, int] | None
    area_px: float | None
    major_axis_px: float | None
    minor_axis_px: float | None
    equiv_diameter_px: float | None
    valid: bool
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def flags_str(self) -> str:
        return ";".join(self.flags)


def _invalid(flags: list[str]) -> GeomResult:
    return GeomResult(None, None, None, None, None, valid=False, flags=tuple(flags))


def polygon_metrics(
    polygon: list[Point] | np.ndarray,
    width_px: int,
    height_px: int,
) -> GeomResult:
    """폴리곤에서 bbox·면적·장축·단축·등가직경을 산출한다.

    좌표 이상은 버리지 않고 이미지 경계로 clip 한 뒤 사유를 flags 에 남긴다.
    clip 후 면적 0 이거나 점이 3개 미만이면 valid=False 로 표시한다(예외를 던지지 않는다).
    """
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    flags: list[str] = []

    if len(pts) < 3:
        flags.append(FLAG_TOO_FEW_POINTS)
        return _invalid(flags)

    if (pts < 0).any():
        flags.append(FLAG_NEGATIVE_COORD)
    if (pts[:, 0] > width_px).any() or (pts[:, 1] > height_px).any():
        flags.append(FLAG_OUT_OF_BOUNDS)

    clipped = pts.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, float(width_px))
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, float(height_px))

    poly = Polygon(clipped)
    if not poly.is_valid:
        flags.append(FLAG_SELF_INTERSECT)
        poly = make_valid(poly)
        if isinstance(poly, MultiPolygon):
            flags.append(FLAG_MULTIPART)
            parts = [g for g in poly.geoms if isinstance(g, Polygon)]
            if not parts:
                flags.append(FLAG_ZERO_AREA)
                return _invalid(flags)
            poly = max(parts, key=lambda g: g.area)
        elif not isinstance(poly, Polygon):
            # GeometryCollection 등 — 면 성분만 추린다
            parts = [g for g in getattr(poly, "geoms", []) if isinstance(g, Polygon)]
            if not parts:
                flags.append(FLAG_ZERO_AREA)
                return _invalid(flags)
            flags.append(FLAG_MULTIPART)
            poly = max(parts, key=lambda g: g.area)

    area = float(poly.area)
    if area <= 0.0:
        flags.append(FLAG_ZERO_AREA)
        return _invalid(flags)

    ring = np.asarray(poly.exterior.coords, dtype=np.float64)[:-1]
    if len(ring) < 3:
        flags.append(FLAG_TOO_FEW_POINTS)
        return _invalid(flags)

    x1 = math.floor(ring[:, 0].min())
    y1 = math.floor(ring[:, 1].min())
    x2 = math.ceil(ring[:, 0].max())
    y2 = math.ceil(ring[:, 1].max())
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width_px), min(y2, height_px)
    # 최소 1px 폭·높이를 보장한다. bbox 는 정수 격자이므로 얇은 결함이 0 폭으로 접힐 수 있다.
    if x2 <= x1:
        x2 = min(x1 + 1, width_px)
        x1 = max(x2 - 1, 0)
    if y2 <= y1:
        y2 = min(y1 + 1, height_px)
        y1 = max(y2 - 1, 0)

    (_, _), (w, h), _ = cv2.minAreaRect(ring.astype(np.float32))
    major = float(max(w, h))
    minor = float(min(w, h))
    equiv = float(math.sqrt(4.0 * area / math.pi))

    return GeomResult(
        bbox_px=(x1, y1, x2, y2),
        area_px=area,
        major_axis_px=major,
        minor_axis_px=minor,
        equiv_diameter_px=equiv,
        valid=True,
        flags=tuple(flags),
    )


def px_to_mm(value_px: float | None, px_per_mm: float | None) -> float | None:
    """픽셀 → mm. 스케일이 없으면 None (가정값으로 채우지 않는다).

    가정값 적용은 채점 시점의 `verdict_mode: conditional` 경로에서 한다 — 데이터에 굽지 않는다.
    """
    if value_px is None or px_per_mm is None:
        return None
    if px_per_mm <= 0:
        raise ValueError(f"px_per_mm 은 양수여야 한다: {px_per_mm}")
    return float(value_px) / float(px_per_mm)
