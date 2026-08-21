"""위치 지표 — BBox-IoU. 스펙 §4-6.

좌표는 **원본 이미지 픽셀**로 들어온다. C가 역변환을 끝내 제출하므로 이 모듈에는 좌표
변환 산식이 없다(§3-4). `letterbox`·`smart_resize`·`1000` 같은 상수가 이 파일에 등장하면
코드 리뷰에서 거부한다 — 그것이 이중 역변환을 구조적으로 막는 장치다.

매칭은 **Hungarian 1:1**이다. greedy 를 쓰지 않는 이유는 결함이 밀집한 이미지에서 순서에
따라 다른 답을 내기 때문이다 — 채점기가 비결정적이면 `check-scorer` 의 비트 단위 일치
요구가 성립하지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

Box = Sequence[float]


def iou(a: Box, b: Box) -> float:
    """xyxy 두 박스의 IoU. **클리핑하지 않는다** — 이미지 밖으로 나간 예측은 자연 벌점을
    받아야 하고, 클램프는 IoU 를 올리는 방향으로만 작동해 답을 고쳐주는 셈이 된다(§3-4)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Match:
    image_id: str
    iso_code: str
    iou: float
    matched: bool


def match_image(
    pred: Sequence[tuple[str, Box]],
    gold: Sequence[tuple[str, Box]],
    image_id: str = "",
) -> tuple[Match, ...]:
    """같은 `iso_code` 안에서만 Hungarian 1:1 매칭한다.

    클래스가 틀리면 위치가 맞아도 매칭하지 않는다. **미매칭 GT 는 IoU 0 으로 남긴다** —
    분모에서 빼면 못 찾은 결함이 지표에서 사라진다.
    """
    out: list[Match] = []
    codes = sorted({c for c, _ in gold} | {c for c, _ in pred})
    for code in codes:
        p = [b for c, b in pred if c == code]
        g = [b for c, b in gold if c == code]
        if not g:
            continue  # GT 가 없는 클래스의 오검출은 검출 지표(FP)가 잡는다
        if not p:
            out.extend(Match(image_id, code, 0.0, False) for _ in g)
            continue
        cost = np.zeros((len(g), len(p)), dtype=float)
        for i, gb in enumerate(g):
            for j, pb in enumerate(p):
                cost[i, j] = -iou(gb, pb)
        rows, cols = linear_sum_assignment(cost)
        assigned = set(rows.tolist())
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            out.append(Match(image_id, code, -cost[i, j], True))
        out.extend(
            Match(image_id, code, 0.0, False)
            for i in range(len(g)) if i not in assigned
        )
    return tuple(out)


@dataclass(frozen=True)
class BBoxIoUReport:
    mean_all: float
    """주 보고값 — 미매칭 GT 를 IoU 0 으로 포함한 전체 GT 박스 기준 평균."""
    mean_matched: float
    """부 보고값 — 매칭쌍만의 평균. 선행 연구(WeldLLM 0.953) 비교용."""
    n_gold: int
    n_matched: int
    coord_suspect: bool
    """매칭쌍 median IoU ≤ 0.1 이면 True. **실패가 아니라 플래그다** — 진짜 저성능을
    채점기가 기각하면 안 되므로 판단은 사람이 오버레이로 한다(§3-3c)."""

    def as_dict(self) -> dict:
        return {
            "bbox_iou": self.mean_all,
            "bbox_iou_matched_only": self.mean_matched,
            "n_gold": self.n_gold,
            "n_matched": self.n_matched,
            "coord_suspect": self.coord_suspect,
        }


COORD_SUSPECT_MEDIAN = 0.1


def score_bbox_iou(
    pred: Mapping[str, Sequence[tuple[str, Box]]],
    gold: Mapping[str, Sequence[tuple[str, Box]]],
) -> BBoxIoUReport:
    """전 이미지의 BBox-IoU. 정의 두 벌을 **함께** 낸다.

    어느 정의로 산출했는지 표에 반드시 병기한다 — 안 밝히면 0.9 와 0.5 가 같은 이름을 단다.
    """
    matches: list[Match] = []
    for img in sorted(gold):
        matches.extend(match_image(pred.get(img, ()), gold[img], img))
    if not matches:
        return BBoxIoUReport(0.0, 0.0, 0, 0, False)

    all_ious = [m.iou for m in matches]
    matched = [m.iou for m in matches if m.matched]
    median = float(np.median(matched)) if matched else 0.0
    return BBoxIoUReport(
        mean_all=float(np.mean(all_ious)),
        mean_matched=float(np.mean(matched)) if matched else 0.0,
        n_gold=len(matches),
        n_matched=len(matched),
        coord_suspect=bool(matched) and median <= COORD_SUSPECT_MEDIAN,
    )


def to_coco_xywh(box: Box) -> tuple[float, float, float, float]:
    """xyxy → COCO xywh. 좌상단 원점, **0-기준 연속 좌표** — Pascal VOC `+1` 규약 혼입 금지.

    지표 파일에서 직접 산술하지 말고 이 함수를 쓴다. 같은 공간 안의 표기 변환이므로
    좌표계 변환(C 소유)이 아니라 D 소유다(§3-4).
    """
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)
