"""C1:C2 Dirichlet 배정. 스펙 §6-7.

`dirichlet_partition` 은 순수함수다 — I/O·전역 상태·전역 np.random 을 쓰지 않는다.
배정은 **묶음 단위**이고 충전 기준은 **이미지 수**(묶음 수가 아니다).

농도 벡터를 그대로 넘긴다. "α=0.5" 축약 표기는 `α·p` 인지 `α·n·p` 인지에 따라 이질성
강도가 달라져 재현이 불가능하다 (§6-0 쟁점 10).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

#: α=0.5, p=(2/3, 1/3) → α·p. configs/base.yaml 의 초기 기입값과 같아야 한다.
DEFAULT_CONCENTRATION: tuple[float, float] = (1.0 / 3.0, 1.0 / 6.0)
#: 실현 C1 지분 수용 밴드 (목표 2/3). 열린질문 Q15.
DEFAULT_SHARE_BAND: tuple[float, float] = (0.60, 0.73)


def rarest_first_order(
    labels: Sequence[str],
    sizes: Sequence[int],
    iso_codes: Mapping[str, str],
) -> tuple[str, ...]:
    """클래스 순회 순서 — 재질 내 이미지 빈도 오름차순, 동률은 ISO 코드 오름차순.

    빈도는 **런타임 실측에서 유도**한다. 공지 수치를 하드코딩하면 AL 의 융합불량 329 대
    균열 332 처럼 3장 차이에서 순서가 뒤집힌다 (§6-0 쟁점 8).
    """
    totals: dict[str, int] = {}
    for label, size in zip(labels, sizes, strict=True):
        totals[label] = totals.get(label, 0) + int(size)
    return tuple(sorted(totals, key=lambda k: (totals[k], iso_codes.get(k, k))))


def dirichlet_partition(
    group_ids: tuple[str, ...],
    group_labels: np.ndarray,
    group_sizes: np.ndarray,
    concentration: tuple[float, float] = DEFAULT_CONCENTRATION,
    seed: int = 0,
    *,
    label_priority: Sequence[str] | None = None,
) -> np.ndarray:
    """묶음을 C1(0) / C2(1) 로 배정한다.

    Args:
        group_ids: 사전순 정렬된 묶음 ID (호출 규약). 정렬을 함수가 다시 강제한다.
        group_labels: (G,) 묶음 대표 클래스 — §6-6 과 동일 정의.
        group_sizes: (G,) 묶음 이미지 수.
        concentration: Dirichlet 농도 벡터 (C1, C2).
        seed: `np.random.default_rng` 시드.
        label_priority: 클래스 순회 순서. 생략 시 `group_labels` 정렬 순서.

    Returns:
        (G,) int 배열, 값 ∈ {0, 1}. 서로소·전체 커버·묶음 원자성이 보장된다.
    """
    ids = np.asarray(group_ids, dtype=object)
    labels = np.asarray(group_labels, dtype=object)
    sizes = np.asarray(group_sizes, dtype=np.int64)
    if not (len(ids) == len(labels) == len(sizes)):
        raise ValueError("group_ids / group_labels / group_sizes 길이가 다르다")
    if len(ids) == 0:
        return np.empty(0, dtype=np.int64)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("group_ids 에 중복이 있다")

    rng = np.random.default_rng(seed)
    order = tuple(label_priority) if label_priority is not None else tuple(
        sorted(set(labels.tolist()))
    )
    unseen = set(labels.tolist()) - set(order)
    if unseen:
        raise ValueError(f"label_priority 에 없는 클래스: {sorted(unseen)}")

    assign = np.full(len(ids), -1, dtype=np.int64)
    for label in order:
        idx = np.flatnonzero(labels == label)
        if idx.size == 0:
            continue
        # 사전순으로 고정한 뒤 셔플 — 입력 순서에 의존하지 않는다
        idx = idx[np.argsort(ids[idx].astype(str), kind="stable")]
        p = rng.dirichlet(np.asarray(concentration, dtype=np.float64))
        rng.shuffle(idx)
        target = float(p[0]) * float(sizes[idx].sum())
        cum = 0.0
        for i in idx:
            if cum < target:
                assign[i] = 0
                cum += float(sizes[i])
            else:
                assign[i] = 1
    if (assign < 0).any():
        raise AssertionError("배정되지 않은 묶음이 있다")
    return assign


@dataclass(frozen=True)
class PartitionResult:
    assignment: np.ndarray
    seed_used: int
    attempts: int
    c1_share: float


def partition_with_acceptance(
    group_ids: tuple[str, ...],
    group_labels: np.ndarray,
    group_sizes: np.ndarray,
    concentration: tuple[float, float] = DEFAULT_CONCENTRATION,
    seed: int = 0,
    *,
    label_priority: Sequence[str] | None = None,
    share_band: tuple[float, float] = DEFAULT_SHARE_BAND,
    max_attempts: int = 50,
) -> PartitionResult:
    """실현 C1 지분이 수용 밴드에 들 때까지 `seed, seed+1, …` 로 재추첨한다.

    **최초 통과를 채택**하고 채택 시드·시도 횟수를 기록한다. 예산 캡·빈패킹·클래스 0장
    재추첨은 쓰지 않는다 — 실현 분포를 명목 α 에서 더 멀어지게 만든다 (§6-0 쟁점 11).
    클래스 0장 배정은 의도된 non-IID 로 허용한다.
    """
    sizes = np.asarray(group_sizes, dtype=np.int64)
    lo, hi = share_band
    for attempt in range(max_attempts):
        s = seed + attempt
        assign = dirichlet_partition(
            group_ids, group_labels, sizes, concentration, s, label_priority=label_priority
        )
        total = float(sizes.sum())
        share = float(sizes[assign == 0].sum()) / total if total else 0.0
        if lo <= share <= hi:
            return PartitionResult(assign, s, attempt + 1, share)
    raise RuntimeError(
        f"{max_attempts}회 재추첨했으나 C1 지분이 {share_band} 밴드에 들지 않았다. "
        "농도 벡터나 밴드를 재검토하라 (CTO 회부)"
    )
