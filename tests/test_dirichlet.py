"""C1:C2 Dirichlet 배정 테스트. 스펙 §6-11 테스트 16~20."""

from __future__ import annotations

import numpy as np
import pytest

from data.split.dirichlet import (
    DEFAULT_CONCENTRATION,
    dirichlet_partition,
    partition_with_acceptance,
    rarest_first_order,
)

LABELS = ("crack", "porosity", "slag_inclusion")
ISO = {"crack": "100", "porosity": "2011", "slag_inclusion": "301"}


def _fixture(n: int = 300, seed: int = 7):
    rng = np.random.default_rng(seed)
    ids = tuple(f"grp_{i:05d}" for i in range(n))
    labels = np.asarray(rng.choice(LABELS, size=n), dtype=object)
    sizes = rng.integers(1, 5, size=n).astype(np.int64)
    return ids, labels, sizes


def test_rarest_first_order_by_image_count() -> None:
    labels = ["crack"] * 3 + ["porosity"] * 10 + ["slag_inclusion"] * 1
    sizes = [1] * len(labels)
    assert rarest_first_order(labels, sizes, ISO) == ("slag_inclusion", "crack", "porosity")


def test_rarest_first_order_tiebreak_by_iso_code() -> None:
    """동률이면 ISO 코드 오름차순. AL 의 융합불량 329 vs 균열 332 같은 살얼음 대비."""
    labels = ["crack", "slag_inclusion"]
    order = rarest_first_order(labels, [5, 5], ISO)
    assert order == ("crack", "slag_inclusion")   # "100" < "301"


def test_rarest_first_order_uses_sizes_not_group_count() -> None:
    labels = ["crack", "porosity", "porosity"]
    # 묶음 수는 porosity 가 많지만 이미지 수는 crack 이 많다
    assert rarest_first_order(labels, [100, 1, 1], ISO)[0] == "porosity"


def test_pure_and_deterministic() -> None:
    ids, labels, sizes = _fixture()
    labels_before = labels.copy()
    sizes_before = sizes.copy()
    a = dirichlet_partition(ids, labels, sizes, DEFAULT_CONCENTRATION, seed=1)
    b = dirichlet_partition(ids, labels, sizes, DEFAULT_CONCENTRATION, seed=1)
    assert np.array_equal(a, b)
    assert np.array_equal(labels, labels_before) and np.array_equal(sizes, sizes_before)
    c = dirichlet_partition(ids, labels, sizes, DEFAULT_CONCENTRATION, seed=2)
    assert not np.array_equal(a, c)


def test_input_order_independence() -> None:
    """입력 순서를 섞어도 같은 배정이 나온다 — group_id 로 정렬을 다시 강제하므로."""
    ids, labels, sizes = _fixture(120, seed=11)
    a = dirichlet_partition(ids, labels, sizes, DEFAULT_CONCENTRATION, seed=5)
    perm = np.random.default_rng(0).permutation(len(ids))
    ids_p = tuple(np.asarray(ids, dtype=object)[perm])
    b = dirichlet_partition(ids_p, labels[perm], sizes[perm], DEFAULT_CONCENTRATION, seed=5)
    lookup = dict(zip(ids_p, b, strict=True))
    assert [lookup[g] for g in ids] == list(a)


def test_disjoint_complete_atomic() -> None:
    ids, labels, sizes = _fixture()
    assign = dirichlet_partition(ids, labels, sizes, DEFAULT_CONCENTRATION, seed=3)
    assert len(assign) == len(ids)
    assert set(np.unique(assign)) <= {0, 1}
    assert (assign >= 0).all()          # 전체 커버
    # 묶음 원자성: 배정은 묶음 단위이므로 이미지 회계가 정확히 나뉜다
    assert sizes[assign == 0].sum() + sizes[assign == 1].sum() == sizes.sum()


def test_prior_convergence_at_high_concentration() -> None:
    """농도를 크게 키우면 C1:C2 가 2:1 로 수렴한다."""
    ids, labels, sizes = _fixture(2000, seed=21)
    scaled = (DEFAULT_CONCENTRATION[0] * 1e6, DEFAULT_CONCENTRATION[1] * 1e6)
    assign = dirichlet_partition(ids, labels, sizes, scaled, seed=4)
    share = sizes[assign == 0].sum() / sizes.sum()
    assert share == pytest.approx(2 / 3, abs=0.02)


def test_skew_larger_than_high_concentration() -> None:
    """확정 농도의 클라이언트 간 클래스 분포 L1 이 고농도 대비 유의하게 크다."""
    ids, labels, sizes = _fixture(900, seed=31)

    def l1(concentration, seed):
        assign = dirichlet_partition(ids, labels, sizes, concentration, seed=seed)
        total = 0.0
        for label in LABELS:
            mask = labels == label
            n0 = sizes[mask & (assign == 0)].sum()
            n1 = sizes[mask & (assign == 1)].sum()
            tot = n0 + n1
            if tot:
                total += abs(n0 / tot - 2 / 3)
        return total

    high = (DEFAULT_CONCENTRATION[0] * 1e6, DEFAULT_CONCENTRATION[1] * 1e6)
    skewed = np.mean([l1(DEFAULT_CONCENTRATION, s) for s in range(20)])
    uniform = np.mean([l1(high, s) for s in range(20)])
    assert skewed > uniform * 3


def test_acceptance_band_and_deterministic_redraw() -> None:
    ids, labels, sizes = _fixture(400, seed=41)
    r1 = partition_with_acceptance(ids, labels, sizes, seed=100)
    r2 = partition_with_acceptance(ids, labels, sizes, seed=100)
    assert r1.seed_used == r2.seed_used and r1.attempts == r2.attempts
    assert np.array_equal(r1.assignment, r2.assignment)
    assert 0.60 <= r1.c1_share <= 0.73
    assert r1.seed_used == 100 + r1.attempts - 1


def test_acceptance_reports_first_pass_seed() -> None:
    """밴드를 아주 좁히면 재추첨이 실제로 일어나고 시드가 기록된다."""
    ids, labels, sizes = _fixture(400, seed=51)
    r = partition_with_acceptance(ids, labels, sizes, seed=0, share_band=(0.664, 0.670))
    assert r.attempts >= 1
    assert 0.664 <= r.c1_share <= 0.670


def test_acceptance_raises_when_impossible() -> None:
    ids, labels, sizes = _fixture(50, seed=61)
    with pytest.raises(RuntimeError, match="밴드"):
        partition_with_acceptance(ids, labels, sizes, seed=0, share_band=(0.999, 1.0),
                                  max_attempts=3)


def test_empty_input() -> None:
    assert dirichlet_partition((), np.array([], dtype=object), np.array([], dtype=np.int64)).size == 0


def test_rejects_duplicate_group_ids() -> None:
    with pytest.raises(ValueError, match="중복"):
        dirichlet_partition(
            ("g1", "g1"), np.asarray(["crack", "crack"], dtype=object),
            np.asarray([1, 1], dtype=np.int64),
        )


def test_rejects_unknown_label_in_priority() -> None:
    ids, labels, sizes = _fixture(30, seed=71)
    with pytest.raises(ValueError, match="label_priority"):
        dirichlet_partition(ids, labels, sizes, seed=1, label_priority=("crack",))
