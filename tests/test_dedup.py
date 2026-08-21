"""중복 묶음 테스트. 스펙 §6-11 테스트 1·3·5·6·7·8."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from data.dedup.phash import (
    EDGE_PHASH,
    EDGE_SHA256,
    HASH_BITS,
    UnionFind,
    build_groups,
    compute_phash,
    distance_histogram,
    hamming_matrix,
    iter_close_pairs,
    pack_hashes,
)

H0 = "0" * 64
H1 = "f" * 64


def _hex(bits: list[int]) -> str:
    """비트 리스트(256개) → 64자 hex."""
    assert len(bits) == HASH_BITS
    return "".join(f"{int(''.join(map(str, bits[i:i+4])), 2):x}" for i in range(0, HASH_BITS, 4))


def test_phash_is_256bit_hex(tmp_path) -> None:
    p = tmp_path / "a.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (224, 224), dtype=np.uint8), mode="L").save(p)
    h = compute_phash(p)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_phash_fixture_stable(tmp_path) -> None:
    """같은 파일 → 같은 해시. 라이브러리 드리프트 감지의 최소 형태."""
    p = tmp_path / "a.png"
    y, x = np.mgrid[0:224, 0:224]
    Image.fromarray(((x + y) % 255).astype(np.uint8), mode="L").save(p)
    assert compute_phash(p) == compute_phash(p)


def test_pack_and_distance() -> None:
    packed = pack_hashes([H0, H1])
    assert packed.shape == (2, 4)
    d = hamming_matrix(packed)
    assert d[0, 0] == 0 and d[0, 1] == HASH_BITS


def test_pack_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="길이"):
        pack_hashes(["abc"])


def test_pairwise_tile_vs_reference() -> None:
    """타일 popcount = 참조 전수 구현. 타일 경계에서 쌍이 빠지지 않는지."""
    rng = np.random.default_rng(7)
    hexes = ["".join(rng.choice(list("0123456789abcdef"), size=64)) for _ in range(300)]
    packed = pack_hashes(hexes)
    ref = hamming_matrix(packed)
    threshold = 100
    expected = {
        (i, j, int(ref[i, j]))
        for i in range(300)
        for j in range(i + 1, 300)
        if ref[i, j] <= threshold
    }
    for tile in (16, 64, 4096):
        assert set(iter_close_pairs(packed, threshold, tile=tile)) == expected


def test_histogram_counts_each_pair_once() -> None:
    rng = np.random.default_rng(3)
    hexes = ["".join(rng.choice(list("0123456789abcdef"), size=64)) for _ in range(120)]
    packed = pack_hashes(hexes)
    for tile in (32, 4096):
        hist = distance_histogram(packed, tile=tile)
        assert int(hist.sum()) == 120 * 119 // 2


def test_unionfind_basic() -> None:
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(3, 4)
    uf.union(1, 3)
    assert uf.find(0) == uf.find(4)
    assert uf.find(0) != uf.find(2)


# ---- build_groups -------------------------------------------------------------------


def _ids(n: int) -> list[str]:
    return [f"img{i}" for i in range(n)]


def _shas(*seeds: int) -> list[str]:
    """서로 다른 **선두** 16자를 갖는 sha256 스텁.

    `f"{i:064x}"` 류는 앞자리가 전부 0 이라 group_id(= min(sha)[:16]) 가 충돌한다 —
    실제 sha256 은 선두가 흩어지므로 스텁도 그렇게 만든다. 충돌 방지 assert 가 이걸 잡았다.
    """
    return [hashlib.sha256(f"stub{s}".encode()).hexdigest() for s in seeds]


#: H0 과 거리 2·3 인 해시 (E3 엣지 시험용)
NEAR2 = _hex([0] * (HASH_BITS - 2) + [1, 1])
NEAR3 = _hex([0] * (HASH_BITS - 3) + [1, 1, 1])
#: H1(전부 1)과 가깝지 않고 H0 과도 먼 중간 해시
MID = _hex([0, 1] * (HASH_BITS // 2))


def test_singleton_gets_own_group() -> None:
    """단독 이미지도 자기 묶음을 갖는다 — group_id 는 전 행 non-null."""
    res = build_groups(_ids(2), _shas(1, 2), [H0, H1], ["ST", "ST"], threshold=8)
    assert res.n_groups == 2
    assert all(g.startswith("grp_") for g in res.group_ids)
    assert res.max_group_size == 1


def test_e1_merges_identical_files() -> None:
    same = _shas(1)[0]
    res = build_groups(_ids(2), [same, same], [H0, H1], ["ST", "ST"], threshold=0)
    assert res.n_groups == 1
    assert res.edge_counts[EDGE_SHA256] == 1


def test_e3_merges_close_hashes() -> None:
    res = build_groups(_ids(2), _shas(1, 2), [H0, NEAR3], ["ST", "ST"], threshold=8)
    assert res.n_groups == 1
    assert res.edge_counts[EDGE_PHASH] == 1


def test_group_id_content_derived_and_order_invariant() -> None:
    """엣지·입력 순서를 섞어도 같은 분할. group_id 는 (재질, min sha256) 에서만 나온다."""
    shas = _shas(3, 1, 2)
    hashes = [H0, NEAR2, H1]        # 0-1 은 가깝고 2 는 멀다
    a = build_groups(_ids(3), shas, hashes, ["ST"] * 3, threshold=8)
    assert a.n_groups == 2
    expected = f"grp_ST_{min(shas[0], shas[1])[:12]}"
    assert a.group_ids[0] == a.group_ids[1] == expected

    perm = [2, 0, 1]
    b = build_groups(
        [_ids(3)[i] for i in perm], [shas[i] for i in perm],
        [hashes[i] for i in perm], ["ST"] * 3, threshold=8,
    )
    lookup = dict(zip([_ids(3)[i] for i in perm], b.group_ids, strict=True))
    assert [lookup[g] for g in _ids(3)] == list(a.group_ids)


def test_cross_material_not_merged_but_reported() -> None:
    """교차 재질은 union 하지 않고 해소 게이트로 보고한다 — 로그만 남기면 누수가 지나간다."""
    same = _shas(9)[0]
    res = build_groups(_ids(2), [same, same], [H0, H0], ["ST", "AL"], threshold=8)
    assert res.n_groups == 2
    assert len(res.cross_material_pairs) >= 1
    assert res.edge_counts[EDGE_SHA256] == 0


def test_material_purity_holds() -> None:
    mats = ["ST", "ST", "AL", "AL"]
    res = build_groups(_ids(4), _shas(1, 2, 3, 4), [H0, NEAR2, H1, H1], mats, threshold=8)
    for gid in set(res.group_ids):
        got = {m for g, m in zip(res.group_ids, mats, strict=True) if g == gid}
        assert len(got) == 1


def test_e2_meta_edge_merges() -> None:
    """메타 ID 가 같으면 해시가 멀어도 묶인다 — 이동 촬영 누수의 실질 방어선."""
    res = build_groups(
        _ids(3), _shas(1, 2, 3), [H0, H1, MID],
        ["ST"] * 3, threshold=0, meta_keys=["film-1", "film-1", None],
    )
    assert res.n_groups == 2
    assert res.edge_counts["E2"] == 1
    assert res.edge_counts[EDGE_PHASH] == 0
    assert res.group_ids[0] == res.group_ids[1] != res.group_ids[2]


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="길이"):
        build_groups(_ids(2), _shas(1), [H0, H1], ["ST", "ST"])
