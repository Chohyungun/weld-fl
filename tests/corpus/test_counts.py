"""counts.json · 정상 이미지 선별 테스트 — 스펙 §4-8 T35 + §7-5 정합 검증.

각 테스트의 스펙 번호는 # T0x 주석으로 표기한다.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from corpus.generate.counts_builder import (
    COUNTS_TOP_KEYS,
    DISCARD_STAGES,
    CountsError,
    build_counts,
    validate_counts,
    write_counts,
)
from corpus.rules.skeleton_gen import select_normal_images

D = Decimal


# ---------------------------------------------------------------------------
# 매니페스트 픽스처 — split=train ∧ 결함 0건 후보 (C1: 30 / C2: 10 / C3: 5)
# ---------------------------------------------------------------------------


def _manifest():
    records = []
    for i in range(30):
        records.append({"image_id": f"c1_n{i:03d}", "split": "train",
                        "client": "C1", "n_defects": 0})
    for i in range(10):
        records.append({"image_id": f"c2_n{i:03d}", "split": "train",
                        "client": "C2", "n_defects": 0})
    for i in range(5):
        records.append({"image_id": f"c3_n{i:03d}", "split": "train",
                        "client": "C3", "n_defects": 0})
    # 배제 대상: eval split, 결함 이미지 — 선별에 절대 들어가면 안 된다
    records.append({"image_id": "c1_eval0", "split": "eval", "client": "C1", "n_defects": 0})
    records.append({"image_id": "c1_d0", "split": "train", "client": "C1", "n_defects": 2})
    return records


# ---------------------------------------------------------------------------
# T35 — select_normal_images 시드 재현 + 클라이언트별 2:1 비율
# ---------------------------------------------------------------------------


def test_t35_normal_selection_seed_reproducible():  # T35
    manifest = _manifest()
    pairs = {"C1": 16, "C2": 6, "C3": 4}
    sel1 = select_normal_images(manifest, D("0.5"), seed=11, defect_pair_counts=pairs)
    sel2 = select_normal_images(manifest, D("0.5"), seed=11, defect_pair_counts=pairs)
    assert sel1.clients == sel2.clients  # 시드 재현
    sel3 = select_normal_images(manifest, D("0.5"), seed=12, defect_pair_counts=pairs)
    assert sel3.clients != sel1.clients  # 시드 변경 → 다른 선별


def test_t35_per_client_half_ratio_and_candidate_filter():  # T35 (클라이언트별 2:1)
    manifest = _manifest()
    pairs = {"C1": 16, "C2": 6, "C3": 4}
    sel = select_normal_images(manifest, D("0.5"), seed=11, defect_pair_counts=pairs)
    # 목표 = floor(결함 페어 수 × 1/2), 클라이언트 단위 (전역 비율 아님)
    assert len(sel.clients["C1"]) == 8
    assert len(sel.clients["C2"]) == 3
    assert len(sel.clients["C3"]) == 2
    # split=train ∧ 결함 0건 후보 안에서만 뽑는다
    train_normals = {r["image_id"] for r in manifest
                     if r["split"] == "train" and r["n_defects"] == 0}
    for ids in sel.clients.values():
        assert set(ids) <= train_normals
    assert "c1_eval0" not in sel.clients["C1"] and "c1_d0" not in sel.clients["C1"]
    # 홀수 페어 수의 내림 (5 × 0.5 → 2)
    sel_odd = select_normal_images(manifest, D("0.5"), seed=11,
                                   defect_pair_counts={"C3": 5})
    assert len(sel_odd.clients["C3"]) == 2


def test_t35_shortfall_reported_not_silent():  # T35 보충 (후보 부족)
    manifest = _manifest()
    sel = select_normal_images(manifest, D("0.5"), seed=11,
                               defect_pair_counts={"C3": 20})  # 목표 10 > 후보 5
    assert len(sel.clients["C3"]) == 5
    assert sel.shortfalls["C3"] == 5


# ---------------------------------------------------------------------------
# T35 — counts.json 합계 = 채택 jsonl 행수 + §7-5 스키마·정합
# ---------------------------------------------------------------------------


def _adopted():
    # 채택 페어 레코드 (dict 형태 — PairGold.to_json_dict 와 동일 최소 필드)
    return [
        {"image_id": "c1_d0", "defects": [{"defect_code": "2011"}]},
        {"image_id": "c1_d1", "defects": [{"defect_code": "100"}]},
        {"image_id": "c1_n0", "defects": []},
        {"image_id": "c2_d0", "defects": [{"defect_code": "401"}]},
        {"image_id": "c2_n0", "defects": []},
        {"image_id": "c3_d0", "defects": [{"defect_code": "301"}]},
    ]


_CLIENT_OF = {"c1_d0": "C1", "c1_d1": "C1", "c1_n0": "C1",
              "c2_d0": "C2", "c2_n0": "C2", "c3_d0": "C3"}

_DISCARDED = {"stage0_numeric_lock": 3, "stage1_rule": 2, "stage2_judge": 1,
              "stage3_expert": 0, "quarantine": 2}


def _build(**over):
    kwargs = dict(
        adopted=_adopted(), client_of=_CLIENT_OF,
        n_generated=6 + 8, discarded=_DISCARDED,
        limits_sha256="a" * 64, manifest_sha256="b" * 64,
        git_commit="deadbeef", date="2026-08-17",
    )
    kwargs.update(over)
    return build_counts(**kwargs)


def test_t35_counts_totals_match_adopted_rows():  # T35 (counts 합계 = 채택 행수)
    counts = _build()
    total = sum(c["n_total"] for c in counts["clients"].values())
    assert total == len(_adopted())
    assert counts["clients"]["C1"] == {"n_total": 3, "n_defect": 2, "n_normal": 1}
    assert counts["clients"]["C2"] == {"n_total": 2, "n_defect": 1, "n_normal": 1}
    assert counts["clients"]["C3"] == {"n_total": 1, "n_defect": 1, "n_normal": 0}
    validate_counts(counts, _adopted())  # 재검증도 통과


def test_counts_schema_keys_frozen():  # §7-5 (스키마 동결)
    counts = _build()
    assert tuple(counts.keys()) == COUNTS_TOP_KEYS
    assert tuple(counts["discarded"].keys()) == DISCARD_STAGES
    assert counts["limits_snapshot"] == "sha256:" + "a" * 64
    assert counts["manifest_snapshot"] == "sha256:" + "b" * 64
    assert counts["aux"] == {"b_qa_total": 0, "c_reasoning_total": 0}


def test_counts_discard_sum_mismatch_fails():  # §7-5 (discarded 합계 = 생성량 − 채택량)
    with pytest.raises(CountsError):
        _build(n_generated=100)  # 8 ≠ 100 − 6


def test_counts_records_n_generated():  # §7-5 (정합 3항을 파일만으로 재검증하기 위한 값)
    counts = _build()
    assert counts["n_generated"] == 14
    assert sum(counts["discarded"].values()) == counts["n_generated"] - len(_adopted())


def test_counts_discarded_tamper_after_build_detected():  # §7-5 정합 3항 (사후 변조)
    counts = _build()
    tampered = json.loads(json.dumps(counts))
    tampered["discarded"]["stage2_judge"] = 999
    with pytest.raises(CountsError):
        validate_counts(tampered, _adopted())


def test_counts_n_generated_tamper_detected():  # §7-5 정합 3항 (생성량 쪽 변조)
    counts = _build()
    tampered = json.loads(json.dumps(counts))
    tampered["n_generated"] = 999
    with pytest.raises(CountsError):
        validate_counts(tampered, _adopted())


def test_counts_n_generated_missing_or_not_int_fails():  # §7-5 (스키마 동결)
    counts = _build()
    with pytest.raises(CountsError):
        validate_counts({k: v for k, v in counts.items() if k != "n_generated"}, _adopted())
    tampered = json.loads(json.dumps(counts))
    tampered["n_generated"] = "14"
    with pytest.raises(CountsError):
        validate_counts(tampered, _adopted())


def test_counts_negative_discarded_fails():  # §7-5 (합계만 맞으면 되는 게 아니다)
    with pytest.raises(CountsError):
        _build(discarded={"stage0_numeric_lock": 9, "stage1_rule": 0, "stage2_judge": 0,
                          "stage3_expert": -1, "quarantine": 0})


def test_counts_unknown_stage_or_client_fails():  # §7-5 (fail-closed)
    with pytest.raises(CountsError):
        _build(discarded={**_DISCARDED, "stage9_bogus": 1})
    with pytest.raises(CountsError):
        _build(client_of={**_CLIENT_OF, "c3_d0": "C9"})
    with pytest.raises(CountsError):
        _build(client_of={k: v for k, v in _CLIENT_OF.items() if k != "c3_d0"})


def test_counts_tampered_totals_detected():  # T35 (n_defect + n_normal = n_total)
    counts = _build()
    tampered = json.loads(json.dumps(counts))
    tampered["clients"]["C1"]["n_normal"] = 9
    with pytest.raises(CountsError):
        validate_counts(tampered, _adopted())


def test_counts_write_roundtrip(tmp_path):  # §7-5 (기계 산출·표기 고정)
    counts = _build()
    path = tmp_path / "counts.json"
    sha1 = write_counts(path, counts)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == counts
    assert write_counts(path, counts) == sha1  # 표기 고정 → 해시 재현
