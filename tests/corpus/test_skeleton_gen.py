"""골격 생성기 테스트 — 스펙 §4-8 T15~T26, T29~T32, T34.

각 테스트의 스펙 번호는 # T0x 주석으로 표기한다. 버킷 구간 검증(T15)은 구현을 재사용하지
않고 테스트 안에서 그리드 인덱스를 독립 계산한다.
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import pytest
from pydantic import ValidationError

from corpus.rules.schema import (
    FAIL,
    PASS,
    InspectionMethod,
    LimitRule,
    Q,
    VerdictMode,
    quantize,
)
from corpus.rules.limit_eval import applicable_row, effective_limit, judge
from corpus.rules.skeleton_gen import (
    BUCKET_BOUNDARY_EQ,
    BUCKET_BOUNDARY_HIGH,
    BUCKET_BOUNDARY_LOW,
    BUCKET_FAIL,
    BUCKET_PASS,
    BOUNDARY_STRATA,
    STRATA,
    D4Assumptions,
    D4Label,
    GenConfig,
    Provenance,
    Quarantined,
    Skeleton,
    SkeletonFlags,
    SkeletonGenError,
    allocate_counts,
    assemble_pair,
    audit_skeletons,
    build_skeleton,
    config_sha256,
    decompose_counts,
    enumerate_combos,
    find_defect_tokens,
    generate_corpus_skeletons,
    load_defect_lexicon,
    normal_pair_violations,
    skeleton_from_json,
    skeleton_from_label,
    skeletons_from_labels,
    skeletons_sha256,
)
from tests.corpus.conftest import make_row, make_table

D = Decimal
CFG = GenConfig()


def _prov() -> Provenance:
    return Provenance(limits_sha256="0" * 64, seed=0, configs_sha256=config_sha256(CFG))


# ---------------------------------------------------------------------------
# 픽스처 테이블 — F1(const)·F2(prop_t)·F3(prop_t_cap)·F4(불허)·F5(비율) 각 1행
# ---------------------------------------------------------------------------

R1 = make_row()  # const 직경 2011 ST [8,25) C L=3 le mm, clause KR-3.2.1
R2 = make_row(rule_id="T-002", defect_code="401", limit_type="길이",
              limit_rule="prop_t", limit_value=None, limit_factor=D("0.1"),
              ratio_basis="t", clause_id="KR-3.3.1")
R3 = make_row(rule_id="T-003", defect_code="401", limit_type="길이",
              limit_rule="prop_t_cap", limit_value=None, limit_factor=D("0.2"),
              limit_cap=D("5"), ratio_basis="t",
              thickness_min=D("25"), thickness_max=D("50"), clause_id="KR-3.3.2")
R4 = make_row(rule_id="T-004", defect_code="100", limit_type="불허",
              limit_rule="none_permitted", limit_value=None, unit=None,
              clause_id="KR-3.1.1")
R5 = make_row(rule_id="T-005", defect_code="2012", limit_type="비율",
              limit_rule="const", limit_value=D("25"), unit="percent",
              clause_id="KR-3.4.1")

TABLE = make_table(R1, R2, R3, R4, R5)


@pytest.fixture(scope="module")
def skeletons():
    # 5조합 × 40 = 200건 (총량 충분 → 전 조합 상한 도달)
    return generate_corpus_skeletons(TABLE, seed=7, total=10_000, cap=40)


def _expected_interval(L: Decimal, bucket: str) -> tuple[int, int]:
    """구현과 독립인 그리드 인덱스 구간 계산 (스펙 §4-3 정의 직접 전사)."""
    Li = int(quantize(L) / Q)
    ceil_ = lambda d: int(d.to_integral_value(rounding=ROUND_CEILING))
    floor_ = lambda d: int(d.to_integral_value(rounding=ROUND_FLOOR))
    if bucket == BUCKET_PASS:
        return 1, ceil_(D("0.9") * Li) - 1
    if bucket == BUCKET_BOUNDARY_LOW:
        return ceil_(D("0.9") * Li), Li - 1
    if bucket == BUCKET_BOUNDARY_EQ:
        return Li, Li
    if bucket == BUCKET_BOUNDARY_HIGH:
        return Li + 1, floor_(D("1.1") * Li)
    return floor_(D("1.1") * Li) + 1, floor_(D("2.5") * Li)


# ---------------------------------------------------------------------------
# T15 — 각 버킷 표본 전수 구간 내 (그리드 인덱스 기준)
# ---------------------------------------------------------------------------


def test_t15_all_samples_inside_bucket_grid_interval(skeletons):  # T15
    for sk in skeletons:
        measured = sk.measured_value if sk.measured_value is not None else sk.size_mm
        mi = int(measured / Q)
        assert measured == (D(mi) * Q).quantize(Q), "측정값이 0.01 그리드 밖"
        if sk.limit_rule is LimitRule.NONE_PERMITTED:
            lo, hi = int(CFG.f4_size_min / Q), int(CFG.f4_size_max / Q)
        else:
            lo, hi = _expected_interval(sk.limit_value, sk.sample_bucket)
            if sk.unit is not None and sk.unit.value == "percent":
                hi = min(hi, int(CFG.percent_max / Q))
        assert lo <= mi <= hi, (
            f"{sk.sample_id}: {sk.sample_bucket} 구간 [{lo},{hi}] 밖 {mi}"
        )


def test_t15_boundary_noneq_at_least_one_grid_step_from_L(skeletons):  # T15 (=L 외 0.01 차이)
    for sk in skeletons:
        if sk.sample_bucket in (BUCKET_BOUNDARY_LOW, BUCKET_BOUNDARY_HIGH):
            m = sk.measured_value if sk.measured_value is not None else sk.size_mm
            assert abs(m - sk.limit_value) >= Q
        elif sk.sample_bucket == BUCKET_BOUNDARY_EQ:
            m = sk.measured_value if sk.measured_value is not None else sk.size_mm
            assert m == sk.limit_value


# ---------------------------------------------------------------------------
# T16 — 최소 보장 분해
# ---------------------------------------------------------------------------


def test_t16_decompose_minimum_guarantee():  # T16
    # n ≥ 3: =L·하·상 각 ≥ 1, 층 합 = n
    for n in (3, 5, 12, 27, 40):
        c = decompose_counts(n, LimitRule.CONST, CFG)
        assert sum(c.values()) == n
        for s in BOUNDARY_STRATA:
            assert c[s] >= 1, f"n={n}: {s} 최소 보장 결손"
    # n=40 은 40/40/20 정확 (경계 내부 8/4/8)
    c40 = decompose_counts(40, LimitRule.CONST, CFG)
    assert c40[BUCKET_PASS] == 16 and c40[BUCKET_FAIL] == 16
    assert sum(c40[s] for s in BOUNDARY_STRATA) == 8


def test_t16_decompose_small_n_priority():  # T16 (n<3 우선순위 =L → 상 → 하)
    c1 = decompose_counts(1, LimitRule.CONST, CFG)
    assert c1[BUCKET_BOUNDARY_EQ] == 1 and sum(c1.values()) == 1
    c2 = decompose_counts(2, LimitRule.CONST, CFG)
    assert c2[BUCKET_BOUNDARY_EQ] == 1 and c2[BUCKET_BOUNDARY_HIGH] == 1


def test_t16_f4_all_fail():  # T16 보충 (F4 는 40/40/20 미적용 — §4-3)
    c = decompose_counts(20, LimitRule.NONE_PERMITTED, CFG)
    assert c[BUCKET_FAIL] == 20 and sum(c.values()) == 20


def test_t16_generated_combos_have_all_boundary_strata(skeletons):  # T16 (생성 결과)
    by_clause: dict[str, set] = {}
    for sk in skeletons:
        if sk.limit_rule is not LimitRule.NONE_PERMITTED:
            by_clause.setdefault(sk.clause_id, set()).add(sk.sample_bucket)
    for clause, buckets in by_clause.items():
        assert set(BOUNDARY_STRATA) <= buckets, f"{clause}: 경계 3층 결손"


# ---------------------------------------------------------------------------
# T17 — 조합 상한·총량 축소
# ---------------------------------------------------------------------------


def test_t17_cap_and_total_reduction():  # T17
    combos = enumerate_combos(TABLE)
    counts = allocate_counts(combos, total=10_000, cap=40)
    assert all(v <= 40 for v in counts.values())
    # 상한 미달 → 상한을 올리지 않고 총량 축소 (5조합 × 40 = 200)
    assert sum(counts.values()) == 200
    counts2 = allocate_counts(combos, total=150, cap=40)
    assert sum(counts2.values()) == 150 and all(v <= 40 for v in counts2.values())
    counts3 = allocate_counts(combos, total=152, cap=40)
    assert sum(counts3.values()) == 152
    assert sorted(counts3.values(), reverse=True)[:2] == [31, 31]  # 잔여 정렬순 라운드로빈


# ---------------------------------------------------------------------------
# T18 — 산출 조합 = 열거 결과 상호 포함 (외삽 0 + 결손 0)
# ---------------------------------------------------------------------------


def test_t18_no_ghost_no_missing_combo(skeletons):  # T18
    expected = {c.row.rule_id for c in enumerate_combos(TABLE)}
    realized = set()
    for sk in skeletons:
        row = applicable_row(TABLE, sk.defect_code, sk.material, sk.inspection_method,
                             sk.quality_scheme, sk.quality_level, sk.thickness_mm,
                             limit_type=sk.limit_type)
        realized.add(row.rule_id)
    assert realized == expected


# ---------------------------------------------------------------------------
# T19 — 결정론: 같은 (table, seed) → sha 동일 / 시드 변경 → 상이
# ---------------------------------------------------------------------------


def test_t19_same_seed_same_sha_diff_seed_diff(skeletons):  # T19
    again = generate_corpus_skeletons(TABLE, seed=7, total=10_000, cap=40)
    assert skeletons_sha256(skeletons) == skeletons_sha256(again)
    other = generate_corpus_skeletons(TABLE, seed=8, total=10_000, cap=40)
    assert skeletons_sha256(other) != skeletons_sha256(skeletons)
    # 시드 변경은 provenance 뿐 아니라 표본 자체를 바꾼다
    m1 = [str(s.size_mm or s.measured_value) for s in skeletons]
    m2 = [str(s.size_mm or s.measured_value) for s in other]
    assert m1 != m2


# ---------------------------------------------------------------------------
# T20 — 행 순서 셔플 → 출력 불변
# ---------------------------------------------------------------------------


def test_t20_row_order_invariance(skeletons):  # T20
    shuffled = make_table(R5, R3, R1, R4, R2)
    out = generate_corpus_skeletons(shuffled, seed=7, total=10_000, cap=40)
    assert [s.to_json_dict() for s in out] == [s.to_json_dict() for s in skeletons]


# ---------------------------------------------------------------------------
# T21 — 무관 행 추가 → 기존 조합 출력 불변 (정체성 시드)
# ---------------------------------------------------------------------------


def test_t21_unrelated_row_leaves_existing_combos_unchanged(skeletons):  # T21
    r6 = make_row(rule_id="T-006", defect_code="301", limit_type="길이",
                  limit_value=D("2"), clause_id="KR-3.5.1")
    bigger = make_table(R1, R2, R3, R4, R5, r6)
    out = generate_corpus_skeletons(bigger, seed=7, total=10_000, cap=40)
    base = {d["sample_id"]: d for s in skeletons for d in [s.to_json_dict()]}
    new = {d["sample_id"]: d for s in out for d in [s.to_json_dict()]
           if d["defect_code"] != "301"}
    assert new == base


# ---------------------------------------------------------------------------
# T22 — 전 골격 verdict·margin judge() 재계산 일치, clause·L 의 CSV 행 일치
# ---------------------------------------------------------------------------


def test_t22_recompute_matches(skeletons):  # T22
    for sk in skeletons:
        row = applicable_row(TABLE, sk.defect_code, sk.material, sk.inspection_method,
                             sk.quality_scheme, sk.quality_level, sk.thickness_mm,
                             limit_type=sk.limit_type)
        assert row.clause_id == sk.clause_id
        basis = sk.thickness_mm if row.limit_rule in (
            LimitRule.PROP_T, LimitRule.PROP_T_CAP) else None
        assert effective_limit(row, basis) == sk.limit_value
        measured = sk.measured_value if sk.measured_value is not None else sk.size_mm
        j = judge(row, measured, basis)
        assert j.verdict == sk.verdict and j.margin == sk.margin


# ---------------------------------------------------------------------------
# T23 — JSON round-trip 동치, verdict enum 강제, NaN·inf 없음
# ---------------------------------------------------------------------------


def _assert_no_float(obj):
    if isinstance(obj, float):
        raise AssertionError(f"float 발견: {obj!r} (NaN·inf 경로)")
    if isinstance(obj, dict):
        for v in obj.values():
            _assert_no_float(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_float(v)


def test_t23_json_roundtrip_and_enum(skeletons):  # T23
    for sk in skeletons[:50]:
        d = sk.to_json_dict()
        _assert_no_float(d)
        d2 = json.loads(json.dumps(d, ensure_ascii=False))
        assert d2 == d
        assert skeleton_from_json(d2) == sk
        assert d["verdict"] in (PASS, FAIL)


def test_t23_verdict_enum_rejected(skeletons):  # T23 (enum 강제)
    bad = skeletons[0].to_json_dict()
    bad["verdict"] = "보류"
    with pytest.raises(ValidationError):
        skeleton_from_json(bad)


# ---------------------------------------------------------------------------
# T24 — 동일 입력을 (c)·D4 경로로 → verdict·margin 동일
# ---------------------------------------------------------------------------


def test_t24_same_input_both_paths_same_judgment():  # T24
    m, t = D("2.50"), D("10")
    flags = SkeletonFlags(source="sampled", verdict_mode=VerdictMode.FULL,
                          sample_bucket=BUCKET_PASS)
    sk_c = build_skeleton(R1, m, t, flags, sample_id="x-000", provenance=_prov())
    label = D4Label(image_id="img1", defect_instance_id="d0", defect_type="porosity",
                    material="ST", size_px=D("250"), thickness_mm=D("10"))
    sk_d4 = skeleton_from_label(
        TABLE, label, scale=D("0.01"),
        assumptions=D4Assumptions(verdict_mode=VerdictMode.FULL),
    )
    assert isinstance(sk_d4, Skeleton)
    assert sk_d4.size_mm == sk_c.size_mm == m
    assert sk_d4.verdict == sk_c.verdict and sk_d4.margin == sk_c.margin


# ---------------------------------------------------------------------------
# T25 — 픽셀→mm 기지값, scale=None → assumed·조건부 플래그
# ---------------------------------------------------------------------------


def test_t25_known_scale_metadata_flags():  # T25
    label = D4Label(image_id="img2", defect_instance_id="d0", defect_type="porosity",
                    material="ST", size_px=D("250"), thickness_mm=D("12"))
    sk = skeleton_from_label(TABLE, label, scale=D("0.01"),
                             assumptions=D4Assumptions(verdict_mode=VerdictMode.FULL))
    assert isinstance(sk, Skeleton)
    assert sk.size_mm == D("2.50")
    assert sk.scale_source == "metadata" and sk.thickness_source == "metadata"
    # 두께·스케일은 메타데이터지만 품질수준은 configs 가정값이다 (§4-5 quality_source).
    # 판정을 가른 허용치가 그 가정에서 선택된 행에서 나오므로 확정이 아니라 조건부다.
    assert sk.quality_source == "assumed"
    assert sk.verdict_type == "조건부"


def test_t25_scale_none_falls_back_to_assumed_conditional():  # T25
    label = D4Label(image_id="img3", defect_instance_id="d0", defect_type="porosity",
                    material="ST", size_px=D("250"), thickness_mm=None)
    sk = skeleton_from_label(
        TABLE, label, scale=None,
        assumptions=D4Assumptions(thickness_mm=D("12"), scale_mm_per_px=D("0.02"),
                                  verdict_mode=VerdictMode.CONDITIONAL),
    )
    assert isinstance(sk, Skeleton)
    assert sk.size_mm == D("5.00")  # 250 px × 0.02
    assert sk.scale_source == "assumed" and sk.thickness_source == "assumed"
    assert sk.verdict_type == "조건부"
    d = sk.to_json_dict()
    assert d["verdict"] is not None  # conditional 은 게이트 대상 아님


# ---------------------------------------------------------------------------
# T26 — (c) 생성 중 이상 주입 → 부분 산출 없이 전체 중단
# ---------------------------------------------------------------------------


def test_t26_internal_invariant_aborts_whole_run():  # T26
    # L=0.05 → 경계 버킷 그리드 공집합 (V10 미통과 상당) — (c) 경로는 전체 중단
    bad = make_table(R1, make_row(rule_id="T-BAD", limit_value=D("0.05"),
                                  clause_id="KR-9.9.9", defect_code="301"))
    with pytest.raises(SkeletonGenError):
        generate_corpus_skeletons(bad, seed=7, total=10_000, cap=40)


# ---------------------------------------------------------------------------
# T29 — 전 골격 저장 t 재조회 → 생성 시와 동일 행·clause_id
# ---------------------------------------------------------------------------


def test_t29_stored_thickness_reselects_same_row(skeletons):  # T29
    for sk in skeletons:
        row = applicable_row(TABLE, sk.defect_code, sk.material, sk.inspection_method,
                             sk.quality_scheme, sk.quality_level, sk.thickness_mm,
                             limit_type=sk.limit_type)
        assert row.clause_id == sk.clause_id
        assert row.limit_rule is sk.limit_rule


# ---------------------------------------------------------------------------
# T30 — assemble_pair: 다중 결함 AND, 인용 합집합 정렬·중복 제거
# ---------------------------------------------------------------------------


def _d4_flags(image_id: str, inst: str) -> SkeletonFlags:
    return SkeletonFlags(source="measured", verdict_mode=VerdictMode.FULL,
                         image_id=image_id, defect_instance_id=inst,
                         thickness_source="metadata", scale_source="metadata",
                         quality_source="assumed")


def test_t30_assemble_pair_and_rule_and_cited_union():  # T30
    a = build_skeleton(R1, D("2.00"), D("10"), _d4_flags("imgX", "d0"),
                       "imgX-d0", _prov())      # 합격
    b = build_skeleton(R1, D("3.50"), D("10"), _d4_flags("imgX", "d1"),
                       "imgX-d1", _prov())      # 불합격 (같은 조항 — 중복 인용)
    c = build_skeleton(R2, D("0.50"), D("10"), _d4_flags("imgX", "d2"),
                       "imgX-d2", _prov())      # 합격 (다른 조항)
    pair = assemble_pair("imgX", [a, b, c], VerdictMode.FULL)
    assert pair.verdict == FAIL  # 한 결함이라도 불합격 → 불합격
    assert pair.cited_clauses == ("KR-3.2.1", "KR-3.3.1")  # 정렬·중복 제거
    all_pass = assemble_pair("imgY", [
        build_skeleton(R1, D("2.00"), D("10"), _d4_flags("imgY", "d0"), "imgY-d0", _prov()),
        build_skeleton(R2, D("0.50"), D("10"), _d4_flags("imgY", "d1"), "imgY-d1", _prov()),
    ], VerdictMode.FULL)
    assert all_pass.verdict == PASS


# ---------------------------------------------------------------------------
# T31 — D4 퇴화 라벨·구간 밖 두께 → quarantine 배출, 생성 계속, 사유 기록
# ---------------------------------------------------------------------------


def test_t31_quarantine_does_not_abort_batch():  # T31
    labels = [
        D4Label("ok1", "d0", "porosity", "ST", D("250"), D("10")),
        D4Label("bad1", "d0", "porosity", "ST", D("0"), D("10")),    # 퇴화 폴리곤
        D4Label("bad2", "d0", "porosity", "ST", D("250"), D("60")),  # 표 구간 밖 두께
    ]
    oks, quarantined = skeletons_from_labels(
        TABLE, labels, scale=D("0.01"),
        assumptions=D4Assumptions(verdict_mode=VerdictMode.FULL),
    )
    assert len(oks) == 1 and oks[0].image_id == "ok1"
    assert len(quarantined) == 2
    reasons = {q.image_id: q.reason for q in quarantined}
    assert reasons["bad1"] == "degenerate_polygon"
    assert reasons["bad2"] == "no_applicable_row"
    assert all(isinstance(q, Quarantined) for q in quarantined)


def test_t31_excluded_2012_quarantined():  # T31 보충 (S2-4: D4 는 2011 확정, 2012 배제)
    label = D4Label("img5", "d0", "2012", "ST", D("250"), D("10"))
    q = skeleton_from_label(TABLE, label, scale=D("0.01"),
                            assumptions=D4Assumptions(verdict_mode=VerdictMode.FULL))
    assert isinstance(q, Quarantined) and q.reason == "excluded_2012"


# ---------------------------------------------------------------------------
# T32 — 널러빌리티 매트릭스 준수 (§4-4)
# ---------------------------------------------------------------------------


def test_t32_nullability_matrix(skeletons):  # T32
    for sk in skeletons:
        d = sk.to_json_dict()
        if sk.limit_rule is LimitRule.NONE_PERMITTED:      # F4
            assert d["size_mm"] is not None
            assert d["measured_value"] is None
            assert d["limit_value"] is None and d["margin"] is None
            assert d["verdict"] == FAIL
        elif sk.unit is not None and sk.unit.value == "percent":  # F5 비율
            assert d["size_mm"] is None
            assert d["measured_value"] is not None and d["measured_unit"] == "percent"
            assert d["limit_value"] is not None and d["margin"] is not None
        else:                                              # 길이·직경 (const·prop 계열)
            assert d["size_mm"] is not None
            assert d["measured_value"] is None
            assert d["limit_value"] is not None and d["margin"] is not None


# ---------------------------------------------------------------------------
# T34 (개정판, §4-7) — 정상 페어 스키마 강제 + 결함 어휘 락
# ---------------------------------------------------------------------------


def test_t34_normal_pair_schema_full_and_clause_only():  # T34
    full = assemble_pair("imgN", [], VerdictMode.FULL, quality_level="C").to_json_dict()
    assert full["defects"] == [] and full["cited_clauses"] == []
    assert full["verdict"] == PASS  # aggregate_verdicts([]) = 합격 (full 모드)
    assert full["sample_id"] == "imgN-normal"
    gated = assemble_pair("imgN", [], VerdictMode.CLAUSE_ONLY,
                          quality_level="C").to_json_dict()
    assert gated["verdict"] is None  # 개정판: clause_only 면 null
    assert gated["cited_clauses"] == []


def test_t34_normal_pair_violation_detection():  # T34
    good = assemble_pair("imgN", [], VerdictMode.FULL, quality_level="C").to_json_dict()
    assert normal_pair_violations(good, VerdictMode.FULL) == ()
    tampered = dict(good, cited_clauses=["KR-3.2.1"])
    assert "cited_clauses_nonempty" in normal_pair_violations(tampered, VerdictMode.FULL)
    wrong_verdict = dict(good, verdict=FAIL)
    assert "verdict_mismatch" in normal_pair_violations(wrong_verdict, VerdictMode.FULL)
    not_null = dict(good)  # clause_only 인데 verdict 가 남아 있음
    assert "verdict_not_null" in normal_pair_violations(not_null, VerdictMode.CLAUSE_ONLY)


def test_t34_defect_lexicon_lock():  # T34 (환각 결함 주입 → 폐기)
    lex = load_defect_lexicon()
    # 부정 문맥도 문맥 불문 검출 (§4-6-1 ③: "기공은 관찰되지 않았다" 금지)
    assert find_defect_tokens("방사선 사진에서 기공은 관찰되지 않았다", lex)
    assert find_defect_tokens("ISO 6520-1 2011 에 해당하는 지시", lex)
    # 깨끗한 정상 서술 + 숫자 경계 (가정 두께 1000 은 코드 100 이 아니다)
    assert find_defect_tokens("검출 한계 내 특기할 결함 지시 없음. 두께 1000 mm 기준", lex) == ()


# ---------------------------------------------------------------------------
# G2 사후 감사 (§4-9) — T15~T22 의 통합 회로가 실제로 작동하는지
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 검사 방법 축 (게이트 #13, 스펙 v1.3 §1-2 5a) — RT 기공과 VT 기공의 병존
# ---------------------------------------------------------------------------

# 같은 결함코드 2011·같은 재질·같은 수준·같은 limit_type 에 두께 구간까지 겹치는 두 행.
# 가르는 것은 inspection_method 뿐이다 (파일럿 CSV 의 표15 ↔ 표면결함표 구도).
RT_POROSITY = make_row(rule_id="T-RT-2011", inspection_method="RT",
                       limit_value=D("3"), clause_id="KR-3.2.1")
VT_POROSITY = make_row(rule_id="T-VT-2011", inspection_method="VT",
                       limit_value=D("6"), clause_id="KR-3.2.2")
MIXED_TABLE = make_table(RT_POROSITY, VT_POROSITY)


def test_inspection_axis_enumerate_takes_rt_only():
    """기본 필터 {RT, ALL} — VT 행은 (c) 조합 열거에 들어오지 않는다."""
    rt = enumerate_combos(MIXED_TABLE)
    assert [c.row.rule_id for c in rt] == ["T-RT-2011"]
    vt = enumerate_combos(MIXED_TABLE, inspection_method=InspectionMethod.VT)
    assert [c.row.rule_id for c in vt] == ["T-VT-2011"]


def test_inspection_axis_generation_does_not_mix():
    """축 없이 열거하면 applicable_row 가 복수 매칭으로 무너진다 — 축이 그 붕괴를 막는다."""
    out = generate_corpus_skeletons(MIXED_TABLE, seed=7, total=40, cap=40)
    assert out
    assert {sk.inspection_method for sk in out} == {InspectionMethod.RT}
    assert {sk.clause_id for sk in out} == {"KR-3.2.1"}
    assert all(sk.limit_value == D("3.00") for sk in out)
    # 축을 끄면(inspection_method=None) 생성 자체를 거부한다. 조용히 한쪽을 고르지 않는
    # 것을 넘어 아예 시작하지 않는다는 것이 요점이다 — 축 없는 corpus 는 표면·내부 기준이
    # 섞여 있어 판정 정합성이 틀린 채로 높게 나온다.
    with pytest.raises(SkeletonGenError, match="축 미지정"):
        generate_corpus_skeletons(MIXED_TABLE, seed=7, total=40, cap=40,
                                  inspection_method=None)


def test_inspection_axis_vt_generation_uses_surface_limit():
    """같은 결함코드라도 VT 축으로 생성하면 표면 기준 한계가 잡힌다."""
    out = generate_corpus_skeletons(MIXED_TABLE, seed=7, total=40, cap=40,
                                    inspection_method="VT")
    assert {sk.clause_id for sk in out} == {"KR-3.2.2"}
    assert all(sk.limit_value == D("6.00") for sk in out)
    assert audit_skeletons(out, MIXED_TABLE, inspection_method="VT").ok


def test_inspection_axis_is_carried_in_skeleton_json(skeletons):
    """D 채점기의 행 재선택 축이 골격에 실려 있어야 한다 (§1-4 rows_for_clause)."""
    d = skeletons[0].to_json_dict()
    assert d["inspection_method"] == "RT"
    assert skeleton_from_json(d) == skeletons[0]


def test_inspection_axis_audit_rejects_wrong_axis():
    """RT 골격을 VT 축으로 감사하면 행 재선택이 실패해 감사가 떨어진다 (조용한 통과 금지)."""
    out = generate_corpus_skeletons(MIXED_TABLE, seed=7, total=40, cap=40)
    report = audit_skeletons(out, MIXED_TABLE, inspection_method="VT")
    assert not report.ok


def test_inspection_axis_d4_selects_rt_row():
    """D4 실측 경로도 같은 축을 지난다 — 표면 기준으로 채점되면 판정이 조용히 뒤집힌다."""
    label = D4Label("imgRT", "d0", "porosity", "ST", D("450"), D("10"))
    sk = skeleton_from_label(MIXED_TABLE, label, scale=D("0.01"),
                             assumptions=D4Assumptions(verdict_mode=VerdictMode.FULL))
    assert isinstance(sk, Skeleton)
    assert sk.size_mm == D("4.50")
    assert sk.inspection_method is InspectionMethod.RT
    assert sk.clause_id == "KR-3.2.1" and sk.verdict == FAIL  # 4.50 > RT 한계 3.00
    vt = skeleton_from_label(
        MIXED_TABLE, label, scale=D("0.01"),
        assumptions=D4Assumptions(inspection_method="VT", verdict_mode=VerdictMode.FULL),
    )
    assert isinstance(vt, Skeleton)
    assert vt.clause_id == "KR-3.2.2" and vt.verdict == PASS  # 4.50 ≤ VT 한계 6.00


def test_g2_audit_green_and_detects_flip(skeletons):  # §4-9 (T28 의 G2 검출 절반)
    report = audit_skeletons(skeletons, TABLE)
    assert report.ok, report.failures
    assert report.n_skeletons == 200
    # 40/40/20 (F4 제외 160건): pass 64 / fail 64 / boundary 32
    assert report.bucket_totals[BUCKET_PASS] == 64
    # verdict 반전 오염본 → 감사가 잡는다
    flipped = skeletons[0].model_copy(
        update={"verdict": FAIL if skeletons[0].verdict == PASS else PASS})
    tampered = (flipped,) + tuple(skeletons[1:])
    assert not audit_skeletons(tampered, TABLE).ok
