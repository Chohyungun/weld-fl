"""수치 잠금 검증 테스트 (스펙 §5-4, §4-6-1 ④) — 재추출·정규화·화이트리스트·판정어."""

from __future__ import annotations

from decimal import Decimal

import pytest

from corpus.generate.numeric_lock import (
    REASON_CHANGED_VALUE,
    REASON_DEFECT_HALLUCINATION,
    REASON_EXTRA_VALUE,
    REASON_FORMAT_VIOLATION,
    REASON_MISSING_VALUE,
    REASON_VERDICT_FLIP,
    check_normal_lock,
    check_numeric_lock,
    check_pair_lock,
    find_defect_tokens,
    to_decimal,
)

# ---------------------------------------------------------------------------
# 골격 픽스처 (§4-5 골격 JSON 형태의 dict — skeleton_gen 비의존)
# ---------------------------------------------------------------------------


def sk_const(**over) -> dict:
    sk = {
        "defect_code": "2011", "material": "ST", "quality_scheme": "iso5817",
        "quality_level": "C", "limit_type": "직경", "limit_rule": "const",
        "thickness_mm": "12.00", "size_mm": "2.00", "measured_value": None,
        "clause_id": "KR-3.2.1", "limit_value": "3.00", "margin": "1.00",
        "verdict": "합격", "limit_op": "le", "unit": "mm",
        "limit_factor": None, "limit_cap": None, "ratio_basis": None,
    }
    sk.update(over)
    return sk


GOOD_TEXT = (
    "관찰: 직경 2.00 mm의 원형 음영. 분류: 기공 (ISO 6520-1 2011). "
    "조회: 모재 두께 12 mm 품질수준 C → 조항 KR-3.2.1 기준 3 mm 이하. "
    "대조: 2.00 ≤ 3.00. 판정: 합격."
)

DEFECT_LEXICON = ("균열", "기공", "융합불량", "슬래그혼입", "2011", "2012", "100", "401", "301")


# ---------------------------------------------------------------------------
# 기본 통과·정규화 (§5-4 1·2단계)
# ---------------------------------------------------------------------------


def test_good_text_passes():
    r = check_numeric_lock(GOOD_TEXT, sk_const())
    assert r.ok, r
    assert r.reasons == ()


def test_fullwidth_and_mm_sign_normalized():
    # 전각 숫자·㎜ 기호 → NFKC 정규화 후 동일 판정 (㎜→mm, ２→2)
    text = GOOD_TEXT.replace("2.00 mm의", "２.００㎜의")
    assert check_numeric_lock(text, sk_const()).ok


def test_trailing_zero_equivalence():
    # 후행 0 동치: 골격 3.00 ↔ 생성문 "3" / 골격 12.00 ↔ "12.0"
    text = GOOD_TEXT.replace("두께 12 mm", "두께 12.0 mm")
    assert check_numeric_lock(text, sk_const()).ok


def test_josa_separation():
    text = GOOD_TEXT.replace("2.00 mm의 원형 음영", "2.00mm를 나타내는 원형 음영")
    assert check_numeric_lock(text, sk_const()).ok


def test_empty_text_is_format_violation():
    r = check_numeric_lock("   ", sk_const())
    assert not r.ok
    assert r.reasons == (REASON_FORMAT_VIOLATION,)


def test_float_in_skeleton_rejected():
    with pytest.raises(TypeError):
        to_decimal(3.0)
    r = check_numeric_lock(GOOD_TEXT, sk_const(size_mm=2.0))
    assert not r.ok
    assert REASON_FORMAT_VIOLATION in r.reasons


# ---------------------------------------------------------------------------
# 화이트리스트 대조 (§5-4 3단계 — 허용 밖 1개면 폐기, 필수 결손도 폐기)
# ---------------------------------------------------------------------------


def test_extra_value_discarded():
    r = check_numeric_lock(GOOD_TEXT + " 참고로 여유는 1.53 mm 이다.", sk_const())
    assert not r.ok
    assert REASON_EXTRA_VALUE in r.reasons


def test_margin_value_is_allowed():
    # margin(1.00)과 그 절대값 표기는 허용 집합에 있다
    r = check_numeric_lock(GOOD_TEXT + " 여유 1.00 mm.", sk_const())
    assert r.ok, r


def test_missing_required_limit_value():
    text = (
        "관찰: 직경 2.00 mm의 원형 음영. 조회: 두께 12 mm 기준 KR-3.2.1. 판정: 합격."
    )  # L(3.00) 미출현
    r = check_numeric_lock(text, sk_const())
    assert not r.ok
    assert REASON_MISSING_VALUE in r.reasons


def test_changed_value():
    # 3 → 4 로 바꾸면 필수 결손 + 허용 밖 출현이 동시에 발생 = changed_value
    text = GOOD_TEXT.replace("기준 3 mm", "기준 4 mm").replace("≤ 3.00", "≤ 4.00")
    r = check_numeric_lock(text, sk_const())
    assert not r.ok
    assert REASON_CHANGED_VALUE in r.reasons


def test_clause_changed():
    text = GOOD_TEXT.replace("KR-3.2.1", "KR-9.9.9")
    r = check_numeric_lock(text, sk_const())
    assert not r.ok
    assert REASON_CHANGED_VALUE in r.reasons


def test_clause_extra():
    r = check_numeric_lock(GOOD_TEXT + " 또한 KR-3.2.2 참조.", sk_const())
    assert not r.ok
    assert REASON_EXTRA_VALUE in r.reasons


def test_std_ref_masked_not_counted():
    # "ISO 6520-1"의 6520·1 은 표준 식별자 — 수치 토큰으로 세지 않는다 (GOOD_TEXT 에 포함)
    assert check_numeric_lock(GOOD_TEXT, sk_const()).ok


def test_thousands_comma_removed():
    sk = sk_const(size_mm="1000.00", limit_value="1200.00", margin="200.00")
    text = (
        "관찰: 길이 1,000.00 mm 지시. 조회: 두께 12 mm 기준 KR-3.2.1 허용 1,200 mm 이하. "
        "판정: 합격."
    )
    assert check_numeric_lock(text, sk).ok


# ---------------------------------------------------------------------------
# 단위 대조 (§5-4 3단계 "수치 토큰(숫자+단위)") — 값이 같아도 단위가 다르면 다른 물리량
# ---------------------------------------------------------------------------


def sk_ratio(**over) -> dict:
    sk = sk_const(
        defect_code="2012", limit_type="비율", limit_rule="const", unit="percent",
        size_mm=None, measured_value="20.00", limit_value="25.00", margin="5.00",
    )
    sk.update(over)
    return sk


RATIO_TEXT = (
    "관찰: 기공 면적 비율 20.00 %. 분류: 2012. 조회: 두께 12 mm 기준 KR-3.2.1 "
    "허용 25 % 이하. 판정: 합격."
)


def test_percent_skeleton_written_in_mm_discarded():
    # 20% 를 20mm 로 적으면 골격과 다른 물리량이다 (§4-5 measured_unit 이 있는 이유)
    text = RATIO_TEXT.replace("비율 20.00 %", "비율 20.00 mm")
    r = check_numeric_lock(text, sk_ratio())
    assert not r.ok
    assert REASON_CHANGED_VALUE in r.reasons


def test_mm_skeleton_written_in_percent_discarded():
    text = GOOD_TEXT.replace("직경 2.00 mm의", "직경 2.00 %의")
    r = check_numeric_lock(text, sk_const())
    assert not r.ok
    assert REASON_CHANGED_VALUE in r.reasons


def test_thickness_is_always_mm():
    text = GOOD_TEXT.replace("두께 12 mm", "두께 12 %")
    r = check_numeric_lock(text, sk_const())
    assert not r.ok
    assert REASON_CHANGED_VALUE in r.reasons


def test_unitless_mention_allowed():
    # 대조 대상은 "다른 단위" 뿐이다 — 단위를 붙이지 않은 표기는 통과한다
    text = GOOD_TEXT.replace("기준 3 mm 이하", "기준 3 이하")
    assert check_numeric_lock(text, sk_const()).ok


def test_ratio_text_with_correct_units_passes():
    assert check_numeric_lock(RATIO_TEXT, sk_ratio()).ok


def test_pair_lock_unit_tamper():
    text = PAIR_TEXT.replace("길이 6.00 mm", "길이 6.00 %")
    r = check_pair_lock(text, PAIR_DEFECTS, image_verdict="불합격")
    assert not r.ok
    assert REASON_CHANGED_VALUE in r.reasons


# ---------------------------------------------------------------------------
# 표준 식별자 화이트리스트 (§5-4 3단계) — 골격에 근거 없는 표준 인용은 환각이다
# ---------------------------------------------------------------------------


def test_defect_code_system_reference_allowed():
    # ISO 6520-1 은 결함 코드 체계(사실 정보)라 골격 지시 없이도 허용된다
    assert check_numeric_lock(GOOD_TEXT, sk_const()).ok


def test_hallucinated_standard_discarded():
    r = check_numeric_lock(GOOD_TEXT + " 아울러 ISO 9999 요건도 충족한다.", sk_const())
    assert not r.ok
    assert REASON_EXTRA_VALUE in r.reasons


def test_standard_from_skeleton_source_doc_allowed():
    text = GOOD_TEXT.replace("판정: 합격.", "출처 IACS Rec.47. 판정: 합격.")
    assert not check_numeric_lock(text, sk_const()).ok
    assert check_numeric_lock(text, sk_const(source_doc="IACS Rec.47")).ok


def test_standard_whitelist_injection():
    text = GOOD_TEXT + " EN 1290 참조."
    assert not check_numeric_lock(text, sk_const()).ok
    assert check_numeric_lock(
        text, sk_const(), allowed_standards=("ISO 6520-1", "EN 1290")
    ).ok


def test_pair_lock_hallucinated_standard():
    r = check_pair_lock(PAIR_TEXT + " ISO 9999 참조.", PAIR_DEFECTS, image_verdict="불합격")
    assert not r.ok
    assert REASON_EXTRA_VALUE in r.reasons


# ---------------------------------------------------------------------------
# 판정어 대조 (§5-4 4단계) · rule 별 필수 출현 (§4-4 매트릭스)
# ---------------------------------------------------------------------------


def test_verdict_flip():
    text = GOOD_TEXT.replace("판정: 합격.", "판정: 불합격.")
    r = check_numeric_lock(text, sk_const())
    assert not r.ok
    assert REASON_VERDICT_FLIP in r.reasons


def test_verdict_missing():
    text = GOOD_TEXT.replace(" 판정: 합격.", "")
    r = check_numeric_lock(text, sk_const())
    assert not r.ok
    assert REASON_MISSING_VALUE in r.reasons


def test_verdict_both_words_is_format_violation():
    text = GOOD_TEXT + " 합격 기준을 넘으면 불합격이다."
    r = check_numeric_lock(text, sk_const())
    assert not r.ok
    assert REASON_FORMAT_VIOLATION in r.reasons


def test_clause_only_gate_no_verdict_ok():
    sk = sk_const(verdict=None, margin=None)
    text = (
        "관찰: 직경 2.00 mm의 원형 음영 (2011). 조회: 두께 12 mm 기준 KR-3.2.1 "
        "허용 3.00 mm 이하."
    )
    assert check_numeric_lock(text, sk).ok


def test_clause_only_gate_verdict_word_is_flip():
    sk = sk_const(verdict=None, margin=None)
    text = (
        "관찰: 직경 2.00 mm의 원형 음영 (2011). 조회: 두께 12 mm 기준 KR-3.2.1 "
        "허용 3.00 mm 이하. 판정: 합격."
    )
    r = check_numeric_lock(text, sk)
    assert not r.ok
    assert REASON_VERDICT_FLIP in r.reasons


def test_f4_requires_defect_code():
    sk = sk_const(
        defect_code="100", limit_type="불허", limit_rule="none_permitted",
        limit_value=None, margin=None, verdict="불합격", size_mm="0.50", unit=None,
    )
    good = "관찰: 길이 0.50 mm의 선형 지시. 분류: 균열(100). 균열은 전 수준 불허 (KR-3.2.1). 판정: 불합격."
    assert check_numeric_lock(good, sk).ok
    bad = "관찰: 길이 0.50 mm의 선형 지시. 균열은 전 수준 불허 (KR-3.2.1). 판정: 불합격."
    r = check_numeric_lock(bad, sk)
    assert not r.ok
    assert REASON_MISSING_VALUE in r.reasons


def test_ratio_requires_measured_percent():
    sk = sk_const(
        defect_code="2012", limit_type="비율", limit_rule="const", unit="percent",
        size_mm=None, measured_value="20.00", limit_value="25.00", margin="5.00",
    )
    good = (
        "관찰: 기공 면적 비율 20.00 %. 분류: 2012. 조회: 두께 12 mm 기준 KR-3.2.1 "
        "허용 25 % 이하. 판정: 합격."
    )
    assert check_numeric_lock(good, sk).ok
    bad = good.replace("20.00 %", "면적 비율 상당")
    r = check_numeric_lock(bad, sk)
    assert not r.ok
    assert REASON_MISSING_VALUE in r.reasons


# ---------------------------------------------------------------------------
# 정상 페어 변형 (§4-6-1 ④ — 결함 어휘 락은 사전 주입)
# ---------------------------------------------------------------------------

NORMAL_TEXT = "방사선 사진 전반에서 검출 한계 내 특기할 결함 지시 없음. 가정 두께 12 mm 품질수준 C 기준. 판정: 합격."


def test_normal_clean_passes():
    r = check_normal_lock(
        NORMAL_TEXT, defect_lexicon=DEFECT_LEXICON,
        thickness_mm=Decimal("12.00"), quality_level="C", expected_verdict="합격",
    )
    assert r.ok, r


def test_normal_defect_word_negative_context_still_discarded():
    # "기공은 관찰되지 않았다" — 부정 문맥도 문맥 불문 폐기 (§4-6-1 ③·④)
    text = NORMAL_TEXT.replace("특기할 결함 지시 없음.", "기공은 관찰되지 않았다.")
    r = check_normal_lock(
        text, defect_lexicon=DEFECT_LEXICON,
        thickness_mm=Decimal("12.00"), quality_level="C", expected_verdict="합격",
    )
    assert not r.ok
    assert REASON_DEFECT_HALLUCINATION in r.reasons


def test_normal_iso_code_token_discarded():
    text = NORMAL_TEXT.replace("특기할 결함 지시 없음.", "코드 2011 해당 없음.")
    r = check_normal_lock(
        text, defect_lexicon=DEFECT_LEXICON,
        thickness_mm=Decimal("12.00"), quality_level="C", expected_verdict="합격",
    )
    assert not r.ok
    assert REASON_DEFECT_HALLUCINATION in r.reasons


def test_find_defect_tokens_is_the_single_lexicon_lock():
    # 정본 검출기: 대소문자 불문 + ISO 코드 숫자 경계
    lex = ("Crack", "기공", "2011")
    assert find_defect_tokens("소문자 crack 성 지시", lex) == ("Crack",)
    assert find_defect_tokens("대문자 CRACK 성 지시", lex) == ("Crack",)
    assert find_defect_tokens("코드 2011 참조", lex) == ("2011",)
    assert find_defect_tokens("코드 12011 참조", lex) == ()  # 숫자 경계
    assert find_defect_tokens("특기할 지시 없음", lex) == ()


def test_normal_lexicon_lock_ignores_case():
    # 영문 결함 명칭의 대소문자 변형이 정상 corpus 로 새면 안 된다
    for term in ("crack", "CRACK", "Crack"):
        text = f"필름 전반에 {term} 성 지시 없음. 판정: 합격."
        r = check_normal_lock(
            text, defect_lexicon=("Crack", "균열"),
            thickness_mm=Decimal("12.00"), quality_level="C", expected_verdict="합격",
        )
        assert not r.ok, term
        assert REASON_DEFECT_HALLUCINATION in r.reasons


def test_normal_standard_reference_discarded():
    # 정상 페어는 인용할 허용치가 없다 — 조항과 마찬가지로 표준 인용도 금지다
    text = NORMAL_TEXT.replace("판정:", "ISO 5817 기준 특기 사항 없음. 판정:")
    r = check_normal_lock(
        text, defect_lexicon=DEFECT_LEXICON,
        thickness_mm=Decimal("12.00"), quality_level="C", expected_verdict="합격",
    )
    assert not r.ok
    assert REASON_EXTRA_VALUE in r.reasons


def test_normal_thickness_unit_tamper():
    text = NORMAL_TEXT.replace("두께 12 mm", "두께 12 %")
    r = check_normal_lock(
        text, defect_lexicon=DEFECT_LEXICON,
        thickness_mm=Decimal("12.00"), quality_level="C", expected_verdict="합격",
    )
    assert not r.ok
    assert REASON_CHANGED_VALUE in r.reasons


def test_normal_stray_number_discarded():
    text = NORMAL_TEXT + " 지시 크기는 3 mm 미만."
    r = check_normal_lock(
        text, defect_lexicon=DEFECT_LEXICON,
        thickness_mm=Decimal("12.00"), quality_level="C", expected_verdict="합격",
    )
    assert not r.ok
    assert REASON_EXTRA_VALUE in r.reasons


def test_normal_clause_citation_discarded():
    text = NORMAL_TEXT + " 근거: KR-3.2.1."
    r = check_normal_lock(
        text, defect_lexicon=DEFECT_LEXICON,
        thickness_mm=Decimal("12.00"), quality_level="C", expected_verdict="합격",
    )
    assert not r.ok
    assert REASON_EXTRA_VALUE in r.reasons


def test_normal_clause_only_gate():
    text = "방사선 사진 전반에서 검출 한계 내 특기할 결함 지시 없음. 가정 두께 12 mm 품질수준 C 기준."
    ok = check_normal_lock(
        text, defect_lexicon=DEFECT_LEXICON,
        thickness_mm=Decimal("12.00"), quality_level="C", expected_verdict=None,
    )
    assert ok.ok
    r = check_normal_lock(
        text + " 판정: 합격.", defect_lexicon=DEFECT_LEXICON,
        thickness_mm=Decimal("12.00"), quality_level="C", expected_verdict=None,
    )
    assert not r.ok
    assert REASON_VERDICT_FLIP in r.reasons


# ---------------------------------------------------------------------------
# 다결함 페어 잠금 (§4-6 이미지 수준)
# ---------------------------------------------------------------------------

PAIR_DEFECTS = (
    sk_const(),  # 2011 합격 (size 2.00, L 3.00, margin 1.00)
    sk_const(
        defect_code="401", limit_type="길이", limit_rule="prop_t", limit_value="1.20",
        limit_factor="0.1", ratio_basis="t", size_mm="6.00", margin="-4.80",
        verdict="불합격", clause_id="KR-3.2.2",
    ),
)

PAIR_TEXT = (
    "관찰: 직경 2.00 mm 기공(2011) 및 길이 6.00 mm 융합불량(401). "
    "조회: 두께 12 mm 기준 KR-3.2.1 허용 3.00 mm 이하 및 KR-3.2.2 허용 1.20 mm 이하. "
    "판정: 불합격."
)


def test_pair_lock_passes():
    r = check_pair_lock(PAIR_TEXT, PAIR_DEFECTS, image_verdict="불합격")
    assert r.ok, r


def test_pair_lock_missing_clause():
    text = PAIR_TEXT.replace(" 및 KR-3.2.2 허용 1.20 mm 이하", "")
    r = check_pair_lock(text, PAIR_DEFECTS, image_verdict="불합격")
    assert not r.ok
    assert REASON_MISSING_VALUE in r.reasons


def test_pair_lock_image_verdict_flip():
    text = PAIR_TEXT.replace("판정: 불합격.", "판정: 합격.")
    r = check_pair_lock(text, PAIR_DEFECTS, image_verdict="불합격")
    assert not r.ok
    assert REASON_VERDICT_FLIP in r.reasons


def test_pair_lock_rejects_empty_defects():
    with pytest.raises(ValueError):
        check_pair_lock(PAIR_TEXT, (), image_verdict="합격")
