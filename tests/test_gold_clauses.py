"""정답 조항 목록 테스트 — 계약 #3 세 소비처 중 두 번째. 스펙 §6·§10.

**검사축이 유일성 키의 첫 축**인지가 핵심이다. 축이 없으면 표면 기준 조항이 내부 검사
정답으로 등재되고, 채점기는 잘못된 정답과 대조하면서도 형식상 정상으로 보인다.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from evaluation.gold import (
    GoldLookupError,
    ImageContext,
    assert_unique,
    build_gold_pairs,
    entries_from_derived,
    lookup,
)


def derived(clause: str, method="RT", code="2011", tmin="8", tmax="25",
            ltype="diameter", material="ST", rule=None) -> dict:
    return {
        "defect_code": code, "material": material, "inspection_method": method,
        "thickness_min": tmin, "thickness_max": tmax,
        "quality_scheme": "iso5817", "quality_level": "C",
        "limit_type": ltype, "clause_id": clause, "rule_id": rule or f"R-{clause}",
    }


def ctx(image="img1", method="RT", t="12") -> ImageContext:
    return ImageContext(
        image_id=image, inspection_method=method, material="ST",
        thickness_mm=None if t is None else Decimal(t),
        quality_scheme="iso5817", quality_level="C",
    )


def look(entries, method="RT", **over):
    kwargs = {
        "defect_code": "2011", "material": "ST", "inspection_method": method,
        "quality_scheme": "iso5817", "quality_level": "C",
        "thickness_mm": Decimal(12),
    }
    kwargs.update(over)
    return lookup(entries, **kwargs)


# --- 검사축 (계약 #3 변경 회귀) ---------------------------------------------------

def test_rt_lookup_does_not_return_vt_clause():
    """표면 기준 조항이 내부 검사 정답으로 등재되는 것을 막는다."""
    e = entries_from_derived([derived("INTERNAL", method="RT"),
                              derived("SURFACE", method="VT")])
    assert look(e, method="RT").clause_id == "INTERNAL"
    assert look(e, method="VT").clause_id == "SURFACE"


def test_same_code_same_thickness_different_axis_is_not_ambiguous():
    """축이 없으면 이 둘이 같은 키로 보여 '복수 매칭'이 되거나 임의 선택된다."""
    e = entries_from_derived([derived("INTERNAL", method="RT"),
                              derived("SURFACE", method="VT")])
    assert_unique(e)                       # 축 덕분에 유일성 성립
    assert look(e, method="RT").clause_id == "INTERNAL"


def test_uniqueness_violation_without_axis_is_caught():
    """같은 축·같은 키에 조항이 둘이면 빌드 실패 + B 회부."""
    e = entries_from_derived([derived("A", method="RT"), derived("B", method="RT")])
    with pytest.raises(ValueError) as exc:
        assert_unique(e)
    assert "재파생" in str(exc.value)


def test_all_axis_entry_answers_both():
    e = entries_from_derived([derived("ANY", method="ALL")])
    assert look(e, method="RT").clause_id == "ANY"
    assert look(e, method="VT").clause_id == "ANY"


# --- 두께 반구간 -----------------------------------------------------------------

def test_thickness_lower_inclusive_upper_exclusive():
    e = entries_from_derived([derived("LOW", tmin="8", tmax="25"),
                              derived("HIGH", tmin="25", tmax="50")])
    assert look(e, thickness_mm=Decimal(8)).clause_id == "LOW"
    assert look(e, thickness_mm=Decimal(25)).clause_id == "HIGH"


def test_open_upper_bound_covers_large_thickness():
    e = entries_from_derived([derived("OPEN", tmin="25", tmax=None)])
    assert look(e, thickness_mm=Decimal(999)).clause_id == "OPEN"


# --- fallback 금지 ---------------------------------------------------------------

def test_missing_combination_raises_not_falls_back():
    """조용한 오답이 명시적 실패보다 위험하다."""
    e = entries_from_derived([derived("A", code="301")])
    with pytest.raises(GoldLookupError):
        look(e, defect_code="2011")


def test_multiple_limit_types_require_disambiguation():
    """같은 조합에 낱개 지름 + 누적 비율은 정당하게 공존한다."""
    e = entries_from_derived([derived("A", ltype="diameter"),
                              derived("B", ltype="ratio")])
    with pytest.raises(GoldLookupError) as exc:
        look(e)
    assert "limit_type" in str(exc.value)
    assert look(e, limit_type="ratio").clause_id == "B"


# --- 쌍 생성 ---------------------------------------------------------------------

def test_pairs_built_per_image_and_code():
    e = entries_from_derived([derived("C-2011", code="2011"),
                              derived("C-301", code="301")])
    pairs, skipped = build_gold_pairs(
        e, [ctx("img1")], {"img1": ["2011", "301"]}
    )
    assert {(p.image_id, p.iso_code, p.clause_id) for p in pairs} == {
        ("img1", "2011", "C-2011"), ("img1", "301", "C-301"),
    }
    assert skipped == {}


def test_pair_carries_rule_id_for_recomputation():
    e = entries_from_derived([derived("C-2011", rule="R-42")])
    pairs, _ = build_gold_pairs(e, [ctx()], {"img1": ["2011"]})
    assert pairs[0].row_id == "R-42"


def test_vt_image_gets_vt_clause_not_rt():
    e = entries_from_derived([derived("INTERNAL", method="RT"),
                              derived("SURFACE", method="VT")])
    pairs, _ = build_gold_pairs(e, [ctx(method="VT")], {"img1": ["2011"]})
    assert pairs[0].clause_id == "SURFACE"


def test_skipped_lookups_are_counted_not_silent():
    """한 조합이 비었다고 전체 채점을 멈추지 않되, 건너뛴 건수는 반드시 보고한다."""
    e = entries_from_derived([derived("C-2011", code="2011")])
    pairs, skipped = build_gold_pairs(e, [ctx()], {"img1": ["2011", "999"]})
    assert len(pairs) == 1
    assert skipped == {"정답 조항 없음": 1}


def test_duplicate_gt_codes_produce_one_pair():
    e = entries_from_derived([derived("C-2011")])
    pairs, _ = build_gold_pairs(e, [ctx()], {"img1": ["2011", "2011"]})
    assert len(pairs) == 1
