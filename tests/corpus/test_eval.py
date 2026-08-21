"""limit_eval 테스트 — 스펙 §4-8 T07~T14 + T33.

각 테스트의 스펙 번호는 # T0x 주석으로 표기한다.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from corpus.rules.limit_eval import (
    aggregate_verdicts,
    applicable_row,
    effective_limit,
    judge,
    quantize,
)
from corpus.rules.schema import FAIL, PASS
from tests.corpus.conftest import make_row, make_table

D = Decimal


# ---------------------------------------------------------------------------
# T07 — t = min → 해당 행, t = max → 다음 행 (반열림 [min, max))
# ---------------------------------------------------------------------------


def _two_interval_table():
    r1 = make_row(rule_id="R1", thickness_min=D("8"), thickness_max=D("25"),
                  limit_value=D("3"))
    r2 = make_row(rule_id="R2", thickness_min=D("25"), thickness_max=D("50"),
                  limit_value=D("5"), clause_id="KR-3.2.2")
    return make_table(r1, r2)


def test_t07_boundary_thickness_selects_half_open():  # T07
    table = _two_interval_table()
    assert applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("8")).rule_id == "R1"
    assert applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("24.99")).rule_id == "R1"
    # t = max → 다음 구간의 행
    assert applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("25")).rule_id == "R2"


def test_t07_t_quantized_before_lookup():  # T07 (S2-3: 저장 t 재조회 규약)
    table = _two_interval_table()
    # 24.996 은 0.01 그리드 양자화로 25.00 — 다음 구간 행이 잡혀야 결정론이 성립
    assert applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("24.996")).rule_id == "R2"


def test_t07_material_all_row_matches_any_material():  # T07 보충 (material=ALL)
    table = make_table(make_row(rule_id="R-ALL", material="ALL"))
    assert applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("10")).rule_id == "R-ALL"
    assert applicable_row(table, "2011", "AL", "RT", "iso5817", "C", D("10")).rule_id == "R-ALL"


# ---------------------------------------------------------------------------
# T08 — 없는 조합 질의 → 예외, fallback 없음
# ---------------------------------------------------------------------------


def test_t08_missing_combo_raises():  # T08
    table = _two_interval_table()
    with pytest.raises(LookupError):
        applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("50"))  # 구간 밖
    with pytest.raises(LookupError):
        applicable_row(table, "100", "ST", "RT", "iso5817", "C", D("10"))  # 없는 결함
    with pytest.raises(LookupError):
        applicable_row(table, "2011", "AL", "RT", "iso5817", "C", D("10"))  # 없는 재질
    with pytest.raises(LookupError):
        applicable_row(table, "2011", "ST", "RT", "iso5817", "B", D("10"))  # 없는 수준


def test_t08_excluded_and_noncanonical_not_matched():  # T08 (active·canonical 만)
    table = make_table(
        make_row(rule_id="R-ex", scope="excluded"),
        make_row(rule_id="R-nc", canonical=False, clause_id="KR-3.2.2"),
    )
    with pytest.raises(LookupError):
        applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("10"))


def test_t08_multiple_limit_types_require_explicit_type():  # T08 (임의 선택 금지)
    table = make_table(
        make_row(rule_id="R-dia", limit_type="직경", limit_value=D("3")),
        make_row(rule_id="R-len", limit_type="길이", limit_value=D("6"),
                 clause_id="KR-3.2.2"),
    )
    with pytest.raises(LookupError):
        applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("10"))
    got = applicable_row(table, "2011", "ST", "RT", "iso5817", "C", D("10"), limit_type="직경")
    assert got.rule_id == "R-dia"


# ---------------------------------------------------------------------------
# T09 — prop_t_cap: factor·t < cap / > cap 각각 min() 선택
# ---------------------------------------------------------------------------


def _cap_row():
    return make_row(rule_id="R-cap", limit_rule="prop_t_cap", limit_value=None,
                    limit_factor=D("0.2"), limit_cap=D("5"), ratio_basis="t",
                    thickness_min=D("0"), thickness_max=None)


def test_t09_cap_selection():  # T09
    row = _cap_row()
    assert effective_limit(row, basis_value=D("20")) == D("4.00")  # 0.2·20 < 5
    assert effective_limit(row, basis_value=D("30")) == D("5.00")  # 0.2·30 > 5 → cap
    assert effective_limit(row, basis_value=D("25")) == D("5.00")  # 경계: 정확히 cap


def test_t09_judge_uses_capped_limit():  # T09
    row = _cap_row()
    assert judge(row, D("4.5"), basis_value=D("30")).verdict == PASS
    j = judge(row, D("4.5"), basis_value=D("20"))
    assert j.verdict == FAIL and j.margin == D("-0.50")


def test_t09_prop_without_basis_raises():  # T09 보충 (fail-closed)
    with pytest.raises(ValueError):
        effective_limit(_cap_row(), basis_value=None)


# ---------------------------------------------------------------------------
# T10 — measured == L → 합격(le)·불합격(lt), L±0.01 대칭 검증 (핵심)
# ---------------------------------------------------------------------------


def test_t10_equality_le():  # T10
    row = make_row(limit_op="le", limit_value=D("3"))
    assert judge(row, D("3")).verdict == PASS  # 경계값 합격
    assert judge(row, D("2.99")).verdict == PASS
    assert judge(row, D("3.01")).verdict == FAIL


def test_t10_equality_lt():  # T10
    row = make_row(limit_op="lt", limit_value=D("3"))
    assert judge(row, D("3")).verdict == FAIL  # "미만" — 경계값 불합격
    assert judge(row, D("2.99")).verdict == PASS
    assert judge(row, D("3.01")).verdict == FAIL


def test_t10_margin_values():  # T10
    row = make_row(limit_op="le", limit_value=D("3"))
    assert judge(row, D("3")).margin == D("0.00")
    assert judge(row, D("2.99")).margin == D("0.01")
    assert judge(row, D("3.01")).margin == D("-0.01")


# ---------------------------------------------------------------------------
# T11 — 0.014999 vs 0.015 류 양자화 결정론 (부동소수 경계 오판 차단)
# ---------------------------------------------------------------------------


def test_t11_quantize_determinism():  # T11
    assert quantize(D("0.014999")) == D("0.01")
    assert quantize(D("0.015")) == D("0.02")  # ROUND_HALF_UP
    assert quantize(D("2.675")) == D("2.68")  # float 였다면 2.67 로 새는 값
    assert quantize(D("0.005")) == D("0.01")
    assert quantize(quantize(D("1.239"))) == quantize(D("1.239"))  # 멱등


def test_t11_float_rejected():  # T11 (float 직접 비교 금지)
    with pytest.raises(TypeError):
        quantize(0.015)


def test_t11_judge_boundary_after_quantization():  # T11
    row = make_row(limit_op="le", limit_value=D("3"))
    assert judge(row, D("3.004999")).verdict == PASS  # → 3.00
    assert judge(row, D("3.005")).verdict == FAIL  # → 3.01


# ---------------------------------------------------------------------------
# T12 — 불허: 양수 크기 → 불합격·margin=null / 크기 0·음수 → 예외
# ---------------------------------------------------------------------------


def _np_row():
    return make_row(rule_id="R-np", defect_code="100", limit_type="불허",
                    limit_rule="none_permitted", limit_value=None, unit=None,
                    thickness_min=D("0"), thickness_max=None)


def test_t12_none_permitted_positive_size_fails_with_null_margin():  # T12
    j = judge(_np_row(), D("1.5"))
    assert j.verdict == FAIL and j.margin is None


def test_t12_none_permitted_nonpositive_size_raises():  # T12 ((c) 예외 / D4 는 호출자 quarantine)
    with pytest.raises(ValueError):
        judge(_np_row(), D("0"))
    with pytest.raises(ValueError):
        judge(_np_row(), D("-1"))
    with pytest.raises(ValueError):
        judge(_np_row(), None)


def test_t12_none_permitted_effective_limit_is_none():  # T12
    assert effective_limit(_np_row()) is None


# ---------------------------------------------------------------------------
# T13 — 비율형: 단위 혼입 차단, 정상 판정·margin
# ---------------------------------------------------------------------------


def _ratio_row(op="le"):
    return make_row(rule_id="R-pct", limit_type="비율", unit="percent",
                    limit_rule="const", limit_value=D("25"), limit_op=op)


def test_t13_ratio_judge_and_margin():  # T13
    row = _ratio_row()
    j = judge(row, D("20"))
    assert j.verdict == PASS and j.margin == D("5.00")
    assert judge(row, D("25")).verdict == PASS
    assert judge(row, D("25.01")).verdict == FAIL


def test_t13_ratio_with_mm_unit_rejected_at_schema():  # T13 (단위 혼입 스키마 차단)
    with pytest.raises(ValidationError):
        make_row(limit_type="비율", unit="mm", limit_value=D("25"))
    with pytest.raises(ValidationError):
        make_row(limit_type="직경", unit="percent", limit_value=D("3"))
    # %가 size_mm 필드로 들어가는 골격 수준 차단(§4-4 매트릭스)은 skeleton_gen 단계 소관


# ---------------------------------------------------------------------------
# T14 — 전 표본 margin 불변식 (le: margin ≥ 0 ⇔ 합격 / lt: margin > 0 ⇔ 합격)
# ---------------------------------------------------------------------------


def test_t14_margin_invariant_sweep():  # T14
    for op in ("le", "lt"):
        row = make_row(limit_op=op, limit_value=D("3"))
        for i in range(0, 601):  # 0.00 ~ 6.00, 0.01 그리드 전수
            m = D(i) / D(100)
            j = judge(row, m)
            if op == "le":
                assert (j.margin >= 0) == (j.verdict == PASS), f"le 불변식 붕괴 at {m}"
            else:
                assert (j.margin > 0) == (j.verdict == PASS), f"lt 불변식 붕괴 at {m}"


def test_t14_margin_invariant_prop_rows():  # T14 (prop 계열에도 동일)
    row = _cap_row()
    for t in (D("10"), D("25"), D("40")):
        for i in range(0, 80):
            m = D(i) / D(10)
            j = judge(row, m, basis_value=t)
            assert (j.margin >= 0) == (j.verdict == PASS)


# ---------------------------------------------------------------------------
# T33 — aggregate_verdicts: 다결함 보수 규칙 (게이트 #6 결정 F)
# ---------------------------------------------------------------------------


def test_t33_all_pass_is_pass():  # T33
    assert aggregate_verdicts([PASS, PASS, PASS]) == PASS


def test_t33_any_fail_is_fail():  # T33
    assert aggregate_verdicts([PASS, FAIL, PASS]) == FAIL
    assert aggregate_verdicts([FAIL]) == FAIL


def test_t33_empty_is_pass():  # T33 (정상 이미지: 결함 0건 ⇒ 합격)
    assert aggregate_verdicts([]) == PASS


def test_t33_enum_enforced():  # T33 (verdict enum 2종 외 금지)
    with pytest.raises(ValueError):
        aggregate_verdicts([PASS, "보류"])
