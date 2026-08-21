"""G0 로더 검증 테스트 — 스펙 §4-8 T01~T06 + 로더 소관 보충 (V5·V6·V8·V9·V10·pilot).

각 테스트의 스펙 번호는 # T0x 주석으로 표기한다.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path

import pytest

from corpus.rules.limits_loader import (
    LimitsValidationError,
    coverage_report,
    load_limits,
)
from corpus.rules.schema import InspectionMethod, Unit
from tests.corpus.conftest import csv_row

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# T01 — 필수 컬럼·타입·enum 위반 행 → 예외 + 행 번호 (오염 CSV 의 조용한 통과 차단)
# ---------------------------------------------------------------------------


def test_t01_missing_column_rejected(load):  # T01
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row()], drop_columns=("unit",))
    assert any(v.check == "V1" and "unit" in v.message for v in ei.value.violations)


def test_t01_bad_enum_rejected_with_row_number(load):  # T01
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(material="XX")])
    assert 2 in ei.value.row_numbers
    assert any("material" in v.message for v in ei.value.violations)


def test_t01_bad_type_rejected(load):  # T01
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(thickness_min="abc")])
    assert 2 in ei.value.row_numbers


def test_t01_missing_required_value_rejected(load):  # T01
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(rule_id="")])
    assert 2 in ei.value.row_numbers


def test_t01_second_row_number_reported(load):  # T01
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(), csv_row(rule_id="KR-3.2.1-002", scope="bogus")])
    assert ei.value.row_numbers == [3]


# ---------------------------------------------------------------------------
# T02 — thickness_min ≥ max 거부 (구간 뒤집힘)
# ---------------------------------------------------------------------------


def test_t02_interval_equal_rejected(load):  # T02
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(thickness_min="25", thickness_max="25")])
    assert any(v.check == "V2" for v in ei.value.violations)
    assert 2 in ei.value.row_numbers


def test_t02_interval_inverted_rejected(load):  # T02
    with pytest.raises(LimitsValidationError):
        load([csv_row(thickness_min="30", thickness_max="25")])


# ---------------------------------------------------------------------------
# T03 — 구간 겹침·틈 주입 → 거부 / 정상 → 통과 (limit_type 별 그룹)
# ---------------------------------------------------------------------------


def _interval_rows(pairs, **common):
    rows = []
    for i, (lo, hi) in enumerate(pairs):
        rows.append(csv_row(
            rule_id=f"KR-3.2.1-{i:03d}", thickness_min=lo, thickness_max=hi, **common
        ))
    return rows


def test_t03_contiguous_intervals_pass(load):  # T03
    table = load(_interval_rows([("0", "10"), ("10", "25"), ("25", "")]))
    assert len(table.rows) == 3


def test_t03_overlap_rejected(load):  # T03
    with pytest.raises(LimitsValidationError) as ei:
        load(_interval_rows([("0", "12"), ("10", "25")]))
    assert any(v.check == "V3" and "겹침" in v.message for v in ei.value.violations)


def test_t03_gap_rejected(load):  # T03
    with pytest.raises(LimitsValidationError) as ei:
        load(_interval_rows([("0", "10"), ("12", "25")]))
    assert any(v.check == "V3" and "틈" in v.message for v in ei.value.violations)


def test_t03_different_limit_type_not_a_conflict(load):  # T03 (V3 그룹 키에 limit_type — D1-8)
    rows = [
        csv_row(rule_id="KR-3.2.1-001", limit_type="직경", limit_value="3"),
        csv_row(rule_id="KR-3.2.1-002", limit_type="길이", limit_value="6"),
    ]
    table = load(rows)  # 같은 조합·같은 구간이라도 limit_type 이 다르면 정당한 복수 제약
    assert len(table.rows) == 2


def test_t03_infinite_interval_before_next_is_overlap(load):  # T03
    with pytest.raises(LimitsValidationError) as ei:
        load(_interval_rows([("0", ""), ("10", "25")]))
    assert any(v.check == "V3" for v in ei.value.violations)


# ---------------------------------------------------------------------------
# T04 — 단조성 위반 주입 → 경보 (동일 스킴, 명시 평가 지점. 예외 아님)
# ---------------------------------------------------------------------------


def test_t04_monotonicity_violation_warns_not_raises(load):  # T04
    rows = [
        csv_row(rule_id="KR-3.2.1-B", quality_level="B", limit_value="4"),
        csv_row(rule_id="KR-3.2.1-C", quality_level="C", limit_value="3"),
    ]
    table = load(rows)  # 예외 없이 로드
    assert table.v4_warnings, "B(4) > C(3) 이면 단조성 경보가 나야 한다"
    msg = table.v4_warnings[0]
    assert "t=" in msg  # 평가 지점 명시
    assert "KR-3.2.1-B" in msg and "KR-3.2.1-C" in msg


def test_t04_conforming_levels_no_warning(load):  # T04
    rows = [
        csv_row(rule_id="KR-3.2.1-B", quality_level="B", limit_value="2"),
        csv_row(rule_id="KR-3.2.1-C", quality_level="C", limit_value="3"),
    ]
    assert load(rows).v4_warnings == ()


def test_t04_prop_t_cap_crossing_point_evaluated(load):  # T04 (t* = cap/factor)
    rows = [
        csv_row(rule_id="KR-3.2.1-B", quality_level="B", thickness_min="2", thickness_max="20",
                limit_rule="prop_t_cap", limit_value="", limit_factor="0.5", limit_cap="4",
                ratio_basis="t"),
        csv_row(rule_id="KR-3.2.1-C", quality_level="C", thickness_min="2", thickness_max="20",
                limit_value="3"),
    ]
    table = load(rows)
    assert table.v4_warnings  # min(0.5t, 4) 가 t>6 에서 3 을 초과 — 경보


def test_t04_other_scheme_not_compared(load):  # T04 (동일 스킴 한정)
    rows = [
        csv_row(rule_id="KR-3.2.1-S", quality_scheme="iacs", quality_level="STD",
                limit_value="4"),
        csv_row(rule_id="KR-3.2.1-L", quality_scheme="iacs", quality_level="LIM",
                limit_value="3"),
    ]
    assert load(rows).v4_warnings == ()


# ---------------------------------------------------------------------------
# T05 — 단위-유형 교차 위반 → 예외, ㎜·전각 정규화
# ---------------------------------------------------------------------------


def test_t05_diameter_with_percent_rejected(load):  # T05
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(unit="percent")])
    assert any("unit" in v.message or "mm" in v.message for v in ei.value.violations)


def test_t05_ratio_with_mm_rejected(load):  # T05
    with pytest.raises(LimitsValidationError):
        load([csv_row(limit_type="비율", unit="mm", limit_value="25")])


def test_t05_legacy_mm_glyph_normalized(load):  # T05 (㎜ → mm)
    table = load([csv_row(unit="㎜")])
    assert table.rows[0].unit is Unit.MM


def test_t05_fullwidth_digits_and_percent_normalized(load):  # T05 (전각 → 반각)
    rows = [
        csv_row(rule_id="KR-3.2.1-001", limit_value="３"),  # 전각 3
        csv_row(rule_id="KR-3.2.1-002", limit_type="비율", unit="％", limit_value="25",
                thickness_min="25", thickness_max="50"),
    ]
    table = load(rows)
    assert table.rows[0].limit_value == Decimal("3.00")
    assert table.rows[1].unit is Unit.PERCENT


# ---------------------------------------------------------------------------
# T06 — none_permitted·prop 행에 limit_value 기재 → 예외 (파생값 중복 기재 차단)
# ---------------------------------------------------------------------------


def test_t06_none_permitted_with_value_rejected(load):  # T06
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(limit_type="불허", limit_rule="none_permitted", unit="", limit_value="3")])
    assert any(v.check == "V7" for v in ei.value.violations)


def test_t06_prop_with_value_rejected(load):  # T06
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(limit_rule="prop_t", limit_factor="0.2", ratio_basis="t", limit_value="5")])
    assert any(v.check == "V7" for v in ei.value.violations)


def test_t06_const_with_factor_rejected(load):  # T06
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(limit_factor="0.2")])
    assert any(v.check == "V7" for v in ei.value.violations)


def test_t06_valid_shapes_pass(load):  # T06 (규칙별 정상형)
    rows = [
        csv_row(rule_id="R-const", thickness_min="0", thickness_max="10"),
        csv_row(rule_id="R-prop", thickness_min="10", thickness_max="25",
                limit_rule="prop_t", limit_value="", limit_factor="0.2", ratio_basis="t"),
        csv_row(rule_id="R-cap", thickness_min="25", thickness_max="",
                limit_rule="prop_t_cap", limit_value="", limit_factor="0.2", limit_cap="5",
                ratio_basis="t"),
        csv_row(rule_id="R-np", defect_code="100", limit_type="불허",
                limit_rule="none_permitted", limit_value="", unit="",
                thickness_min="0", thickness_max=""),
    ]
    assert len(load(rows).rows) == 4


# ---------------------------------------------------------------------------
# 로더 소관 보충 — V5 · V6 · V8 · V9 · V10 · pilot · fail-closed
# ---------------------------------------------------------------------------


def test_v5_paid_source_rejected_even_in_pilot(load):
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(source_doc="PAID-DOC")], pilot=True)
    assert any(v.check == "V5" and "PAID" in v.message for v in ei.value.violations)


def test_v5_unregistered_source_rejected(load):
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(source_doc="NOPE-DOC")])
    assert any(v.check == "V1" for v in ei.value.violations)


def test_v6_duplicate_combo_rejected(load):
    rows = [csv_row(), csv_row(rule_id="KR-3.2.1-002")]  # rule_id 만 다르고 조합 동일
    with pytest.raises(LimitsValidationError) as ei:
        load(rows)
    assert any(v.check == "V6" for v in ei.value.violations)


def test_v6_double_canonical_rejected(load):
    rows = [
        csv_row(rule_id="A-1", clause_id="KR-3.2.1"),
        csv_row(rule_id="A-2", clause_id="KR-3.2.2"),  # clause 만 다른 같은 조합, 둘 다 canonical
    ]
    with pytest.raises(LimitsValidationError) as ei:
        load(rows)
    assert any(v.check == "V6" and "canonical" in v.message for v in ei.value.violations)


def test_v1_duplicate_rule_id_rejected(load):
    rows = [csv_row(), csv_row(thickness_min="25", thickness_max="50")]  # 같은 rule_id
    with pytest.raises(LimitsValidationError) as ei:
        load(rows)
    assert any("rule_id 중복" in v.message for v in ei.value.violations)


def test_v8_unregistered_clause_rejected(load):
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(clause_id="KR-NOPE")])
    assert any(v.check == "V8" for v in ei.value.violations)


def test_v1_defect_code_outside_label_map_rejected(load):
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(defect_code="9999")])
    assert any(v.check == "V1" and "사상표" in v.message for v in ei.value.violations)


def test_v9_unresolved_cross_doc_conflict_rejected(load):
    rows = [
        csv_row(rule_id="KR-A", canonical="false", quality_scheme="none", quality_level="ALL",
                limit_value="3"),
        csv_row(rule_id="IACS-A", canonical="false", quality_scheme="none", quality_level="ALL",
                limit_value="4", source_doc="IACS47", clause_id="IACS47-3.2.1"),
    ]
    with pytest.raises(LimitsValidationError) as ei:
        load(rows)
    assert any(v.check == "V9" and "미해소" in v.message for v in ei.value.violations)


def test_v9_resolved_by_canonical_is_recorded(load):
    rows = [
        csv_row(rule_id="KR-A", canonical="true", quality_scheme="none", quality_level="ALL",
                limit_value="3"),
        csv_row(rule_id="IACS-A", canonical="false", quality_scheme="none", quality_level="ALL",
                limit_value="4", source_doc="IACS47", clause_id="IACS47-3.2.1"),
    ]
    table = load(rows)
    assert len(table.v9_conflicts) == 1
    assert "KR-A" in table.v9_conflicts[0]


def test_v10_infeasible_row_flagged_not_rejected(load):
    rows = [csv_row(limit_rule="prop_t", limit_value="", limit_factor="0.01",
                    ratio_basis="t", thickness_min="1", thickness_max="10")]
    table = load(rows)
    assert table.v10_flags and "V10" in table.v10_flags[0]


def test_pilot_flag_and_warning(load, caplog):
    with caplog.at_level(logging.WARNING, logger="corpus.rules.limits_loader"):
        table = load([csv_row()])
    assert table.pilot is True
    assert "V0" in table.checks_skipped
    assert any("pilot" in rec.message for rec in caplog.records)


def test_non_pilot_fail_closed_without_references(write_limits):
    path = write_limits([csv_row()])
    with pytest.raises(LimitsValidationError) as ei:
        load_limits(path, pilot=False)  # sources·조항 목록·V0 산출물 전부 부재
    checks = {v.check for v in ei.value.violations}
    assert {"V5", "V8", "V0"} <= checks


def test_v11_sha256_recorded(load):
    table = load([csv_row()])
    assert len(table.sha256) == 64


def test_coverage_report_gaps(load):
    table = load([csv_row()])
    report = coverage_report(table)
    assert report["n_active_canonical"] == 1
    assert f"ST|2011|iso5817:C" in report["cells"]
    assert report["cells"]["ST|2011|iso5817:C"] == "present"
    # AL 쪽은 행이 없으므로 전부 공백으로 보고돼야 한다
    assert any(g["material"] == "AL" and g["reason"] == "missing" for g in report["gaps"])


def test_real_pilot_csv_loads():
    """저장소의 limits_v0_pilot.csv 가 pilot 모드 G0 를 통과하는지 (스모크)."""
    csv_path = REPO_ROOT / "corpus" / "rules" / "limits_v0_pilot.csv"
    sources = REPO_ROOT / "corpus" / "parse" / "sources.yaml"
    table = load_limits(csv_path, sources_yaml=sources, clause_registry=None, pilot=True)
    assert len(table.rows) == 12
    # 게이트 #13 이후 KRA27-S-01(표면 기공)은 VT 로 분리돼 active 로 승격됐다 — 배제 0건
    assert len(table.active_canonical) == 12
    assert table.pilot is True


def test_real_pilot_csv_rt_vt_axis():
    """파일럿 CSV 의 RT 내부 기준과 VT 표면 기준이 같은 결함코드로 병존한다 (게이트 #13).

    두 행을 가르는 것은 inspection_method 뿐이다 — 재질·결함코드·수준·limit_type 이 같고
    두께 구간도 겹친다. 주 실험(RT) 행 선택에서 표면 행이 딸려오면 안 된다.
    """
    csv_path = REPO_ROOT / "corpus" / "rules" / "limits_v0_pilot.csv"
    sources = REPO_ROOT / "corpus" / "parse" / "sources.yaml"
    table = load_limits(csv_path, sources_yaml=sources, clause_registry=None, pilot=True)

    porosity = [r for r in table.rows if r.defect_code == "2011"]
    methods = {r.inspection_method for r in porosity}
    assert methods == {InspectionMethod.RT, InspectionMethod.VT}
    surface = next(r for r in porosity if r.inspection_method is InspectionMethod.VT)
    internal = [r for r in porosity if r.inspection_method is InspectionMethod.RT]
    assert surface.clause_id == "KRA27-S"
    # 표면 행 [0, +inf) 는 내부 행 전 구간과 겹친다 — 축이 없으면 V3 겹침으로 로드 불가였다
    assert surface.thickness_max is None
    assert all(r.material is surface.material and r.limit_type is surface.limit_type
               for r in internal)

    rt_view = table.for_inspection(InspectionMethod.RT)
    assert len(rt_view.rows) == 11
    assert all(r.inspection_method is not InspectionMethod.VT for r in rt_view.rows)
    vt_view = table.for_inspection("VT")
    assert [r.rule_id for r in vt_view.rows] == ["KRA27-S-01"]
    # sha256 은 원천 파일 해시 그대로 — 뷰가 provenance 를 갈아치우지 않는다
    assert rt_view.sha256 == table.sha256


def test_for_inspection_rejects_all_as_query():
    """ALL 은 '검사 방법과 무관한 조항' 표시이지 '전 행' 이 아니다 (혼입 재발 차단)."""
    csv_path = REPO_ROOT / "corpus" / "rules" / "limits_v0_pilot.csv"
    sources = REPO_ROOT / "corpus" / "parse" / "sources.yaml"
    table = load_limits(csv_path, sources_yaml=sources, clause_registry=None, pilot=True)
    with pytest.raises(ValueError, match="ALL 금지"):
        table.for_inspection("ALL")


# ---------------------------------------------------------------------------
# 검사 방법 축 (게이트 #13, 스펙 v1.3 §1-2 5a) — V1·V3·V6 전파
# ---------------------------------------------------------------------------


def test_inspection_method_column_required(load):
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row()], drop_columns=("inspection_method",))
    assert any(v.check == "V1" and "inspection_method" in v.message
               for v in ei.value.violations)


def test_inspection_method_bad_enum_rejected(load):
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(inspection_method="UT")])
    assert 2 in ei.value.row_numbers
    assert any("inspection_method" in v.message for v in ei.value.violations)


def test_inspection_method_blank_rejected(load):
    """공란 허용 컬럼이 아니다 — 미상을 RT 로 추정해 채우면 축을 만든 이유가 사라진다."""
    with pytest.raises(LimitsValidationError) as ei:
        load([csv_row(inspection_method="")])
    assert 2 in ei.value.row_numbers


def test_v3_rt_vt_same_defect_and_interval_is_not_overlap(load):
    """RT 기공과 VT 기공이 같은 defect_code 2011 · 같은 두께 구간으로 병존한다."""
    rt = csv_row(rule_id="KR-RT-001", inspection_method="RT",
                 clause_id="KR-3.2.1", limit_value="3")
    vt = csv_row(rule_id="KR-VT-001", inspection_method="VT",
                 clause_id="KR-3.2.2", limit_value="4")
    table = load([rt, vt])
    assert len(table.rows) == 2
    assert {r.inspection_method for r in table.rows} == {
        InspectionMethod.RT, InspectionMethod.VT
    }


def test_v3_same_method_same_interval_still_overlaps(load):
    """축이 같으면 겹침 판정은 그대로다 — 축 추가가 V3 를 무력화하지 않았다."""
    a = csv_row(rule_id="KR-RT-001", inspection_method="RT", clause_id="KR-3.2.1")
    b = csv_row(rule_id="KR-RT-002", inspection_method="RT", clause_id="KR-3.2.2",
                limit_value="4")
    with pytest.raises(LimitsValidationError) as ei:
        load([a, b])
    assert any(v.check in ("V3", "V6") for v in ei.value.violations)


def test_v6_duplicate_ignoring_method_rejected(load):
    """inspection_method 까지 같으면 여전히 중복이다 (V6 유일성 키 확장의 역방향 확인)."""
    rows = [
        csv_row(rule_id="KR-RT-001", inspection_method="VT"),
        csv_row(rule_id="KR-RT-002", inspection_method="VT"),
    ]
    with pytest.raises(LimitsValidationError) as ei:
        load(rows)
    assert any(v.check == "V6" for v in ei.value.violations)


def test_v9_rt_vt_cross_doc_is_not_a_conflict(load):
    """다른 문서의 RT 행과 VT 행은 한계가 달라도 충돌이 아니다 — 정본 지정을 요구하면
    표면·내부 중 한쪽을 눕히게 된다."""
    rows = [
        csv_row(rule_id="KR-A", inspection_method="RT", canonical="false",
                quality_scheme="none", quality_level="ALL", limit_value="3"),
        csv_row(rule_id="IACS-A", inspection_method="VT", canonical="false",
                quality_scheme="none", quality_level="ALL", limit_value="4",
                source_doc="IACS47", clause_id="IACS47-3.2.1"),
    ]
    table = load(rows)
    assert table.v9_conflicts == ()


def test_coverage_report_inspection_axis(load):
    """RT 격자의 공백을 VT 행이 가리지 않는다."""
    rows = [
        csv_row(rule_id="KR-VT-001", inspection_method="VT", quality_scheme="none",
                quality_level="ALL", clause_id="KR-3.2.2"),
    ]
    table = load(rows)
    mixed = coverage_report(table)
    assert mixed["inspection_method"] is None
    assert mixed["inspection_method_counts"] == {"VT": 1}
    assert mixed["cells"]["ST|2011|none:ALL"] == "present"
    rt_only = coverage_report(table, inspection_method="RT")
    assert rt_only["inspection_method"] == "RT"
    assert rt_only["n_active_canonical"] == 0
    assert rt_only["cells"]["ST|2011|none:ALL"] == "missing"
