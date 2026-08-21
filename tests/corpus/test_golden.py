"""T27 골든 케이스 러너 + T28 뮤테이션 감도 (스펙 §1-4 공유 평가기 3중 교차 검증 ①·②).

golden_cases.csv 는 사람이 스펙 §1-3·§4-4 규약 텍스트만으로 손 계산한 표다 —
정답 소스가 코드가 아니라 사람이라는 것이 이 표의 존재 이유다 (공통 모드 실패 방어).
"""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

import pytest

from corpus.rules.limit_eval import applicable_row, judge
from corpus.validate.stage1_rules import R_RECOMPUTE, run_stage1
from tests.corpus.conftest import make_row, make_table

GOLDEN_PATH = (
    Path(__file__).resolve().parents[2] / "corpus" / "validate" / "golden_cases.csv"
)


def _load_cases() -> list[dict[str, str]]:
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


CASES = _load_cases()


def _dec(s: str) -> Decimal | None:
    return None if s == "" else Decimal(s)


def _row_from_case(c: dict[str, str]):
    return make_row(
        rule_id=c["case_id"],
        defect_code=c["defect_code"],
        material=c["material"],
        thickness_min=Decimal(c["thickness_min"]),
        thickness_max=_dec(c["thickness_max"]),
        quality_scheme=c["quality_scheme"],
        quality_level=c["quality_level"],
        limit_type=c["limit_type"],
        limit_rule=c["limit_rule"],
        limit_value=_dec(c["limit_value"]),
        limit_factor=_dec(c["limit_factor"]),
        limit_cap=_dec(c["limit_cap"]),
        ratio_basis=c["ratio_basis"] or None,
        limit_op=c["limit_op"],
        unit=c["unit"] or None,
    )


# ---------------------------------------------------------------------------
# T27 — 골든 표 전건 일치 (judge 는 사람 손 계산과 한 건도 어긋나면 안 된다)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_t27_golden_case(case: dict[str, str]):
    row = _row_from_case(case)
    table = make_table(row)
    t = Decimal(case["t_mm"])

    if case["expected_verdict"] == "no_match":
        # 반열림 [min, max) 규약: t = max 는 이 행에서 선택 불가 (fallback 금지)
        with pytest.raises(LookupError):
            applicable_row(
                table, case["defect_code"], case["material"], "RT",
                case["quality_scheme"], case["quality_level"], t,
                limit_type=case["limit_type"],
            )
        return

    selected = applicable_row(
        table, case["defect_code"], case["material"], "RT",
        case["quality_scheme"], case["quality_level"], t,
        limit_type=case["limit_type"],
    )
    j = judge(selected, Decimal(case["measured"]), basis_value=t)
    assert j.verdict == case["expected_verdict"], case["rationale"]
    expected_margin = _dec(case["expected_margin"])
    assert j.margin == expected_margin, (
        f"{case['case_id']}: margin={j.margin} ↔ 손 계산 {expected_margin} — {case['rationale']}"
    )


def test_golden_table_composition():
    """골든 표 구성 요건: 30건+ · 등호/경계 ±0.01/prop_t/prop_t_cap 교차점/비율/불허/
    구간 경계 두께/lt 행이 전부 포함돼야 한다 (스펙 §1-4 ①)."""
    assert len(CASES) >= 30
    rules = {c["limit_rule"] for c in CASES}
    assert {"const", "prop_t", "prop_t_cap", "none_permitted"} <= rules
    assert any(c["limit_op"] == "lt" for c in CASES)
    assert any(c["limit_type"] == "비율" for c in CASES)
    assert any(c["expected_verdict"] == "no_match" for c in CASES)  # 구간 경계 t=max
    # 등호 정확 일치 (measured == L → margin 0.00) 가 le·lt 양쪽에 존재
    zero_margin = [c for c in CASES if c["expected_margin"] == "0.00"]
    assert any(c["limit_op"] == "le" and c["expected_verdict"] == "합격" for c in zero_margin)
    assert any(c["limit_op"] == "lt" and c["expected_verdict"] == "불합격" for c in zero_margin)
    # 경계 ±0.01
    assert any(c["expected_margin"] == "-0.01" for c in CASES)
    assert any(c["expected_margin"] == "0.01" for c in CASES)
    # prop_t_cap 교차점 (t* = cap/factor) 케이스
    assert any(
        c["limit_rule"] == "prop_t_cap"
        and c["limit_cap"] and c["limit_factor"]
        and Decimal(c["t_mm"]) == Decimal(c["limit_cap"]) / Decimal(c["limit_factor"])
        for c in CASES
    )
    # 구간 경계: t = thickness_min 포함 케이스
    assert any(
        c["expected_verdict"] != "no_match" and Decimal(c["t_mm"]) == Decimal(c["thickness_min"])
        for c in CASES
    )
    # case_id 유일성
    ids = [c["case_id"] for c in CASES]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# T28 — 뮤테이션 감도: verdict 반전 오염 레코드는 stage1 이 전건 검출해야 한다
# ---------------------------------------------------------------------------

LABEL_CODES = {"2011", "2012", "100", "401", "301"}
REGISTRY = ["KR-3.2.1", "KR-3.2.2", "KR-3.2.3", "KR-3.2.4"]


def _mutation_table():
    r1 = make_row(rule_id="R1")  # 2011 직경 const 3, KR-3.2.1
    r2 = make_row(
        rule_id="R2", defect_code="401", limit_type="길이", limit_rule="prop_t",
        limit_value=None, limit_factor=Decimal("0.1"), ratio_basis="t",
        clause_id="KR-3.2.2",
    )
    r3 = make_row(
        rule_id="R3", defect_code="100", limit_type="불허", limit_rule="none_permitted",
        limit_value=None, unit=None, clause_id="KR-3.2.3",
    )
    r4 = make_row(
        rule_id="R4", defect_code="2012", limit_type="비율", limit_rule="const",
        limit_value=Decimal("25"), unit="percent", clause_id="KR-3.2.4",
    )
    return make_table(r1, r2, r3, r4)


def _base(**over) -> dict:
    rec = {
        "sample_id": "c-000", "source": "sampled",
        "defect_code": "2011", "material": "ST", "inspection_method": "RT",
        "quality_scheme": "iso5817",
        "quality_level": "C", "limit_type": "직경", "limit_rule": "const",
        "thickness_mm": "12.00", "size_mm": "2.00", "measured_value": None,
        "clause_id": "KR-3.2.1", "limit_value": "3.00", "margin": "1.00",
        "verdict": "합격",
    }
    rec.update(over)
    return rec


def _valid_records() -> list[dict]:
    return [
        _base(
            sample_id="c-001",
            text=(
                "관찰: 직경 2.00 mm의 원형 음영. 분류: 기공(2011). 조회: 모재 두께 12 mm "
                "품질수준 C 조항 KR-3.2.1 허용 3.00 mm 이하. 대조: 2.00 ≤ 3.00. 판정: 합격."
            ),
        ),
        _base(
            sample_id="c-002", defect_code="401", limit_type="길이", limit_rule="prop_t",
            size_mm="1.00", limit_value="1.20", margin="0.20", clause_id="KR-3.2.2",
            text=(
                "관찰: 길이 1.00 mm의 선형 지시. 분류: 융합불량(401). 조회: 두께 12 mm "
                "기준 KR-3.2.2 허용 1.20 mm 이하. 대조: 1.00 ≤ 1.20. 판정: 합격."
            ),
        ),
        _base(
            sample_id="c-003", defect_code="100", limit_type="불허",
            limit_rule="none_permitted", size_mm="0.50", limit_value=None, margin=None,
            verdict="불합격", clause_id="KR-3.2.3",
            text=(
                "관찰: 길이 0.50 mm의 선형 지시. 분류: 균열(100). 조회: KR-3.2.3 — "
                "균열은 전 수준 불허. 판정: 불합격."
            ),
        ),
        _base(
            sample_id="c-004", defect_code="2012", limit_type="비율",
            size_mm=None, measured_value="20.00", limit_value="25.00", margin="5.00",
            clause_id="KR-3.2.4",
            text=(
                "관찰: 기공 면적 비율 20.00 %. 분류: 균일 분포 기공(2012). 조회: 두께 "
                "12 mm 기준 KR-3.2.4 허용 25 % 이하. 대조: 20.00 ≤ 25. 판정: 합격."
            ),
        ),
    ]


def _flip(v: str) -> str:
    return "불합격" if v == "합격" else "합격"


def test_t28_baseline_all_pass():
    report = run_stage1(
        _valid_records(), asset="c", table=_mutation_table(),
        label_codes=LABEL_CODES, clause_registry=REGISTRY,
    )
    assert report.n_fail == 0, report.results


def test_t28_verdict_field_flip_all_detected():
    """골격 verdict 필드 반전 → 부등식 재계산이 전건 잡아야 한다 (지표 회로 작동 증명)."""
    mutated = [{**rec, "verdict": _flip(rec["verdict"])} for rec in _valid_records()]
    report = run_stage1(
        mutated, asset="c", table=_mutation_table(),
        label_codes=LABEL_CODES, clause_registry=REGISTRY,
    )
    assert report.n_pass == 0, report.results  # 전건 검출
    for r in report.results:
        assert any(x == R_RECOMPUTE for x in r.reasons), r


def test_t28_text_verdict_flip_all_detected():
    """생성문 판정어만 반전 → 수치 잠금 재실행(판정어 대조)이 전건 잡아야 한다."""
    mutated = []
    for rec in _valid_records():
        old = f"판정: {rec['verdict']}."
        new = f"판정: {_flip(rec['verdict'])}."
        mutated.append({**rec, "text": rec["text"].replace(old, new)})
    report = run_stage1(
        mutated, asset="c", table=_mutation_table(),
        label_codes=LABEL_CODES, clause_registry=REGISTRY,
    )
    assert report.n_pass == 0, report.results
    for r in report.results:
        assert any(x.startswith("numeric_lock:") for x in r.reasons), r


def test_t28_margin_corruption_detected():
    """margin 오염 (부호 반전) 도 재계산이 잡는다."""
    rec = _valid_records()[0]
    mutated = {**rec, "margin": "-1.00"}
    report = run_stage1(
        [mutated], asset="c", table=_mutation_table(),
        label_codes=LABEL_CODES, clause_registry=REGISTRY,
    )
    assert report.n_pass == 0
    assert any(R_RECOMPUTE in r.reasons for r in report.results)
