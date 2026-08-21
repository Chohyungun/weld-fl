"""1단계 규칙 검사 (§6-1) + validation_report 작성기 (§6-4·§7-6) 테스트."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from corpus.validate.report import (
    build_pilot_report,
    build_validation_report,
    stage_summary,
    write_pilot_report,
    write_validation_report,
)
from corpus.validate.stage1_rules import (
    R_CLAUSE,
    R_CLAUSE_MISMATCH,
    R_COMBO,
    R_DUP,
    R_EVIDENCE,
    R_ISO,
    R_LONG,
    R_NORMAL,
    R_PAIR,
    R_RECOMPUTE,
    R_SCHEMA,
    check_record_c,
    check_record_pair,
    check_record_qa,
    normalize_question,
    run_stage1,
)
from tests.corpus.conftest import make_row, make_table

LABEL_CODES = {"2011", "2012", "100", "401", "301"}
REGISTRY = ["KR-3.2.1", "KR-3.2.2", "KR-3.2.3", "KR-3.2.4"]
LEXICON = ("균열", "기공", "융합불량", "슬래그혼입", "2011", "2012", "100", "401", "301")


@pytest.fixture
def table():
    return make_table(
        make_row(rule_id="R1"),  # 2011 직경 const 3 le, [8,25), C, KR-3.2.1
        make_row(
            rule_id="R2", defect_code="401", limit_type="길이", limit_rule="prop_t",
            limit_value=None, limit_factor=Decimal("0.1"), ratio_basis="t",
            clause_id="KR-3.2.2",
        ),
        make_row(
            rule_id="R3", defect_code="100", limit_type="불허",
            limit_rule="none_permitted", limit_value=None, unit=None,
            clause_id="KR-3.2.3",
        ),
        make_row(
            rule_id="R4", defect_code="2012", limit_type="비율", limit_rule="const",
            limit_value=Decimal("25"), unit="percent", clause_id="KR-3.2.4",
        ),
    )


def c_record(**over) -> dict:
    rec = {
        "sample_id": "c-000",
        "defect_code": "2011", "material": "ST", "inspection_method": "RT",
        "quality_scheme": "iso5817",
        "quality_level": "C", "limit_type": "직경", "limit_rule": "const",
        "thickness_mm": "12.00", "size_mm": "2.00", "measured_value": None,
        "clause_id": "KR-3.2.1", "limit_value": "3.00", "margin": "1.00",
        "verdict": "합격",
        "text": (
            "관찰: 직경 2.00 mm의 원형 음영. 분류: 기공(2011). 조회: 모재 두께 12 mm "
            "품질수준 C 조항 KR-3.2.1 허용 3.00 mm 이하. 대조: 2.00 ≤ 3.00. 판정: 합격."
        ),
    }
    rec.update(over)
    return rec


def _check(rec, table, **kw):
    return check_record_c(rec, table, label_codes=LABEL_CODES, clause_registry=REGISTRY, **kw)


# ---------------------------------------------------------------------------
# (c) 경로 — 골격 + 생성문
# ---------------------------------------------------------------------------


def test_c_valid_passes(table):
    r = _check(c_record(), table)
    assert r.ok, r


def test_c_unknown_iso_code(table):
    r = _check(c_record(defect_code="999"), table)
    assert not r.ok
    assert R_ISO in r.reasons


def test_c_clause_not_registered(table):
    rec = c_record(clause_id="KR-8.8.8")
    rec["text"] = rec["text"].replace("KR-3.2.1", "KR-8.8.8")
    r = _check(rec, table)
    assert not r.ok
    assert R_CLAUSE in r.reasons
    assert R_CLAUSE_MISMATCH in r.reasons  # 실제 행의 clause 와도 불일치


def test_c_combo_not_found(table):
    # t=30 은 [8,25) 밖 — fallback 없이 거부돼야 한다
    rec = c_record(thickness_mm="30.00")
    rec["text"] = rec["text"].replace("두께 12 mm", "두께 30 mm")
    r = _check(rec, table)
    assert not r.ok
    assert R_COMBO in r.reasons


def test_c_limit_value_recompute_mismatch(table):
    rec = c_record(limit_value="4.00")
    rec["text"] = rec["text"].replace("3.00 mm 이하", "4.00 mm 이하").replace("≤ 3.00", "≤ 4.00")
    r = _check(rec, table)
    assert not r.ok
    assert R_RECOMPUTE in r.reasons


def test_c_margin_recompute_mismatch(table):
    r = _check(c_record(margin="0.90"), table)
    assert not r.ok
    assert R_RECOMPUTE in r.reasons


def test_c_nullability_matrix(table):
    # 비율(F5) 골격에 size_mm 기재 = 단위 혼입 차단 (§4-4 매트릭스)
    rec = c_record(
        defect_code="2012", limit_type="비율", size_mm="20.00", measured_value="20.00",
        limit_value="25.00", margin="5.00", clause_id="KR-3.2.4",
    )
    r = _check(rec, table)
    assert not r.ok
    assert R_SCHEMA in r.reasons


def test_c_missing_text(table):
    r = _check(c_record(text=None), table)
    assert not r.ok
    assert R_SCHEMA in r.reasons


def test_c_numeric_lock_rerun(table):
    rec = c_record()
    rec["text"] += " 추가로 7.77 mm 지점 참조."
    r = _check(rec, table)
    assert not r.ok
    assert any(x.startswith("numeric_lock:") for x in r.reasons)


def test_c_unit_mixup_discarded(table):
    # 골격 단위는 mm 인데 생성문이 % 로 적었다 — 값이 같아도 다른 물리량이다 (§5-4)
    rec = c_record()
    rec["text"] = rec["text"].replace("허용 3.00 mm 이하", "허용 3.00 % 이하")
    r = _check(rec, table)
    assert not r.ok
    assert "numeric_lock:changed_value" in r.reasons


def test_c_hallucinated_standard_discarded(table):
    rec = c_record()
    rec["text"] += " 아울러 ISO 9999 요건도 충족한다."
    r = _check(rec, table)
    assert not r.ok
    assert "numeric_lock:extra_value" in r.reasons


def test_c_standard_whitelist_can_be_injected(table):
    # 템플릿이 다른 표준을 인용해야 하면 코드가 아니라 인자로 넓힌다
    rec = c_record()
    rec["text"] += " 근거 문서 IACS Rec.47."
    assert not _check(rec, table).ok
    assert _check(rec, table, allowed_standards=("ISO 6520-1", "IACS Rec.47")).ok


def test_c_too_long(table):
    r = _check(c_record(), table, max_len=10)
    assert not r.ok
    assert R_LONG in r.reasons


def test_c_prop_t_valid(table):
    rec = c_record(
        sample_id="c-401", defect_code="401", limit_type="길이", limit_rule="prop_t",
        size_mm="1.00", limit_value="1.20", margin="0.20", clause_id="KR-3.2.2",
        text=(
            "관찰: 길이 1.00 mm의 선형 지시. 분류: 융합불량(401). 조회: 두께 12 mm 기준 "
            "KR-3.2.2 허용 1.20 mm 이하. 대조: 1.00 ≤ 1.20. 판정: 합격."
        ),
    )
    assert _check(rec, table).ok


def test_c_f4_size_zero_fails(table):
    # 불허 행 measured ≤ 0 → judge 예외 → 평가 불능으로 폐기 (T12 의 (c) 경로 대응)
    rec = c_record(
        defect_code="100", limit_type="불허", limit_rule="none_permitted",
        size_mm="0.00", limit_value=None, margin=None, verdict="불합격",
        clause_id="KR-3.2.3",
        text="관찰: 길이 0.00 mm. 분류: 균열(100). KR-3.2.3 불허. 판정: 불합격.",
    )
    r = _check(rec, table)
    assert not r.ok
    assert R_SCHEMA in r.reasons


# ---------------------------------------------------------------------------
# D4 페어 — verdict_mode 게이팅 (§4-7) · 정상 페어 (§4-6-1) · 조립 (§4-6)
# ---------------------------------------------------------------------------


def _pair_defect(**over) -> dict:
    d = c_record()
    d.pop("text")
    d.update(
        sample_id="IMG001-0", source="measured",
        image_id="IMG001", defect_instance_id="0",
    )
    d.update(over)
    return d


def test_pair_full_mode_valid(table):
    d1 = _pair_defect()
    d2 = _pair_defect(
        sample_id="IMG001-1", defect_instance_id="1", defect_code="401",
        limit_type="길이", limit_rule="prop_t", size_mm="6.00", limit_value="1.20",
        margin="-4.80", verdict="불합격", clause_id="KR-3.2.2",
    )
    rec = {
        "image_id": "IMG001", "sample_id": "IMG001", "defects": [d1, d2],
        "verdict": "불합격",  # aggregate_verdicts: 하나라도 불합격 → 불합격
        "cited_clauses": ["KR-3.2.1", "KR-3.2.2"],
        "text": (
            "관찰: 직경 2.00 mm 기공(2011) 및 길이 6.00 mm 융합불량(401). "
            "조회: 두께 12 mm 기준 KR-3.2.1 허용 3.00 mm 이하 및 KR-3.2.2 허용 1.20 mm "
            "이하. 판정: 불합격."
        ),
    }
    r = check_record_pair(
        rec, table, verdict_mode="full", label_codes=LABEL_CODES,
        clause_registry=REGISTRY, defect_lexicon=LEXICON,
    )
    assert r.ok, r


def test_pair_full_mode_aggregate_mismatch(table):
    d1 = _pair_defect()
    d2 = _pair_defect(
        sample_id="IMG001-1", defect_code="401", limit_type="길이", limit_rule="prop_t",
        size_mm="6.00", limit_value="1.20", margin="-4.80", verdict="불합격",
        clause_id="KR-3.2.2",
    )
    rec = {
        "image_id": "IMG001", "sample_id": "IMG001", "defects": [d1, d2],
        "verdict": "합격",  # 보수 규칙 위반
        "cited_clauses": ["KR-3.2.1", "KR-3.2.2"],
        "text": (
            "관찰: 직경 2.00 mm 기공(2011) 및 길이 6.00 mm 융합불량(401). "
            "조회: 두께 12 mm 기준 KR-3.2.1 허용 3.00 mm 이하 및 KR-3.2.2 허용 1.20 mm "
            "이하. 판정: 합격."
        ),
    }
    r = check_record_pair(
        rec, table, verdict_mode="full", label_codes=LABEL_CODES,
        clause_registry=REGISTRY, defect_lexicon=LEXICON,
    )
    assert not r.ok
    assert R_PAIR in r.reasons


def test_pair_cited_clauses_must_be_sorted_union(table):
    d1 = _pair_defect()
    rec = {
        "image_id": "IMG001", "sample_id": "IMG001", "defects": [d1],
        "verdict": "합격", "cited_clauses": [],  # 결손
        "text": (
            "관찰: 직경 2.00 mm 기공(2011). 조회: 두께 12 mm 기준 KR-3.2.1 허용 "
            "3.00 mm 이하. 판정: 합격."
        ),
    }
    r = check_record_pair(
        rec, table, verdict_mode="full", label_codes=LABEL_CODES,
        clause_registry=REGISTRY,
    )
    assert not r.ok
    assert R_PAIR in r.reasons


def test_pair_clause_only_gate(table):
    d1 = _pair_defect(verdict=None, margin=None)
    rec = {
        "image_id": "IMG001", "sample_id": "IMG001", "defects": [d1],
        "verdict": None, "cited_clauses": ["KR-3.2.1"],
        "text": (
            "관찰: 직경 2.00 mm 기공(2011). 조회: 두께 12 mm 기준 KR-3.2.1 허용 "
            "3.00 mm 이하."
        ),
    }
    r = check_record_pair(
        rec, table, verdict_mode="clause_only", label_codes=LABEL_CODES,
        clause_registry=REGISTRY,
    )
    assert r.ok, r
    # clause_only 인데 verdict 기재 → 게이트 위반
    bad = {**rec, "defects": [_pair_defect(verdict="합격", margin="1.00")], "verdict": "합격"}
    r2 = check_record_pair(
        bad, table, verdict_mode="clause_only", label_codes=LABEL_CODES,
        clause_registry=REGISTRY,
    )
    assert not r2.ok
    assert R_SCHEMA in r2.reasons or R_PAIR in r2.reasons


def test_normal_pair_full_mode(table):
    rec = {
        "image_id": "IMG002", "sample_id": "IMG002-normal", "defects": [],
        "verdict": "합격",  # aggregate_verdicts([]) = 합격
        "cited_clauses": [], "thickness_mm": "12.00", "quality_level": "C",
        "text": "검출 한계 내 특기할 결함 지시 없음. 가정 두께 12 mm 품질수준 C 기준. 판정: 합격.",
    }
    r = check_record_pair(
        rec, table, verdict_mode="full", defect_lexicon=LEXICON,
        label_codes=LABEL_CODES, clause_registry=REGISTRY,
    )
    assert r.ok, r


def test_normal_pair_cited_clause_violation(table):
    rec = {
        "image_id": "IMG002", "sample_id": "IMG002-normal", "defects": [],
        "verdict": "합격", "cited_clauses": ["KR-3.2.1"],
        "thickness_mm": "12.00", "quality_level": "C",
        "text": "검출 한계 내 특기할 결함 지시 없음. 판정: 합격.",
    }
    r = check_record_pair(
        rec, table, verdict_mode="full", defect_lexicon=LEXICON,
        label_codes=LABEL_CODES, clause_registry=REGISTRY,
    )
    assert not r.ok
    assert R_NORMAL in r.reasons


def test_normal_pair_clause_only_verdict_gate(table):
    rec = {
        "image_id": "IMG002", "sample_id": "IMG002-normal", "defects": [],
        "verdict": "합격",  # clause_only 에서는 None 이어야 한다 (§4-7 T34 개정)
        "cited_clauses": [], "thickness_mm": "12.00", "quality_level": "C",
        "text": "검출 한계 내 특기할 결함 지시 없음. 가정 두께 12 mm 기준.",
    }
    r = check_record_pair(
        rec, table, verdict_mode="clause_only", defect_lexicon=LEXICON,
        label_codes=LABEL_CODES, clause_registry=REGISTRY,
    )
    assert not r.ok
    assert R_NORMAL in r.reasons


def test_normal_pair_defect_lexicon_lock(table):
    rec = {
        "image_id": "IMG002", "sample_id": "IMG002-normal", "defects": [],
        "verdict": "합격", "cited_clauses": [], "thickness_mm": "12.00",
        "quality_level": "C",
        "text": "기공은 관찰되지 않았다. 가정 두께 12 mm 기준. 판정: 합격.",
    }
    r = check_record_pair(
        rec, table, verdict_mode="full", defect_lexicon=LEXICON,
        label_codes=LABEL_CODES, clause_registry=REGISTRY,
    )
    assert not r.ok
    assert "numeric_lock:defect_hallucination" in r.reasons


def test_normal_pair_defect_lexicon_lock_ignores_case(table):
    rec = {
        "image_id": "IMG002", "sample_id": "IMG002-normal", "defects": [],
        "verdict": "합격", "cited_clauses": [], "thickness_mm": "12.00",
        "quality_level": "C",
        "text": "필름 전반에 crack 성 지시 없음. 가정 두께 12 mm 기준. 판정: 합격.",
    }
    r = check_record_pair(
        rec, table, verdict_mode="full", defect_lexicon=(*LEXICON, "Crack"),
        label_codes=LABEL_CODES, clause_registry=REGISTRY,
    )
    assert not r.ok
    assert "numeric_lock:defect_hallucination" in r.reasons


def test_normal_pair_standard_citation_discarded(table):
    rec = {
        "image_id": "IMG002", "sample_id": "IMG002-normal", "defects": [],
        "verdict": "합격", "cited_clauses": [], "thickness_mm": "12.00",
        "quality_level": "C",
        "text": "ISO 5817 기준 특기할 지시 없음. 가정 두께 12 mm 기준. 판정: 합격.",
    }
    r = check_record_pair(
        rec, table, verdict_mode="full", defect_lexicon=LEXICON,
        label_codes=LABEL_CODES, clause_registry=REGISTRY,
    )
    assert not r.ok
    assert "numeric_lock:extra_value" in r.reasons


def test_pair_invalid_verdict_mode(table):
    with pytest.raises(ValueError):
        check_record_pair({"defects": [], "cited_clauses": []}, table, verdict_mode="loose")


# ---------------------------------------------------------------------------
# (b) QA
# ---------------------------------------------------------------------------

PASSAGES = {"P1", "P2"}


def test_qa_valid_and_duplicate():
    seen: set[str] = set()
    r1 = check_record_qa(
        {"question": "기공의 허용 기준은?", "answer": "표준의 허용치를 따른다.",
         "evidence_passage_ids": ["P1"]},
        passage_ids=PASSAGES, seen=seen,
    )
    assert r1.ok, r1
    # 정규화 후 완전 일치 (공백·전각 차이는 동일 질문)
    r2 = check_record_qa(
        {"question": "기공의  허용 기준은?", "answer": "다른 답.",
         "evidence_passage_ids": ["P2"]},
        passage_ids=PASSAGES, seen=seen,
    )
    assert not r2.ok
    assert R_DUP in r2.reasons


def test_qa_evidence_not_found():
    r = check_record_qa(
        {"question": "질문?", "answer": "답.", "evidence_passage_ids": ["P9"]},
        passage_ids=PASSAGES, seen=set(),
    )
    assert not r.ok
    assert R_EVIDENCE in r.reasons


def test_qa_schema_violation():
    r = check_record_qa(
        {"question": "", "answer": "답.", "evidence_passage_ids": []},
        passage_ids=PASSAGES, seen=set(),
    )
    assert not r.ok
    assert R_SCHEMA in r.reasons


def test_normalize_question():
    assert normalize_question("기공의  허용 기준은?") == normalize_question("기공의 허용 기준은?")


# ---------------------------------------------------------------------------
# 배치 러너
# ---------------------------------------------------------------------------


def test_run_stage1_counts(table):
    records = [c_record(), c_record(defect_code="999"), c_record(margin="0.00")]
    rep = run_stage1(
        records, asset="c", table=table, label_codes=LABEL_CODES, clause_registry=REGISTRY
    )
    assert (rep.n_in, rep.n_pass, rep.n_fail) == (3, 1, 2)
    assert rep.fail_reasons.get(R_ISO) == 1
    assert rep.fail_reasons.get(R_RECOMPUTE) == 1
    summary = rep.stage_summary()
    assert summary["n_in"] == 3 and summary["n_pass"] == 1


def test_run_stage1_requires_table():
    with pytest.raises(ValueError):
        run_stage1([], asset="c", table=None)
    with pytest.raises(ValueError):
        run_stage1([], asset="x")


# ---------------------------------------------------------------------------
# validation_report (§6-4) · 파일럿 변형 (§7-6)
# ---------------------------------------------------------------------------


def test_stage_summary_math():
    s = stage_summary(200, 180, {"extra_value": 12, "verdict_flip": 8})
    assert s["n_fail"] == 20
    assert s["pass_rate"] == 0.9
    with pytest.raises(ValueError):
        stage_summary(10, 11)


def test_build_report_chain_enforced():
    stages = {
        "stage0_numeric_lock": {"n_in": 520, "n_pass": 480, "fail_reasons": {"extra_value": 40}},
        "stage1_rule": {"n_in": 480, "n_pass": 470},
        "stage2_judge": {"n_in": 470, "n_pass": 450},
    }
    rep = build_validation_report(
        "c_reasoning", "v1", stages, quarantine=3, git_commit="abc123",
        limits_sha256="f" * 64, configs_sha256="e" * 64,
    )
    assert rep["n_generated"] == 520
    assert rep["n_final"] == 450
    assert rep["overall_pass_rate"] == round(450 / 520, 4)
    # 단계 연쇄 위반 → 거부 (각 단계는 전 단계 통과분에만)
    bad = dict(stages)
    bad["stage1_rule"] = {"n_in": 500, "n_pass": 470}
    with pytest.raises(ValueError):
        build_validation_report("c_reasoning", "v1", bad)
    # 미정의 단계명 거부
    with pytest.raises(ValueError):
        build_validation_report("c_reasoning", "v1", {"stageX": {"n_in": 1, "n_pass": 1}})


def test_write_report_roundtrip(tmp_path: Path):
    rep = build_validation_report(
        "d4_pairs", "v1",
        {"stage0_numeric_lock": {"n_in": 100, "n_pass": 90},
         "stage1_rule": {"n_in": 90, "n_pass": 88}},
        quarantine=2, git_commit="deadbeef", limits_sha256="a" * 64,
    )
    p = write_validation_report(rep, tmp_path)
    assert p.name == "validation_report_d4_pairs_v1.json"
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["stages"]["stage1_rule"]["pass_rate"] == round(88 / 90, 4)
    assert loaded["quarantine"] == 2
    assert loaded["limits_snapshot"] == "a" * 64


def test_pilot_report(tmp_path: Path):
    r_c = build_validation_report(
        "c_reasoning", "v0-pilot",
        {"stage0_numeric_lock": {"n_in": 400, "n_pass": 360},
         "stage1_rule": {"n_in": 360, "n_pass": 350}},
        pilot=True,
    )
    r_b = build_validation_report(
        "b_qa", "v0-pilot", {"stage1_rule": {"n_in": 100, "n_pass": 95}}, pilot=True
    )
    pilot = build_pilot_report([r_c, r_b], git_commit="abc", limits_sha256="b" * 64)
    assert pilot["pilot"] is True
    assert pilot["snapshot_eligible"] is False  # §7-6: 파일럿 산출물은 스냅샷 비대상
    assert set(pilot["assets"]) == {"c_reasoning", "b_qa"}
    p = write_pilot_report(pilot, tmp_path)
    assert p.name == "validation_report_pilot.json"
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["assets"]["c_reasoning"]["stages"]["stage0_numeric_lock"]["pass_rate"] == 0.9
    with pytest.raises(ValueError):
        write_pilot_report({"pilot": False}, tmp_path)
