"""검출·근거접지 지표 테스트. 스펙 §10 "Macro-F1 / 결함 recall / Class-Jaccard" 행."""

from __future__ import annotations

import pytest

from evaluation.metrics.detection import (
    class_jaccard,
    confusion_pairs,
    score_detection,
)

CLASSES = ("100", "2011", "301", "401")  # 균열·기공·슬래그·융합불량 (label_map 유래)


def test_perfect_prediction_scores_one():
    gold = {"a": ["2011"], "b": ["100", "301"]}
    r = score_detection(gold, gold, CLASSES)
    assert r.macro_f1 == pytest.approx(1.0)
    assert r.defect_recall == pytest.approx(1.0)


def test_complete_miss_scores_zero():
    gold = {"a": ["2011"], "b": ["100"]}
    r = score_detection({"a": [], "b": []}, gold, CLASSES)
    assert r.macro_f1 == pytest.approx(0.0)
    assert r.defect_recall == pytest.approx(0.0)


def test_normal_image_is_not_a_fifth_class():
    """정상을 클래스로 세면 다수 클래스가 하나 더 들어가 Macro-F1 이 부풀려진다."""
    gold = {"n1": [], "n2": [], "n3": [], "d": ["2011"]}
    r = score_detection(gold, gold, CLASSES)
    scored = {c.iso_code for c in r.per_class if c.support > 0}
    assert scored == {"2011"}
    assert r.macro_f1 == pytest.approx(1.0)


def test_absent_class_is_skipped_and_reported():
    """표본 0인 클래스를 0.0 으로 평균에 넣으면 지표가 조용히 내려간다.
    빼되 **뺀 사실을 보고**해야 한다 (A 리뷰 Minor-2 처방)."""
    gold = {"a": ["2011"]}
    r = score_detection(gold, gold, CLASSES)
    assert r.macro_f1 == pytest.approx(1.0)
    assert set(r.skipped_classes) == {"100", "301", "401"}


def test_parse_failure_counted_as_miss_not_dropped():
    """파싱 실패를 통계에서 빼면 오답보다 낙관적으로 잡힌다."""
    gold = {"a": ["2011"], "b": ["2011"]}
    r = score_detection({"a": ["2011"], "b": []}, gold, CLASSES)
    assert r.defect_recall == pytest.approx(0.5)


def test_partial_detection_on_multi_defect_image():
    gold = {"a": ["2011", "301"]}
    r = score_detection({"a": ["2011"]}, gold, CLASSES)
    assert r.defect_recall == pytest.approx(0.5)


def test_precision_undefined_when_no_positive_prediction():
    """0으로 두면 macro 평균이 왜곡된다 — 없는 것과 틀린 것은 다르다."""
    gold = {"a": ["2011"]}
    r = score_detection({"a": []}, gold, CLASSES)
    porosity = next(c for c in r.per_class if c.iso_code == "2011")
    assert porosity.precision is None
    assert porosity.recall == pytest.approx(0.0)


def test_duplicate_mentions_collapse_to_set():
    """통합형이 같은 결함을 두 번 언급해도 이미지 수준 집합은 하나다."""
    gold = {"a": ["2011"]}
    r = score_detection({"a": ["2011", "2011"]}, gold, CLASSES)
    porosity = next(c for c in r.per_class if c.iso_code == "2011")
    assert (porosity.tp, porosity.fp) == (1, 0)


def test_false_positive_counted():
    gold = {"a": []}
    r = score_detection({"a": ["100"]}, gold, CLASSES)
    crack = next(c for c in r.per_class if c.iso_code == "100")
    assert (crack.tp, crack.fp, crack.fn) == (0, 1, 0)


def test_confusion_matrix_is_per_class_2x2():
    gold = {"a": ["2011"], "b": []}
    m = confusion_pairs({"a": ["2011"], "b": ["100"]}, gold, CLASSES)
    assert m["2011"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}
    assert m["100"] == {"tp": 0, "fp": 1, "fn": 0, "tn": 1}


# --- Class-Jaccard --------------------------------------------------------------

def test_jaccard_both_empty_is_one():
    """정상을 정상으로 맞힌 것은 만점이다. 0/0 을 0 으로 두면 정상 비율이 지표를 지배한다."""
    assert class_jaccard({"a": []}, {"a": []}) == pytest.approx(1.0)


def test_jaccard_one_side_empty_is_zero():
    assert class_jaccard({"a": []}, {"a": ["2011"]}) == pytest.approx(0.0)


def test_jaccard_partial_overlap():
    j = class_jaccard({"a": ["2011", "100"]}, {"a": ["2011", "301"]})
    assert j == pytest.approx(1 / 3)
