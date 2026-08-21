"""원문 PDF 정찰 1단계 — 텍스트 레이어·키워드 페이지 스캔 (pdfplumber).

스펙 §3-2 경로 결정 트리의 입력을 만든다:
  - 페이지당 추출 문자수 < 50 → 스캔본 의심 (경로 M 후보)
  - 허용치 관련 키워드 적중 페이지 → Docling 정찰(survey_docling.py)의 --pages 후보

사용: uv run python -m corpus.parse.survey_textscan <doc_id>
산출: corpus/parse/survey/{doc_id}/textscan.json + 콘솔 요약
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pdfplumber
import yaml

from data.label_map import load_label_map

PARSE_DIR = Path(__file__).resolve().parent
SCAN_TEXT_THRESHOLD = 50  # 문자/페이지 — 스펙 §3-2 스캔본 판정 기준

# 허용치 표·판정 조항 탐색 키워드.
# 결함 명칭은 사상표(계약 #1)에서 읽는다 — 라벨 문자열 하드코딩 금지(불변조건 1-8).
# 사상표가 바뀌면 문서 스캔 대상도 따라 바뀌는 것이 옳다.
RULE_KEYWORDS = [
    "방사선", "판정", "등급", "허용", "결함", "비파괴",
    "radiograph", "acceptance", "limit", "imperfection", "defect",
]
# 사상표에 없는 결함 유형 (RT 주 실험 밖이지만 문서에는 등장한다)
EXTRA_DEFECT_KEYWORDS = ["용입", "언더컷", "undercut", "porosity", "crack", "slag"]


def build_keywords() -> list[str]:
    """스캔 키워드 = 사상표 결함 명칭 + 규정 어휘 + 문서 등장 결함 어휘."""
    lm = load_label_map()
    names = sorted({dt.name_ko for dt in lm.defect_types.values()})
    return names + RULE_KEYWORDS + EXTRA_DEFECT_KEYWORDS


KEYWORDS = build_keywords()
# 허용치 판정 조합 판별용 — 결함 축과 규칙 축이 같은 페이지에 있는지 본다
DEFECT_KEYWORDS = frozenset(build_keywords()) - frozenset(RULE_KEYWORDS)
RULE_AXIS = frozenset(["허용", "판정", "등급", "limit", "acceptance"])


def load_source(doc_id: str) -> dict:
    reg = yaml.safe_load((PARSE_DIR / "sources.yaml").read_text(encoding="utf-8"))
    for d in reg["documents"]:
        if d["doc_id"] == doc_id:
            return d
    raise SystemExit(f"sources.yaml 에 미등록: {doc_id} — 미등록 문서는 파싱 금지 (스펙 §3-1)")


def verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != expected:
        raise SystemExit(f"sha256 불일치: {path}\n  등록부 {expected}\n  실파일 {h}")


def group_pages(pages: list[int]) -> list[list[int]]:
    """연속 페이지를 [시작, 끝] 구간으로 묶는다."""
    ranges: list[list[int]] = []
    for p in sorted(pages):
        if ranges and p <= ranges[-1][1] + 2:  # 1쪽 간격은 같은 구간으로
            ranges[-1][1] = p
        else:
            ranges.append([p, p])
    return ranges


def main() -> None:
    doc_id = sys.argv[1]
    src = load_source(doc_id)
    pdf_path = PARSE_DIR.parent.parent / src["file_local"]
    verify_sha256(pdf_path, src["sha256"])

    pages_report = []
    hit_pages: dict[str, list[int]] = {k: [] for k in KEYWORDS}
    low_text_pages: list[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text() or ""
            n_chars = len(text)
            n_tables_hint = len(page.find_tables())
            if n_chars < SCAN_TEXT_THRESHOLD:
                low_text_pages.append(page.page_number)
            hits = [k for k in KEYWORDS if k.lower() in text.lower()]
            for k in hits:
                hit_pages[k].append(page.page_number)
            pages_report.append(
                {"page": page.page_number, "chars": n_chars,
                 "tables_hint": n_tables_hint, "hits": hits}
            )

    # 허용치 표 후보: 결함 축과 규칙 축이 같은 페이지에서 동시 적중
    candidates = [
        p["page"] for p in pages_report
        if set(p["hits"]) & DEFECT_KEYWORDS and set(p["hits"]) & RULE_AXIS
    ]

    report = {
        "doc_id": doc_id,
        "sha256": src["sha256"],
        "n_pages": n_pages,
        "scan_text_threshold": SCAN_TEXT_THRESHOLD,
        "low_text_pages": low_text_pages,
        "scan_suspect": len(low_text_pages) > n_pages * 0.5,
        "candidate_pages": candidates,
        "candidate_ranges": group_pages(candidates),
        "keyword_pages": {k: v for k, v in hit_pages.items() if v},
        "pages": pages_report,
    }

    out_dir = PARSE_DIR / "survey" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "textscan.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[{doc_id}] {n_pages}쪽, 저텍스트 {len(low_text_pages)}쪽"
          f" (스캔본 의심: {report['scan_suspect']})")
    print(f"  허용치 표 후보 구간: {report['candidate_ranges']}")
    print(f"  보고서: {out}")


if __name__ == "__main__":
    main()
