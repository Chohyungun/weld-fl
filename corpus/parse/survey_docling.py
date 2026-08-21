"""원문 PDF 정찰 2단계 — Docling 변환 실측 (함정 #7 게이트 실측, 스펙 §3-7).

본 파이프라인이 아니다. 수용 판정·정규화 없이 Docling 원시 출력만 뜬다:
  - 문서(또는 페이지 구간) markdown
  - 표별 원시 CSV (가공 없음 — fill·정규화 금지, 스펙 §3-3)
  - meta.json (docling 버전·표 수·표별 형상·페이지·소요 시간)

사용: uv run python corpus/parse/survey_docling.py <doc_id> [--pages A-B]
산출: corpus/parse/survey/{doc_id}/
설정: TableFormer ACCURATE, OCR off (스펙 §3-3 고정)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

import yaml

# sm_120/Windows 실측 (2026-08-17): triton 휠이 없어 torch.compile이 docling layout
# 모델(object_detection transformers engine)에서 치명 오류로 승격된다. eager 강제.
# torch/docling import 전에 설정해야 효력이 있다.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

PARSE_DIR = Path(__file__).resolve().parent


def load_source(doc_id: str) -> dict:
    reg = yaml.safe_load((PARSE_DIR / "sources.yaml").read_text(encoding="utf-8"))
    for d in reg["documents"]:
        if d["doc_id"] == doc_id:
            return d
    raise SystemExit(f"sources.yaml 에 미등록: {doc_id} — 미등록 문서는 파싱 금지 (스펙 §3-1)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id")
    ap.add_argument("--pages", help="정찰 페이지 구간 A-B (1-기준, 포함)", default=None)
    args = ap.parse_args()

    src = load_source(args.doc_id)
    pdf_path = PARSE_DIR.parent.parent / src["file_local"]
    h = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    if h != src["sha256"]:
        raise SystemExit(f"sha256 불일치: {pdf_path}")

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False                      # 스펙 §3-2: OCR 미도입
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

    page_range = None
    if args.pages:
        a, b = args.pages.split("-")
        page_range = (int(a), int(b))

    t0 = time.time()
    if page_range:
        result = converter.convert(pdf_path, page_range=page_range)
    else:
        result = converter.convert(pdf_path)
    elapsed = time.time() - t0
    doc = result.document

    tag = f"p{page_range[0]}-{page_range[1]}" if page_range else "full"
    out_dir = PARSE_DIR / "survey" / args.doc_id
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{args.doc_id}_{tag}.md"
    md_path.write_text(doc.export_to_markdown(), encoding="utf-8")

    tables_meta = []
    for i, table in enumerate(doc.tables):
        df = table.export_to_dataframe()
        page_no = table.prov[0].page_no if table.prov else None
        csv_path = out_dir / f"{args.doc_id}_{tag}_t{i:03d}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        tables_meta.append(
            {"table_idx": i, "page": page_no, "rows": int(df.shape[0]),
             "cols": int(df.shape[1]), "csv": csv_path.name,
             "empty_cell_ratio": round(float(df.isna().sum().sum() + (df == "").sum().sum())
                                        / max(df.size, 1), 3)}
        )

    meta = {
        "doc_id": args.doc_id,
        "sha256": src["sha256"],
        "page_range": list(page_range) if page_range else "full",
        "docling_version": pkg_version("docling"),
        "tableformer_mode": "ACCURATE",
        "ocr": False,
        "elapsed_sec": round(elapsed, 1),
        "conversion_status": str(result.status),
        "n_tables": len(tables_meta),
        "tables": tables_meta,
    }
    (out_dir / f"meta_{tag}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"[{args.doc_id} {tag}] status={result.status} {elapsed:.0f}s, 표 {len(tables_meta)}개")
    for t in tables_meta:
        print(f"  t{t['table_idx']:03d} p{t['page']} {t['rows']}x{t['cols']}"
              f" empty={t['empty_cell_ratio']}")


if __name__ == "__main__":
    main()
