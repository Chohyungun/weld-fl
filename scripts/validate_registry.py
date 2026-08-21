"""데이터셋 등록부 검증기 — 손으로 쓴 JSON은 반드시 깨진다는 전제로 만든 관문.

    uv run python scripts/validate_registry.py

두 층으로 검사한다.

1. **스키마** (`data/registry/_schema/dataset_registry.schema.json`)
   키 오타·타입·enum·status 별 필수 동반 필드. `jsonschema` 가 설치돼 있을 때만 돈다.
2. **의미 검사** (이 파일) — 스키마가 표현할 수 없는 것. 항상 돈다.
   계약 #1 대조, 파일명 일치, 레드라인 정합, 계산값 침입, 절대경로 등.

실패는 두 등급이다. **ERROR 는 등록부를 무효로 만들고, WARN 은 8/23 초안 단계에서
허용되지만 파이프라인 투입(`pipeline.eligible: true`) 전에는 반드시 해소해야 한다.**
`--strict` 를 주면 WARN 도 실패로 친다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# Windows 콘솔 기본 코드페이지(cp949)가 em-dash 같은 문자를 못 찍어 죽는다.
# 팀원이 매번 밟을 자리라 출력 인코딩을 강제한다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = REPO_ROOT / "data" / "registry" / "datasets"
SCHEMA_PATH = REPO_ROOT / "data" / "registry" / "_schema" / "dataset_registry.schema.json"

sys.path.insert(0, str(REPO_ROOT))

#: 사람이 적으면 안 되는 키. 스키마의 additionalProperties:false 가 이미 막지만,
#: 중첩 어디에 숨어도 잡히도록 전 트리를 훑는다.
FORBIDDEN_KEYS = {
    "sha256", "phash_hex", "group_id", "strata_key", "split", "client", "eval_subset",
    "width_px", "height_px", "area_px", "equiv_diameter_px", "equiv_diameter_mm",
    "major_axis_px", "minor_axis_px", "has_localization", "has_defect", "n_defects",
    "verdict_mode", "assumptions", "counts_measured", "label_map_version",
    "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
}
#: 게이트 #7 결정 G — 등록부는 "어디를 보라"만 받는다. 값을 적으면 실측 동기가 사라진다.
NO_VALUE_FIELDS = ("thickness_mm", "px_per_mm", "quality_level")

ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
ISO_CODE_RE = re.compile(r"^\d{3,4}$")
HANGUL_RE = re.compile(r"[가-힣]")


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warns: list[str] = []

    def error(self, ds: str, msg: str) -> None:
        self.errors.append(f"[{ds}] {msg}")

    def warn(self, ds: str, msg: str) -> None:
        self.warns.append(f"[{ds}] {msg}")


def _walk(obj: Any, path: str = ""):
    """(경로, 키, 값) 전수 순회."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


SMART_QUOTES = "“”‘’"


def check_bytes(path: Path, rep: Report) -> bool:
    """S0 — 파싱보다 먼저. 인코딩·개행·스마트쿼트. **자동 수정하지 않는다.**

    손으로 쓴 JSON이 깨지는 첫 자리가 여기다. 메모장이 BOM을 붙이고, 워드가 따옴표를
    둥글게 바꾸고, git 설정이 CRLF로 바꾼다. 어느 파일 몇 행인지만 알려준다.
    """
    ds = path.stem
    raw = path.read_bytes()
    ok = True

    if raw.startswith(b"\xef\xbb\xbf"):
        rep.error(ds, "UTF-8 BOM 이 있다 — BOM 없이 저장한다")
        ok = False

    if b"\r\n" in raw:
        line = raw.split(b"\r\n")[0].count(b"\n") + 1
        rep.error(ds, f"개행이 CRLF 다 (첫 발생 {line}행) — LF 로 저장한다")
        ok = False

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        rep.error(ds, f"UTF-8 디코드 실패: {exc}")
        return False

    for i, line_text in enumerate(text.splitlines(), 1):
        hit = [c for c in SMART_QUOTES if c in line_text]
        if hit:
            rep.error(ds, f"{i}행에 스마트쿼트 {hit} — 곧은 따옴표로 바꾼다")
            ok = False
    return ok


def check_schema(doc: dict, ds: str, rep: Report) -> bool:
    """스키마 검증. jsonschema 미설치면 건너뛰고 알린다."""
    try:
        import jsonschema
    except ImportError:
        return False
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(doc), key=lambda e: list(e.path)):
        loc = ".".join(str(p) for p in err.path) or "(최상위)"
        rep.error(ds, f"스키마 위반 {loc}: {err.message}")
    return True


def check_semantics(doc: dict, path: Path, rep: Report) -> None:
    ds = doc.get("dataset_id", path.stem)

    # 파일명 == dataset_id
    if doc.get("dataset_id") != path.stem:
        rep.error(ds, f"dataset_id({doc.get('dataset_id')}) 가 파일명({path.stem}) 과 다르다")

    # 계산값이 어디에도 있으면 안 된다
    for parent, key, _ in _walk(doc):
        if key in FORBIDDEN_KEYS and not parent.startswith("acquisition.archives"):
            where = f"{parent}.{key}" if parent else key
            rep.error(ds, f"계산 결과를 손으로 적었다: {where} — 어댑터가 만든다")

    fields = doc.get("fields_declared", {})

    # 게이트 #7 결정 G — 두께·스케일·품질수준의 '값'을 적으면 안 된다
    for name in NO_VALUE_FIELDS:
        decl = fields.get(name)
        if not isinstance(decl, dict):
            continue
        if decl.get("status") == "constant" and decl.get("constant_value") is not None:
            rep.error(
                ds,
                f"fields_declared.{name} 에 값({decl['constant_value']!r})을 적었다 — "
                "등록부는 위치만 받는다 (게이트 #7 결정 G)",
            )

    # status 별 동반 필드 (jsonschema 없이도 걸리게 이중으로)
    for name, decl in fields.items():
        if not isinstance(decl, dict):
            continue
        st = decl.get("status")
        need = {
            "constant": ("constant_value", "evidence"),
            "absent": ("evidence", "evidence_level"),
            "out_of_contract": ("observed_value", "evidence"),
            "unverified": ("how_to_verify", "owner", "due"),
            "present": ("source_key", "evidence"),
        }.get(st, ())
        for k in need:
            if k not in decl or decl[k] in (None, "") and k != "source_key":
                rep.error(ds, f"fields_declared.{name}: status={st} 인데 {k} 가 없다")
        if decl.get("coverage") is not None:
            rep.error(ds, f"fields_declared.{name}.coverage 는 null 이어야 한다 (실측은 probe_result)")

    # 라벨 어휘 — 계약 #1 대조
    vocab = doc.get("label_vocabulary")
    if isinstance(vocab, dict):
        _check_vocabulary(vocab, doc, ds, rep)

    # 절대경로 금지
    for parent, key, val in _walk(doc):
        if isinstance(val, str) and ABS_PATH_RE.match(val):
            where = f"{parent}.{key}" if parent else key
            rep.error(ds, f"절대경로를 적었다: {where} = {val!r} — configs/paths.local.yaml 로")

    # 레드라인 정합
    _check_redlines(doc, ds, rep)

    # 투입 자격
    _check_eligibility(doc, ds, rep)

    auth = doc.get("authoring", {})
    if auth.get("review_status") == "reviewed" and auth.get("reviewer") == auth.get("owner"):
        rep.error(ds, "작성자와 검토자가 같다 — 8/16 '결과는 사람이 1회 검증' 조항 위반")

    # 미확인 항목은 기한이 있어야 한다
    for item in doc.get("open_items") or []:
        if isinstance(item, dict) and not item.get("due"):
            rep.warn(ds, f"open_items 에 기한이 없다: {item.get('item')!r}")


def _check_vocabulary(vocab: dict, doc: dict, ds: str, rep: Report) -> None:
    try:
        from data.label_map import load_label_map

        lm = load_label_map()
        l2_keys = set(lm.defect_types)
    except (OSError, ValueError, KeyError) as exc:  # 계약 #1 을 못 읽으면 대조를 건너뛰되 알린다
        rep.warn(ds, f"계약 #1 을 읽지 못해 라벨 대조를 건너뛴다: {exc}")
        return

    for lab in vocab.get("labels", []):
        raw, l2 = lab.get("raw"), lab.get("proposed_l2")
        if lab.get("is_normal"):
            if l2 is not None:
                rep.error(ds, f"라벨 {raw!r}: 정상인데 proposed_l2 가 있다 — 정상은 L2 에 없다")
            continue
        if l2 is None:
            rep.warn(ds, f"라벨 {raw!r}: proposed_l2 미정")
            continue
        if ISO_CODE_RE.match(str(l2)) or HANGUL_RE.search(str(l2)):
            rep.error(ds, f"라벨 {raw!r}: proposed_l2 에 ISO 코드/한글({l2!r}) — L2 키여야 한다")
        elif l2 not in l2_keys:
            rep.error(ds, f"라벨 {raw!r}: proposed_l2 {l2!r} 가 계약 #1 L2 에 없다")

    # eligible 인데 계약 #1 사상표와 어휘가 어긋나면 ingest 가 즉사한다
    if doc.get("pipeline", {}).get("eligible"):
        src = doc.get("dataset_id")
        spec = lm.sources.get(src)
        if spec is None:
            rep.error(ds, f"eligible=true 인데 계약 #1 sources 에 {src!r} 가 없다")
        else:
            declared = {lab["raw"] for lab in vocab.get("labels", []) if "raw" in lab}
            known = set(spec.mapping) | set(spec.normal_labels)
            missing = declared - known
            if missing:
                rep.error(
                    ds,
                    f"eligible=true 인데 실물 라벨 {sorted(missing)} 가 계약 #1 sources.{src} 에 "
                    "없다 — unmapped_policy=fail 이므로 ingest 가 즉사한다",
                )


def _check_redlines(doc: dict, ds: str, rep: Report) -> None:
    con = doc.get("constraints", {})
    acq = doc.get("acquisition", {})
    uri = str(acq.get("staging_uri", ""))
    cloud_like = any(t in uri.lower() for t in ("gdrive:", "drive.google", "s3:", "gs:", "onedrive"))
    allowed = con.get("cloud_upload")
    if cloud_like and allowed is False:
        rep.error(
            ds,
            f"클라우드 업로드가 금지인데 staging_uri 가 클라우드다: {uri!r} — 보안 레드라인 위반",
        )
    if cloud_like and allowed == "unverified":
        rep.warn(ds, f"staging_uri 가 클라우드인데 cloud_upload 가 미확인이다: {uri!r}")
    if doc.get("provenance", {}).get("citation_required") and not doc["provenance"].get("citations"):
        rep.error(ds, "citation_required=true 인데 citations 가 비어 있다")


def _check_eligibility(doc: dict, ds: str, rep: Report) -> None:
    pipe = doc.get("pipeline", {})
    if not pipe.get("eligible"):
        if not pipe.get("exclusion_reason"):
            rep.warn(ds, "eligible=false 인데 exclusion_reason 이 없다")
        return
    for key in ("raw_subdir", "adapter"):
        if not pipe.get(key):
            rep.error(ds, f"eligible=true 인데 pipeline.{key} 가 비어 있다")
    if pipe.get("adapter_state") != "ready":
        rep.error(ds, f"eligible=true 인데 adapter_state={pipe.get('adapter_state')!r} 다")
    if doc.get("provenance", {}).get("access_state") != "in_hand":
        rep.error(ds, "eligible=true 인데 실물을 보유하지 않았다 (access_state != in_hand)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry-dir", type=Path, default=REGISTRY_DIR)
    ap.add_argument("--strict", action="store_true", help="WARN 도 실패로 친다")
    args = ap.parse_args()

    files = sorted(Path(args.registry_dir).glob("*.json"))
    if not files:
        print(f"등록부가 없다: {args.registry_dir}")
        return 1

    rep = Report()
    schema_ran = False
    for path in files:
        check_bytes(path, rep)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            rep.error(path.stem, f"JSON 파싱 실패 {exc.lineno}행: {exc.msg}")
            continue
        schema_ran = check_schema(doc, doc.get("dataset_id", path.stem), rep) or schema_ran
        check_semantics(doc, path, rep)

    print(f"등록부 {len(files)}건 검사: {', '.join(p.stem for p in files)}")
    if not schema_ran:
        print("  ! jsonschema 미설치 — 스키마 층을 건너뛰었다 (`uv add jsonschema` 로 켠다)")
    for w in rep.warns:
        print(f"  WARN  {w}")
    for e in rep.errors:
        print(f"  ERROR {e}")

    print(f"\n결과: ERROR {len(rep.errors)} / WARN {len(rep.warns)}")
    if rep.errors:
        return 1
    if rep.warns and args.strict:
        print("--strict 이므로 WARN 을 실패로 처리한다")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
