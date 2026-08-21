"""validation_report JSON 작성기 (스펙 §6-4) + 파일럿 보고 변형 (§7-6).

- 경로: corpus/validate/validation_report_{asset}_{version}.json
- 내용: 단계별 {n_in, n_pass, n_fail, fail_reasons{}, pass_rate} + quarantine 건수 +
  git commit + limits·configs 스냅샷 해시.
- 논문 표는 이 JSON 에서 스크립트로 추출한다 — 손 전기 금지.
- 각 단계는 전 단계 통과분에만 적용된다 (§6) — 인접 단계의 n_in ↔ n_pass 연쇄가
  어긋나면 보고서 생성 자체를 거부한다 (fail-closed).
- 파일럿 변형 (§7-6): validation_report_pilot.json — 자산별 단계 통과율 + 폐기 사유
  분포, pilot=True 표기 (limits v0-pilot 유래 — 스냅샷 비대상, 감독 보고 필수).
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Optional, Sequence

__all__ = [
    "STAGE_ORDER",
    "stage_summary",
    "build_validation_report",
    "write_validation_report",
    "build_pilot_report",
    "write_pilot_report",
]

# counts.json 의 discarded 키(§7-5)와 정렬된 표준 단계명
STAGE_ORDER: tuple[str, ...] = (
    "stage0_numeric_lock",
    "stage1_rule",
    "stage2_judge",
    "stage3_expert",
)

_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def stage_summary(
    n_in: int, n_pass: int, fail_reasons: Optional[Mapping[str, int]] = None
) -> dict:
    """단계 1개의 통계 블록. n_fail·pass_rate 는 여기서만 계산한다 (손 계산 금지)."""
    if n_in < 0 or n_pass < 0 or n_pass > n_in:
        raise ValueError(f"단계 통계 위반: n_in={n_in}, n_pass={n_pass}")
    reasons = {str(k): int(v) for k, v in (fail_reasons or {}).items()}
    if any(v < 0 for v in reasons.values()):
        raise ValueError(f"fail_reasons 음수: {reasons}")
    return {
        "n_in": n_in,
        "n_pass": n_pass,
        "n_fail": n_in - n_pass,
        "fail_reasons": dict(sorted(reasons.items())),
        "pass_rate": round(n_pass / n_in, 4) if n_in else None,
    }


def _normalize_stage(name: str, raw: Mapping) -> dict:
    try:
        return stage_summary(
            int(raw["n_in"]), int(raw["n_pass"]), raw.get("fail_reasons")
        )
    except KeyError as e:
        raise ValueError(f"단계 {name}: 필수 키 결손 {e}") from e


def build_validation_report(
    asset: str,
    version: str,
    stages: Mapping[str, Mapping],
    *,
    quarantine: int = 0,
    git_commit: Optional[str] = None,
    limits_sha256: Optional[str] = None,
    configs_sha256: Optional[str] = None,
    manifest_sha256: Optional[str] = None,
    pilot: bool = False,
    notes: Optional[Sequence[str]] = None,
    date: Optional[str] = None,
) -> dict:
    """자산 1개의 검증 보고서 dict (§6-4 형식).

    stages: 단계명 → {n_in, n_pass, fail_reasons?} (삽입 순서 = 단계 순서.
    STAGE_ORDER 밖 단계명은 거부). 인접 단계 연쇄(다음 n_in == 이전 n_pass)가
    어긋나면 ValueError — "각 단계는 전 단계 통과분에만"의 기계 강제다.
    """
    if not _NAME_RE.match(asset) or not _NAME_RE.match(version):
        raise ValueError(f"asset/version 파일명 불가 문자: {asset!r}, {version!r}")
    if not stages:
        raise ValueError("stages 비어 있음")
    if quarantine < 0:
        raise ValueError(f"quarantine 음수: {quarantine}")

    normalized: dict[str, dict] = {}
    for name in stages:
        if name not in STAGE_ORDER:
            raise ValueError(f"미정의 단계명: {name!r} (허용: {STAGE_ORDER})")
    ordered = [n for n in STAGE_ORDER if n in stages]
    for name in ordered:
        normalized[name] = _normalize_stage(name, stages[name])
    for prev, nxt in zip(ordered, ordered[1:]):
        if normalized[nxt]["n_in"] != normalized[prev]["n_pass"]:
            raise ValueError(
                f"단계 연쇄 위반: {nxt}.n_in={normalized[nxt]['n_in']} ≠ "
                f"{prev}.n_pass={normalized[prev]['n_pass']} (각 단계는 전 단계 통과분에만)"
            )

    n_generated = normalized[ordered[0]]["n_in"]
    n_final = normalized[ordered[-1]]["n_pass"]
    return {
        "asset": asset,
        "version": version,
        "pilot": bool(pilot),
        "stages": normalized,
        "quarantine": int(quarantine),
        "n_generated": n_generated,
        "n_final": n_final,
        "overall_pass_rate": round(n_final / n_generated, 4) if n_generated else None,
        "git_commit": git_commit or "unknown",
        "limits_snapshot": limits_sha256,
        "configs_snapshot": configs_sha256,
        "manifest_snapshot": manifest_sha256,
        "notes": list(notes or []),
        "date": date or _dt.date.today().isoformat(),
    }


def _json_default(o: object) -> str:
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, (_dt.date, _dt.datetime)):
        return o.isoformat()
    raise TypeError(f"직렬화 불가 타입: {type(o).__name__}")


def _write_json(report: Mapping, path: Path) -> Path:
    """바이트로 기록한다 — write_text 의 플랫폼 개행 변환이 같은 내용에 서로 다른 파일
    바이트를 만들어 스냅샷 해시가 플랫폼마다 갈린다 (counts_builder.write_counts 와 동일 규약)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default
    )
    path.write_bytes((payload + "\n").encode("utf-8"))
    return path


def write_validation_report(report: Mapping, out_dir: str | Path) -> Path:
    """corpus/validate/validation_report_{asset}_{version}.json 으로 기록."""
    asset = str(report["asset"])
    version = str(report["version"])
    if not _NAME_RE.match(asset) or not _NAME_RE.match(version):
        raise ValueError(f"asset/version 파일명 불가 문자: {asset!r}, {version!r}")
    return _write_json(report, Path(out_dir) / f"validation_report_{asset}_{version}.json")


def build_pilot_report(
    asset_reports: Sequence[Mapping],
    *,
    git_commit: Optional[str] = None,
    limits_sha256: Optional[str] = None,
    configs_sha256: Optional[str] = None,
    notes: Optional[Sequence[str]] = None,
    date: Optional[str] = None,
) -> dict:
    """소표본 관통 시험 보고 (§7-6) — 자산별 보고서 묶음 + pilot 표기.

    입력은 build_validation_report 산출물이어야 하며 전부 pilot=True 로 강제 재표기된다
    (limits v0-pilot 유래 — 스냅샷 비대상, 감독 보고 필수).
    """
    if not asset_reports:
        raise ValueError("asset_reports 비어 있음")
    assets: dict[str, dict] = {}
    for rep in asset_reports:
        a = str(rep["asset"])
        if a in assets:
            raise ValueError(f"자산 중복: {a}")
        assets[a] = {**rep, "pilot": True}
    return {
        "pilot": True,
        "snapshot_eligible": False,  # §7-6: 파일럿 산출물은 스냅샷 비대상
        "assets": dict(sorted(assets.items())),
        "git_commit": git_commit or "unknown",
        "limits_snapshot": limits_sha256,
        "configs_snapshot": configs_sha256,
        "notes": list(notes or []),
        "date": date or _dt.date.today().isoformat(),
    }


def write_pilot_report(report: Mapping, out_dir: str | Path) -> Path:
    """corpus/validate/validation_report_pilot.json 으로 기록."""
    if not report.get("pilot"):
        raise ValueError("pilot 보고서가 아니다 (pilot=True 필수)")
    return _write_json(report, Path(out_dir) / "validation_report_pilot.json")
