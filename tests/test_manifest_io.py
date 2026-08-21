"""계약 #2 로더·조인·잠금 테스트. 스펙 §7-6 + 게이트 #4 조건 1·2."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from data.manifest_io import (
    ANNOTATIONS_FILENAME,
    CAPABILITIES_FILENAME,
    MANIFEST_COLUMNS,
    MANIFEST_FILENAME,
    ROW_KIND_NORMAL,
    SNAPSHOT_FILENAME,
    SNAPSHOT_MEMBERS,
    ManifestError,
    MetricStatus,
    Snapshot,
    SnapshotVerificationError,
    VerdictMode,
    defect_free_images,
    join_defects,
    load_snapshot,
    localizable,
    read_manifest,
    split_view,
    verify_snapshot,
    write_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AIHUB = REPO_ROOT / "data" / "mock" / "mock_aihub_v1"
RIAWELC = REPO_ROOT / "data" / "mock" / "mock_riawelc_v1"


@pytest.fixture(scope="module")
def snap() -> Snapshot:
    return load_snapshot(AIHUB)


@pytest.fixture(scope="module")
def snap_riawelc() -> Snapshot:
    return load_snapshot(RIAWELC)


def test_mock_snapshots_exist() -> None:
    for root in (AIHUB, RIAWELC):
        for name in (*SNAPSHOT_MEMBERS, SNAPSHOT_FILENAME):
            assert (root / name).exists(), f"{root / name} 이 없다"


def test_column_order_is_contract(snap: Snapshot) -> None:
    assert tuple(snap.manifest.columns) == MANIFEST_COLUMNS


def test_empty_string_not_nan(snap: Snapshot) -> None:
    """정상 이미지의 defect_types 는 "" 로 읽힌다 (NaN 이 아니다)."""
    normals = snap.manifest.loc[~snap.manifest["has_defect"].astype(bool)]
    assert len(normals) > 0
    assert (normals["defect_types"] == "").all()
    assert normals["defect_types"].notna().all()
    assert (normals["iso_codes"] == "").all()


def test_client_null_only_for_eval(snap: Snapshot) -> None:
    m = snap.manifest
    assert m.loc[m["split"] == "eval", "client"].isna().all()
    assert m.loc[m["split"] != "eval", "client"].notna().all()


def test_bool_columns_are_boolean_dtype(snap: Snapshot) -> None:
    assert str(snap.manifest["has_defect"].dtype) == "boolean"
    assert str(snap.manifest["has_localization"].dtype) == "boolean"


def test_join_defects_keeps_normal_images(snap: Snapshot) -> None:
    joined = join_defects(snap)
    # 결함 인스턴스 수 + 정상 이미지 수 = 조인 행 수
    n_normal = int((~snap.manifest["has_defect"].astype(bool)).sum())
    assert len(joined) == len(snap.annotations) + n_normal
    normals = joined.loc[joined["row_kind"] == ROW_KIND_NORMAL]
    assert len(normals) == n_normal
    assert normals["ann_id"].isna().all()


def test_join_defects_row_kind_distinguishes_n1_n2(
    snap: Snapshot, snap_riawelc: Snapshot
) -> None:
    """N1(위치 라벨 없음)과 N2(정상)가 확실히 갈린다 — 결측을 뭉뚱그리면 mAP 가 왜곡된다."""
    j = join_defects(snap_riawelc)
    defects = j.loc[j["row_kind"] != ROW_KIND_NORMAL]
    assert len(defects) > 0
    # N1: 결함 행인데 bbox 가 전부 null
    assert defects["bbox_x1_px"].isna().all()
    assert (~defects["has_localization"].astype(bool)).all()
    # 반면 AI허브 mock 은 결함 행에 bbox 가 있다
    j2 = join_defects(snap)
    d2 = j2.loc[j2["row_kind"] != ROW_KIND_NORMAL]
    assert d2["bbox_x1_px"].notna().all()


def test_join_rejects_overlapping_columns(snap: Snapshot) -> None:
    bad = snap.annotations.assign(split="x")
    with pytest.raises(ManifestError, match="겹친다"):
        join_defects(manifest=snap.manifest, annotations=bad)


def test_capabilities_gate_metrics(snap: Snapshot, snap_riawelc: Snapshot) -> None:
    assert snap.can_score("map") is True
    assert snap_riawelc.can_score("map") is False
    assert snap_riawelc.can_score("bbox_iou") is False
    # clause_only 이므로 판정 정합성은 두 쪽 다 산출 불가
    assert snap.verdict_mode is VerdictMode.CLAUSE_ONLY
    assert snap.can_score("verdict_consistency") is False
    assert MetricStatus.NO_LOCALIZATION.value.startswith("N/A")


def test_capabilities_matches_row_level(snap: Snapshot, snap_riawelc: Snapshot) -> None:
    for s in (snap, snap_riawelc):
        expected = bool(s.manifest["has_localization"].astype(bool).all())
        assert s.has_localization == expected
        n_scale = int(s.manifest["px_per_mm"].notna().sum())
        assert s.capabilities["counts"]["with_pixel_scale"] == n_scale
        assert s.capabilities["capabilities"]["pixel_scale"] == (n_scale == len(s.manifest) and n_scale > 0)


def test_views(snap: Snapshot) -> None:
    assert len(defect_free_images(snap.manifest)) > 0
    assert len(localizable(snap.manifest)) == len(snap.manifest)
    assert (split_view(snap.manifest, "train", "C1")["client"] == "C1").all()
    assert len(split_view(snap.manifest, "eval")) > 0
    assert len(localizable(load_snapshot(RIAWELC).manifest)) == 0


# ---- 잠금 (게이트 #4 조건 1) ---------------------------------------------------------


def test_snapshot_locks_both_files_together(snap: Snapshot) -> None:
    recorded = {
        line.split(maxsplit=1)[1].strip()
        for line in (AIHUB / SNAPSHOT_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert recorded == set(SNAPSHOT_MEMBERS)
    assert MANIFEST_FILENAME in recorded and ANNOTATIONS_FILENAME in recorded


def test_verify_detects_tamper(tmp_path: Path, snap: Snapshot) -> None:
    root = tmp_path / "snap"
    shutil.copytree(AIHUB, root)
    verify_snapshot(root)                       # 사본은 통과해야 한다
    target = root / ANNOTATIONS_FILENAME
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SnapshotVerificationError, match="해시가 다르다"):
        verify_snapshot(root)


def test_verify_detects_partial_lock(tmp_path: Path) -> None:
    """annotations 를 뺀 잠금은 거부한다 — 두 파일은 함께 잠긴다."""
    root = tmp_path / "snap"
    shutil.copytree(AIHUB, root)
    lines = [
        line
        for line in (root / SNAPSHOT_FILENAME).read_text(encoding="utf-8").splitlines()
        if ANNOTATIONS_FILENAME not in line
    ]
    (root / SNAPSHOT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(SnapshotVerificationError, match="함께 잠겨야"):
        verify_snapshot(root)


def test_load_requires_lock(tmp_path: Path) -> None:
    root = tmp_path / "snap"
    shutil.copytree(AIHUB, root)
    (root / SNAPSHOT_FILENAME).unlink()
    with pytest.raises(SnapshotVerificationError, match="잠기지 않은"):
        load_snapshot(root)
    load_snapshot(root, verify=False)   # 생성 중간 단계에서만 허용


def test_byte_reproducible_roundtrip(tmp_path: Path, snap: Snapshot) -> None:
    """write → read → write 왕복 후 바이트가 동일해야 한다 (스펙 §6-9)."""
    a, b = tmp_path / "a", tmp_path / "b"
    d1 = write_snapshot(a, snap.manifest, snap.annotations, snap.capabilities)
    reread = load_snapshot(a)
    d2 = write_snapshot(b, reread.manifest, reread.annotations, reread.capabilities)
    assert d1 == d2
    for name in SNAPSHOT_MEMBERS:
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_written_files_use_lf(snap: Snapshot) -> None:
    """win32 CRLF 자동 변환이 재현성을 깨뜨리므로 LF 를 강제한다."""
    for root in (AIHUB, RIAWELC):
        for name in (MANIFEST_FILENAME, ANNOTATIONS_FILENAME, CAPABILITIES_FILENAME):
            assert b"\r\n" not in (root / name).read_bytes(), f"{root/name} 에 CRLF"


def test_read_rejects_wrong_column_order(tmp_path: Path) -> None:
    df = pd.read_csv(AIHUB / MANIFEST_FILENAME, dtype=str, keep_default_na=False)
    shuffled = df.loc[:, [*MANIFEST_COLUMNS[1:], MANIFEST_COLUMNS[0]]]
    p = tmp_path / "m.csv"
    shuffled.to_csv(p, index=False, lineterminator="\n")
    with pytest.raises(ManifestError, match="컬럼 순서"):
        read_manifest(p)
