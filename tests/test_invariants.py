"""불변식 IV1~IV12 테스트. 스펙 §7-4.

각 불변식에 **통과 케이스 1 + 위반 케이스 1**. 위반 케이스는 고의로 깨뜨린 픽스처로
검증기가 실제로 잡는지 확인한다 — 특히 IV7·IV8(누수)은 이 테스트가 마지막 방어선이다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.invariants import check_invariants
from data.label_map import load_label_map
from data.manifest_io import load_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCKS = ("mock_aihub_v1", "mock_riawelc_v1")


@pytest.fixture(scope="module")
def lm():
    return load_label_map()


@pytest.fixture(params=MOCKS, scope="module")
def snap(request):
    return load_snapshot(REPO_ROOT / "data" / "mock" / request.param)


def _rules(violations) -> set[str]:
    return {v.rule for v in violations}


def test_mock_snapshots_pass_all_invariants(snap, lm) -> None:
    violations = check_invariants(snap.manifest, snap.annotations, lm)
    assert not violations, "\n".join(str(v) for v in violations)


# ---- 위반 케이스 ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def base(lm):
    s = load_snapshot(REPO_ROOT / "data" / "mock" / "mock_aihub_v1")
    return s.manifest, s.annotations


def test_IV1_duplicate_image_id(base, lm) -> None:
    m, a = base
    bad = pd.concat([m, m.iloc[[0]]], ignore_index=True)
    assert "IV1" in _rules(check_invariants(bad, a, lm))


def test_IV1_orphan_annotation(base, lm) -> None:
    m, a = base
    bad = a.copy()
    bad.loc[bad.index[0], "image_id"] = "ghost:does/not/exist.png"
    assert "IV1" in _rules(check_invariants(m, bad, lm))


def test_IV2_n_defects_mismatch(base, lm) -> None:
    m, a = base
    bad = m.copy()
    idx = bad.index[bad["has_defect"].astype(bool)][0]
    bad.loc[idx, "n_defects"] = 99
    assert "IV2" in _rules(check_invariants(bad, a, lm))


def test_IV2_has_defect_lies(base, lm) -> None:
    m, a = base
    bad = m.copy()
    idx = bad.index[~bad["has_defect"].astype(bool)][0]
    bad.loc[idx, "has_defect"] = True
    assert "IV2" in _rules(check_invariants(bad, a, lm))


def test_IV3_defect_types_mismatch(base, lm) -> None:
    m, a = base
    bad = m.copy()
    idx = bad.index[bad["has_defect"].astype(bool)][0]
    bad.loc[idx, "defect_types"] = "crack;porosity;slag_inclusion"
    assert "IV3" in _rules(check_invariants(bad, a, lm))


def test_IV3_unknown_defect_type(base, lm) -> None:
    m, a = base
    bad = a.copy()
    bad.loc[bad.index[0], "defect_type"] = "undercut"
    assert "IV3" in _rules(check_invariants(m, bad, lm))


def test_IV3_wrong_iso_code(base, lm) -> None:
    m, a = base
    bad = a.copy()
    bad.loc[bad.index[0], "iso_code"] = "9999"
    assert "IV3" in _rules(check_invariants(m, bad, lm))


def test_IV4_localization_false_with_bbox(lm) -> None:
    s = load_snapshot(REPO_ROOT / "data" / "mock" / "mock_riawelc_v1")
    bad = s.annotations.copy()
    bad.loc[bad.index[0], "bbox_x1_px"] = 10
    bad.loc[bad.index[0], "bbox_x2_px"] = 50
    assert "IV4" in _rules(check_invariants(s.manifest, bad, lm))


def test_IV5_mm_without_scale(base, lm) -> None:
    m, a = base
    bad = a.copy()
    bad.loc[bad.index[0], "major_axis_mm"] = 2.5
    assert "IV5" in _rules(check_invariants(m, bad, lm))


def test_IV6_eval_with_client(base, lm) -> None:
    m, a = base
    bad = m.copy()
    idx = bad.index[bad["split"] == "eval"][0]
    bad.loc[idx, "client"] = "C1"
    assert "IV6" in _rules(check_invariants(bad, a, lm))


def test_IV6_train_without_client(base, lm) -> None:
    m, a = base
    bad = m.copy()
    idx = bad.index[bad["split"] == "train"][0]
    bad.loc[idx, "client"] = pd.NA
    assert "IV6" in _rules(check_invariants(bad, a, lm))


def test_IV7_group_split_leak_detected(base, lm) -> None:
    """묶음이 eval 과 train 에 갈라지면 잡아야 한다. 이게 곧 누수다."""
    m, a = base
    multi = m.loc[m["group_size"] > 1]
    assert len(multi) > 0, "mock 에 다중 묶음이 없다 — 픽스처가 잘못됐다"
    gid = multi["group_id"].iloc[0]
    bad = m.copy()
    idx = bad.index[bad["group_id"] == gid][0]
    bad.loc[idx, "split"] = "eval" if bad.loc[idx, "split"] != "eval" else "train"
    assert "IV7" in _rules(check_invariants(bad, a, lm))


def test_IV8_group_client_leak_detected(base, lm) -> None:
    m, a = base
    pool = m.loc[(m["group_size"] > 1) & (m["split"] != "eval")]
    gid = pool["group_id"].iloc[0]
    bad = m.copy()
    rows = bad.index[bad["group_id"] == gid]
    bad.loc[rows[0], "client"] = "C1"
    bad.loc[rows[1], "client"] = "C2"
    assert "IV8" in _rules(check_invariants(bad, a, lm))


def test_IV8_group_size_mismatch(base, lm) -> None:
    m, a = base
    bad = m.copy()
    bad.loc[bad.index[0], "group_size"] = 77
    assert "IV8" in _rules(check_invariants(bad, a, lm))


def test_IV9_eval_subset_outside_eval(base, lm) -> None:
    m, a = base
    bad = m.copy()
    idx = bad.index[bad["split"] == "train"][0]
    bad.loc[idx, "eval_subset"] = "judgment_2000"
    assert "IV9" in _rules(check_invariants(bad, a, lm))


def test_IV10_bbox_out_of_bounds(base, lm) -> None:
    m, a = base
    bad = a.copy()
    idx = bad.index[bad["bbox_x1_px"].notna()][0]
    bad.loc[idx, "bbox_x2_px"] = 999999
    assert "IV10" in _rules(check_invariants(m, bad, lm))


def test_IV10_inverted_bbox(base, lm) -> None:
    m, a = base
    bad = a.copy()
    idx = bad.index[bad["bbox_x1_px"].notna()][0]
    x1 = bad.loc[idx, "bbox_x1_px"]
    bad.loc[idx, "bbox_x1_px"] = bad.loc[idx, "bbox_x2_px"]
    bad.loc[idx, "bbox_x2_px"] = x1
    assert "IV10" in _rules(check_invariants(m, bad, lm))


def test_IV11_label_map_version_mismatch(base, lm) -> None:
    m, a = base
    bad = m.copy()
    bad["label_map_version"] = 999
    assert "IV11" in _rules(check_invariants(bad, a, lm))


def test_IV12_hash_mismatch_when_raw_root_given(base, lm, tmp_path) -> None:
    """실파일이 없으면 IV12 가 파일 부재를 잡아야 한다."""
    m, a = base
    violations = check_invariants(m, a, lm, raw_root=tmp_path, hash_sample_frac=0.01)
    assert "IV12" in _rules(violations)


def test_IV12_passes_with_matching_files(base, lm, tmp_path) -> None:
    import hashlib

    m, a = base
    sample = m.head(3)
    for row in sample.itertuples():
        p = tmp_path / str(row.rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # mock 의 sha256 은 rel_path 로부터 만들어졌다 — 같은 내용을 못 만들므로
        # 여기서는 파일 존재 + 해시 일치를 직접 구성한다
        p.write_bytes(b"")
        assert hashlib.sha256(b"").hexdigest() != row.sha256
    fixed = m.head(3).copy()
    fixed["sha256"] = hashlib.sha256(b"").hexdigest()
    violations = check_invariants(
        fixed, a.loc[a["image_id"].isin(fixed["image_id"])], lm,
        raw_root=tmp_path, hash_sample_frac=1.0,
    )
    assert "IV12" not in _rules(violations)
