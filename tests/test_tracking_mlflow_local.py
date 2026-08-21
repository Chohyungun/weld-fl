"""MLflow 로컬 추적 테스트. 스펙 §8·§10 "격리" 행.

금지를 **문서가 아니라 실행 시점에 강제**하는지가 핵심이다. 경고로 넘기면 로그에 묻히고,
묻힌 채로 한 번 돌면 그 run 의 기록이 어디로 갔는지 사후에 증명할 수 없다.
"""

from __future__ import annotations

import pytest

from tracking.mlflow_local import (
    REQUIRED_TAGS,
    CellFingerprint,
    CloudLoggingBlocked,
    MissingRunMetadata,
    check_cells_identical,
    check_required_tags,
    detect_cloud_logging,
    guard_no_cloud_logging,
    reject_best_checkpoint,
    require_tags,
    run_name,
    tracking_uri,
)

# --- 외부 클라우드 로깅 차단 --------------------------------------------------

def test_wandb_env_blocks_immediately():
    with pytest.raises(CloudLoggingBlocked) as e:
        guard_no_cloud_logging({"WANDB_API_KEY": "x"})
    assert "WANDB_API_KEY" in str(e.value)


@pytest.mark.parametrize("var", [
    "WANDB_PROJECT", "WANDB_MODE", "COMET_API_KEY", "NEPTUNE_API_TOKEN", "CLEARML_API_HOST",
])
def test_other_cloud_loggers_blocked(var):
    with pytest.raises(CloudLoggingBlocked):
        guard_no_cloud_logging({var: "x"})


def test_clean_env_passes():
    guard_no_cloud_logging({"PATH": "/usr/bin", "HOME": "/home/x"})


def test_detect_reports_all_offenders_sorted():
    found = detect_cloud_logging({"WANDB_MODE": "1", "COMET_API_KEY": "2", "PATH": "3"})
    assert found == ("COMET_API_KEY", "WANDB_MODE")


def test_empty_value_still_blocks():
    """빈 값이라도 설정돼 있으면 도구가 켜질 수 있다."""
    with pytest.raises(CloudLoggingBlocked):
        guard_no_cloud_logging({"WANDB_MODE": ""})


# --- 로컬 저장소 구성 ---------------------------------------------------------

def test_tracking_uri_is_local_sqlite():
    uri = tracking_uri("tracking/mlruns.db")
    assert uri.startswith("sqlite:///")
    assert "://" not in uri.removeprefix("sqlite:///")


def test_tracking_uri_has_no_remote_host():
    assert "http" not in tracking_uri()


# --- run 명명 -----------------------------------------------------------------

def test_run_name_format():
    assert run_name("sep_fed", 0, "260910") == "sep_fed_s0_260910"


def test_run_name_rejects_unknown_cell():
    with pytest.raises(ValueError):
        run_name("uni_local", 0, "260910")   # 5칸에 없는 조합


def test_run_name_rejects_bad_date():
    with pytest.raises(ValueError):
        run_name("sep_fed", 0, "2026-09-10")


# --- 필수 태그 ----------------------------------------------------------------

def full_tags() -> dict[str, str]:
    return {t: "v" for t in REQUIRED_TAGS}


def test_complete_tags_pass():
    require_tags(full_tags())


def test_missing_tag_fails_loudly():
    tags = full_tags()
    del tags["coord_cfg_hash"]
    with pytest.raises(MissingRunMetadata) as e:
        require_tags(tags)
    assert "coord_cfg_hash" in str(e.value)


def test_blank_tag_counts_as_missing():
    tags = full_tags()
    tags["git_commit"] = "   "
    assert "git_commit" in check_required_tags(tags)


def test_all_missing_reported_at_once():
    """한 번에 고칠 수 있게 전부 모아 보고한다."""
    assert set(check_required_tags({})) == set(REQUIRED_TAGS)


# --- 5칸 동일성 ---------------------------------------------------------------

def fp(cell: str, **over) -> CellFingerprint:
    base = {"base_ckpt_sha256": "ckpt", "coords_sha256": "co",
            "coord_cfg_hash": "cfg", "rag_snapshot_sha256": "rag"}
    base.update(over)
    return CellFingerprint(cell=cell, **base)


def test_identical_cells_pass():
    assert check_cells_identical([fp("sep_central"), fp("sep_fed")]) == {}


def test_diverging_base_checkpoint_detected():
    """다섯 칸이 같은 출발점에서 시작했다는 것이 논문의 전제다."""
    d = check_cells_identical([fp("sep_central"), fp("sep_fed", base_ckpt_sha256="다름")])
    assert "base_ckpt_sha256" in d


def test_diverging_coord_cfg_detected():
    """프로파일 override 로 좌표 공간이 조용히 갈라지는 경로(Q15)."""
    d = check_cells_identical([fp("uni_central"), fp("uni_fed", coord_cfg_hash="다름")])
    assert d["coord_cfg_hash"] == {"cfg", "다름"}


def test_diverging_rag_snapshot_detected():
    d = check_cells_identical([fp("sep_local"), fp("sep_fed", rag_snapshot_sha256="다름")])
    assert "rag_snapshot_sha256" in d


# --- best 체크포인트 거부 ------------------------------------------------------

def test_best_checkpoint_rejected():
    """best 선택은 val 기준의 암묵적 조기 종료다."""
    with pytest.raises(ValueError):
        reject_best_checkpoint("runs/detect/train/weights/best.pt")


def test_last_checkpoint_accepted():
    reject_best_checkpoint("runs/detect/train/weights/last.pt")


def test_best_detected_case_insensitively():
    with pytest.raises(ValueError):
        reject_best_checkpoint("BEST.pt")
