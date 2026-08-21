"""매니페스트 불변식 IV1~IV12. 스펙 §2-4.

IV7·IV8 은 **누수 여부를 직접 판정하는 불변식**이다. 실패하면 실험 결과 전체가 무효다.
검증기는 첫 위반에서 멈추지 않고 전부 모아 보고한다 — 한 번에 고칠 수 있게.

    from data.invariants import check_invariants
    violations = check_invariants(snap.manifest, snap.annotations, load_label_map())
    assert not violations, violations
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data.label_map import LabelMap
from data.manifest_io import ANNOTATION_COLUMNS, file_sha256

_GEOM_PX_COLUMNS = (
    "polygon_json", "bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px",
    "area_px", "major_axis_px", "minor_axis_px", "equiv_diameter_px",
)
_MM_COLUMNS = ("major_axis_mm", "equiv_diameter_mm")


@dataclass(frozen=True)
class Violation:
    rule: str
    message: str
    examples: tuple[str, ...] = ()

    def __str__(self) -> str:
        tail = f" 예: {list(self.examples)}" if self.examples else ""
        return f"[{self.rule}] {self.message}{tail}"


def _head(values) -> tuple[str, ...]:
    return tuple(str(v) for v in list(values)[:5])


def _is_null(series: pd.Series) -> pd.Series:
    """빈 문자열도 결측으로 본다 (문자열 컬럼과 수치 컬럼을 함께 다루기 위해)."""
    if str(series.dtype) == "string":
        return series.isna() | (series == "")
    return series.isna()


def check_invariants(
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    label_map: LabelMap,
    *,
    raw_root: Path | str | None = None,
    hash_sample_frac: float = 0.01,
    seed: int = 20260825,
) -> list[Violation]:
    """IV1~IV12 를 전부 확인하고 위반 목록을 반환한다. 빈 리스트 = 통과."""
    v: list[Violation] = []
    m, a = manifest, annotations

    # IV1 — PK 유일 + FK 무결성
    dup = m.loc[m["image_id"].duplicated(), "image_id"]
    if len(dup):
        v.append(Violation("IV1", f"image_id 중복 {len(dup)}건", _head(dup)))
    if a["ann_id"].duplicated().any():
        v.append(Violation("IV1", "ann_id 중복", _head(a.loc[a["ann_id"].duplicated(), "ann_id"])))
    orphan = set(a["image_id"]) - set(m["image_id"])
    if orphan:
        v.append(Violation("IV1", f"annotations 에 고아 image_id {len(orphan)}건", _head(sorted(orphan))))

    # IV2 — has_defect == (n_defects > 0) == 실제 annotation 행 수 > 0
    counts = a.groupby("image_id").size()
    actual = m["image_id"].map(counts).fillna(0).astype(int)
    bad = m.loc[(m["n_defects"].astype(int) != actual) | (m["has_defect"].astype(bool) != (actual > 0))]
    if len(bad):
        v.append(Violation("IV2", f"has_defect/n_defects 와 실제 결함 행 수 불일치 {len(bad)}건",
                           _head(bad["image_id"])))

    # IV3 — defect_types 는 L2 키의 정렬·중복제거 조인과 정확히 일치
    expected = (
        a.groupby("image_id")["defect_type"]
        .apply(lambda s: ";".join(sorted(set(s.dropna()))))
    )
    got = m.set_index("image_id")["defect_types"].fillna("")
    exp = got.index.map(lambda i: expected.get(i, ""))
    mism = [i for i, g, e in zip(got.index, got, exp) if str(g) != str(e)]
    if mism:
        v.append(Violation("IV3", f"defect_types 가 annotations 와 불일치 {len(mism)}건", _head(mism)))
    unknown = set(a["defect_type"].dropna()) - set(label_map.defect_types)
    if unknown:
        v.append(Violation("IV3", f"사상표에 없는 defect_type {sorted(unknown)}"))
    wrong_code = a.loc[
        a["defect_type"].notna()
        & (a["defect_type"].map(lambda k: label_map.defect_types[k].iso_code
                               if k in label_map.defect_types else None) != a["iso_code"])
    ]
    if len(wrong_code):
        v.append(Violation("IV3", f"iso_code 가 사상표와 다르다 {len(wrong_code)}건",
                           _head(wrong_code["ann_id"])))

    # IV4 — has_localization=false ⇒ 기하 컬럼 전부 null
    no_loc = set(m.loc[~m["has_localization"].astype(bool), "image_id"])
    if no_loc:
        rows = a.loc[a["image_id"].isin(no_loc)]
        present = [c for c in _GEOM_PX_COLUMNS if c in rows and (~_is_null(rows[c])).any()]
        if present:
            v.append(Violation("IV4", f"has_localization=false 인데 기하 컬럼에 값이 있다: {present}",
                               _head(rows["ann_id"])))

    # IV5 — px_per_mm null ⇒ *_mm 전부 null
    no_scale = set(m.loc[_is_null(m["px_per_mm"]), "image_id"])
    if no_scale:
        rows = a.loc[a["image_id"].isin(no_scale)]
        present = [c for c in _MM_COLUMNS if (~_is_null(rows[c])).any()]
        if present:
            v.append(Violation("IV5", f"px_per_mm 이 없는데 mm 컬럼에 값이 있다: {present}",
                               _head(rows["ann_id"])))

    # IV6 — split=='eval' ⇔ client is null
    is_eval = m["split"] == "eval"
    client_null = _is_null(m["client"])
    bad = m.loc[is_eval != client_null]
    if len(bad):
        v.append(Violation("IV6", f"split=='eval' 과 client is null 이 동치가 아니다 {len(bad)}건",
                           _head(bad["image_id"])))

    # IV7 — 같은 group_id 의 split 이 동일 (누수 방지의 핵심)
    straddle = m.groupby("group_id")["split"].nunique()
    bad_g = straddle.loc[straddle > 1]
    if len(bad_g):
        v.append(Violation("IV7", f"묶음이 split 을 걸쳤다 — 누수 {len(bad_g)}묶음", _head(bad_g.index)))

    # IV8 — 같은 group_id 의 client 가 동일
    cg = m.assign(_c=m["client"].fillna("__EVAL__")).groupby("group_id")["_c"].nunique()
    bad_g = cg.loc[cg > 1]
    if len(bad_g):
        v.append(Violation("IV8", f"묶음이 client 를 걸쳤다 — 누수 {len(bad_g)}묶음", _head(bad_g.index)))

    # group_size 정합 (IV7/IV8 의 보조)
    sizes = m.groupby("group_id")["image_id"].transform("size")
    bad = m.loc[m["group_size"].astype(int) != sizes]
    if len(bad):
        v.append(Violation("IV8", f"group_size 가 실제 묶음 크기와 다르다 {len(bad)}건",
                           _head(bad["image_id"])))

    # IV9 — eval_subset non-null ⇒ split == 'eval'
    bad = m.loc[(~_is_null(m["eval_subset"])) & (m["split"] != "eval")]
    if len(bad):
        v.append(Violation("IV9", f"eval_subset 이 있는데 split!='eval' {len(bad)}건",
                           _head(bad["image_id"])))

    # IV10 — bbox 가 이미지 경계 안, x1<x2 / y1<y2
    dims = m.set_index("image_id")[["width_px", "height_px"]]
    ab = a.loc[~_is_null(a["bbox_x1_px"])].join(dims, on="image_id")
    if len(ab):
        bad = ab.loc[
            (ab["bbox_x1_px"] < 0) | (ab["bbox_y1_px"] < 0)
            | (ab["bbox_x1_px"] >= ab["bbox_x2_px"]) | (ab["bbox_y1_px"] >= ab["bbox_y2_px"])
            | (ab["bbox_x2_px"] > ab["width_px"]) | (ab["bbox_y2_px"] > ab["height_px"])
        ]
        if len(bad):
            v.append(Violation("IV10", f"bbox 좌표 범위 위반 {len(bad)}건", _head(bad["ann_id"])))

    # IV11 — label_map_version 일치
    versions = set(m["label_map_version"].astype(int))
    if versions != {label_map.version}:
        v.append(Violation("IV11", f"label_map_version 불일치. manifest={sorted(versions)} "
                                   f"configs={label_map.version}"))

    # IV12 — 이미지 파일 해시 표본 재계산 (raw_root 가 주어질 때만)
    if raw_root is not None:
        root = Path(raw_root)
        n = max(1, int(len(m) * hash_sample_frac))
        sample = m.sample(n=min(n, len(m)), random_state=seed)
        mism2 = []
        for row in sample.itertuples():
            p = root / str(row.rel_path)
            if not p.exists():
                mism2.append(f"{row.image_id}(파일없음)")
            elif file_sha256(p) != row.sha256:
                mism2.append(str(row.image_id))
        if mism2:
            v.append(Violation("IV12", f"sha256 재계산 불일치 {len(mism2)}/{len(sample)}건",
                               _head(mism2)))

    # 스키마 위생 — annotations 컬럼 순서
    if tuple(a.columns) != ANNOTATION_COLUMNS:
        v.append(Violation("SCHEMA", "annotations 컬럼 순서가 계약과 다르다"))

    return v
