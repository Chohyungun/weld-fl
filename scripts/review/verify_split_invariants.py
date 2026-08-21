"""R-재현 리뷰(A): 분할 불변식 검증 — 묶음 누수 / 평가셋 선분리 / 층화 / 어댑터 컬럼 동일성."""
import pandas as pd, pathlib, collections

for snap in ["mock_aihub_v1", "mock_riawelc_v1"]:
    d = pathlib.Path("data/mock") / snap
    m = pd.read_csv(d / "manifest.csv", dtype=str, keep_default_na=False)
    a = pd.read_csv(d / "annotations.csv", dtype=str, keep_default_na=False)
    print(f"\n=== {snap} · images={len(m)} anns={len(a)} ===")

    # 1) 묶음(group) 단위 분할 — 같은 group_id가 두 split에 걸치면 누수
    g = m.groupby("group_id")["split"].nunique()
    bad = g[g > 1]
    print(f"[1] group_id가 여러 split에 걸친 묶음: {len(bad)}건 -> {'OK' if len(bad)==0 else 'LEAK'}")

    # 1b) 같은 group_id가 여러 client에 걸치는가
    gc = m[m.split == "train"].groupby("group_id")["client"].nunique()
    print(f"[1b] train 내 group이 여러 client에 걸침: {(gc>1).sum()}건 "
          f"-> {'OK' if (gc>1).sum()==0 else 'LEAK'}")

    # 2) split 비율
    print(f"[2] split 분포: {dict(m['split'].value_counts())}")
    eval_ratio = (m.split == "eval").mean()
    print(f"    eval 비율 {eval_ratio:.3f} (목표 0.20)")

    # 3) 평가셋이 client 배정을 받았는가 (선분리 순서 위반 신호)
    ev = m[m.split == "eval"]["client"]
    print(f"[3] eval 행의 client 값: {sorted(set(ev))} "
          f"-> {'OK (미배정)' if set(ev) <= {''} else '주의: eval에 client 배정됨'}")

    # 4) 층화 — strata_key 별 eval 비율 편차
    st = m.groupby("strata_key").apply(
        lambda x: (x.split == "eval").mean(), include_groups=False)
    print(f"[4] strata별 eval 비율 min={st.min():.3f} max={st.max():.3f} "
          f"(n_strata={len(st)}) -> {'OK' if st.max()-st.min() < 0.15 else '편차 큼'}")

    # 5) annotations ↔ manifest 참조 무결성
    orphan = set(a.image_id) - set(m.image_id)
    print(f"[5] manifest에 없는 image_id를 가진 annotation: {len(orphan)}건 "
          f"-> {'OK' if not orphan else 'FAIL'}")
    declared = m.set_index("image_id")["n_defects"].astype(int)
    actual = a.groupby("image_id").size()
    joined = declared.to_frame("declared").join(actual.rename("actual")).fillna(0)
    mism = (joined.declared != joined.actual).sum()
    print(f"[6] n_defects 선언값 ≠ 실제 annotation 수: {mism}건 -> {'OK' if mism==0 else 'FAIL'}")

    # 7) has_localization=False면 bbox 계열 전부 비어야
    loc_false = set(m[m.has_localization == "False"].image_id)
    if loc_false:
        sub = a[a.image_id.isin(loc_false)]
        nonempty = (sub[["bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px"]] != "").any(axis=1).sum()
        print(f"[7] has_localization=False인데 bbox가 채워진 annotation: {nonempty}건 "
              f"-> {'OK' if nonempty==0 else 'FAIL'}")
    else:
        print("[7] has_localization=False 행 없음 (해당 없음)")

    # 8) bbox 유효성 (x1<x2, y1<y2, 이미지 범위 내)
    wh = m.set_index("image_id")[["width_px", "height_px"]].astype(int)
    b = a[a.bbox_x1_px != ""].copy()
    if len(b):
        for c in ["bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px"]:
            b[c] = b[c].astype(float)
        b = b.join(wh, on="image_id")
        deg = ((b.bbox_x1_px >= b.bbox_x2_px) | (b.bbox_y1_px >= b.bbox_y2_px)).sum()
        oob = ((b.bbox_x1_px < 0) | (b.bbox_y1_px < 0) |
               (b.bbox_x2_px > b.width_px) | (b.bbox_y2_px > b.height_px)).sum()
        print(f"[8] 퇴화 bbox {deg}건 / 이미지 범위 이탈 {oob}건 "
              f"-> {'OK' if deg==0 and oob==0 else 'FAIL'}")
    else:
        print("[8] bbox 없음 (해당 없음)")

    # 9) 컬럼 집합 (두 어댑터 동일성 비교용)
    globals().setdefault("cols", {})[snap] = (tuple(m.columns), tuple(a.columns))

c = globals()["cols"]
same_m = c["mock_aihub_v1"][0] == c["mock_riawelc_v1"][0]
same_a = c["mock_aihub_v1"][1] == c["mock_riawelc_v1"][1]
print(f"\n[9] 두 어댑터 manifest 컬럼 동일: {same_m} / annotations 컬럼 동일: {same_a}")
