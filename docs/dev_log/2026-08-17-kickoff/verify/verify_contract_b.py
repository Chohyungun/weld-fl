"""R-재현: 트랙 B 스펙의 계약 #3 ↔ 계약 #1·#2(구현본) 정합성 실측.

리뷰 문서: docs/dev_log/2026-08-17-kickoff/20_review_재현_B.md §0 표의 근거.
저장소 루트에서 실행한다. 외부 의존 없음 — 계약 #1·#2 구현본과 data/mock/ 만 읽는다.

    uv run python docs/dev_log/2026-08-17-kickoff/verify/verify_contract_b.py
"""

import io
import sys
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(R))

from data.label_map import load_label_map
from data.manifest_io import load_snapshot

lm = load_label_map(R / "configs" / "label_map.yaml")

print("=" * 70)
print("[1] 계약 #1 코드 집합 — B의 G0 V1 'defect_code ∈ 사상표' 가 참조할 API")
print("=" * 70)
print("대표 코드 :", sorted(d.iso_code for d in lm.defect_types.values()))
alts = sorted(c for d in lm.defect_types.values() for c in d.iso_code_alt)
print("별칭 코드 :", alts)
print("공개 API  :", [a for a in dir(lm) if not a.startswith("_")])
has_accessor = hasattr(lm, "iso_codes")
print("→ 코드 집합 접근자 존재?", has_accessor,
      "  (리뷰 시점: 없음 → Important-7 로 지적하고 A 가 추가함)")
if has_accessor:
    print("   iso_codes()               =", sorted(lm.iso_codes()))
    print("   iso_codes(include_alt=F)  =", sorted(lm.iso_codes(include_alt=False)))
    print("   → 별칭 2012 포함 여부의 판단이 소비 트랙이 아니라 계약 소유자에게 있다")

print()
print("=" * 70)
print("[2] material 값 대조 — B: ST|AL|ALL  vs  계약 #2 실측")
print("=" * 70)
for name in ("mock_aihub_v1", "mock_riawelc_v1"):
    s = load_snapshot(R / "data" / "mock" / name)
    vals = sorted(s.manifest["material"].dropna().unique().tolist())
    print(f"{name:18s} material = {vals}")

print()
print("=" * 70)
print("[3] thickness_source / scale_source 값 — B: {metadata, assumed}")
print("=" * 70)
for name in ("mock_aihub_v1", "mock_riawelc_v1"):
    s = load_snapshot(R / "data" / "mock" / name)
    print(f"{name:18s} thickness_source={sorted(s.manifest['thickness_source'].unique().tolist())}"
          f"  scale_source={sorted(s.manifest['scale_source'].unique().tolist())}"
          f"  verdict_mode={s.verdict_mode.value}")

print()
print("=" * 70)
print("[4] 공란 = +무한 을 pandas 기본값으로 읽으면 V2 검사가 어떻게 되는가")
print("=" * 70)
csv = (
    "rule_id,thickness_min,thickness_max,limit_value\n"
    "KR-1-01,3,8,2.0\n"
    "KR-1-02,8,,3.0\n"        # max 공란 = +INF (정상 행)
    "KR-1-03,25,12,4.0\n"     # min > max (반드시 걸려야 하는 오염 행)
)
df = pd.read_csv(io.StringIO(csv))          # B 스펙에 인코딩·na 규약 없음 → 기본값
print(df.to_string(index=False))
print("dtypes:", dict(df.dtypes.astype(str)))
print()
print("V2 를 '이상이면 오류'로 쓰면:  (min >= max) →",
      df.apply(lambda r: bool(r.thickness_min >= r.thickness_max), axis=1).tolist())
print("V2 를 '미만이 아니면 오류'로: not(min < max) →",
      df.apply(lambda r: not bool(r.thickness_min < r.thickness_max), axis=1).tolist())
print("→ 2행(정상 +INF)과 3행(오염)의 판정이 작성 방식에 따라 뒤바뀐다. NaN 비교는 항상 False.")

print()
print("=" * 70)
print("[5] B의 D4 수량 산정 재계산")
print("=" * 70)
st = {"정상": 26287, "균열": 2277, "기공": 27040, "융합불량": 3256, "슬래그": 1140}
al = {"정상": 5242, "균열": 332, "기공": 2945, "융합불량": 329, "슬래그": 1152}
tot = sum(st.values()) + sum(al.values())
defect = tot - st["정상"] - al["정상"]
pool = tot * 0.8
pool_defect = pool * defect / tot
normals = pool_defect / 2
print(f"RT 총           {tot:,}")
print(f"결함 이미지     {defect:,}  ({defect/tot:.2%})")
print(f"학습 풀 80%     {pool:,.0f}")
print(f"풀 내 결함      {pool_defect:,.0f}   (B 주장 ~30,800)")
print(f"정상 다운샘플   {normals:,.0f}   (B 주장 ~15,400)")
print(f"D4 합계         {pool_defect + normals:,.0f}   (B 주장 ~46,000)")

print()
print("=" * 70)
print("[6] ISO 코드 2011 과 연도 2011 의 동형 — 수치 잠금 화이트리스트 충돌")
print("=" * 70)
print("허용 집합에 들어가는 defect_code:", sorted({d.iso_code for d in lm.defect_types.values()}))
print("→ '2011' 은 기공 코드이면서 연도 표기로도 자연 등장한다.")
print("→ 반대로 '2012'(배제 코드)가 연도로 등장하면 허용 집합 밖 → 전량 폐기.")
