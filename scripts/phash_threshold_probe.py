"""pHash 임계 탐색 — 거리 히스토그램 + (정답 있을 때) 과분리/과병합 실측. 스펙 §6-4.

실데이터에서는 정답이 없으므로 §6-4 는 표본 100쌍 육안 확인으로 t* 를 정한다. 픽스처에는
정답(같은 용접부 = 파일명의 `w####`)이 있으므로, **육안 확인이 무엇을 보게 되는지를 미리
정량화**할 수 있다. 손실 비대칭(§6-1)에 따라 과분리(누수)를 0 으로 만드는 최소 임계를 찾는다.

    uv run python scripts/phash_threshold_probe.py --raw-root <픽스처> --truth-from-filename
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dedup.phash import (
    HASH_BITS,
    THRESHOLD_CAP,
    UnionFind,
    compute_phash,
    distance_histogram,
    iter_close_pairs,
    pack_hashes,
)
from data.ingest.riawelc import IMAGE_SUFFIXES

#: 픽스처 파일명 규약 `{CLASS}_w{weld:04d}_f{frame}.png`
TRUTH_RE = re.compile(r"^(?P<cls>[A-Z]+)_w(?P<weld>\d+)_f\d+$")


def collect(raw_root: Path, *, use_clahe: bool = True) -> tuple[list[Path], list[str]]:
    root = raw_root / "riawelc"
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    return paths, [compute_phash(p, use_clahe=use_clahe) for p in paths]


def truth_groups(paths: list[Path]) -> list[str]:
    out: list[str] = []
    for p in paths:
        m = TRUTH_RE.match(p.stem)
        if not m:
            raise ValueError(f"정답을 파일명에서 못 읽는다: {p.name}")
        out.append(f"{m['cls']}_{m['weld']}")
    return out


def evaluate(packed: np.ndarray, truth: list[str], threshold: int) -> dict[str, int]:
    """임계 t 에서의 과분리·과병합 쌍 수. 과분리가 곧 누수다."""
    n = len(truth)
    uf = UnionFind(n)
    for i, j, _ in iter_close_pairs(packed, threshold):
        uf.union(i, j)
    same_pred = {(i, j) for i in range(n) for j in range(i + 1, n) if uf.find(i) == uf.find(j)}
    same_true = {(i, j) for i in range(n) for j in range(i + 1, n) if truth[i] == truth[j]}
    return {
        "threshold": threshold,
        "true_pairs": len(same_true),
        "merged": len(same_pred),
        "under_merge": len(same_true - same_pred),   # 같은 용접부인데 안 묶임 = 누수
        "over_merge": len(same_pred - same_true),    # 다른 용접부인데 묶임 = 층화 오차
        "n_groups": len({uf.find(i) for i in range(n)}),
        "true_groups": len(set(truth)),
    }


def _local_hashes(paths: list[Path], hash_size: int, *, use_clahe: bool) -> np.ndarray:
    """(N, W) uint64. 임의 hash_size 비교용 — 프로덕션 모듈(256bit 고정)을 건드리지 않는다."""
    import imagehash as ih

    from data.dedup.phash import HIGHFREQ_FACTOR
    from data.dedup.phash import preprocess as pre

    bits = hash_size * hash_size
    words = (bits + 63) // 64
    out = np.zeros((len(paths), words), dtype=np.uint64)
    for i, p in enumerate(paths):
        flat = ih.phash(
            pre(p, use_clahe=use_clahe), hash_size=hash_size, highfreq_factor=HIGHFREQ_FACTOR
        ).hash.ravel()
        packed = np.packbits(flat.astype(np.uint8))
        buf = np.zeros(words * 8, dtype=np.uint8)
        buf[: len(packed)] = packed
        out[i] = buf.view(np.uint64)
    return out


def compare_hash_sizes(paths: list[Path], truth: list[str], *, use_clahe: bool) -> None:
    """쟁점 1(64bit vs 256bit) 실측 — **과분리를 맞춘 뒤 과병합을 비교**한다.

    비트 수가 다르면 같은 절대 임계를 비교하는 것이 의미 없다. 손실 비대칭(§6-1)에 따라
    과분리(누수)를 기준선으로 고정하고 그 지점의 과병합으로 판별력을 비교한다.
    """
    print("\n" + "=" * 62)
    print("쟁점 1 실측 — hash_size 64bit vs 256bit (과분리 기준선 정렬 비교)")
    print("=" * 62)
    n = len(truth)
    same_true = {(i, j) for i in range(n) for j in range(i + 1, n) if truth[i] == truth[j]}

    for hash_size in (8, 16):
        packed = _local_hashes(paths, hash_size, use_clahe=use_clahe)
        bits = hash_size * hash_size
        print(f"\n  {bits}bit (hash_size={hash_size})  최대 임계 {bits}")
        rows = []
        for t in range(bits + 1):
            uf = UnionFind(n)
            for i, j, _ in iter_close_pairs(packed, t):
                uf.union(i, j)
            pred = {(i, j) for i in range(n) for j in range(i + 1, n) if uf.find(i) == uf.find(j)}
            under, over = len(same_true - pred), len(pred - same_true)
            rows.append((t, under, over))
            if under == 0:
                break
        for target in (60, 30, 10, 0):
            hit = next((r for r in rows if r[1] <= target), None)
            if hit:
                t, under, over = hit
                print(f"    과분리 ≤{target:3d} 최초 지점: t={t:3d} ({t/bits:.1%}) "
                      f"과분리={under:3d} 과병합={over:4d}")
            else:
                print(f"    과분리 ≤{target:3d} 도달 못 함 (최소 {min(r[1] for r in rows)})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--truth-from-filename", action="store_true")
    ap.add_argument("--max-threshold", type=int, default=THRESHOLD_CAP)
    ap.add_argument("--no-clahe", action="store_true",
                    help="열린질문 Q14 비교 경로 — 이중모드 형성 실패 시 1회 비교용")
    ap.add_argument("--compare-hash-sizes", action="store_true",
                    help="쟁점 1 실측 — 64bit vs 256bit 를 과분리 기준선 정렬로 비교")
    args = ap.parse_args()

    paths, hexes = collect(args.raw_root, use_clahe=not args.no_clahe)
    packed = pack_hashes(hexes)
    print(f"이미지 {len(paths)}장, 해시 {HASH_BITS}bit, "
          f"CLAHE {'off (Q14 비교)' if args.no_clahe else 'on'}\n")

    hist = distance_histogram(packed)
    nz = np.nonzero(hist)[0]
    print("거리 히스토그램 (전수·무편향 — 인덱스 산물이 아니다)")
    print(f"  거리 범위 {int(nz.min())}..{int(nz.max())}, 전체 쌍 {int(hist.sum()):,}")
    lo = hist[: args.max_threshold + 1]
    print("  d≤임계 구간 분포:")
    for d in range(0, args.max_threshold + 1, 4):
        c = int(lo[d : d + 4].sum())
        if c:
            print(f"    [{d:3d}, {min(d+4, args.max_threshold+1):3d})  {c:8,d}  {'#' * min(c // 2, 60)}")

    if not args.truth_from_filename:
        return 0

    truth = truth_groups(paths)
    print(f"\n정답 묶음 {len(set(truth))}개 (파일명 유래)")
    print(f"\n{'t':>4} {'묶음':>6} {'병합쌍':>7} {'과분리':>7} {'과병합':>7}  판정")
    print("-" * 52)
    first_clean = None
    for t in range(0, args.max_threshold + 1, 2):
        r = evaluate(packed, truth, t)
        verdict = "누수 있음" if r["under_merge"] else ("깨끗" if not r["over_merge"] else "과병합")
        if r["under_merge"] == 0 and first_clean is None:
            first_clean = t
            verdict += "  ← 과분리 0 최소 임계"
        print(f"{t:4d} {r['n_groups']:6d} {r['merged']:7d} {r['under_merge']:7d} "
              f"{r['over_merge']:7d}  {verdict}")

    print()
    if first_clean is None:
        print(f"과분리 0 을 달성하는 임계가 {args.max_threshold} 이하에 없다 — pHash 부적합 신호.")
        print("§6-4 절차대로 감독 에스컬레이션.")
    else:
        r = evaluate(packed, truth, first_clean)
        print(f"과분리 0 최소 임계 t = {first_clean}  (과병합 {r['over_merge']}쌍)")
        print("손실 비대칭(§6-1)에 따라 과분리 0 을 우선한다 — 과병합은 층화 밴드가 흡수한다.")

    if args.compare_hash_sizes:
        compare_hash_sizes(paths, truth, use_clahe=not args.no_clahe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
