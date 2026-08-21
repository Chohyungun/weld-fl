"""RIAWELC 형상 픽스처 생성 — 파이프라인 전 구간 검증용.

**실데이터가 아니다.** `data/raw/riawelc/` 가 비어 있는 동안(RAR 미확보) ingest → convert →
dedup → split → 잠금 전 구간을 **실제 PNG 파일과 실제 pHash 로** 완주시키기 위한 픽스처다.
원본 구조(클래스 폴더 + 224×224 8bit PNG)와 중복 구조(같은 용접부 연속 촬영)를 재현한다.

RAR 이 도착하면 이 스크립트만 쓰지 않게 되고 어댑터·dedup·split 코드는 그대로다.

    uv run python scripts/make_riawelc_fixture.py --out <경로> --per-class 60
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

#: RIAWELC 원본 클래스 폴더명. 사상표 L3 (configs/label_map.yaml sources.riawelc) 가 해석한다.
CLASS_DIRS = ("CR", "PO", "LP", "ND")
SIZE = 224


def _weld_seam(rng: np.random.Generator) -> np.ndarray:
    """방사선 사진 흉내 — 균일한 배경 + 가로 용접 비드. 대비가 낮다(pHash 판별력 시험)."""
    y, x = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    base = 120.0 + 8.0 * np.sin(x / 37.0 + float(rng.uniform(0, 6.28)))
    seam_y = SIZE / 2 + float(rng.uniform(-12, 12))
    bead = 40.0 * np.exp(-((y - seam_y) ** 2) / (2 * 18.0**2))
    grain = rng.normal(0.0, 3.0, size=(SIZE, SIZE))
    return base + bead + grain


def _stamp_defect(img: np.ndarray, kind: str, rng: np.random.Generator) -> None:
    """클래스별로 다른 음영을 찍는다. 위치 라벨은 만들지 않는다 — RIAWELC 은 분류 패치다."""
    if kind == "ND":
        return
    cy = SIZE / 2 + float(rng.uniform(-8, 8))
    if kind == "PO":                                    # 기공 — 원형 음영 몇 개
        for _ in range(int(rng.integers(2, 6))):
            r = float(rng.uniform(3, 7))
            cx = float(rng.uniform(20, SIZE - 20))
            yy, xx = np.mgrid[0:SIZE, 0:SIZE]
            img -= 45.0 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r**2)))
    elif kind == "CR":                                  # 균열 — 얇고 긴 선
        x0 = float(rng.uniform(30, SIZE - 90))
        length = float(rng.uniform(40, 80))
        angle = float(rng.uniform(-0.4, 0.4))
        for t in np.linspace(0, length, int(length * 3)):
            px = int(x0 + t * math.cos(angle))
            py = int(cy + t * math.sin(angle) + rng.normal(0, 0.6))
            if 0 <= px < SIZE and 0 <= py < SIZE:
                img[py, px] -= 60.0
    else:                                               # LP 용입불량 — 심 따라 연속 밴드
        yy = np.arange(SIZE)
        band = 35.0 * np.exp(-((yy - cy) ** 2) / (2 * 2.5**2))
        img -= band[:, None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="raw 루트 (여기에 riawelc/ 를 만든다)")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--dup-ratio", type=float, default=0.25,
                    help="같은 용접부를 2~4프레임 연속 촬영한 것으로 만들 비율")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = args.out / "riawelc"
    total = 0
    for kind in CLASS_DIRS:
        d = root / kind
        d.mkdir(parents=True, exist_ok=True)
        remaining = args.per_class
        weld = 0
        while remaining > 0:
            n_frames = (
                int(rng.integers(2, 5)) if rng.random() < args.dup_ratio else 1
            )
            n_frames = min(n_frames, remaining)
            remaining -= n_frames

            canvas = _weld_seam(rng)
            _stamp_defect(canvas, kind, rng)
            for f in range(n_frames):
                # 연속 촬영 = 같은 장면 + 노출 드리프트 + 미세 노이즈. 기하 변형은 주지 않는다
                # (pHash 는 평행이동 불변이 아니므로, 이동을 주면 원본과 다른 현상을 시험하게 된다)
                frame = canvas * float(rng.uniform(0.985, 1.015)) + rng.normal(0, 1.2, canvas.shape)
                arr = np.clip(frame, 0, 255).astype(np.uint8)
                Image.fromarray(arr, mode="L").save(d / f"{kind}_w{weld:04d}_f{f}.png")
                total += 1
            weld += 1

    print(f"픽스처 {total}장 → {root}")
    print(f"클래스 {list(CLASS_DIRS)} / {SIZE}x{SIZE} 8bit PNG / dup_ratio={args.dup_ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
