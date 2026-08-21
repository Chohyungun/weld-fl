"""실행 환경 실측 확인 — sm_120(Blackwell) / cu128 검증용.

docs/개발규약.md 실행환경 절의 요구사항.
설명서를 믿지 않고 실제로 커널을 돌려 확인한다.

    uv run python scripts/check_env.py
"""

import platform
import sys

import torch


def main() -> int:
    print(f"python            : {sys.version.split()[0]} ({platform.machine()})")
    print(f"torch             : {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cuda available    : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("FAIL: CUDA 사용 불가")
        return 1

    idx = torch.cuda.current_device()
    cap = torch.cuda.get_device_capability(idx)
    props = torch.cuda.get_device_properties(idx)
    print(f"device            : {props.name}")
    print(f"compute capability: {cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})")
    print(f"total memory      : {props.total_memory / 1024**3:.1f} GiB")
    print(f"arch list         : {torch.cuda.get_arch_list()}")

    # 실제 커널 실행 — 아키텍처 미지원이면 여기서 터진다.
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    c = a @ b
    torch.cuda.synchronize()
    ref = (a.cpu() @ b.cpu())
    err = (c.cpu() - ref).abs().max().item()
    print(f"fp32 matmul       : ok, max abs err = {err:.3e}")

    # AMP 경로도 확인 (학습은 전부 AMP)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        d = a @ b
    torch.cuda.synchronize()
    print(f"bf16 autocast     : ok, dtype = {d.dtype}")

    print(f"peak allocated    : {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
