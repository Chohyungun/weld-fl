"""계약 #4 JSON Schema 생성. 손으로 고치지 말고 이 스크립트를 돌린다.

    uv run python scripts/gen_prediction_schema.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.schema import json_schema

OUT = Path("evaluation/prediction.schema.json")


def main() -> None:
    # newline="\n" 고정 — Windows 기본 CRLF 로 쓰면 플랫폼마다 파일 해시가 갈린다.
    # 스냅샷 해시는 논문에 싣는 값이라, 줄끝 하나로 재현 검증이 거짓 실패한다.
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(json_schema(), ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
