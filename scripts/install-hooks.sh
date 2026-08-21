#!/usr/bin/env bash
# git 훅은 clone 으로 따라오지 않는다. 새 환경에서 1회 실행해라.
#   bash scripts/install-hooks.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
DEST="$(git rev-parse --git-common-dir)/hooks"
mkdir -p "$DEST"
for h in scripts/hooks/*; do
  cp "$h" "$DEST/$(basename "$h")"
  chmod +x "$DEST/$(basename "$h")"
  echo "설치: $DEST/$(basename "$h")"
done
echo "완료. worktree 전체에 적용된다(공용 hooks 디렉터리)."
