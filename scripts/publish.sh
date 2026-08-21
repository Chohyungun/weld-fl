#!/usr/bin/env bash
# =============================================================================
# publish.sh — 기존 이력에서 AI 어시스턴트 트레일러를 제거하고 GitHub 원격에 푸시한다.
#
#   저장소 생성은 하지 않는다. GitHub에서 직접 만든 빈 저장소의 URL을 인자로 넘겨라.
#
#   사용:
#     bash scripts/publish.sh <원격URL>            # 미리보기 (아무것도 바꾸지 않음)
#     bash scripts/publish.sh <원격URL> --apply    # 실제 수행
#
#   예:
#     bash scripts/publish.sh git@github.com:USER/weld-fl.git
#     bash scripts/publish.sh https://github.com/USER/weld-fl.git --apply
# =============================================================================
set -euo pipefail

REMOTE_URL="${1:-}"
APPLY="${2:-}"

die() { printf '\n[중단] %s\n' "$1" >&2; exit 1; }
ok()  { printf '  [OK] %s\n' "$1"; }

[ -n "$REMOTE_URL" ] || die "원격 URL을 인자로 넘겨라.  사용법: bash scripts/publish.sh <원격URL> [--apply]"

cd "$(git rev-parse --show-toplevel)"
printf '저장소: %s\n원격:   %s\n' "$(pwd)" "$REMOTE_URL"

# ---------------------------------------------------------------------------
# 1. 사전 점검 — 하나라도 걸리면 아무것도 하지 않는다
# ---------------------------------------------------------------------------
printf '\n[1/5] 사전 점검\n'

BR="$(git rev-parse --abbrev-ref HEAD)"
[ "$BR" = "main" ] || die "main 에서 실행해라 (현재: $BR)"
ok "브랜치 main"

[ -z "$(git status --porcelain)" ] || die "커밋되지 않은 변경이 있다. 먼저 정리해라."
ok "작업 트리 깨끗함"

# 이력 재작성은 main 으로만 한정한다(아래 4단계). 따라서 wt/* 의 ref 와 워크트리는
# 건드리지 않으므로 작업 중인 트랙이 있어도 안전하다. 다만 무엇이 푸시에서 빠지는지는
# 반드시 알려준다 — 조용히 빠지는 것이 사고다.
UNMERGED=""
for b in $(git for-each-ref --format='%(refname:short)' refs/heads/wt/ 2>/dev/null || true); do
  n="$(git rev-list --count "main..$b")"
  [ "$n" -gt 0 ] && UNMERGED="$UNMERGED\n    $b ($n 커밋이 main 에 없음)"
done
if [ -n "$UNMERGED" ]; then
  printf '  [주의] 아래 트랙 커밋은 이번 푸시에 포함되지 않는다:%b\n' "$UNMERGED"
  printf '         (squash 로 이미 반영된 경우도 여기 표시된다. 내용 반영 여부는 감독이 판단한다)\n'
  printf '         main 재작성 후 이 브랜치들은 main 과 갈라진다. 푸시 후 재동기화할 것.\n'
else
  ok "wt/* 전부 main 에 머지됨"
fi

# 데이터가 실수로 추적되고 있지 않은지 — 반출 금지 원칙
LEAK="$(git ls-files | grep -E '^(data/(raw|interim|processed)/|outputs/|checkpoints/|mlruns/)' || true)"
[ -z "$LEAK" ] || die "추적되면 안 되는 데이터 파일이 있다:
$LEAK"
ok "데이터 파일 미추적 (반출 금지 원칙)"

BIG="$(git ls-files -z | xargs -0 -I{} sh -c 'test -f "{}" && find "{}" -size +20M -print' 2>/dev/null || true)"
[ -z "$BIG" ] || printf '  [경고] 20MB 초과 추적 파일:\n%s\n' "$BIG"

# ---------------------------------------------------------------------------
# 2. 제거 대상 확인
# ---------------------------------------------------------------------------
printf '\n[2/5] 트레일러 제거 대상\n'
PATTERN='Co-[Aa]uthored-[Bb]y:.*(Claude|Anthropic|noreply@anthropic)|Generated with \[?Claude|Claude-Session:|https://claude\.ai/code/'
HITS="$(git log main --format='%H' | while read -r h; do
  if git log -1 --format='%B' "$h" | grep -qE "$PATTERN"; then git log -1 --format='  %h %s' "$h"; fi
done)"
if [ -z "$HITS" ]; then
  ok "제거할 트레일러 없음 — 이력 재작성을 건너뛴다"
  NEED_REWRITE=0
else
  printf '%s\n' "$HITS"
  printf '  총 %s개 커밋\n' "$(printf '%s\n' "$HITS" | wc -l | tr -d ' ')"
  NEED_REWRITE=1
fi

# ---------------------------------------------------------------------------
# 미리보기 종료
# ---------------------------------------------------------------------------
if [ "$APPLY" != "--apply" ]; then
  printf '\n───────────────────────────────────────────────\n'
  printf '미리보기입니다. 아무것도 바뀌지 않았습니다.\n'
  printf '실제로 수행하려면 뒤에 --apply 를 붙이세요:\n\n'
  printf '  bash scripts/publish.sh %s --apply\n\n' "$REMOTE_URL"
  exit 0
fi

# ---------------------------------------------------------------------------
# 3. 백업 — 되돌릴 수 있게
# ---------------------------------------------------------------------------
printf '\n[3/5] 백업\n'
BK="backup/pre-publish-$(date +%Y%m%d-%H%M%S)"
git branch "$BK" main
ok "백업 브랜치 $BK (문제 시 git reset --hard $BK 로 복구)"
git log --all --format='%H %s' > .git/pre-publish-shas.txt
ok "재작성 전 SHA 목록 .git/pre-publish-shas.txt"

# ---------------------------------------------------------------------------
# 4. 이력에서 트레일러 제거
# ---------------------------------------------------------------------------
printf '\n[4/5] 이력 재작성\n'
if [ "$NEED_REWRITE" -eq 1 ]; then
  FILTER_BRANCH_SQUELCH_WARNING=1 \
  git filter-branch -f --msg-filter '
    grep -v -E -i \
      -e "^[[:space:]]*Co-[Aa]uthored-[Bb]y:.*(Claude|Anthropic|noreply@anthropic)" \
      -e "^[[:space:]]*(🤖[[:space:]]*)?Generated with \[?Claude" \
      -e "^[[:space:]]*Claude-Session:" \
      -e "^[[:space:]]*https://claude\.ai/code/" \
    | awk "BEGIN{n=0} {l[NR]=\$0} END{e=NR; while(e>0 && l[e] ~ /^[[:space:]]*\$/) e--; for(i=1;i<=e;i++) print l[i]}"
  ' -- main
  ok "트레일러 제거 완료 (main 한정 — wt/* 는 건드리지 않는다)"

  REMAIN="$(git log main --format='%B' | grep -cE "$PATTERN" || true)"
  [ "$REMAIN" = "0" ] || die "재작성 후에도 트레일러가 $REMAIN 줄 남아 있다. 푸시하지 않는다."
  ok "검증: 잔존 0건"

  rm -rf .git/refs/original
  git reflog expire --expire=now --all
  git gc --prune=now --quiet
  ok "정리 완료"
else
  ok "재작성 불필요 — 건너뜀"
fi

# ---------------------------------------------------------------------------
# 5. 원격 등록 + 푸시
# ---------------------------------------------------------------------------
printf '\n[5/5] 원격 등록 및 푸시\n'
if git remote | grep -qx origin; then
  git remote set-url origin "$REMOTE_URL"; ok "origin URL 갱신"
else
  git remote add origin "$REMOTE_URL"; ok "origin 등록"
fi

git push -u origin main
ok "main 푸시 완료"

# ---------------------------------------------------------------------------
# 6. wt/* 재동기화 — 이력 재작성으로 끊긴 조상 관계 복구
#
#    이 단계를 사람 손에 맡겼다가 2026-08-17 에 사고가 났다. 트랙들이 끊긴 브랜치
#    위에서 계속 커밋해 공통 조상 없는 상태가 됐다. 자동화하되, 고유 작업이 있는
#    브랜치는 건드리지 않고 복구 명령만 출력한다.
# ---------------------------------------------------------------------------
if [ "$NEED_REWRITE" -eq 1 ]; then
  printf '\n[6/6] 트랙 브랜치 재동기화\n'
  MANUAL=""
  for b in $(git for-each-ref --format='%(refname:short)' refs/heads/wt/ 2>/dev/null || true); do
    # 워크트리에 체크아웃돼 있으면 ref 를 강제로 못 옮긴다 — 명령만 안내한다
    WTPATH="$(git worktree list --porcelain | awk -v br="refs/heads/$b" '
      /^worktree /{p=$2} /^branch /{ if($2==br) print p }')"
    if [ -n "$(git diff "main..$b" --name-only)" ]; then
      MANUAL="$MANUAL\n    $b — 고유 내용 있음. 백업 후 수동 복구 필요"
      continue
    fi
    if [ -n "$WTPATH" ]; then
      if git -C "$WTPATH" diff --quiet && git -C "$WTPATH" diff --cached --quiet; then
        git -C "$WTPATH" checkout -q -B "$b" main && ok "$b 재동기화 (내용 동일)"
      else
        MANUAL="$MANUAL\n    $b — 워크트리에 미커밋 변경 있음"
      fi
    else
      git branch -f "$b" main && ok "$b 재동기화 (내용 동일)"
    fi
  done
  if [ -n "$MANUAL" ]; then
    printf '  [수동 처리 필요]%b\n' "$MANUAL"
    printf '  → docs/dev_log/*/dispatch_전트랙_브랜치복구.md 절차를 따를 것\n'
  fi
fi

printf '\n───────────────────────────────────────────────\n'
printf '완료.  %s\n' "$REMOTE_URL"
printf '\n남은 것:\n'
printf '  · 트랙 브랜치(wt/*)는 푸시하지 않았다. 필요하면: git push origin wt/A ...\n'
printf '  · 백업 브랜치 %s 는 로컬에만 있다. 확인 후 지워라: git branch -D %s\n' "$BK" "$BK"
printf '  · 이력 재작성으로 SHA 가 바뀌었다. 게이트 기록의 커밋 해시 참조는\n'
printf '    .git/pre-publish-shas.txt 로 대조할 수 있다.\n\n'
