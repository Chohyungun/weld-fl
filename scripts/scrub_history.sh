#!/usr/bin/env bash
# =============================================================================
# scrub_history.sh — 공개 저장소 이력에서 비공개 대상을 제거한다.
#
#   HEAD 에서 지우는 것만으로는 부족하다. 이미 푸시된 커밋에 남아 있으면
#   누구나 이력을 뒤져 꺼낼 수 있다. main 이력 전체를 다시 쓴다.
#
#   대상 (공개 저장소 정비 방침)
#     · 회의록·액션아이템 파일   — 팀 내부 논의
#     · 제3자 실명               — 동의 없이 올라간 협의 예정 인물
#     · 개인 이메일 주소         — 선행연구 저자 연락처
#
#   사용:  bash scripts/scrub_history.sh          미리보기
#          bash scripts/scrub_history.sh --apply  실제 수행
#
#   재작성 범위는 main 뿐이다. wt/* 는 건드리지 않는다.
# =============================================================================
set -euo pipefail

APPLY="${1:-}"
cd "$(git rev-parse --show-toplevel)"

die(){ printf '\n[중단] %s\n' "$1" >&2; exit 1; }
ok(){ printf '  [OK] %s\n' "$1"; }

NAME='이수동'
MAIL='cassunqs@gmail\.com'
PATHS='docs/회의록_2026-08-09_킥오프.md docs/회의록_2026-08-16_연구설계리뷰.md docs/액션아이템.md'

[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || die "main 에서 실행해라"
[ -z "$(git status --porcelain)" ] || die "커밋되지 않은 변경이 있다"

printf '검사 대상: main (%s 커밋)\n\n' "$(git rev-list --count main)"

printf '[1/3] 이력에 남아 있는 것\n'
HITF=0
for p in $PATHS; do
  n="$(git log main --oneline -- "$p" | wc -l | tr -d ' ')"
  [ "$n" -gt 0 ] && { printf '  파일 %s — %s개 커밋\n' "$p" "$n"; HITF=1; }
done
HITN="$(git grep -l "$NAME" $(git rev-list main) -- docs 2>/dev/null | wc -l | tr -d ' ')"
HITM="$(git grep -lE "$MAIL" $(git rev-list main) -- docs 2>/dev/null | wc -l | tr -d ' ')"
printf '  실명 %s — %s개 (커밋×파일)\n' "$NAME" "$HITN"
printf '  이메일 — %s개 (커밋×파일)\n' "$HITM"

if [ "$HITF" -eq 0 ] && [ "$HITN" = "0" ] && [ "$HITM" = "0" ]; then
  ok "이력이 이미 깨끗하다. 할 일 없음"; exit 0
fi

if [ "$APPLY" != "--apply" ]; then
  printf '\n미리보기입니다. 실제 수행: bash scripts/scrub_history.sh --apply\n\n'
  exit 0
fi

printf '\n[2/3] 백업\n'
BK="backup/pre-scrub-$(git rev-parse --short main)"
git branch -f "$BK" main
ok "백업 브랜치 $BK"

printf '\n[3/3] 이력 재작성 (main 한정)\n'
export SCRUB_PATHS="$PATHS"
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --prune-empty \
  --tree-filter '
    for p in $SCRUB_PATHS; do rm -f "$p"; done
    find . -name "*.md" -not -path "./.git/*" -print0 2>/dev/null | while IFS= read -r -d "" f; do
      sed -i "s/이수동 교수님/외부 도메인 전문가(협의 예정)/g; s/이수동 교수/외부 도메인 전문가(협의 예정)/g; s/저자 cassunqs@gmail\.com 메일 발송/저자에게 메일 발송/g; s/cassunqs@gmail\.com//g" "$f" 2>/dev/null || true
    done
  ' -- main

ok "재작성 완료"

REM_N="$(git grep -l "$NAME" $(git rev-list main) -- docs 2>/dev/null | wc -l | tr -d ' ')"
REM_M="$(git grep -lE "$MAIL" $(git rev-list main) -- docs 2>/dev/null | wc -l | tr -d ' ')"
REM_F=0
for p in $PATHS; do
  [ "$(git log main --oneline -- "$p" | wc -l | tr -d ' ')" -gt 0 ] && REM_F=1
done
[ "$REM_N" = "0" ] && [ "$REM_M" = "0" ] && [ "$REM_F" -eq 0 ] \
  || die "잔존이 있다 (실명 $REM_N / 메일 $REM_M / 파일 $REM_F). 푸시하지 마라."
ok "검증: 잔존 0건"

rm -rf .git/refs/original
git reflog expire --expire=now --all
git gc --prune=now --quiet
ok "정리 완료"

printf '\n[4/4] 트랙 브랜치 재동기화\n'
# 2026-08-17·18 에 두 번 같은 사고가 났다. main 이력을 다시 쓰면 wt/* 의 공통 조상이
# 끊기는데, 안내만 하고 사람 손에 맡겼더니 그대로 반복됐다. 실행 단계로 올린다.
for b in $(git for-each-ref --format='%(refname:short)' refs/heads/wt/ 2>/dev/null || true); do
  WT="$(git worktree list --porcelain | awk -v br="refs/heads/$b" '/^worktree /{p=$2} /^branch /{ if($2==br) print p }')"
  TARGET="${WT:-.}"
  UNIQ="$(git -C "$TARGET" diff main --name-status --diff-filter=A 2>/dev/null | grep -vcE '회의록|액션아이템' || true)"
  DIRTY="$(git -C "$TARGET" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  if [ "${UNIQ:-0}" -gt 0 ] || [ "${DIRTY:-0}" -gt 0 ]; then
    printf '  [수동] %s — 고유 파일 %s개 / 미커밋 %s건. 백업 후 소관 경로만 복원할 것\n' \
      "$b" "${UNIQ:-0}" "${DIRTY:-0}"
  elif [ -n "$WT" ]; then
    git -C "$WT" branch -f "backup/${b#wt/}-prescrub" "$b" 2>/dev/null || true
    git -C "$WT" checkout -q -B "$b" main && ok "$b 재생성"
  else
    git branch -f "$b" main && ok "$b 재생성"
  fi
done

printf '\n───────────────────────────────────────────────\n'
printf '이력 정리 완료. 원격 반영은 강제 푸시가 필요하다:\n\n'
printf '  git push --force-with-lease origin main\n\n'
printf '주의 — 이미 공개된 내용은 GitHub 캐시·포크·검색엔진에 남아 있을 수 있다.\n'
printf '이력 정리는 앞으로의 열람을 막을 뿐 이미 퍼진 사본까지 되돌리지 못한다.\n'
printf '백업 브랜치 %s 는 로컬에만 있다.\n\n' "$BK"
