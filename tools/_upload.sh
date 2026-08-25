#!/usr/bin/env bash
# 실행 하나가 끝나면 그 결과만 구글시트로 올린다. 체인 스크립트에서 부른다.
#
#   ./tools/_upload.sh <실행명> [실행명 ...]
#
# 실패해도 0 을 돌려준다 — 업로드 실패로 학습 체인이 멈추면 안 된다.
# 전체를 다시 정렬하고 싶으면 gspread_upload.py 를 --replace 로 직접 부를 것.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
# 체인 밖에서 단독 실행될 수도 있으므로 환경을 직접 잡는다
if [ "${CONDA_DEFAULT_ENV:-}" != "pancrafter" ]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate pancrafter
fi
[ $# -ge 1 ] || { echo "usage: $0 <실행명> [...]" >&2; exit 0; }
LOG="$REPO/work_dir/gspread_upload.log"
{
  echo "--- $(date -Iseconds)  $* ---"
  python gspread/gspread_upload.py "$@" 2>&1
} >> "$LOG" || echo "[upload] 실패 — $LOG 참고"
tail -2 "$LOG" | grep -E "업로드 완료|실패" || true
exit 0
