#!/usr/bin/env bash
# 새 서버에 clone 한 뒤 한 번 실행한다.
# config/*.yaml 에 박힌 절대경로를 현재 저장소 위치로 바꾼다.
#
#   ./tools/setup_paths.sh                    # 확인만 (dry-run)
#   ./tools/setup_paths.sh --apply            # 실제 반영
#
# 형제 저장소(CANConv, DLPan-Toolbox)는 코드를 고치지 않고 환경변수로 지정한다.
#   export PANCRAFTER_CANCONV=/path/to/CANConv
#   export PANCRAFTER_DLPAN=/path/to/DLPan-Toolbox
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD="/home/knuvi/Desktop/song/PAN-Crafter"     # 원 개발 환경의 저장소 루트
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

if [ "$REPO" = "$OLD" ]; then
  echo "[setup] 저장소 위치가 원본과 같다 ($REPO). 바꿀 것이 없다."
  exit 0
fi

N=$(grep -rl "$OLD" "$REPO"/config/*.yaml 2>/dev/null | wc -l)
C=$(grep -rho "$OLD" "$REPO"/config/*.yaml 2>/dev/null | wc -l)
echo "[setup] 저장소 루트 : $REPO"
echo "[setup] 치환 대상   : $OLD -> $REPO"
echo "[setup] config 파일 $N 개, 총 $C 곳"

if [ "$APPLY" -eq 0 ]; then
  echo "[setup] dry-run 이다. 실제로 바꾸려면: ./tools/setup_paths.sh --apply"
  exit 0
fi

grep -rl "$OLD" "$REPO"/config/*.yaml 2>/dev/null | while read -r f; do
  sed -i "s|$OLD|$REPO|g" "$f"
done
echo "[setup] 반영 완료. 남은 항목: $(grep -rho "$OLD" "$REPO"/config/*.yaml 2>/dev/null | wc -l) 곳"

# 데이터셋 존재 확인
D="$REPO/data/PanCollection/WV3/train_wv3.h5"
if [ -e "$D" ]; then echo "[setup] 데이터 확인 OK : $D"
else
  echo "[setup] !! 데이터 없음 : $D"
  echo "[setup]    PanCollection 을 data/PanCollection/ 아래 두거나 심볼릭 링크를 건다."
fi
for v in PANCRAFTER_CANCONV PANCRAFTER_DLPAN; do
  eval "p=\${$v:-}"
  if [ -n "$p" ] && [ -d "$p" ]; then echo "[setup] $v = $p (OK)"
  else echo "[setup] !! $v 미설정 또는 경로 없음 — 평가 도구(tools/eval_dlpan*.py, collect_*.py)가 동작하지 않는다"; fi
done
