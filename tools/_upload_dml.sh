#!/usr/bin/env bash
# DML 실행 하나가 끝나면 두 peer 를 바로 구글시트로 올린다.
#
#   ./tools/_upload_dml.sh dml_m0
#
# 업로더는 work_dir/<이름> 을 실행 단위로 본다. DML 은 work_dir/<이름>/peer{A,B} 로
# 중첩돼 있어 그대로는 실행명이 "peerA" 로 겹친다(M0/M1 이 충돌한다).
# 그래서 형제 위치에 <이름>_peerA / <이름>_peerB 심볼릭 링크를 만들어 올린다.
# 링크라 디스크를 쓰지 않고, 원본 디렉터리 구조도 그대로 둔다.
#
# 실패해도 0 을 돌려준다 — 업로드 실패로 체인이 멈추면 안 된다.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
[ $# -ge 1 ] || { echo "usage: $0 <실행명>" >&2; exit 0; }
TAG="$1"

for P in peerA peerB; do
  SRC="$REPO/work_dir/$TAG/$P"
  [ -f "$SRC/results/reduced_best_val.mat" ] || {
    echo "[upload] $TAG/$P — mat 이 없다. 건너뛴다"; continue; }
  # 업로더의 collect() 는 meta/config.yaml (없으면 config/<이름>.yaml) 을 요구한다.
  # main_dml 은 peer 별 meta 를 만들지 않으므로 여기서 합성한다. peerB 는 seed 를
  # seed_b 값으로 바꿔 넣어 시트의 seed 표기가 실제 초기화와 일치하게 한다.
  if [ ! -f "$SRC/meta/config.yaml" ]; then
    mkdir -p "$SRC/meta"
    if [ "$P" = "peerB" ]; then
      SB=$(grep -m1 '^seed_b:' "$REPO/config/$TAG.yaml" | awk '{print $2}')
      sed "s/^seed: .*/seed: ${SB:-2026}/" "$REPO/config/$TAG.yaml" > "$SRC/meta/config.yaml"
    else
      cp "$REPO/config/$TAG.yaml" "$SRC/meta/config.yaml"
    fi
  fi
  LINK="$REPO/work_dir/${TAG}_${P}"
  [ -L "$LINK" ] && rm -f "$LINK"
  ln -s "$SRC" "$LINK"
done

TARGETS=()
for P in peerA peerB; do
  [ -e "$REPO/work_dir/${TAG}_${P}" ] && TARGETS+=("${TAG}_${P}")
done
[ ${#TARGETS[@]} -gt 0 ] || { echo "[upload] 올릴 것이 없다"; exit 0; }

if [ "${CONDA_DEFAULT_ENV:-}" != "pancrafter" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null && conda activate pancrafter
fi
LOG="$REPO/work_dir/gspread_upload.log"
{
  echo "--- $(date -Iseconds)  ${TARGETS[*]} ---"
  python gspread/gspread_upload.py "${TARGETS[@]}" 2>&1
} >> "$LOG"
tail -3 "$LOG" | grep -E "업로드 완료|실패|Error" || echo "[upload] $LOG 참고"
exit 0
