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
  CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/knuvi/miniconda3)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null && conda activate pancrafter
fi
[ $# -ge 1 ] || { echo "usage: $0 <실행명> [...]" >&2; exit 0; }
LOG="$REPO/work_dir/gspread_upload.log"
{
  echo "--- $(date -Iseconds)  $* ---"
  # 논문 세트(.mat FR 20장) 평가 — 시트의 FR 열. run 의 센서에 맞는 h5 가 없으면 스크립트가 건너뛴다 (KNOWN_ISSUES F-2).
  python tools/eval_fr_paperset.py "$@" 2>&1 | grep -v Warning || true
  # 아키텍처 고정 다중 데이터셋 캠페인: WV3 학습 run 이 끝나면 WV2 zero-shot run 도 만들어 함께 올린다
  ZS=()
  for t in "$@"; do
    case "$t" in ARCH_W168_D123_DUAL_WV3_*|S1_T05_W168_D123_DUAL)
      if [ -f "$REPO/data/PanCollection/WV2/reduced_examples_h5/test_wv2_multiExm1_pan.h5" ]; then
        python tools/make_zeroshot_run.py --run "$t" --sensor wv2 2>&1 | grep -v Warning && ZS+=("${t}_zs_wv2") || echo "[upload] zero-shot 실패: $t"
      else
        echo "[upload] WV2 데이터 없음 — zero-shot 생략 (tools/setup_wv2.py)"
      fi;;
    esac
  done
  set -- "$@" "${ZS[@]}"
  # 구글 API 가 간헐적으로 503 을 낸다. 몇 번 다시 시도한다.
  for k in 1 2 3; do
    python gspread/gspread_upload.py "$@" 2>&1 && break
    echo "[upload] 시도 $k 실패 — 60초 후 재시도"; sleep 60
  done
} >> "$LOG" || echo "[upload] 실패 — $LOG 참고"
tail -2 "$LOG" | grep -E "업로드 완료|실패" || true
exit 0
