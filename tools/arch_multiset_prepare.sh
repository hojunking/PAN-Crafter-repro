#!/usr/bin/env bash
# 아키텍처 고정(S1_T05_W168_D123_DUAL 구조) 다중 데이터셋 3-seed 캠페인 — 서버마다 이 한 줄로 준비하고 기동한다.
#
#   ./tools/arch_multiset_prepare.sh                # 준비 + 체인 기동 (48h 마감)
#   ./tools/arch_multiset_prepare.sh --no-start     # 준비만
#   ./tools/arch_multiset_prepare.sh --no-download  # Drive 시도 없이 (이미 받은 .mat 만 사용; 속도제한 시)
#
# 하는 일:
#   1. QB full-res lpan 복구 (F-1: 배포 lpan 손상) → data/PanCollection/QB/full_examples_h5_repaired/
#   2. WV2 zero-shot 데이터 (없으면 PanCollection Drive 에서 H5 를 받아 lpan 생성 — tools/setup_wv2.py)
#   3. 센서별 논문 FR 세트 h5 (tools/build_paperset_all.sh: wv3 qb gf2 wv2)
#   4. 지표 이식 검사 (tools/verify_metrics.py) · config 8벌 smoke (tools/smoke_cases.py)
#   5. 체인 기동: config/queues/arch_w168_multiset_3seed.txt (QB×3 → GF2×3 → WV3 seed 1234·7777).
#      WV3 seed 2025 는 기존 S1_T05_W168_D123_DUAL 을 그대로 쓴다 (없는 서버는 큐에 추가할 것).
#      run 이 끝날 때마다 tools/_upload.sh 가 논문 세트 평가 → 시트 업로드, WV3 run 은 WV2 zero-shot 까지 만든다.
# 전제: 지표 v2 준비(tools/metric_v2_prepare.sh)가 끝나 있고, gspread/server.txt · PANCRAFTER_DLPAN 이 있다.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
START=1; HOURS=48; NODL=""
while [ $# -gt 0 ]; do case "$1" in --no-start) START=0;; --no-download) NODL="--no-download";; --hours) HOURS="$2"; shift;; *) echo "unknown arg $1" >&2; exit 2;; esac; shift; done
# (tools/setup_paths.sh 는 대화형 sourcing 용이라 여기서는 부르지 않는다 — set -e 아래에서 스크립트를 끝내 버린다)
: "${PANCRAFTER_DLPAN:=/home/knuvi/Desktop/song/DLPan-Toolbox}"; export PANCRAFTER_DLPAN
PY="${PYTHON:-python}"; command -v "$PY" >/dev/null || PY=/home/knuvi/miniconda3/envs/pancrafter/bin/python
for d in QB GF2; do
  for f in train_${d,,}.h5 train_${d,,}_pan.h5 valid_${d,,}.h5 valid_${d,,}_pan.h5; do
    [ -f "data/PanCollection/$d/$f" ] || { echo "!! data/PanCollection/$d/$f 없음 — SETUP.md §4 (PanCollection + pan_h5.zip)"; exit 1; }
  done
done
echo "[arch] 1/5 QB full-res lpan 복구"
[ -f data/PanCollection/QB/full_examples_h5_repaired/test_qb_OrigScale_multiExm1_pan.h5 ] && echo "  있음" || "$PY" tools/repair_lpan.py --sensor qb
echo "[arch] 2/5 WV2 zero-shot 데이터"
if [ -f data/PanCollection/WV2/reduced_examples_h5/test_wv2_multiExm1_pan.h5 ] && [ -f data/PanCollection/WV2/full_examples_h5/test_wv2_OrigScale_multiExm1_pan.h5 ]; then
  echo "  있음"
else
  SRC=data/PanCollection/WV2/_src_h5; mkdir -p "$SRC"
  if [ ! -f "$SRC/test_wv2_multiExm1.h5" ] || [ ! -f "$SRC/test_wv2_OrigScale_multiExm1.h5" ]; then
    "$PY" -c "import gdown" 2>/dev/null || "$PY" -m pip install -q gdown
    ( cd "$SRC" && "$PY" -m gdown --folder "https://drive.google.com/drive/folders/1bpx99ewDRhe8jgAnr4XUMxvb2PqMxKk4" && \
      "$PY" -m gdown --folder "https://drive.google.com/drive/folders/1uIYVnceftT3KjM_WRLwMpy0IOgpNmMIv" ) || true
    find "$SRC" -mindepth 2 -name 'test_wv2_*.h5' -exec mv -n {} "$SRC"/ \; 2>/dev/null || true
  fi
  "$PY" tools/setup_wv2.py --src "$(cd "$SRC" && pwd)" || echo "  !! WV2 준비 실패 — zero-shot 은 생략된다 (학습은 영향 없음)"
fi
echo "[arch] 3/5 논문 FR 세트 (wv3 qb gf2 wv2)"; PYTHON="$PY" ./tools/build_paperset_all.sh wv3 qb gf2 wv2 $NODL
echo "[arch] 4/5 지표 이식 검사 + smoke"; "$PY" tools/verify_metrics.py
"$PY" tools/smoke_cases.py $(grep -v '^#' config/queues/arch_w168_multiset_3seed.txt | grep -v '^$' | tr '\n' ' ')
if [ "$START" = 1 ]; then
  echo "[arch] 5/5 체인 기동 (${HOURS}h)"; ./tools/campaign_start.sh --queue config/queues/arch_w168_multiset_3seed.txt --hours "$HOURS" --label arch-multiset
else
  echo "[arch] 준비 완료 — 기동: ./tools/campaign_start.sh --queue config/queues/arch_w168_multiset_3seed.txt --hours $HOURS --label arch-multiset"
fi
