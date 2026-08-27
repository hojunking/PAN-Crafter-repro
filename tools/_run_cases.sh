#!/usr/bin/env bash
# 경량화 Tier A + M1 — 장애 대비판. 명세: research_log/lightweight_case_specs_v1.md
#   setsid nohup ./tools/_run_cases.sh > work_dir/cases_chain.log 2>&1 < /dev/null &
#
# 장애 대응
#   - 완료 판정: results/reduced_best_hqnr.mat. 재기동하면 완료분은 건너뛰되 시트 업로드는 보장
#   - 실패(비정상 종료) 시: 최신 epoch-* 체크포인트가 있으면 --resume 으로 1회 재시도
#   - exit 3 (NaN/Inf 손실) 은 재시도하지 않는다 — 재개해도 다시 발산한다
#   - 체인 자체가 죽는 경우는 tools/_watchdog.sh(cron) 가 재기동한다
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
# 수동 재기동과 cron 감시자가 겹쳐도 체인은 하나만 뜬다
exec 8>"$REPO/work_dir/.cases_chain.lock"; flock -n 8 || { echo "[cases] 이미 실행 중 — 종료"; exit 0; }
# 주의: fd 8 이 자식(run.sh -> python)에 상속되면, 체인 셸이 죽어도 자식이 잠금을
# 쥐고 있어 재기동이 막힌다. 실제로 겪었다. 자식 호출마다 8>&- 로 닫는다.
ORDER=(c0_hqnr c1_nopan c4_noattn c3b_btl c3e_enc c2_encbtl m1_single
       c5_c2d124 c6_c4d124 c7_c1w96 c8_c4w96)
CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/knuvi/miniconda3)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate pancrafter
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
running(){ ps -eo args | grep -v grep | grep -qE "^python .*main\.py --config"; }
while running; do echo "[cases] 대기 $(date -Iseconds)"; sleep 300; done
echo "[cases] 시작 $(date -Iseconds)  총 ${#ORDER[@]}개"

latest_ckpt(){ ls -d "work_dir/$1"/epoch-* 2>/dev/null | sort -V | tail -1; }

i=0
for TAG in "${ORDER[@]}"; do
    i=$((i+1))
    if [ -f "work_dir/${TAG}/results/reduced_best_hqnr.mat" ]; then
        echo "[cases] ($i/${#ORDER[@]}) $TAG 완료됨 — 업로드만 확인"
        ./tools/_upload.sh "$TAG"; continue
    fi
    echo "[cases] ($i/${#ORDER[@]}) === $TAG 시작 $(date -Iseconds) ==="
    set +e; ./tools/run.sh "$TAG" 8>&-; rc=$?; set -e
    if [ $rc -eq 3 ]; then
        echo "[cases] ($i/${#ORDER[@]}) $TAG NaN 중단 — 재시도하지 않는다"
    elif [ $rc -ne 0 ]; then
        CK=$(latest_ckpt "$TAG")
        if [ -n "$CK" ]; then
            echo "[cases] ($i/${#ORDER[@]}) $TAG 실패(rc=$rc) — $CK 에서 재개 재시도"
            set +e; ./tools/run.sh "$TAG" --resume "$CK" 8>&-; rc=$?; set -e
        else
            echo "[cases] ($i/${#ORDER[@]}) $TAG 실패(rc=$rc) — 체크포인트 없음, 처음부터 재시도"
            set +e; ./tools/run.sh "$TAG" 8>&-; rc=$?; set -e
        fi
    fi
    if [ $rc -eq 0 ] && [ -f "work_dir/${TAG}/results/reduced_best_hqnr.mat" ]; then
        echo "[cases] ($i/${#ORDER[@]}) $TAG 완료 $(date -Iseconds)"
        python tools/eval_dlpan.py "work_dir/$TAG/results/reduced_best_hqnr.mat" --preset wv3 2>/dev/null | grep "$TAG" || true
        ./tools/_upload.sh "$TAG"
    else
        echo "[cases] ($i/${#ORDER[@]}) FAILED $TAG (rc=$rc) $(date -Iseconds)"
        FAILED="${FAILED:-}$TAG "
    fi
done
# DONE 은 감시자의 종료 신호다. 실패가 있어도 재기동 루프를 막기 위해 DONE 은 찍되
# 실패 목록을 함께 남긴다 — 사람이 로그만 보고 정상 완료로 오인하지 않게.
echo "[cases] DONE $(date -Iseconds)${FAILED:+  !! 실패: $FAILED}"
