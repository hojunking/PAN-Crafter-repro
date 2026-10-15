#!/usr/bin/env bash
# UVS-KD 캠페인 원샷 (s2 에서 실행). teacher 준비 → gate → cache → 큐 기동.
#   ./tools/uvs_prepare.sh [teacher=c0_hqnr] [hours=30]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
T="${1:-c0_hqnr}"; H="${2:-30}"
CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/knuvi/miniconda3)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate pancrafter
[ -d "work_dir/$T/best_hqnr" ] || { echo "teacher checkpoint 없음: work_dir/$T/best_hqnr" >&2; exit 1; }
[ -f outputs/global_shift_cache/wv3_train.csv ] || python tools/build_shift_cache.py      # audit pseudo-label·FR 비교 target
python tools/test_uvs.py
set +e; python tools/uvs_prepare_teacher.py --teacher "$T"; rc=$?; set -e
echo "[uvs] teacher gate rc=$rc (0=PASS; shift gate FAIL 이면 S0/M1/M2/M3 는 gate 가 닫는다)"
python tools/uvs_build_cache.py --teacher "$T"
python tools/smoke_cases.py $(grep -v '^#' config/queues/s2_uvs_kd.txt | tr '\n' ' ')
./tools/campaign_start.sh --queue config/queues/s2_uvs_kd.txt --hours "$H"
