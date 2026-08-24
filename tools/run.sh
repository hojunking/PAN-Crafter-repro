#!/usr/bin/env bash
# PAN-Crafter 학습 실행 헬퍼. 실행 조건을 work_dir/meta/ 에 함께 남긴다.
#
#   ./tools/run.sh wv3                 # 배포본 그대로 (baseline)
#   ./tools/run.sh wv3_fixed           # KNOWN_ISSUES.md A-1/A-2 적용
#   ./tools/run.sh wv3 --resume work_dir/wv3_baseline/checkpoint-20000
#   nohup ./tools/run.sh wv3 > /dev/null 2>&1 &
#
# 인자는 config/pancrafter_<이름>.yaml 의 <이름> 부분이다.
# 나머지 인자는 main.py 로 그대로 넘어간다.
set -euo pipefail

# ${1:?...} 는 쓰지 않는다. 메시지 안의 '}' 가 파라미터 확장을 끊어 인자가 오염된다.
if [ $# -lt 1 ]; then
  echo "usage: $0 {wv3|qb|gf2|wv3_fixed|qb_fixed|gf2_fixed} [extra main.py args]" >&2
  exit 1
fi
DS="$1"
shift

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# config/pancrafter_<이름>.yaml 을 먼저 찾고, 없으면 config/<이름>.yaml 을 쓴다
CFG="$REPO/config/pancrafter_${DS}.yaml"
[ -f "$CFG" ] || CFG="$REPO/config/${DS}.yaml"
[ -f "$CFG" ] || { echo "no such config: config/pancrafter_${DS}.yaml or config/${DS}.yaml" >&2; exit 1; }

WORK_DIR="$(grep -m1 '^work_dir:' "$CFG" | awk '{print $2}')"
META="$WORK_DIR/meta"
mkdir -p "$META"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pancrafter

cd "$REPO"

# --- 실행 조건 스냅샷 (results_log/CONVENTION.md 의 "설정값과 그 근거") ---
cp -f "$CFG" "$META/config.yaml"
{ echo "./tools/run.sh $DS $*"; echo "python -u main.py --config $CFG $*"; } > "$META/command.txt"
git rev-parse HEAD                      > "$META/git_commit.txt" 2>/dev/null || echo "n/a" > "$META/git_commit.txt"
git status --porcelain                  > "$META/git_status.txt" 2>/dev/null || true
git diff                                > "$META/git_diff.patch" 2>/dev/null || true
date -Iseconds                          > "$META/started_at.txt"
nvidia-smi                              > "$META/nvidia_smi.txt" 2>/dev/null || true
{ echo "python: $(python -V 2>&1)"
  python -c "import torch;print('torch:',torch.__version__,'cuda:',torch.version.cuda)" || true
  python -c "import torch;print('gpu:',torch.cuda.get_device_name(0))" 2>/dev/null || true
  echo "conda env: ${CONDA_DEFAULT_ENV:-?}"
  conda list --export || true
} > "$META/environment.txt" 2>&1 || true

echo "[run] dataset/variant : $DS"
echo "[run] config          : $CFG"
echo "[run] work_dir        : $WORK_DIR"
echo "[run] commit          : $(cut -c1-8 "$META/git_commit.txt")"
grep -E '^[[:space:]]+fix_(key_alias|local_attn):' "$CFG" | sed 's/^/[run] /' || true

trap 'date -Iseconds > "$META/finished_at.txt"' EXIT

python -u main.py --config "$CFG" "$@" 2>&1 | tee -a "$WORK_DIR/console.log"
