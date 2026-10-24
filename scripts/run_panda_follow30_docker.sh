#!/usr/bin/env bash
# One explicit command per local server. Never stop/remove/replace old work.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/run_panda_follow30_docker.sh s1|s3 (WV3, mature donor + positive-scale and PAN-difference Stage2, unlimited queue)' >&2
  exit 2
fi
panda_follow30_server="$1"
panda_follow30_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
panda_follow30_image="${PAN_FOLLOW30_IMAGE:-${PAN_SCHEDULE30_IMAGE:-${PAN_SCALE30_IMAGE:-${PAN_MATURE30_IMAGE:-${PAN_DIRECTSHIFT_IMAGE:-}}}}}"
panda_follow30_name="panda-follow30-hardonly-wv3-${panda_follow30_server}-v1"
panda_follow30_python="${PAN_HOST_PYTHON:-}"
cd "$panda_follow30_root"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -f gspread/server.txt || "$(tr -d '[:space:]' < gspread/server.txt)" != "$panda_follow30_server" ]]; then
  echo 'Local server marker must match requested s1/s3; nothing started.' >&2
  exit 1
fi
# Reuse the already-used s1 image (reuse the proven local CUDA build).
# Never pull/change environments implicitly or default to an incompatible cu118 image.
if [[ -z "$panda_follow30_image" ]]; then
  for panda_follow30_previous in "panda-next30-hardonly-wv3-${panda_follow30_server}-v1" "panda-schedule30-hardonly-wv3-${panda_follow30_server}-v1" "panda-scale30-hardonly-wv3-${panda_follow30_server}-v1" "panda-zero30-hardonly-wv3-${panda_follow30_server}-v1" "panda-mature30-hardonly-wv3-${panda_follow30_server}-v2" "panda-band30-wv3-${panda_follow30_server}-v1" "panda-warp30-wv3-${panda_follow30_server}-v1" "panda30r-wv3-${panda_follow30_server}" "panda40-wv3-${panda_follow30_server}"; do
    if docker container inspect "$panda_follow30_previous" >/dev/null 2>&1; then
      panda_follow30_image="$(docker inspect --format '{{.Image}}' "$panda_follow30_previous")"
      break
    fi
  done
fi
if [[ -z "$panda_follow30_image" ]]; then
  echo 'Set PAN_FOLLOW30_IMAGE to the same locally installed Docker image used by this server/Panda30R.' >&2
  exit 1
fi
command -v setsid >/dev/null
df -h "$panda_follow30_root/work_dir"
if [[ -z "$panda_follow30_python" ]]; then
  panda_follow30_base_python="$(command -v python || command -v python3)"
  panda_follow30_base_prefix="$("$panda_follow30_base_python" -c 'import sys; print(sys.base_prefix)')"
  for panda_follow30_candidate in "$panda_follow30_base_python" "$panda_follow30_base_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$panda_follow30_candidate" ]] && "$panda_follow30_candidate" -c 'import google.oauth2.service_account; import google.auth.transport.requests; import requests' >/dev/null 2>&1; then
      panda_follow30_python="$panda_follow30_candidate"
      break
    fi
  done
  if [[ -z "$panda_follow30_python" ]]; then
    echo 'Set PAN_HOST_PYTHON to the existing server uploader Python (google-auth and requests required). Old experiments are untouched.' >&2
    exit 1
  fi
fi
# Validate only local authentication material; never print key contents.
"$panda_follow30_python" -c 'import google.auth.transport.requests, requests; from google.oauth2 import service_account; service_account.Credentials.from_service_account_file("gspread/account.json", scopes=["https://www.googleapis.com/auth/spreadsheets"])'
docker image inspect "$panda_follow30_image" >/dev/null
if docker container inspect "$panda_follow30_name" >/dev/null 2>&1; then
  echo "Container $panda_follow30_name already exists; inspect it. Nothing removed or reset." >&2
  exit 1
fi
panda_follow30_dlpan_input="${PANCRAFTER_DLPAN:-$panda_follow30_root/../DLPan-Toolbox}"
if [[ ! -d "$panda_follow30_dlpan_input" ]]; then
  echo 'DLPan evaluator directory is unavailable; set PANCRAFTER_DLPAN to its existing local checkout.' >&2
  exit 1
fi
panda_follow30_dlpan="$(cd "$panda_follow30_dlpan_input" && pwd -P)"
test -f "$panda_follow30_dlpan/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py"
for panda_follow30_path in "$panda_follow30_root" "$panda_follow30_dlpan"; do
  if [[ "$panda_follow30_path" == *','* || "$panda_follow30_path" == *$'\n'* || "$panda_follow30_path" == *$'\r'* ]]; then
    echo 'Unsafe Docker mount delimiter' >&2
    exit 1
  fi
done
panda_follow30_files="$("$panda_follow30_python" -m panda_follow30 mounts)"
test -n "$panda_follow30_files"
panda_follow30_mounts=()
while IFS= read -r panda_follow30_file; do
  # Normally enforced by mounts as well; fail closed if its output is invalid.
  if [[ ! -f "$panda_follow30_file" || "$panda_follow30_file" == *','* || "$panda_follow30_file" == *$'\r'* ]]; then
    echo 'Invalid registered data mount target' >&2
    exit 1
  fi
  panda_follow30_mounts+=(--mount "type=bind,src=$panda_follow30_file,dst=$panda_follow30_file,readonly")
done <<< "$panda_follow30_files"
panda_follow30_common=(--user "$(id -u):$(id -g)" --env PYTHONDONTWRITEBYTECODE=1
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env MKL_THREADING_LAYER=GNU
  --env PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox
  --mount "type=bind,src=$panda_follow30_root,dst=/workspace,readonly"
  --mount "type=bind,src=$panda_follow30_dlpan,dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly"
  "${panda_follow30_mounts[@]}" --workdir /workspace)
# Read-only preflight with the actual image. No GPU/training/Sheet mutation.
docker run --rm "${panda_follow30_common[@]}" "$panda_follow30_image" \
  python -m panda_follow30 check --server "$panda_follow30_server"
"$panda_follow30_python" -c 'from directshift48.__main__ import require_idle_gpu; require_idle_gpu("cuda")'
panda_follow30_campaign_id="$("$panda_follow30_python" -c 'from panda_follow30.contract import CAMPAIGN_ID; print(CAMPAIGN_ID)')"
if [[ "$panda_follow30_campaign_id" != PANDA_S13_FOLLOW_HARDONLY_20261024_v1 ]]; then
  echo 'Invalid next30 campaign identity; nothing started.' >&2
  exit 1
fi
panda_follow30_campaign_root="$panda_follow30_root/work_dir/panda_follow30/$panda_follow30_campaign_id/$panda_follow30_server/WV3"
mkdir -p "$panda_follow30_campaign_root"
# Automatic real-data CUDA validation; its Students are disposable and
# never enter the result ledger. Calibration is retained for production runs.
docker run --rm --init --gpus all --pid=host --shm-size=8g \
  --env PYTHONUNBUFFERED=1 "${panda_follow30_common[@]}" \
  --mount "type=bind,src=$panda_follow30_root/work_dir/panda_follow30,dst=/workspace/work_dir/panda_follow30" \
  "$panda_follow30_image" python -m panda_follow30.preflight --server "$panda_follow30_server"
docker run --detach --init --gpus all --pid=host --shm-size=8g \
  --name "$panda_follow30_name" --env PYTHONUNBUFFERED=1 "${panda_follow30_common[@]}" \
  --mount "type=bind,src=$panda_follow30_root/work_dir/panda_follow30,dst=/workspace/work_dir/panda_follow30" \
  "$panda_follow30_image" bash scripts/run_panda_follow30.sh "$panda_follow30_server"
# Start this campaign's all-passes plus representative Sheet publisher.
# Its local watcher validates the existing destination before writing.
PAN_HOST_PYTHON="$panda_follow30_python" bash scripts/start_panda_follow30_best_publisher.sh "$panda_follow30_server"
echo "Started $panda_follow30_name (WV3, mature donor + positive-scale and PAN-difference hard-only Stage2, unlimited queue). Monitor: docker logs -f $panda_follow30_name"
