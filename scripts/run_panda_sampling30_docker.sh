#!/usr/bin/env bash
# One explicit command per local server. Never stop/remove/replace old work.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/run_panda_sampling30_docker.sh s1|s3 (WV3, mature donor + positive-scale sampling ablation Stage2, unlimited queue)' >&2
  exit 2
fi
panda_sampling30_server="$1"
panda_sampling30_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
panda_sampling30_image="${PAN_SAMPLING30_IMAGE:-${PAN_SCHEDULE30_IMAGE:-${PAN_SCALE30_IMAGE:-${PAN_MATURE30_IMAGE:-${PAN_DIRECTSHIFT_IMAGE:-}}}}}"
panda_sampling30_name="panda-sampling30-hardonly-wv3-${panda_sampling30_server}-v1"
panda_sampling30_python="${PAN_HOST_PYTHON:-}"
cd "$panda_sampling30_root"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -f gspread/server.txt || "$(tr -d '[:space:]' < gspread/server.txt)" != "$panda_sampling30_server" ]]; then
  echo 'Local server marker must match requested s1/s3; nothing started.' >&2
  exit 1
fi
# Reuse the already-used s1 image (reuse the proven local CUDA build).
# Never pull/change environments implicitly or default to an incompatible cu118 image.
if [[ -z "$panda_sampling30_image" ]]; then
  for panda_sampling30_previous in "panda-schedule30-hardonly-wv3-${panda_sampling30_server}-v1" "panda-scale30-hardonly-wv3-${panda_sampling30_server}-v1" "panda-zero30-hardonly-wv3-${panda_sampling30_server}-v1" "panda-mature30-hardonly-wv3-${panda_sampling30_server}-v2" "panda-band30-wv3-${panda_sampling30_server}-v1" "panda-warp30-wv3-${panda_sampling30_server}-v1" "panda30r-wv3-${panda_sampling30_server}" "panda40-wv3-${panda_sampling30_server}"; do
    if docker container inspect "$panda_sampling30_previous" >/dev/null 2>&1; then
      panda_sampling30_image="$(docker inspect --format '{{.Image}}' "$panda_sampling30_previous")"
      break
    fi
  done
fi
if [[ -z "$panda_sampling30_image" ]]; then
  echo 'Set PAN_SAMPLING30_IMAGE to the same locally installed Docker image used by this server/Panda30R.' >&2
  exit 1
fi
command -v setsid >/dev/null
df -h "$panda_sampling30_root/work_dir"
if [[ -z "$panda_sampling30_python" ]]; then
  panda_sampling30_base_python="$(command -v python || command -v python3)"
  panda_sampling30_base_prefix="$("$panda_sampling30_base_python" -c 'import sys; print(sys.base_prefix)')"
  for panda_sampling30_candidate in "$panda_sampling30_base_python" "$panda_sampling30_base_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$panda_sampling30_candidate" ]] && "$panda_sampling30_candidate" -c 'import google.oauth2.service_account; import google.auth.transport.requests; import requests' >/dev/null 2>&1; then
      panda_sampling30_python="$panda_sampling30_candidate"
      break
    fi
  done
  if [[ -z "$panda_sampling30_python" ]]; then
    echo 'Set PAN_HOST_PYTHON to the existing server uploader Python (google-auth and requests required). Old experiments are untouched.' >&2
    exit 1
  fi
fi
# Validate only local authentication material; never print key contents.
"$panda_sampling30_python" -c 'import google.auth.transport.requests, requests; from google.oauth2 import service_account; service_account.Credentials.from_service_account_file("gspread/account.json", scopes=["https://www.googleapis.com/auth/spreadsheets"])'
docker image inspect "$panda_sampling30_image" >/dev/null
if docker container inspect "$panda_sampling30_name" >/dev/null 2>&1; then
  echo "Container $panda_sampling30_name already exists; inspect it. Nothing removed or reset." >&2
  exit 1
fi
panda_sampling30_dlpan_input="${PANCRAFTER_DLPAN:-$panda_sampling30_root/../DLPan-Toolbox}"
if [[ ! -d "$panda_sampling30_dlpan_input" ]]; then
  echo 'DLPan evaluator directory is unavailable; set PANCRAFTER_DLPAN to its existing local checkout.' >&2
  exit 1
fi
panda_sampling30_dlpan="$(cd "$panda_sampling30_dlpan_input" && pwd -P)"
test -f "$panda_sampling30_dlpan/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py"
for panda_sampling30_path in "$panda_sampling30_root" "$panda_sampling30_dlpan"; do
  if [[ "$panda_sampling30_path" == *','* || "$panda_sampling30_path" == *$'\n'* || "$panda_sampling30_path" == *$'\r'* ]]; then
    echo 'Unsafe Docker mount delimiter' >&2
    exit 1
  fi
done
panda_sampling30_files="$("$panda_sampling30_python" -m panda_sampling30 mounts)"
test -n "$panda_sampling30_files"
panda_sampling30_mounts=()
while IFS= read -r panda_sampling30_file; do
  # Normally enforced by mounts as well; fail closed if its output is invalid.
  if [[ ! -f "$panda_sampling30_file" || "$panda_sampling30_file" == *','* || "$panda_sampling30_file" == *$'\r'* ]]; then
    echo 'Invalid registered data mount target' >&2
    exit 1
  fi
  panda_sampling30_mounts+=(--mount "type=bind,src=$panda_sampling30_file,dst=$panda_sampling30_file,readonly")
done <<< "$panda_sampling30_files"
panda_sampling30_common=(--user "$(id -u):$(id -g)" --env PYTHONDONTWRITEBYTECODE=1
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env MKL_THREADING_LAYER=GNU
  --env PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox
  --mount "type=bind,src=$panda_sampling30_root,dst=/workspace,readonly"
  --mount "type=bind,src=$panda_sampling30_dlpan,dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly"
  "${panda_sampling30_mounts[@]}" --workdir /workspace)
# Read-only preflight with the actual image. No GPU/training/Sheet mutation.
docker run --rm "${panda_sampling30_common[@]}" "$panda_sampling30_image" \
  python -m panda_sampling30 check --server "$panda_sampling30_server"
"$panda_sampling30_python" -c 'from directshift48.__main__ import require_idle_gpu; require_idle_gpu("cuda")'
panda_sampling30_campaign_id="$("$panda_sampling30_python" -c 'from panda_sampling30.contract import CAMPAIGN_ID; print(CAMPAIGN_ID)')"
if [[ "$panda_sampling30_campaign_id" != PANDA_S13_SAMPLING_HARDONLY_20261023_v1 ]]; then
  echo 'Invalid sampling30 campaign identity; nothing started.' >&2
  exit 1
fi
panda_sampling30_campaign_root="$panda_sampling30_root/work_dir/panda_sampling30/$panda_sampling30_campaign_id/$panda_sampling30_server/WV3"
mkdir -p "$panda_sampling30_campaign_root"
# Automatic real-data CUDA validation; its Students are disposable and
# never enter the result ledger. Calibration is retained for production runs.
docker run --rm --init --gpus all --pid=host --shm-size=8g \
  --env PYTHONUNBUFFERED=1 "${panda_sampling30_common[@]}" \
  --mount "type=bind,src=$panda_sampling30_root/work_dir,dst=/workspace/work_dir" \
  "$panda_sampling30_image" python -m panda_sampling30.preflight --server "$panda_sampling30_server"
docker run --detach --init --gpus all --pid=host --shm-size=8g \
  --name "$panda_sampling30_name" --env PYTHONUNBUFFERED=1 "${panda_sampling30_common[@]}" \
  --mount "type=bind,src=$panda_sampling30_root/work_dir,dst=/workspace/work_dir" \
  "$panda_sampling30_image" bash scripts/run_panda_sampling30.sh "$panda_sampling30_server"
# Start this campaign's all-passes plus representative Sheet publisher.
# Its local watcher validates the existing destination before writing.
PAN_HOST_PYTHON="$panda_sampling30_python" bash scripts/start_panda_sampling30_best_publisher.sh "$panda_sampling30_server"
echo "Started $panda_sampling30_name (WV3, mature donor + positive-scale sampling ablation hard-only Stage2, unlimited queue). Monitor: docker logs -f $panda_sampling30_name"
