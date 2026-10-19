#!/usr/bin/env bash
# One explicit command per local server. Never stop/remove/replace old work.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s3$ ]]; then
  echo 'Usage: bash scripts/run_panda_midpoint_docker.sh s3 (WV3, three cases; R1 only)' >&2
  exit 2
fi
panda_midpoint_server="$1"
panda_midpoint_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
panda_midpoint_image="${PAN_MIDPOINT_IMAGE:-${PAN_DIRECTSHIFT_IMAGE:-}}"
panda_midpoint_name="panda-midpoint-wv3-${panda_midpoint_server}"
panda_midpoint_python="${PAN_HOST_PYTHON:-}"
cd "$panda_midpoint_root"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -f gspread/server.txt || "$(tr -d '[:space:]' < gspread/server.txt)" != s3 ]]; then
  echo 'Local server marker must be s3; nothing started.' >&2
  exit 1
fi
# Reuse the already-used s3 image (RTX5090 needs a compatible CUDA build).
# Never pull/change environments implicitly or default to an incompatible cu118 image.
if [[ -z "$panda_midpoint_image" ]]; then
  for panda_midpoint_previous in panda30r-wv3-s3 panda40-wv3-s3; do
    if docker container inspect "$panda_midpoint_previous" >/dev/null 2>&1; then
      panda_midpoint_image="$(docker inspect --format '{{.Image}}' "$panda_midpoint_previous")"
      break
    fi
  done
fi
if [[ -z "$panda_midpoint_image" ]]; then
  echo 'Set PAN_MIDPOINT_IMAGE to the same locally installed Docker image used by s3/Panda30R.' >&2
  exit 1
fi
command -v setsid >/dev/null
df -h "$panda_midpoint_root/work_dir"
if [[ -z "$panda_midpoint_python" ]]; then
  panda_midpoint_base_python="$(command -v python || command -v python3)"
  panda_midpoint_base_prefix="$("$panda_midpoint_base_python" -c 'import sys; print(sys.base_prefix)')"
  for panda_midpoint_candidate in "$panda_midpoint_base_python" "$panda_midpoint_base_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$panda_midpoint_candidate" ]] && "$panda_midpoint_candidate" -c 'import google.oauth2.service_account; import google.auth.transport.requests; import requests' >/dev/null 2>&1; then
      panda_midpoint_python="$panda_midpoint_candidate"
      break
    fi
  done
  if [[ -z "$panda_midpoint_python" ]]; then
    echo 'Set PAN_HOST_PYTHON to the existing server uploader Python (google-auth and requests required). Old experiments are untouched.' >&2
    exit 1
  fi
fi
# Validate only local authentication material; never print key contents.
"$panda_midpoint_python" -c 'import google.auth.transport.requests, requests; from google.oauth2 import service_account; service_account.Credentials.from_service_account_file("gspread/account.json", scopes=["https://www.googleapis.com/auth/spreadsheets"])'
docker image inspect "$panda_midpoint_image" >/dev/null
if docker container inspect "$panda_midpoint_name" >/dev/null 2>&1; then
  echo "Container $panda_midpoint_name already exists; inspect it. Nothing removed or reset." >&2
  exit 1
fi
panda_midpoint_dlpan_input="${PANCRAFTER_DLPAN:-$panda_midpoint_root/../DLPan-Toolbox}"
if [[ ! -d "$panda_midpoint_dlpan_input" ]]; then
  echo 'DLPan evaluator directory is unavailable; set PANCRAFTER_DLPAN to its existing local checkout.' >&2
  exit 1
fi
panda_midpoint_dlpan="$(cd "$panda_midpoint_dlpan_input" && pwd -P)"
test -f "$panda_midpoint_dlpan/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py"
for panda_midpoint_path in "$panda_midpoint_root" "$panda_midpoint_dlpan"; do
  if [[ "$panda_midpoint_path" == *','* || "$panda_midpoint_path" == *$'\n'* || "$panda_midpoint_path" == *$'\r'* ]]; then
    echo 'Unsafe Docker mount delimiter' >&2
    exit 1
  fi
done
panda_midpoint_files="$("$panda_midpoint_python" -m panda_midpoint mounts)"
test -n "$panda_midpoint_files"
panda_midpoint_mounts=()
while IFS= read -r panda_midpoint_file; do
  # Normally enforced by mounts as well; fail closed if its output is invalid.
  if [[ ! -f "$panda_midpoint_file" || "$panda_midpoint_file" == *','* || "$panda_midpoint_file" == *$'\r'* ]]; then
    echo 'Invalid registered data mount target' >&2
    exit 1
  fi
  panda_midpoint_mounts+=(--mount "type=bind,src=$panda_midpoint_file,dst=$panda_midpoint_file,readonly")
done <<< "$panda_midpoint_files"
panda_midpoint_common=(--user "$(id -u):$(id -g)" --env PYTHONDONTWRITEBYTECODE=1
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env MKL_THREADING_LAYER=GNU
  --env PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox
  --mount "type=bind,src=$panda_midpoint_root,dst=/workspace,readonly"
  --mount "type=bind,src=$panda_midpoint_dlpan,dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly"
  "${panda_midpoint_mounts[@]}" --workdir /workspace)
# Read-only preflight with the actual image. No GPU/training/Sheet mutation.
docker run --rm "${panda_midpoint_common[@]}" "$panda_midpoint_image" \
  python -m panda_midpoint check --server "$panda_midpoint_server"
"$panda_midpoint_python" -c 'from directshift48.__main__ import require_idle_gpu; require_idle_gpu("cuda")'
panda_midpoint_campaign_root="$panda_midpoint_root/work_dir/panda_midpoint/PANDA_S3_MIDPOINT_STAGE2_20261019_v1/$panda_midpoint_server/WV3"
mkdir -p "$panda_midpoint_campaign_root"
docker run --detach --init --gpus all --pid=host --shm-size=8g \
  --name "$panda_midpoint_name" --env PYTHONUNBUFFERED=1 "${panda_midpoint_common[@]}" \
  --mount "type=bind,src=$panda_midpoint_root/work_dir,dst=/workspace/work_dir" \
  "$panda_midpoint_image" bash scripts/run_panda_midpoint.sh "$panda_midpoint_server"
nohup setsid "$panda_midpoint_python" -u -m panda_midpoint.publication --root "$panda_midpoint_campaign_root" \
  --server "$panda_midpoint_server" --apply --watch --poll-seconds 60 \
  >> "$panda_midpoint_root/work_dir/panda_midpoint/publisher-${panda_midpoint_server}.log" 2>&1 < /dev/null &
echo "Started $panda_midpoint_name (WV3, three cases; R1 only); publisher PID=$!. Monitor: docker logs -f $panda_midpoint_name"
