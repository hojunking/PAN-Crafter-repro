#!/usr/bin/env bash
# One explicit command per local server. Never stop/remove/replace old work.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s3$ ]]; then
  echo 'Usage: bash scripts/run_panda_warp30_docker.sh s3 (WV3, fresh Stage1 + warp cases, 30h)' >&2
  exit 2
fi
panda_warp30_server="$1"
panda_warp30_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
panda_warp30_image="${PAN_WARP30_IMAGE:-${PAN_DIRECTSHIFT_IMAGE:-}}"
panda_warp30_name="panda-warp30-wv3-${panda_warp30_server}-v1"
panda_warp30_python="${PAN_HOST_PYTHON:-}"
cd "$panda_warp30_root"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -f gspread/server.txt || "$(tr -d '[:space:]' < gspread/server.txt)" != s3 ]]; then
  echo 'Local server marker must be s3; nothing started.' >&2
  exit 1
fi
# Reuse the already-used s3 image (RTX5090 needs a compatible CUDA build).
# Never pull/change environments implicitly or default to an incompatible cu118 image.
if [[ -z "$panda_warp30_image" ]]; then
  for panda_warp30_previous in panda-midpoint-wv3-s3-v2 panda-midpoint-wv3-s3-v1 panda30r-wv3-s3 panda40-wv3-s3; do
    if docker container inspect "$panda_warp30_previous" >/dev/null 2>&1; then
      panda_warp30_image="$(docker inspect --format '{{.Image}}' "$panda_warp30_previous")"
      break
    fi
  done
fi
if [[ -z "$panda_warp30_image" ]]; then
  echo 'Set PAN_WARP30_IMAGE to the same locally installed Docker image used by s3/Panda30R.' >&2
  exit 1
fi
command -v setsid >/dev/null
df -h "$panda_warp30_root/work_dir"
if [[ -z "$panda_warp30_python" ]]; then
  panda_warp30_base_python="$(command -v python || command -v python3)"
  panda_warp30_base_prefix="$("$panda_warp30_base_python" -c 'import sys; print(sys.base_prefix)')"
  for panda_warp30_candidate in "$panda_warp30_base_python" "$panda_warp30_base_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$panda_warp30_candidate" ]] && "$panda_warp30_candidate" -c 'import google.oauth2.service_account; import google.auth.transport.requests; import requests' >/dev/null 2>&1; then
      panda_warp30_python="$panda_warp30_candidate"
      break
    fi
  done
  if [[ -z "$panda_warp30_python" ]]; then
    echo 'Set PAN_HOST_PYTHON to the existing server uploader Python (google-auth and requests required). Old experiments are untouched.' >&2
    exit 1
  fi
fi
# Validate only local authentication material; never print key contents.
"$panda_warp30_python" -c 'import google.auth.transport.requests, requests; from google.oauth2 import service_account; service_account.Credentials.from_service_account_file("gspread/account.json", scopes=["https://www.googleapis.com/auth/spreadsheets"])'
docker image inspect "$panda_warp30_image" >/dev/null
if docker container inspect "$panda_warp30_name" >/dev/null 2>&1; then
  echo "Container $panda_warp30_name already exists; inspect it. Nothing removed or reset." >&2
  exit 1
fi
panda_warp30_dlpan_input="${PANCRAFTER_DLPAN:-$panda_warp30_root/../DLPan-Toolbox}"
if [[ ! -d "$panda_warp30_dlpan_input" ]]; then
  echo 'DLPan evaluator directory is unavailable; set PANCRAFTER_DLPAN to its existing local checkout.' >&2
  exit 1
fi
panda_warp30_dlpan="$(cd "$panda_warp30_dlpan_input" && pwd -P)"
test -f "$panda_warp30_dlpan/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py"
for panda_warp30_path in "$panda_warp30_root" "$panda_warp30_dlpan"; do
  if [[ "$panda_warp30_path" == *','* || "$panda_warp30_path" == *$'\n'* || "$panda_warp30_path" == *$'\r'* ]]; then
    echo 'Unsafe Docker mount delimiter' >&2
    exit 1
  fi
done
panda_warp30_files="$("$panda_warp30_python" -m panda_warp30 mounts)"
test -n "$panda_warp30_files"
panda_warp30_mounts=()
while IFS= read -r panda_warp30_file; do
  # Normally enforced by mounts as well; fail closed if its output is invalid.
  if [[ ! -f "$panda_warp30_file" || "$panda_warp30_file" == *','* || "$panda_warp30_file" == *$'\r'* ]]; then
    echo 'Invalid registered data mount target' >&2
    exit 1
  fi
  panda_warp30_mounts+=(--mount "type=bind,src=$panda_warp30_file,dst=$panda_warp30_file,readonly")
done <<< "$panda_warp30_files"
panda_warp30_common=(--user "$(id -u):$(id -g)" --env PYTHONDONTWRITEBYTECODE=1
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env MKL_THREADING_LAYER=GNU
  --env PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox
  --mount "type=bind,src=$panda_warp30_root,dst=/workspace,readonly"
  --mount "type=bind,src=$panda_warp30_dlpan,dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly"
  "${panda_warp30_mounts[@]}" --workdir /workspace)
# Read-only preflight with the actual image. No GPU/training/Sheet mutation.
docker run --rm "${panda_warp30_common[@]}" "$panda_warp30_image" \
  python -m panda_warp30 check --server "$panda_warp30_server"
"$panda_warp30_python" -c 'from directshift48.__main__ import require_idle_gpu; require_idle_gpu("cuda")'
panda_warp30_campaign_id="$("$panda_warp30_python" -c 'from panda_warp30.contract import CAMPAIGN_ID; print(CAMPAIGN_ID)')"
if [[ ! "$panda_warp30_campaign_id" =~ ^PANDA_S3_WARP_STAGE12_30H_[0-9]{8}_v1$ ]]; then
  echo 'Invalid warp30 campaign identity; nothing started.' >&2
  exit 1
fi
panda_warp30_campaign_root="$panda_warp30_root/work_dir/panda_warp30/$panda_warp30_campaign_id/$panda_warp30_server/WV3"
mkdir -p "$panda_warp30_campaign_root"
docker run --detach --init --gpus all --pid=host --shm-size=8g \
  --name "$panda_warp30_name" --env PYTHONUNBUFFERED=1 "${panda_warp30_common[@]}" \
  --mount "type=bind,src=$panda_warp30_root/work_dir,dst=/workspace/work_dir" \
  "$panda_warp30_image" bash scripts/run_panda_warp30.sh "$panda_warp30_server"
nohup setsid "$panda_warp30_python" -u -m panda_warp30.publication --root "$panda_warp30_campaign_root" \
  --server "$panda_warp30_server" --apply --watch --poll-seconds 60 \
  >> "$panda_warp30_root/work_dir/panda_warp30/publisher-${panda_warp30_server}-v1.log" 2>&1 < /dev/null &
echo "Started $panda_warp30_name (WV3, fresh Stage1 + warp cases, 30h); publisher PID=$!. Monitor: docker logs -f $panda_warp30_name"
