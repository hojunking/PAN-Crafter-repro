#!/usr/bin/env bash
# One explicit command per local server. Never stop/remove/replace old work.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s(1|3|4)$ ]]; then
  echo 'Usage: bash scripts/run_panda30r_docker.sh s1|s3|s4 (WV3, max30h)' >&2
  exit 2
fi
panda30r_server="$1"
panda30r_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
panda30r_image="${PAN_DIRECTSHIFT_IMAGE:-hojunqueen/pancrafter-env:torch2.4.0-cu118}"
panda30r_name="panda30r-wv3-${panda30r_server}"
panda30r_python="${PAN_HOST_PYTHON:-}"
cd "$panda30r_root"
export PYTHONDONTWRITEBYTECODE=1
command -v setsid >/dev/null
if [[ -z "$panda30r_python" ]]; then
  panda30r_base_python="$(command -v python || command -v python3)"
  panda30r_base_prefix="$("$panda30r_base_python" -c 'import sys; print(sys.base_prefix)')"
  for panda30r_candidate in "$panda30r_base_python" "$panda30r_base_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$panda30r_candidate" ]] && "$panda30r_candidate" -c 'import google.oauth2.service_account; import google.auth.transport.requests; import requests' >/dev/null 2>&1; then
      panda30r_python="$panda30r_candidate"
      break
    fi
  done
  if [[ -z "$panda30r_python" ]]; then
    echo 'Set PAN_HOST_PYTHON to the existing server uploader Python (google-auth and requests required). Old experiments are untouched.' >&2
    exit 1
  fi
fi
# Validate only local authentication material; never print key contents.
"$panda30r_python" -c 'import google.auth.transport.requests, requests; from google.oauth2 import service_account; service_account.Credentials.from_service_account_file("gspread/account.json", scopes=["https://www.googleapis.com/auth/spreadsheets"])'
docker image inspect "$panda30r_image" >/dev/null
if docker container inspect "$panda30r_name" >/dev/null 2>&1; then
  echo "Container $panda30r_name already exists; inspect it. Nothing removed or reset." >&2
  exit 1
fi
panda30r_dlpan_input="${PANCRAFTER_DLPAN:-$panda30r_root/../DLPan-Toolbox}"
if [[ ! -d "$panda30r_dlpan_input" ]]; then
  echo 'DLPan evaluator directory is unavailable; set PANCRAFTER_DLPAN to its existing local checkout.' >&2
  exit 1
fi
panda30r_dlpan="$(cd "$panda30r_dlpan_input" && pwd -P)"
test -f "$panda30r_dlpan/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py"
for panda30r_path in "$panda30r_root" "$panda30r_dlpan"; do
  if [[ "$panda30r_path" == *','* || "$panda30r_path" == *$'\n'* || "$panda30r_path" == *$'\r'* ]]; then
    echo 'Unsafe Docker mount delimiter' >&2
    exit 1
  fi
done
panda30r_files="$("$panda30r_python" -m panda30r mounts)"
test -n "$panda30r_files"
panda30r_mounts=()
while IFS= read -r panda30r_file; do
  # Normally enforced by mounts as well; fail closed if its output is invalid.
  if [[ ! -f "$panda30r_file" || "$panda30r_file" == *','* || "$panda30r_file" == *$'\r'* ]]; then
    echo 'Invalid registered data mount target' >&2
    exit 1
  fi
  panda30r_mounts+=(--mount "type=bind,src=$panda30r_file,dst=$panda30r_file,readonly")
done <<< "$panda30r_files"
panda30r_common=(--user "$(id -u):$(id -g)" --env PYTHONDONTWRITEBYTECODE=1
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env MKL_THREADING_LAYER=GNU
  --env PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox
  --mount "type=bind,src=$panda30r_root,dst=/workspace,readonly"
  --mount "type=bind,src=$panda30r_dlpan,dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly"
  "${panda30r_mounts[@]}" --workdir /workspace)
# Read-only preflight with the actual image. No GPU/training/Sheet mutation.
docker run --rm "${panda30r_common[@]}" "$panda30r_image" \
  python -m panda30r check --server "$panda30r_server"
"$panda30r_python" -c 'from directshift48.__main__ import require_idle_gpu; require_idle_gpu("cuda")'
panda30r_campaign_root="$panda30r_root/work_dir/panda30r/PANDA_S134_DONOR_SCHEDULE_30H_20261018_v1/$panda30r_server/WV3"
mkdir -p "$panda30r_campaign_root"
docker run --detach --init --gpus all --pid=host --shm-size=8g \
  --name "$panda30r_name" --env PYTHONUNBUFFERED=1 "${panda30r_common[@]}" \
  --mount "type=bind,src=$panda30r_root/work_dir,dst=/workspace/work_dir" \
  "$panda30r_image" bash scripts/run_panda30r.sh "$panda30r_server"
nohup setsid "$panda30r_python" -u -m panda30r.publication --root "$panda30r_campaign_root" \
  --server "$panda30r_server" --apply --watch --poll-seconds 60 \
  >> "$panda30r_root/work_dir/panda30r/publisher-${panda30r_server}.log" 2>&1 < /dev/null &
echo "Started $panda30r_name (WV3, max30h); publisher PID=$!. Monitor: docker logs -f $panda30r_name"
