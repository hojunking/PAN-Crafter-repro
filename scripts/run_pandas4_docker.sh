#!/usr/bin/env bash
# Explicit invocation only; never stop, remove, or replace another experiment.
set -euo pipefail
if [[ $# -ne 0 ]]; then
  echo 'Usage: bash scripts/run_pandas4_docker.sh (s4 WV3, max40h)' >&2
  exit 2
fi
pandas4_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pandas4_image="${PAN_DIRECTSHIFT_IMAGE:-hojunqueen/pancrafter-env:torch2.4.0-cu118}"
pandas4_name='pandas4-wv3-s4'
pandas4_python="${PAN_HOST_PYTHON:-}"
cd "$pandas4_root"
export PYTHONDONTWRITEBYTECODE=1
command -v setsid >/dev/null
if [[ -z "$pandas4_python" ]]; then
  pandas4_base_python="$(command -v python)"
  pandas4_base_prefix="$("$pandas4_base_python" -c 'import sys; print(sys.base_prefix)')"
  for pandas4_candidate in "$pandas4_base_python" "$pandas4_base_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$pandas4_candidate" ]] && "$pandas4_candidate" -c 'import google.oauth2.service_account; import google.auth.transport.requests; import requests' >/dev/null 2>&1; then
      pandas4_python="$pandas4_candidate"
      break
    fi
  done
  if [[ -z "$pandas4_python" ]]; then
    echo 'Set PAN_HOST_PYTHON to the existing server uploader Python (google-auth and requests required).' >&2
    exit 1
  fi
fi
"$pandas4_python" -c 'import google.oauth2.service_account; import google.auth.transport.requests; import requests'
test -f gspread/account.json
docker image inspect "$pandas4_image" >/dev/null
if docker container inspect "$pandas4_name" >/dev/null 2>&1; then
  echo "Container $pandas4_name already exists; inspect it. Nothing removed or reset." >&2
  exit 1
fi
pandas4_dlpan_input="${PANCRAFTER_DLPAN:-$pandas4_root/../DLPan-Toolbox}"
pandas4_dlpan="$(cd "$pandas4_dlpan_input" && pwd -P)"
test -f "$pandas4_dlpan/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py"
for pandas4_path in "$pandas4_root" "$pandas4_dlpan"; do
  if [[ "$pandas4_path" == *','* || "$pandas4_path" == *$'\n'* || "$pandas4_path" == *$'\r'* ]]; then
    echo 'Unsafe Docker mount delimiter' >&2
    exit 1
  fi
done
pandas4_files="$("$pandas4_python" -m pandas4 mounts)"
test -n "$pandas4_files"
pandas4_mounts=()
while IFS= read -r pandas4_file; do
  pandas4_mounts+=(--mount "type=bind,src=$pandas4_file,dst=$pandas4_file,readonly")
done <<< "$pandas4_files"
pandas4_common=(--user "$(id -u):$(id -g)" --env PYTHONDONTWRITEBYTECODE=1
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env MKL_THREADING_LAYER=GNU
  --env PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox
  --mount "type=bind,src=$pandas4_root,dst=/workspace,readonly"
  --mount "type=bind,src=$pandas4_dlpan,dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly"
  "${pandas4_mounts[@]}" --workdir /workspace)
# All four WV3 splits and actual image imports are checked without GPU training.
docker run --rm "${pandas4_common[@]}" "$pandas4_image" python -m pandas4 check
"$pandas4_python" -c 'from directshift48.__main__ import require_idle_gpu; require_idle_gpu("cuda")'
pandas4_campaign_root="$pandas4_root/work_dir/pandas4/PANDA_S4_DONOR_STAGE2_40H_20261017_v1/s4/WV3"
mkdir -p "$pandas4_campaign_root"
docker run --detach --init --gpus all --pid=host --shm-size=8g \
  --name "$pandas4_name" --env PYTHONUNBUFFERED=1 "${pandas4_common[@]}" \
  --mount "type=bind,src=$pandas4_root/work_dir,dst=/workspace/work_dir" \
  "$pandas4_image" bash scripts/run_pandas4.sh
nohup setsid "$pandas4_python" -u -m pandas4.publication --root "$pandas4_campaign_root" \
  --server s4 --apply --watch --poll-seconds 60 \
  >> "$pandas4_root/work_dir/pandas4/publisher-s4.log" 2>&1 < /dev/null &
echo "Started $pandas4_name (WV3, max40h); publisher PID=$!. Monitor: docker logs -f $pandas4_name"
