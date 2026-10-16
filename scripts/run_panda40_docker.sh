#!/usr/bin/env bash
# Only an explicit invocation starts training. --replace-donor24 targets one
# known old controller; no process-pattern killing or container removal.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ! "$1" =~ ^s(1|3)$ ]]; then
  echo 'Usage: bash scripts/run_panda40_docker.sh s1|s3 [--replace-donor24]' >&2
  exit 2
fi
if [[ $# -eq 2 && "$2" != '--replace-donor24' ]]; then
  echo 'Unknown option; only --replace-donor24 is supported' >&2
  exit 2
fi
panda40_server="$1"
panda40_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
panda40_image="${PAN_DIRECTSHIFT_IMAGE:-hojunqueen/pancrafter-env:torch2.4.0-cu118}"
panda40_name="panda40-wv3-${panda40_server}"
panda40_python="${PAN_HOST_PYTHON:-}"
cd "$panda40_root"
export PYTHONDONTWRITEBYTECODE=1
command -v setsid >/dev/null
if [[ -z "$panda40_python" ]]; then
  panda40_base_python="$(command -v python)"
  panda40_base_prefix="$("$panda40_base_python" -c 'import sys; print(sys.base_prefix)')"
  for panda40_candidate in "$panda40_base_python" "$panda40_base_prefix/envs/pancrafter/bin/python"; do
    if [[ -x "$panda40_candidate" ]] && "$panda40_candidate" -c 'import google.oauth2.service_account; import google.auth.transport.requests; import requests' >/dev/null 2>&1; then
      panda40_python="$panda40_candidate"
      break
    fi
  done
  if [[ -z "$panda40_python" ]]; then
    echo 'No existing publisher Python found. Set PAN_HOST_PYTHON to the server uploader environment; old training is untouched.' >&2
    exit 1
  fi
fi
"$panda40_python" -c 'import google.oauth2.service_account; import google.auth.transport.requests; import requests'
test -f gspread/account.json
docker image inspect "$panda40_image" >/dev/null
if docker container inspect "$panda40_name" >/dev/null 2>&1; then
  echo "Container $panda40_name already exists; inspect it. Nothing removed or reset." >&2
  exit 1
fi
panda40_dlpan_input="${PANCRAFTER_DLPAN:-$panda40_root/../DLPan-Toolbox}"
panda40_dlpan="$(cd "$panda40_dlpan_input" && pwd -P)"
test -f "$panda40_dlpan/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py"
for panda40_path in "$panda40_root" "$panda40_dlpan"; do
  if [[ "$panda40_path" == *','* || "$panda40_path" == *$'\n'* || "$panda40_path" == *$'\r'* ]]; then
    echo 'Unsafe Docker mount delimiter' >&2
    exit 1
  fi
done
panda40_files="$("$panda40_python" -m panda40 mounts)"
test -n "$panda40_files"
panda40_mounts=()
while IFS= read -r panda40_file; do
  panda40_mounts+=(--mount "type=bind,src=$panda40_file,dst=$panda40_file,readonly")
done <<< "$panda40_files"
panda40_common=(--user "$(id -u):$(id -g)" --env PYTHONDONTWRITEBYTECODE=1
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env MKL_THREADING_LAYER=GNU
  --env PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox
  --mount "type=bind,src=$panda40_root,dst=/workspace,readonly"
  --mount "type=bind,src=$panda40_dlpan,dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly"
  "${panda40_mounts[@]}" --workdir /workspace)
# Check the actual image/imports and all four WV3 splits BEFORE stopping old work.
docker run --rm "${panda40_common[@]}" "$panda40_image" \
  python -m panda40 check --server "$panda40_server"
if [[ "${2:-}" == '--replace-donor24' ]]; then
  panda40_old="panda-donor24-wv3-${panda40_server}"
  if [[ "$(docker container inspect --format '{{.State.Running}}' "$panda40_old" 2>/dev/null || true)" == 'true' ]]; then
    docker stop --time 60 "$panda40_old"
  fi
fi
"$panda40_python" -c 'from directshift48.__main__ import require_idle_gpu; require_idle_gpu("cuda")'
mkdir -p "$panda40_root/work_dir/panda40/PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1/$panda40_server/WV3"
docker run --detach --init --gpus all --pid=host --shm-size=8g \
  --name "$panda40_name" --env PYTHONUNBUFFERED=1 "${panda40_common[@]}" \
  --mount "type=bind,src=$panda40_root/work_dir,dst=/workspace/work_dir" \
  "$panda40_image" bash scripts/run_panda40.sh "$panda40_server"
panda40_campaign_root="$panda40_root/work_dir/panda40/PANDA_S13_ALIGNER_STAGE2_40H_20261016_v1/$panda40_server/WV3"
nohup setsid "$panda40_python" -u -m panda40.publication --root "$panda40_campaign_root" \
  --server "$panda40_server" --apply --watch --poll-seconds 60 \
  >> "$panda40_root/work_dir/panda40/publisher-${panda40_server}.log" 2>&1 < /dev/null &
echo "Started $panda40_name (WV3, max40h); publisher PID=$!. Monitor: docker logs -f $panda40_name"
