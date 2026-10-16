#!/usr/bin/env bash
# Explicit launch only. Existing training is never killed or auto-replaced.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s(1|3)$ ]]; then
  echo 'Usage: bash scripts/run_donor24_docker.sh s1|s3 (both WV3)' >&2
  exit 2
fi
donor24_server="$1"
donor24_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
donor24_image="${PAN_DIRECTSHIFT_IMAGE:-hojunqueen/pancrafter-env:torch2.4.0-cu118}"
donor24_name="panda-donor24-wv3-${donor24_server}"
donor24_python="${PAN_HOST_PYTHON:-python}"
cd "$donor24_root"
export PYTHONDONTWRITEBYTECODE=1
command -v setsid >/dev/null
"$donor24_python" -c 'from directshift48.__main__ import require_idle_gpu; require_idle_gpu("cuda")'
docker image inspect "$donor24_image" >/dev/null
if docker container inspect "$donor24_name" >/dev/null 2>&1; then
  echo "Container $donor24_name exists; inspect it. Nothing removed." >&2
  exit 1
fi
donor24_dlpan_input="${PANCRAFTER_DLPAN:-$donor24_root/../DLPan-Toolbox}"
donor24_dlpan="$(cd "$donor24_dlpan_input" && pwd -P)"
test -f "$donor24_dlpan/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py"
for donor24_path in "$donor24_root" "$donor24_dlpan"; do
  if [[ "$donor24_path" == *','* || "$donor24_path" == *$'\n'* || "$donor24_path" == *$'\r'* ]]; then
    echo 'Unsafe Docker mount delimiter' >&2
    exit 1
  fi
done
donor24_files="$("$donor24_python" -m donor24 mounts)"
test -n "$donor24_files"
donor24_mounts=()
while IFS= read -r donor24_file; do
  donor24_mounts+=(--mount "type=bind,src=$donor24_file,dst=$donor24_file,readonly")
done <<< "$donor24_files"
mkdir -p "$donor24_root/work_dir/donor24"
docker run --detach --init --gpus all --pid=host --shm-size=8g \
  --name "$donor24_name" --user "$(id -u):$(id -g)" \
  --env PYTHONUNBUFFERED=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env MKL_THREADING_LAYER=GNU \
  --env PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox \
  --mount "type=bind,src=$donor24_root,dst=/workspace,readonly" \
  --mount "type=bind,src=$donor24_root/work_dir,dst=/workspace/work_dir" \
  --mount "type=bind,src=$donor24_dlpan,dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly" \
  "${donor24_mounts[@]}" --workdir /workspace "$donor24_image" \
  bash scripts/run_donor24.sh "$donor24_server"
nohup setsid "$donor24_python" -u -m donor24 publish-auto --server "$donor24_server" --apply \
  >> "$donor24_root/work_dir/donor24/publisher-bootstrap-${donor24_server}.log" 2>&1 < /dev/null &
echo "Started $donor24_name (WV3); publisher PID=$!. Monitor: docker logs -f $donor24_name"
