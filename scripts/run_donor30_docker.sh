#!/usr/bin/env bash
# One command per local server: automatic preflight, donor, Student and uploader.
# Never terminates training, removes containers, or starts another server.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s(1|3)$ ]]; then
  echo 'Usage: bash scripts/run_donor30_docker.sh s1|s3' >&2
  exit 2
fi
donor_server="$1"
donor_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
donor_image="${PAN_DIRECTSHIFT_IMAGE:-hojunqueen/pancrafter-env:torch2.4.0-cu118}"
donor_name="panda-donor30-${donor_server}"
donor_python="${PAN_HOST_PYTHON:-python}"
cd "$donor_root"
export PYTHONDONTWRITEBYTECODE=1
command -v setsid >/dev/null
"$donor_python" -c 'from directshift48.__main__ import require_idle_gpu; require_idle_gpu("cuda")'
docker image inspect "$donor_image" >/dev/null
if docker container inspect "$donor_name" >/dev/null 2>&1; then
  echo "Container $donor_name exists. Nothing removed; inspect before restarting." >&2
  exit 1
fi
donor_dlpan_input="${PANCRAFTER_DLPAN:-$donor_root/../DLPan-Toolbox}"
if [[ ! -d "$donor_dlpan_input" ]]; then
  echo 'Audited DLPan missing. Set PANCRAFTER_DLPAN to the existing toolbox.' >&2
  exit 1
fi
donor_dlpan="$(cd "$donor_dlpan_input" && pwd -P)"
if [[ ! -f "$donor_dlpan/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py" ]]; then
  echo 'Required audited DLPan evaluator is missing.' >&2
  exit 1
fi
for donor_path in "$donor_root" "$donor_dlpan"; do
  if [[ "$donor_path" == *','* || "$donor_path" == *$'\n'* || "$donor_path" == *$'\r'* ]]; then
    echo 'Docker mount paths may not contain commas/newlines.' >&2
    exit 1
  fi
done
donor_files="$("$donor_python" -m edgeres48.docker_mounts "$donor_server")"
if [[ -z "$donor_files" ]]; then
  echo 'No pinned local dataset mounts. Nothing launched.' >&2
  exit 1
fi
donor_mounts=()
while IFS= read -r donor_file; do
  donor_mounts+=(--mount "type=bind,src=$donor_file,dst=$donor_file,readonly")
done <<< "$donor_files"
mkdir -p "$donor_root/work_dir/donor30"
docker run --detach --init --gpus all --pid=host --shm-size=8g \
  --name "$donor_name" --user "$(id -u):$(id -g)" \
  --env PYTHONUNBUFFERED=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env MKL_THREADING_LAYER=GNU \
  --env PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox \
  --mount "type=bind,src=$donor_root,dst=/workspace,readonly" \
  --mount "type=bind,src=$donor_root/work_dir,dst=/workspace/work_dir" \
  --mount "type=bind,src=$donor_dlpan,dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly" \
  "${donor_mounts[@]}" --workdir /workspace "$donor_image" \
  bash scripts/run_donor30.sh "$donor_server"
# Only the same-server legacy SHEET writer may be handed over after preflight;
# all training is untouched. Failed upload stays in local durable outbox.
nohup setsid "$donor_python" -u -m donor30 publish-auto --server "$donor_server" --apply \
  >> "$donor_root/work_dir/donor30/publisher-bootstrap-${donor_server}.log" 2>&1 < /dev/null &
echo "Started $donor_name; publisher bootstrap PID=$!. Monitor: docker logs -f $donor_name"
