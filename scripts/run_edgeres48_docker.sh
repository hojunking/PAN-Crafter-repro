#!/usr/bin/env bash
# Independent per-server launch; no image pull, shared teacher, or remote lock.
set -euo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^s[123]$ ]]; then
  echo 'Usage: bash scripts/run_edgeres48_docker.sh s1|s2|s3 [extra run arguments]' >&2
  exit 2
fi
server="$1"
shift
edgeres_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
edgeres_image="${PAN_EDGERES_IMAGE:-hojunqueen/pancrafter-env:torch2.4.0-cu118}"
edgeres_name="panda-edgeres48-${server}"
cd "$edgeres_root"
docker image inspect "$edgeres_image" >/dev/null
if docker container inspect "$edgeres_name" >/dev/null 2>&1; then
  echo "Container ${edgeres_name} already exists; inspect it before launching a duplicate." >&2
  exit 1
fi
mkdir -p "$edgeres_root/work_dir/edgeres48"
edgeres_files="$("${PAN_HOST_PYTHON:-python}" -m edgeres48.docker_mounts "$server")"
edgeres_mounts=()
while IFS= read -r edgeres_file; do
  edgeres_mounts+=(--mount "type=bind,src=$edgeres_file,dst=$edgeres_file,readonly")
done <<< "$edgeres_files"
exec docker run --detach --init --gpus all --shm-size=8g \
  --name "$edgeres_name" --user "$(id -u):$(id -g)" \
  --env PYTHONUNBUFFERED=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
  --env MKL_THREADING_LAYER=GNU \
  --mount "type=bind,src=$edgeres_root,dst=/workspace,readonly" \
  --mount "type=bind,src=$edgeres_root/work_dir,dst=/workspace/work_dir" \
  "${edgeres_mounts[@]}" \
  --workdir /workspace "$edgeres_image" \
  bash scripts/run_edgeres48.sh "$server" "$@"
