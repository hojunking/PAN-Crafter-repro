#!/usr/bin/env bash
# Isolated cost sidecars only; never signal/stop an existing trainer.
set -euo pipefail
if [[ $# -ne 1 || ! "$1" =~ ^s[13]$ ]]; then
  echo 'Usage: bash scripts/prepare_panda_zero30_costs.sh s1|s3' >&2
  exit 2
fi
panda_cost_server="$1"
panda_cost_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$panda_cost_root"
if [[ ! -f gspread/server.txt || "$(tr -d '[:space:]' < gspread/server.txt)" != "$panda_cost_server" ]]; then
  echo 'Local server marker mismatch; nothing measured.' >&2
  exit 1
fi
panda_cost_image="${PAN_ZERO30_IMAGE:-${PAN_MATURE30_IMAGE:-${PAN_DIRECTSHIFT_IMAGE:-}}}"
if [[ -z "$panda_cost_image" ]]; then
  for panda_cost_old in "panda-zero30-hardonly-wv3-${panda_cost_server}-v1" "panda-mature30-hardonly-wv3-${panda_cost_server}-v2" "panda-band30-wv3-${panda_cost_server}-v1" "panda-warp30-wv3-${panda_cost_server}-v1" "panda30r-wv3-${panda_cost_server}" "panda40-wv3-${panda_cost_server}"; do
    if docker container inspect "$panda_cost_old" >/dev/null 2>&1; then
      panda_cost_image="$(docker inspect --format '{{.Image}}' "$panda_cost_old")"
      break
    fi
  done
fi
if [[ -z "$panda_cost_image" ]]; then
  echo 'Set PAN_ZERO30_IMAGE to the existing local training image.' >&2
  exit 1
fi
docker image inspect "$panda_cost_image" >/dev/null
panda_cost_python="${PAN_HOST_PYTHON:-python}"
mapfile -t panda_cost_data < <("$panda_cost_python" -m edgeres48.docker_mounts s1)
if [[ ${#panda_cost_data[@]} -eq 0 ]]; then
  echo 'No registered WV3 data mounts.' >&2
  exit 1
fi
panda_cost_mounts=()
for panda_cost_path in "$panda_cost_root" "${panda_cost_data[@]}"; do
  if [[ "$panda_cost_path" == *','* || "$panda_cost_path" == *$'\n'* || "$panda_cost_path" == *$'\r'* ]]; then
    echo 'Unsafe mount path.' >&2
    exit 1
  fi
done
for panda_cost_path in "${panda_cost_data[@]}"; do
  test -f "$panda_cost_path"
  panda_cost_mounts+=(--mount "type=bind,src=$panda_cost_path,dst=$panda_cost_path,readonly")
done
docker run --rm --init --gpus all --pid=host --shm-size=8g \
  --user "$(id -u):$(id -g)" --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONUNBUFFERED=1 \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env MKL_THREADING_LAYER=GNU \
  --mount "type=bind,src=$panda_cost_root,dst=/workspace,readonly" \
  --mount "type=bind,src=$panda_cost_root/work_dir,dst=/workspace/work_dir" \
  "${panda_cost_mounts[@]}" --workdir /workspace "$panda_cost_image" \
  python -m panda_zero30.cost --server "$panda_cost_server"

