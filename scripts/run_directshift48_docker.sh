#!/usr/bin/env bash
# Independent S1/WV3 or S3/GF2 training. Never kills a prior run/publisher.
set -euo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^s(1|3)$ ]]; then
  echo 'Usage: bash scripts/run_directshift48_docker.sh s1|s3 [extra run arguments]' >&2
  exit 2
fi
direct_server="$1"
shift
direct_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
direct_image="${PAN_DIRECTSHIFT_IMAGE:-hojunqueen/pancrafter-env:torch2.4.0-cu118}"
direct_name="panda-directshift48-${direct_server}"
cd "$direct_root"
direct_args=("$@")
direct_device=cuda
for ((i=0; i<${#direct_args[@]}; i++)); do
  case "${direct_args[i]}" in
    --device|--sources|--output|--baseline)
      if ((i+1 >= ${#direct_args[@]})); then
        echo "Missing value for ${direct_args[i]}." >&2
        exit 2
      fi
      direct_flag="${direct_args[i]}"
      direct_value="${direct_args[i+1]}"
      if [[ "$direct_flag" == --device ]]; then
        direct_device="$direct_value"
      elif [[ "$direct_value" == "$direct_root/"* ]]; then
        direct_args[i+1]="/workspace/${direct_value#"$direct_root/"}"
      fi
      if [[ "$direct_flag" == --sources && "$direct_value" != ablr2/sensor_sources.json && "$direct_value" != ./ablr2/sensor_sources.json && "$direct_value" != "$direct_root/ablr2/sensor_sources.json" && "$direct_value" != /workspace/ablr2/sensor_sources.json ]]; then
        echo 'Docker launcher mounts the pinned ablr2/sensor_sources.json catalog only.' >&2
        exit 2
      fi
      if [[ "$direct_flag" == --output && "${direct_args[i+1]}" != work_dir/* && "${direct_args[i+1]}" != ./work_dir/* && "${direct_args[i+1]}" != /workspace/work_dir/* ]]; then
        echo 'Docker output must remain under mounted work_dir/ for persistent results.' >&2
        exit 2
      fi
      i=$((i+1))
      ;;
    --device=*) direct_device="${direct_args[i]#--device=}";;
    --sources=*|--output=*|--baseline=*)
      echo 'For Docker use separated --sources VALUE / --output VALUE / --baseline VALUE arguments.' >&2
      exit 2
      ;;
  esac
done
if [[ "$direct_device" == cuda* ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 'Cannot verify GPU ownership: nvidia-smi unavailable. Nothing launched.' >&2
    exit 1
  fi
  if ! direct_gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"; then
    echo 'GPU ownership query failed. Nothing launched.' >&2
    exit 1
  fi
  if [[ -n "${direct_gpu_processes//[[:space:]]/}" ]]; then
    echo 'GPU compute workload active; refusing an overlapping experiment. No process was stopped.' >&2
    echo "$direct_gpu_processes" >&2
    exit 1
  fi
fi
docker image inspect "$direct_image" >/dev/null
if docker container inspect "$direct_name" >/dev/null 2>&1; then
  echo "Container $direct_name already exists; inspect it before a new launch. It was not removed." >&2
  exit 1
fi
direct_dlpan_input="${PANCRAFTER_DLPAN:-$direct_root/../DLPan-Toolbox}"
if [[ ! -d "$direct_dlpan_input" ]]; then
  echo 'DLPan evaluation source missing. Set PANCRAFTER_DLPAN to the existing audited toolbox directory.' >&2
  exit 1
fi
direct_dlpan="$(cd "$direct_dlpan_input" && pwd -P)"
if [[ ! -f "$direct_dlpan/01-DL-toolbox(Pytorch)/UDL/pansharpening/models/APNN/wald_utilities.py" ]]; then
  echo 'Audited DLPan wald_utilities.py was not found; no evaluation code is downloaded or substituted.' >&2
  exit 1
fi
for direct_path in "$direct_root" "$direct_dlpan"; do
  if [[ "$direct_path" == *','* || "$direct_path" == *$'\n'* || "$direct_path" == *$'\r'* ]]; then
    echo 'Docker mount paths must not contain commas or newlines.' >&2
    exit 1
  fi
done
mkdir -p "$direct_root/work_dir/directshift48"
# This unchanged helper resolves only registered H5 symlinks; it imports no
# matcher/model or trainer and makes every external target read-only.
direct_files="$("${PAN_HOST_PYTHON:-python}" -m edgeres48.docker_mounts "$direct_server")"
if [[ -z "$direct_files" ]]; then
  echo 'No registered H5 mounts were resolved; nothing launched.' >&2
  exit 1
fi
direct_mounts=()
while IFS= read -r direct_file; do
  direct_mounts+=(--mount "type=bind,src=$direct_file,dst=$direct_file,readonly")
done <<< "$direct_files"
direct_gpu_args=()
if [[ "$direct_device" == cuda* ]]; then
  direct_gpu_args=(--gpus all)
fi
exec docker run --detach --init "${direct_gpu_args[@]}" --shm-size=8g \
  --name "$direct_name" --user "$(id -u):$(id -g)" \
  --env PYTHONUNBUFFERED=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 --env MKL_THREADING_LAYER=GNU \
  --env PANCRAFTER_DLPAN=/opt/pancrafter-eval/DLPan-Toolbox \
  --mount "type=bind,src=$direct_root,dst=/workspace,readonly" \
  --mount "type=bind,src=$direct_root/work_dir,dst=/workspace/work_dir" \
  --mount "type=bind,src=$direct_dlpan,dst=/opt/pancrafter-eval/DLPan-Toolbox,readonly" \
  "${direct_mounts[@]}" \
  --workdir /workspace "$direct_image" \
  bash scripts/run_directshift48.sh "$direct_server" "${direct_args[@]}"
