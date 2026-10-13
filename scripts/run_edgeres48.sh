#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || ! "$1" =~ ^s[123]$ ]]; then
  echo 'Usage: bash scripts/run_edgeres48.sh s1|s2|s3 [extra run arguments]' >&2
  exit 2
fi
server="$1"
shift
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MKL_THREADING_LAYER=GNU
edgeres_publish_args=(--server "$server" --watch)
edgeres_run_args=("$@")
for ((i=0; i<${#edgeres_run_args[@]}; i++)); do
  case "${edgeres_run_args[i]}" in
    --output|--credentials)
      if ((i+1 >= ${#edgeres_run_args[@]})); then
        echo "Missing value for ${edgeres_run_args[i]}" >&2
        exit 2
      fi
      edgeres_publish_args+=("${edgeres_run_args[i]}" "${edgeres_run_args[i+1]}")
      i=$((i+1))
      ;;
    --output=*|--credentials=*) edgeres_publish_args+=("${edgeres_run_args[i]}");;
  esac
done
"${PAN_PYTHON:-python}" -m edgeres48 run --server "$server" "$@"
# Training closes at its deadline. Keep the container alive only to drain its
# durable outbox; an existing publisher and this one use the same host-local guard.
exec "${PAN_PYTHON:-python}" -m edgeres48 publish "${edgeres_publish_args[@]}"
