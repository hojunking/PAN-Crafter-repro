#!/usr/bin/env bash
# Finite, restartable PANPV R1 pipeline. Does not stop or modify other jobs.
set -euo pipefail
panpv_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
panpv_expected=/home/knuvi/Desktop/song/PAN-Crafter
panpv_name=panpv-s1-wv2-20261004-r1
panpv_image=sha256:32a1e69bfadaf7f838a0f19361e726aee1ee8bfce5a7eab11fc06433ed1f806d
panpv_work="$panpv_repo/work_dir/PANPV_S1_WV2_20261004_R1/s1"
panpv_prep="$panpv_repo/work_dir/PANPV_S1_WV2_20261003_v3/s1/preparation_status.json"
if [[ "$panpv_repo" != "$panpv_expected" || "$(hostname)" != knuvi-hojun-local ]]; then
    echo 'This launcher is frozen to the verified s1 repository and host.' >&2
    exit 1
fi
test -f "$panpv_prep"
test -d /home/knuvi/Desktop/song/CANConv/data/datasets/wv3
mkdir -p -- "$panpv_work"
if docker container inspect "$panpv_name" >/dev/null 2>&1; then
    if [[ "$(docker inspect -f '{{.Image}}' "$panpv_name")" != "$panpv_image" ]]; then
        echo 'Existing named container has a different image; refusing to reuse it.' >&2
        exit 1
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$panpv_name")" == true ]]; then
        echo "$panpv_name is already running. Use docker logs --tail 5 $panpv_name"
        exit 0
    fi
    # The frozen manifest and fullstates enforce same-source exact resume.
    exec docker start "$panpv_name"
fi
exec docker run -d --name "$panpv_name" --init --restart no \
    --read-only --user "$(id -u):$(id -g)" --stop-timeout 300 \
    --gpus device=GPU-7b245231-a17d-06d7-2475-02bc7bb31ce2 \
    --tmpfs /tmp:rw,nosuid,size=2g --shm-size 1g \
    --log-opt max-size=20m --log-opt max-file=5 \
    -e PANPV_HOSTNAME=knuvi-hojun-local -e PANPV_IMAGE_ID="$panpv_image" \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 \
    -v "$panpv_repo:$panpv_repo:ro" -v "$panpv_work:$panpv_work:rw" \
    -v /home/knuvi/Desktop/song/CANConv/data/datasets/wv3:/home/knuvi/Desktop/song/CANConv/data/datasets/wv3:ro \
    -w "$panpv_repo" "$panpv_image" \
    python -m panpv.pipeline --repo-root "$panpv_repo" --work-root "$panpv_work" \
    --preparation "$panpv_prep" --device cuda
