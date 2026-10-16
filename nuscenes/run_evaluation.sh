#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${ACTWORLD_WEIGHTS:?Set the downloaded Hugging Face artifact directory}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p work_dirs
python -m torch.distributed.launch --nproc_per_node=1 --master_port="${MASTER_PORT:-29519}" \
  tools/test.py projects/configs/actworld_nuscenes.py \
  "$ACTWORLD_WEIGHTS/actworld_nuscenes_r13.pth" \
  --out work_dirs/actworld_nuscenes_predictions.pkl --eval bbox \
  --launcher pytorch --seed 41 --deterministic \
  --cfg-options data.workers_per_gpu=4 work_dir=work_dirs/actworld_nuscenes "$@"
