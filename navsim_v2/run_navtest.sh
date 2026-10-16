#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${ACTWORLD_WEIGHTS:?Set the downloaded Hugging Face artifact directory}"
: "${OPENSCENE_DATA_ROOT:?Set the NAVSIM data directory}"
: "${NAVSIM_EXP_ROOT:?Set the experiment / Drive-JEPA-cache directory}"
: "${NUPLAN_MAPS_ROOT:?Set the nuPlan maps directory}"
: "${METRIC_CACHE_PATH:?Set the NAVSIM v2 NAVTEST metric cache}"
source scripts/release_environment.sh
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest agent=actworld_jepa_agent \
  agent.config.latent=false agent.config.actworld_future_latent_grounding=false \
  agent.config.actworld_training_stage=actor agent.config.actworld_critic_variant=f2_base_residual \
  agent.checkpoint_path="$ACTWORLD_WEIGHTS/actworld_navsim_v2.ckpt" \
  metric_cache_path="$METRIC_CACHE_PATH" \
  experiment_name=actworld_navtest_release "$@"
