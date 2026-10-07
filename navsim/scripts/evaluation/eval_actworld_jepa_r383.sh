#!/usr/bin/env bash
set -euo pipefail

required_env=(
  ACTWORLD_CHECKPOINT
  ACTWORLD_LATENT_VERIFIER
  ACTWORLD_CANDIDATE_SCORER
  ACTWORLD_TREE_META_SCORER
  ACTWORLD_R383_SELECTOR
  ACTWORLD_OUTPUT_DIR
  TEAM_NAME
  AUTHORS
  EMAIL
  INSTITUTION
  COUNTRY
)

for name in "${required_env[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

for name in \
  ACTWORLD_CHECKPOINT \
  ACTWORLD_LATENT_VERIFIER \
  ACTWORLD_CANDIDATE_SCORER \
  ACTWORLD_TREE_META_SCORER \
  ACTWORLD_R383_SELECTOR; do
  if [[ ! -f "${!name}" ]]; then
    echo "Required artifact does not exist: ${name}=${!name}" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAVSIM_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${NAVSIM_ROOT}"

python navsim/planning/script/create_r383_submission_parallel.py \
  train_test_split=navtest \
  agent=actworld_jepa_agent \
  worker=single_machine_thread_pool \
  worker.max_workers="${ACTWORLD_WORKERS:-10}" \
  worker.use_process_pool=true \
  output_dir="${ACTWORLD_OUTPUT_DIR}" \
  +team_name="${TEAM_NAME}" \
  +authors="${AUTHORS}" \
  +email="${EMAIL}" \
  +institution="${INSTITUTION}" \
  +country="${COUNTRY}" \
  agent.checkpoint_path="${ACTWORLD_CHECKPOINT}" \
  +agent.config.actworld_selection_mode=conservative \
  +agent.config.actworld_latent_verifier_path="${ACTWORLD_LATENT_VERIFIER}" \
  +agent.config.actworld_candidate_set_scorer_path="${ACTWORLD_CANDIDATE_SCORER}" \
  +agent.config.actworld_tree_meta_scorer_path="${ACTWORLD_TREE_META_SCORER}" \
  +agent.config.actworld_candidate_set_max_proxy_acceleration=6.1125 \
  +agent.config.actworld_candidate_set_max_proxy_jerk=16.74 \
  +agent.config.actworld_latent_risk_veto=false \
  +agent.config.actworld_risk_aware_v3_selection=false \
  +agent.config.actworld_current_future_rank_selection=true \
  +agent.config.actworld_current_future_rank_cascade_on_candidate_set=true \
  +agent.config.actworld_current_future_rank_top_k=8 \
  +agent.config.actworld_current_future_rank_probability_weight=0.0 \
  +agent.config.actworld_current_future_rank_min_probability=0.5 \
  +agent.config.actworld_current_future_rank_min_gain=0.0 \
  +agent.config.actworld_current_future_rank_min_safety_probability=0.2 \
  +agent.config.actworld_current_future_rank_max_candidate_gap=0.01 \
  +agent.config.actworld_current_future_rank_min_frozen_critical=-0.02 \
  +agent.config.actworld_current_future_rank_min_frozen_progress=0.0 \
  +agent.config.actworld_current_future_rank_min_frozen_safety=-0.05 \
  +agent.config.actworld_current_future_rank_min_frozen_ttc=-1.0 \
  +agent.config.actworld_current_future_rank_max_initial_lateral_acceleration_increase=1.0e9 \
  +agent.config.actworld_current_future_rank_min_positive_factor_count=0 \
  +agent.config.actworld_current_future_rank_positive_factor_floor=-1.0e9 \
  +agent.config.actworld_decoupled_current_future_branch=true \
  +agent.config.actworld_post_r94_selector_path="${ACTWORLD_R383_SELECTOR}" \
  +agent.config.actworld_post_r94_selector_allow_navtest_tuning=false
