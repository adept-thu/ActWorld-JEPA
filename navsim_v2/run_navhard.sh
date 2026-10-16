#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${ACTWORLD_WEIGHTS:?Set the downloaded artifact directory (absolute path)}"
: "${OPENSCENE_DATA_ROOT:?Set the two-stage NAVHARD data root}"
: "${NAVSIM_EXP_ROOT:?Set the experiment / Drive-JEPA-cache directory}"
: "${NUPLAN_MAPS_ROOT:?Set the nuPlan maps directory}"
: "${METRIC_CACHE_PATH:?Set the reactive two-stage metric cache}"
: "${OUTPUT_DIR:?Set a new absolute output directory}"
export OFFICIAL_NAVSIM_ROOT="$PWD/navhard_devkit"
export ACTWORLD_V2_ROOT="$PWD"
export ACTWORLD_NAVHARD_BASE_BRIDGE="$PWD/ops/run_actworld_jepa_v2_official_navhard_compat.py"
export ACTWORLD_SELECTOR_DIAGNOSTICS_DIR="$OUTPUT_DIR/diagnostics"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
source scripts/release_environment.sh
test ! -e "$OUTPUT_DIR" || { echo 'Use a new output directory'; exit 2; }
mkdir -p "$ACTWORLD_SELECTOR_DIAGNOSTICS_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python ops/run_actworld_jepa_v2_navhard_transfer_gpu.py \
  "hydra.searchpath=[file://$OFFICIAL_NAVSIM_ROOT/navsim/planning/script/config/common,file://$PWD/navsim/planning/script/config/common]" \
  train_test_split=navhard_two_stage agent=actworld_jepa_agent \
  agent.config.latent=false agent.config.actworld_future_latent_grounding=false \
  agent.config.actworld_training_stage=actor agent.config.actworld_critic_variant=f2_base_residual \
  agent.checkpoint_path="$ACTWORLD_WEIGHTS/actworld_navsim_v2.ckpt" \
  worker=single_machine_thread_pool worker.use_process_pool=true \
  worker.max_workers="${WORKERS:-2}" \
  +traffic_agents_policy.reactive.map_root_override="$NUPLAN_MAPS_ROOT" \
  metric_cache_path="$METRIC_CACHE_PATH" experiment_name=actworld_navhard_capture \
  output_dir="$OUTPUT_DIR/capture"
mapping="$OFFICIAL_NAVSIM_ROOT/navsim/planning/script/config/common/train_test_split/navhard_two_stage.yaml"
python ops/build_actworld_navhard_temporal_fcts_model.py \
  --diagnostics "$ACTWORLD_SELECTOR_DIAGNOSTICS_DIR" \
  --artifact "$ACTWORLD_WEIGHTS/temporal_fcts_pair_head.pkl" \
  --mapping-yaml "$mapping" --threshold .05 --fraction .02 \
  --output "$OUTPUT_DIR/pair.pkl" --report "$OUTPUT_DIR/pair.json"
python ops/apply_actworld_temporal_fcts_head_navhard.py \
  --diagnostics "$ACTWORLD_SELECTOR_DIAGNOSTICS_DIR" \
  --artifact "$ACTWORLD_WEIGHTS/temporal_fcts_head.pt" \
  --mapping-yaml "$mapping" --base-trajectories "$OUTPUT_DIR/pair.pkl" \
  --output "$OUTPUT_DIR/head.pkl" --report "$OUTPUT_DIR/head.json" \
  --max-variant 3 --policy-margin .02 --policy-safety-threshold .95 \
  --policy-progress-threshold .95 --policy-component-margin .01 \
  --policy-hard-metric-threshold .98 --policy-quality-margin-threshold -.05 \
  --target-fraction .01804812834224599
python ops/build_v2_temporal_joint_channel_deployment.py \
  --base-trajectories "$OUTPUT_DIR/head.pkl" --mapping-yaml "$mapping" \
  --navtrain-report ops/policies/channel_policy.json \
  --output "$OUTPUT_DIR/channel.pkl" --report "$OUTPUT_DIR/channel.json"
python ops/build_v2_temporal_dense_rollout_deployment.py \
  --base-trajectories "$OUTPUT_DIR/channel.pkl" --mapping-yaml "$mapping" \
  --navtrain-report ops/policies/dense_policy.json \
  --output "$OUTPUT_DIR/dense.pkl" --report "$OUTPUT_DIR/dense.json"
python ops/build_v2_stage1_arc250_deployment.py \
  --base-trajectories "$OUTPUT_DIR/dense.pkl" --mapping-yaml "$mapping" \
  --navtrain-report ops/policies/stage1_policy.json --strength .25 --variant arc250 \
  --output "$OUTPUT_DIR/trajectories.pkl" --report "$OUTPUT_DIR/deployment.json"
# Compatibility names retained by the frozen evaluation snapshot; no model
# from the snapshot is invoked in score-only mode.
export MOMWORLD_SCORE_ONLY=true
export MOMWORLD_MERGED_TRAJECTORY_PATH="$OUTPUT_DIR/trajectories.pkl"
export PYTHONPATH="$PWD/ops/score_only_compat:$OFFICIAL_NAVSIM_ROOT:$PWD:${PYTHONPATH:-}"
python "$OFFICIAL_NAVSIM_ROOT/navsim/planning/script/run_pdm_score_gpu_v2.py" \
  train_test_split=navhard_two_stage agent=constant_velocity_agent \
  worker=single_machine_thread_pool worker.use_process_pool=true \
  worker.max_workers="${SCORE_WORKERS:-8}" \
  +traffic_agents_policy.reactive.map_root_override="$NUPLAN_MAPS_ROOT" \
  metric_cache_path="$METRIC_CACHE_PATH" experiment_name=actworld_navhard_release \
  output_dir="$OUTPUT_DIR/scored"
