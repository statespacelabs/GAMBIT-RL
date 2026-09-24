#!/usr/bin/env bash
# Production Phase 3A distillation gates. This script cannot start PPO training.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
TRANSITION_MANIFEST="$PROJECT_ROOT/artifacts/demonstrations/transitions/manifests/phase2_transition_manifest.csv"
WINDOW_MANIFEST="$PROJECT_ROOT/artifacts/demonstrations/transitions/manifests/phase2_window_manifest.csv"
TELEMETRY_STATS="$PROJECT_ROOT/artifacts/demonstrations/packed/telemetry_stats.npz"
TELEMETRY_DIR="$PROJECT_ROOT/artifacts/demonstrations/commands/telemetry_npz_v2"
TELEMETRY_MANIFEST="$PROJECT_ROOT/artifacts/demonstrations/commands/window_manifest_with_tel.csv"
ACTION_DIR="$PROJECT_ROOT/artifacts/demonstrations/commands/action_npz_v2"
ACTION_MANIFEST="$PROJECT_ROOT/artifacts/demonstrations/commands/phase2_transition_manifest_command_labels.csv"
SPLIT_TRANSITIONS="$PROJECT_ROOT/artifacts/demonstrations/commands/phase2_transition_manifest_command_labels_split.csv"
SPLIT_WINDOWS="$PROJECT_ROOT/artifacts/demonstrations/commands/window_manifest_with_tel_split.csv"
UNITY_BUILD="${UNITY_BUILD:-$PROJECT_ROOT/artifacts/unity/combat}"

DISTILL_CONFIG="configs/distill_production.yaml"
PPO_CONFIG="configs/ppo_production.yaml"
OUTPUT_ROOT="experiments/phase3/distill_production"
LOG_DIR="$OUTPUT_ROOT/logs"
REPORT="$OUTPUT_ROOT/REPORT.md"

mkdir -p "$LOG_DIR"

for required in "$PYTHON" "$TRANSITION_MANIFEST" "$WINDOW_MANIFEST" \
    "$TELEMETRY_STATS" "checkpoints_v2_large/best_checkpoint.pt"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required production input: $required" >&2
        exit 1
    fi
done

echo "[1/8] Building production telemetry"
"$PYTHON" -m gambit.rl.online_rl.build_telemetry_npz \
    --window_manifest "$WINDOW_MANIFEST" \
    --telemetry_stats "$TELEMETRY_STATS" \
    --out_dir "$TELEMETRY_DIR" \
    --out_manifest "$TELEMETRY_MANIFEST" \
    --report "$LOG_DIR/telemetry_build_report.json" \
    --resume --skip-existing \
    2>&1 | tee "$LOG_DIR/telemetry_build.log"

echo "[2/8] Rebuilding Phase 3 command labels from source events"
"$PYTHON" scripts/build_command_targets.py \
    --transition_manifest "$TRANSITION_MANIFEST" \
    --window_manifest "$WINDOW_MANIFEST" \
    --out_dir "$ACTION_DIR" \
    --out_manifest "$ACTION_MANIFEST" \
    --report "$LOG_DIR/action_build_report.json" \
    --resume --skip-existing \
    2>&1 | tee "$LOG_DIR/action_build.log"

echo "[3/8] Creating leakage-safe splits"
"$PYTHON" scripts/split_demonstration_sessions.py \
    --transition_manifest "$ACTION_MANIFEST" \
    --window_manifest "$TELEMETRY_MANIFEST" \
    --out_transition_manifest "$SPLIT_TRANSITIONS" \
    --out_window_manifest "$SPLIT_WINDOWS" \
    --seed 42 \
    --report "$LOG_DIR/split_report.json" \
    2>&1 | tee "$LOG_DIR/split.log"

echo "[4/8] Checking production dataset and alignment"
"$PYTHON" scripts/check_command_dataset.py \
    --transition_manifest "$SPLIT_TRANSITIONS" \
    --window_manifest "$SPLIT_WINDOWS" \
    --sample_count 100 \
    --seed 42 \
    --report "$LOG_DIR/dataset_check_report.json" \
    2>&1 | tee "$LOG_DIR/dataset_check.log"

echo "[5/8] Feedforward command distillation"
"$PYTHON" -m gambit.rl.online_rl.train_ppo \
    --ppo_config "$PPO_CONFIG" \
    --distill_config "$DISTILL_CONFIG" \
    --stage feedforward_distill_only \
    2>&1 | tee "$LOG_DIR/feedforward_distill.log"

echo "[6/8] Recurrent behavior-cloning warmup"
"$PYTHON" -m gambit.rl.online_rl.train_ppo \
    --ppo_config "$PPO_CONFIG" \
    --distill_config "$DISTILL_CONFIG" \
    --stage recurrent_distill_only \
    2>&1 | tee "$LOG_DIR/recurrent_distill.log"

echo "[7/8] Verifying recurrent checkpoint deltas"
"$PYTHON" scripts/check_recurrent_initialization.py \
    --distill_config "$DISTILL_CONFIG" \
    --report "$LOG_DIR/recurrent_delta_report.json" \
    2>&1 | tee "$LOG_DIR/recurrent_delta.log"

echo "[8/8] PPO invariant and 100-step Unity rollout, without optimization"
xvfb-run -a "$PYTHON" -m gambit.rl.online_rl.train_ppo \
    --ppo_config "$PPO_CONFIG" \
    --distill_config "$DISTILL_CONFIG" \
    --env_path "$UNITY_BUILD" \
    --stage invariant_only \
    2>&1 | tee "$LOG_DIR/invariant.log"

"$PYTHON" scripts/write_distillation_report.py \
    --root "$OUTPUT_ROOT" \
    --output "$REPORT"

echo "Feedforward checkpoint: $OUTPUT_ROOT/checkpoints/command_best.pt"
echo "Recurrent checkpoint: $OUTPUT_ROOT/checkpoints/command_recurrent_best.pt"
echo "Final report: $REPORT"
echo "STOPPED: PPO training was not started."
