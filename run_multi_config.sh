#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
START_DF="${1:-2}"
MAX_DF="${2:-10}"
OUT_ROOT="${3:-$ROOT_DIR/Exps_multi_config}"

run_experiment() {
  local name="$1"
  local config="$2"
  local base_dir="$OUT_ROOT/$name"

  mkdir -p "$base_dir"

  echo "=== [$name] Initializing ${START_DF}df with $(basename "$config") ==="
  python3 "$ROOT_DIR/auto_drop_zeros/pipeline.py" \
    --config "$config" \
    --base "$base_dir" \
    --start-df "$START_DF" \
    --init

  echo "=== [$name] Running ${START_DF}df through ${MAX_DF}df ==="
  bash "$ROOT_DIR/auto_drop_zeros/run_all.sh" "$START_DF" "$MAX_DF" "$base_dir" "$config"
}
run_experiment "cfg0_pos7_14_neg16_23" "$ROOT_DIR/auto_drop_zeros/configs/pipeline_config_cfg0.json"
# run_experiment "cfg1_pos7_12_neg16_20" "$ROOT_DIR/auto_drop_zeros/configs/pipeline_config_cfg1.json"
run_experiment "cfg2_pos7_12_neg18_22" "$ROOT_DIR/auto_drop_zeros/configs/pipeline_config_cfg2.json"
# run_experiment "cfg3_pos0_3_neg3_5" "$ROOT_DIR/auto_drop_zeros/configs/pipeline_config_cfg3.json"

echo "All multi-config experiments completed under $OUT_ROOT."
