#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <config_dir> <start_df> <max_df> [out_root]"
  echo "Example: $0 auto_drop_zeros/configs 2 10"
  echo "Example: $0 auto_drop_zeros/configs_double_peak 2 11 ./Exps_double_peak"
  exit 1
fi

CONFIG_DIR="$1"
START_DF="$2"
MAX_DF="$3"
OUT_ROOT="${4:-$ROOT_DIR/Exps_multi_config/$(basename "$CONFIG_DIR")}"

if [[ "$CONFIG_DIR" != /* ]]; then
  if [[ -d "$CONFIG_DIR" ]]; then
    CONFIG_DIR="$(cd "$CONFIG_DIR" && pwd)"
  else
    CONFIG_DIR="$ROOT_DIR/$CONFIG_DIR"
  fi
fi
CONFIG_DIR="$(cd "$CONFIG_DIR" && pwd)"

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

shopt -s nullglob
configs=("$CONFIG_DIR"/pipeline_config*.json)
shopt -u nullglob

if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No pipeline_config*.json files found under: $CONFIG_DIR"
  exit 1
fi

for config in "${configs[@]}"; do
  run_experiment "$(basename "$config" .json)" "$config"
done

echo "All multi-config experiments completed under $OUT_ROOT."
