#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${ROOT_DIR}/auto_drop_zeros/configs/pipeline_config_cfg1.json"

START_DF="${1:-2}"
END_DF="${2:-10}"
OUT_ROOT="${3:-$ROOT_DIR/Exps_multi_seed}"

if [[ $# -ge 3 ]]; then
  shift 3
else
  shift $#
fi

if [[ $# -eq 0 ]]; then
  SEEDS=(1 2 3 4)
else
  SEEDS=("$@")
fi

write_seed_config() {
  local src_config="$1"
  local dst_config="$2"
  local seed="$3"
  local max_df="$4"

  python3 - "$src_config" "$dst_config" "$seed" "$max_df" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
seed = int(sys.argv[3])
max_df = int(sys.argv[4])

cfg = json.loads(src.read_text())
cfg["x0_seed"] = seed
cfg["ou_seed"] = seed
cfg["one_step_seed"] = seed
cfg["design_seed"] = seed
cfg["max_df"] = max_df

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(cfg, indent=2) + "\n")
PY
}

run_df_training_and_bo() {
  local df="$1"
  local base_dir="$2"

  local scripts_dir="${base_dir}/${df}df/scripts"
  local bo_script="${scripts_dir}/BO_${df}d.sh"
  local train_script="${scripts_dir}/run_shadowgp_trainer.sh"

  if [[ ! -f "$train_script" ]]; then
    echo "Missing trainer script: $train_script"
    exit 1
  fi
  if [[ ! -f "$bo_script" ]]; then
    echo "Missing BO script: $bo_script"
    exit 1
  fi

  echo "=== [${df}d] Running initial ShadowGP samples ==="
  (cd "$scripts_dir" && bash "$(basename "$train_script")")

  echo "=== [${df}d] Running BO + trainer ==="
  if (cd "$scripts_dir" && bash "$(basename "$bo_script")"); then
    echo "=== [${df}d] BO finished ==="
  else
    echo "=== [${df}d] BO terminated early (no more candidates) ==="
  fi
}

split_and_generate_next_df() {
  local df="$1"
  local base_dir="$2"
  local config="$3"

  echo "=== [${df}d] Computing split + generating $((df + 1))df ==="
  set +u
  if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda.sh not found under ~/miniforge3 or ~/miniconda3."
    exit 1
  fi
  conda activate test
  set -u

  python3 "$ROOT_DIR/auto_drop_zeros/pipeline.py" \
    --config "$config" \
    --base "$base_dir" \
    --start-df "$df" \
    --max-df "$((df + 1))"
}

for seed in "${SEEDS[@]}"; do
  exp_name="seed_${seed}"
  exp_base="${OUT_ROOT}/${exp_name}"
  exp_config="${exp_base}/pipeline_config_seed_${seed}.json"

  mkdir -p "$exp_base"
  write_seed_config "$BASE_CONFIG" "$exp_config" "$seed" "$((END_DF + 1))"

  echo "======================================================="
  echo "Running ${exp_name}"
  echo "Base dir: $exp_base"
  echo "Config:   $exp_config"
  echo "DF range: ${START_DF}df through ${END_DF}df"
  echo "Seeds:    x0=ou=one_step=design=${seed}"
  echo "======================================================="

  echo "=== [${exp_name}] Initializing ${START_DF}df ==="
  python3 "$ROOT_DIR/auto_drop_zeros/pipeline.py" \
    --config "$exp_config" \
    --base "$exp_base" \
    --start-df "$START_DF" \
    --init

  for ((df=START_DF; df<=END_DF; df++)); do
    run_df_training_and_bo "$df" "$exp_base"

    if [[ "$df" -lt "$END_DF" ]]; then
      split_and_generate_next_df "$df" "$exp_base" "$exp_config"
    else
      echo "=== [${df}d] Final df reached; skipping split so $((df + 1))df is not generated ==="
    fi
  done
done

echo "Completed seed experiments: ${SEEDS[*]}"
echo "Outputs written under: $OUT_ROOT"
