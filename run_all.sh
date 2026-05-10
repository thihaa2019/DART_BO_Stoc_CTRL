#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_DIR="$ROOT_DIR"
CONFIG="$ROOT_DIR/auto_drop_zeros/pipeline_config.json"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <start_df> <max_df> [base_dir] [config_path]"
  echo "Example: $0 2 7"
  echo "Example: $0 2 10 ./seed1"
  echo "Example: $0 2 10 ./exp1 ./auto_drop_zeros/configs/pipeline_config_cfg1.json"
  exit 1
fi

START_DF="$1"
MAX_DF="$2"
if [[ $# -ge 3 ]]; then
  BASE_DIR="$(cd "$3" && pwd)"
fi
if [[ $# -ge 4 ]]; then
  CONFIG="$(cd "$(dirname "$4")" && pwd)/$(basename "$4")"
fi
export PIPELINE_CONFIG_PATH="$CONFIG"

INCLUDE_0DF="$(python3 - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path(__import__("os").environ["PIPELINE_CONFIG_PATH"]).read_text())
print("true" if cfg.get("include_0df", False) else "false")
PY
)"

if [[ "$INCLUDE_0DF" == "true" ]]; then
  ZERO_SCRIPTS_DIR="$BASE_DIR/0df/scripts"
  ZERO_TRAIN_SCRIPT="$ZERO_SCRIPTS_DIR/run_shadowgp_trainer.sh"
  if [[ -f "$ZERO_TRAIN_SCRIPT" ]]; then
    echo "=== [0d] Running RT-only ShadowGP (B_DA=0) ==="
    (cd "$ZERO_SCRIPTS_DIR" && bash "run_shadowgp_trainer.sh")
  else
    echo "Missing 0df trainer script: $ZERO_TRAIN_SCRIPT"
    echo "Run pipeline first to initialize 0df."
    exit 1
  fi
fi

for ((df=START_DF; df<MAX_DF; df++)); do
  SCRIPTS_DIR="$BASE_DIR/${df}df/scripts"
  BO_SCRIPT="$SCRIPTS_DIR/BO_${df}d.sh"
  TRAIN_SAMPLES_SCRIPT="$SCRIPTS_DIR/run_shadowgp_trainer.sh"

  echo "=== [${df}d] Running 5-sample ShadowGP ==="
  (cd "$SCRIPTS_DIR" && bash "run_shadowgp_trainer.sh")

  echo "=== [${df}d] Running BO + trainer ==="
  if (cd "$SCRIPTS_DIR" && bash "$(basename "$BO_SCRIPT")"); then
    echo "=== [${df}d] BO finished ==="
  else
    echo "=== [${df}d] BO terminated early (no more candidates) ==="
  fi

  echo "=== [${df}d] Computing split + generating next df ==="
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
    --config "$CONFIG" \
    --base "$BASE_DIR" \
    --start-df "$df" \
    --max-df "$((df+1))"

done

echo "All done up to ${MAX_DF}d."
