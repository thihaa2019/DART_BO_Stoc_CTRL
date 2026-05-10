#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-auto_drop_zeros/configs/pipeline_config_cfg1.json}"
BASE="${BASE:-Exp_two_splits}"
START_DF="${START_DF:-2}"
END_DF="${END_DF:-10}"
PIPELINE_ENV="${PIPELINE_ENV:-test}"

cd "$(dirname "$0")/.."

if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found under ~/miniforge3 or ~/miniconda3."
    exit 1
fi

activate_pipeline_env () {
    conda activate "$PIPELINE_ENV"
}

has_results_index () {
    local outdir="$1"
    [[ -f "$outdir/results_index.pkl" ]]
}

has_final_postmean () {
    local outdir="$1"
    python3 - "$outdir" <<'PY'
import os
import pickle
import sys

outdir = sys.argv[1]
path = os.path.join(outdir, "results_index.pkl")
if not os.path.exists(path):
    raise SystemExit(1)
with open(path, "rb") as f:
    index = pickle.load(f)
records = index.get("records", {}) if isinstance(index, dict) else {}
final_id = index.get("final_postmean_id") if isinstance(index, dict) else None
if final_id is None or str(final_id) not in records:
    raise SystemExit(1)
rec = records[str(final_id)]
if "profit" not in rec:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

run_pipeline () {
    activate_pipeline_env
    python3 auto_drop_zeros/pipeline.py \
        --config "$CONFIG" \
        --base "$BASE" \
        --start-df "$START_DF" \
        --max-df "$END_DF" "$@"
}

echo "======================================================="
echo "Two-splits-per-round end-to-end run"
echo "CONFIG   = $CONFIG"
echo "BASE     = $BASE"
echo "START_DF = $START_DF"
echo "END_DF   = $END_DF"
echo "======================================================="

run_pipeline --init

df="$START_DF"
while [[ "$df" -le "$END_DF" ]]; do
    scripts_dir="$BASE/${df}df/scripts"
    outdir="$BASE/${df}df/outputs/ndec96nstep96_${df}d"

    if [[ ! -d "$scripts_dir" ]]; then
        echo "ERROR: missing scripts dir: $scripts_dir"
        echo "Expected the previous pipeline split to generate ${df}df."
        exit 1
    fi

    echo "======================================================="
    echo "[${df}df] Running sampler/initial ShadowGP if needed"
    echo "======================================================="
    if has_results_index "$outdir"; then
        echo "[${df}df] results_index.pkl exists; skipping initial sampler."
    else
        (cd "$scripts_dir" && bash run_shadowgp_trainer.sh)
    fi

    echo "======================================================="
    echo "[${df}df] Running BO loop/final postmean if needed"
    echo "======================================================="
    if has_final_postmean "$outdir"; then
        echo "[${df}df] final_postmean_id exists; skipping BO loop."
    else
        bo_script="BO_${df}d.sh"
        if [[ ! -f "$scripts_dir/$bo_script" ]]; then
            echo "ERROR: missing BO script: $scripts_dir/$bo_script"
            exit 1
        fi
        (cd "$scripts_dir" && bash "$bo_script")
    fi

    if [[ "$df" -lt "$END_DF" ]]; then
        echo "======================================================="
        echo "[${df}df] Splitting and generating next df"
        echo "======================================================="
        run_pipeline
    fi

    df=$((df + 2))
done

echo "======================================================="
echo "Done. Finished BO through ${END_DF}df in $BASE."
echo "======================================================="
