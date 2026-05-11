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

run_pipeline () {
    local start_df="$1"
    local max_df="$2"
    shift 2

    activate_pipeline_env
    python3 auto_drop_zeros/pipeline.py \
        --config "$CONFIG" \
        --base "$BASE" \
        --start-df "$start_df" \
        --max-df "$max_df" "$@"
}

echo "======================================================="
echo "Two-splits-per-round end-to-end run"
echo "CONFIG   = $CONFIG"
echo "BASE     = $BASE"
echo "START_DF = $START_DF"
echo "END_DF   = $END_DF"
echo "======================================================="

run_pipeline "$START_DF" "$END_DF" --init

df="$START_DF"
while [[ "$df" -le "$END_DF" ]]; do
    scripts_dir="$BASE/${df}df/scripts"
    if [[ ! -d "$scripts_dir" ]]; then
        echo "ERROR: missing scripts dir: $scripts_dir"
        echo "Expected the previous pipeline split to generate ${df}df."
        exit 1
    fi

    echo "======================================================="
    echo "[${df}df] Running sampler/initial ShadowGP if needed"
    echo "======================================================="
    (cd "$scripts_dir" && bash run_shadowgp_trainer.sh)

    echo "======================================================="
    echo "[${df}df] Running BO loop/final postmean"
    echo "======================================================="
    bo_script="BO_${df}d.sh"
    if [[ ! -f "$scripts_dir/$bo_script" ]]; then
        echo "ERROR: missing BO script: $scripts_dir/$bo_script"
        exit 1
    fi
    if (cd "$scripts_dir" && bash "$bo_script"); then
        echo "=== [${df}d] BO finished ==="
    else
        echo "=== [${df}d] BO terminated early (no more candidates) ==="
    fi

    if [[ "$df" -lt "$END_DF" ]]; then
        echo "======================================================="
        echo "[${df}df] Splitting and generating next df"
        echo "======================================================="
        run_pipeline "$df" "$((df + 1))"
    fi

    df=$((df + 2))
done

echo "======================================================="
echo "Done. Finished BO through ${END_DF}df in $BASE."
echo "======================================================="
