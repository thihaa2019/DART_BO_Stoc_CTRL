# Auto Drop Zeros Pipeline

`auto_drop_zeros/` generates and advances the BO/ShadowGP experiment folders under a chosen base directory. In the current workspace, the main experiment directory is:

```bash
/home/taung/Desktop/BO_exps/Exp_drop_zeros
```

The generated folders look like:

```text
BO_exps/Exp_drop_zeros/
  auto/
  2df/
    blocks.json
    split_metrics.json
    split_metrics.txt
    scripts/
      run_shadowgp_trainer.sh
      BO_2d.sh
      bo_2d.py
      run_shadowgp_Trainer.py
      sampler.py
  3df/
  ...
```

## Conda Setup

Generated scripts now source conda from either:

```bash
~/miniforge3/etc/profile.d/conda.sh
~/miniconda3/etc/profile.d/conda.sh
```

On this machine, `~/miniforge3` exists. The generated scripts use:

- `gpytorchCPU` for sampler and BO/Gaussian-process steps.
- `test` for ShadowGP training and pipeline split generation.

## Initialize 2df

From `/home/taung/Desktop`:

```bash
python3 BO_exps/auto_drop_zeros/pipeline.py \
  --config BO_exps/auto_drop_zeros/configs/pipeline_config_cfg1.json \
  --base BO_exps/Exp_drop_zeros \
  --start-df 2 \
  --init
```

This creates:

```bash
BO_exps/Exp_drop_zeros/2df
```

## Run One df Manually

For the initial 2D experiment:

```bash
cd /home/taung/Desktop/BO_exps/Exp_drop_zeros/2df/scripts
bash run_shadowgp_trainer.sh
bash BO_2d.sh
```

Then compute the split and generate the next df:

```bash
cd /home/taung/Desktop
source ~/miniforge3/etc/profile.d/conda.sh
conda activate test

python3 BO_exps/auto_drop_zeros/pipeline.py \
  --config BO_exps/auto_drop_zeros/configs/pipeline_config_cfg1.json \
  --base BO_exps/Exp_drop_zeros \
  --start-df 2 \
  --max-df 3
```

## Run the Full Chain

The helper script runs each df's initial ShadowGP samples, BO loop, split computation, and next-folder generation.

```bash
cd /home/taung/Desktop
bash BO_exps/auto_drop_zeros/run_all.sh \
  2 10 \
  BO_exps/Exp_drop_zeros \
  BO_exps/auto_drop_zeros/configs/pipeline_config_cfg1.json
```

Arguments:

```text
run_all.sh <start_df> <max_df> [base_dir] [config_path]
```

`max_df` is exclusive in `run_all.sh`; `2 10` runs `2df` through `9df` and generates `10df`.

## Multi-Seed Runs

```bash
cd /home/taung/Desktop
bash BO_exps/auto_drop_zeros/run_multi_seed.sh 2 10 BO_exps/Exps_multi_seed 1 2 3 4
```

Outputs are written under:

```bash
BO_exps/Exps_multi_seed/seed_1
BO_exps/Exps_multi_seed/seed_2
...
```

## Zero-Drop Rule

After each df finishes, the pipeline checks the final posterior-mean record:

- If a block's final posterior-mean alpha is zero within `drop_zero_tol`, that block is marked `fixed_zero`.
- Fixed-zero blocks stay in `blocks.json` and `B_DA.py`.
- Fixed-zero blocks are removed from the BO decision vector in later dfs.
- If a fixed-zero block is split later, both children remain fixed at zero.

This lets the structural partition keep refining while the BO dimension does not grow for regions already fixed at zero.

## Adaptive BO Budget

Configs can enable dimension-dependent BO stopping with:

```json
"adaptive_bo_budget": true,
"bo_extra_iter_per_dim": 10,
"bo_epsilon_base": 0.1,
"bo_epsilon_ref_dim": 2
```

When enabled, each generated `BO_<df>d.sh` uses the current BO dimension `d`, where `d` is the number of non-`fixed_zero` decision variables:

```text
n0(d)      = floor(6 * sqrt(d))
nmax(d)    = n0(d) + 10d
MAX_ITER   = nmax(d) - n0(d) = 10d
epsilon(d) = 0.1 * 2^-(d - 2)
THRESH     = epsilon(d)
```

So the generated BO loop runs more iterations and uses a stricter UCB-LCB stopping threshold as the BO dimension increases:

```text
d   n0   nmax   MAX_ITER   THRESH
2    8     28      20      0.1
3   10     40      30      0.05
4   12     52      40      0.025
5   13     63      50      0.0125
6   14     74      60      0.00625
7   15     85      70      0.003125
8   16     96      80      0.0015625
```

If `"adaptive_bo_budget": false`, the pipeline falls back to the fixed config values:

```json
"thresh": 1e-1,
"max_iter": 30
```

## Split Metrics

`choose_split` and `compute_split_metrics` evaluate candidate splits using RT adjustment energy on each side.

For a candidate split at hour `k` inside segment `[start, end]`:

```python
lhs = B_96[:, start * 4 : k * 4].sum(axis=1) * 0.25
rhs = B_96[:, k * 4 : end * 4].sum(axis=1) * 0.25
```

So `lhs_mean`, `rhs_mean`, and `abs_mean_diff` are not divided by side length. The metric is the unnormalized side energy score.

The reports are saved in each df folder:

```bash
BO_exps/Exp_drop_zeros/<df>df/split_metrics.txt
BO_exps/Exp_drop_zeros/<df>df/split_metrics.json
```
