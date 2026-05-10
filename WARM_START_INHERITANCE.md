# BO Warm-Start Inheritance

Note: this copy lives under `auto_drop_zeros/`. The warm-start behavior is the same as `auto/`, but `auto_drop_zeros/` adds the separate zero-fixing rule documented in [README.md](README.md).

When the pipeline advances from `kdf` to `(k+1)df`, the new df can reuse evaluated samples from the previous df instead of starting BO from only the new LHS samples.

## Folder Versions

There are now two `auto` folders:

```text
auto/
auto_no_reuse/
```

`auto/` is the active warm-start version. It includes the inheritance logic described in this document.

`auto_no_reuse/` is the baseline version immediately before warm-start inheritance was added. It does **not** reuse evaluated samples from previous dfs.

The `auto_no_reuse/` folder still includes the earlier pipeline improvements made before warm-start inheritance:

- QMC/LHS sampler
- UCB-only BO
- posterior-mean final evaluation
- `results_index.pkl` index-based result naming
- cleanup that keeps all profit records but only the best available policy map
- repeated-candidate BO stopping
- `B_96 * 0.25` split-metric scaling

So `auto_no_reuse/` means "no cross-df sample reuse", not "the original untouched pipeline."

## Existing Pipeline Features

These features are present in both `auto/` and `auto_no_reuse/`.

### QMC/LHS sampler

Initial samples are generated with SciPy's `qmc.LatinHypercube` over the sign-dependent bounding box:

```text
pos block -> [0, Bmax]
neg block -> [-Bmax, 0]
```

The sampler generates a fixed candidate pool and keeps feasible points that pass the SoC inequality constraints.

### UCB-only BO

BO uses UCB only. The acquisition type is no longer selected from config.

The UCB beta is computed dynamically from the current number of observations and dimension:

```python
beta_iter = 2.0 * log(num_params * iter_num**2 * pi**2 / (6.0 * delta))
```

### Posterior-mean final evaluation

Each `BO_Xd.sh` loop stops when either:

- `MAX_ITER` is reached
- `ucb_minus_lcb < THRESH`
- the repeated-candidate heuristic triggers

After stopping, it runs one final ShadowGP evaluation at the posterior-mean maximizer.

This final run is tagged in `results_index.pkl`:

```python
results_index["final_postmean_id"] = record_id
```

The next split uses this final posterior-mean evaluated policy when it exists and has a policy map.

### `results_index.pkl` index naming

Evaluated results are stored in `results_index.pkl` instead of encoding all alpha values into long filenames.

Policy maps use compact index filenames:

```text
policy_maps_000001.pkl
policy_maps_000002.pkl
...
```

Each record stores alphas at high precision:

```python
"alphas": [...]
"alpha_repr": [format(alpha, ".17g"), ...]
```

### Keep all profits, only one policy map

Cleanup preserves all alpha/profit records in `results_index.pkl`.

It keeps the final posterior-mean policy map when available, because that policy is used for the next split. If no final posterior-mean policy map is available, it keeps the best available policy map by observed profit.

All other policy map files are removed to save space. Records whose maps are removed stay in the index with their profit, but their `policy_maps_file` is set to `None`.

### Repeat-candidate stopping

BO records the suggested candidate in history. If the last 5 BO-suggested candidates match after rounding to 4 decimals, the BO shell stops and runs the final posterior-mean evaluation.

### `B_96 * 0.25` split metric fix

Split metrics use the 15-minute decision grid correctly by multiplying summed `B_96` values by `0.25`.

This applies to split scoring and metric reporting.

The inherited samples are written into the new df output directory's `results_index.pkl` before the new df BO loop runs. Therefore, on the first BO call for the new df:

```python
bo_history[0]["X_arr"]
bo_history[0]["y_arr"]
```

contain:

```text
new df LHS evaluations
+ inherited valid evaluations from previous df
```

depending on whether the LHS trainer script has already been run for that df.

## Alpha Augmentation Rule

Inheritance is based on block intervals, not alpha names.

For each current/new block:

- If the new block is contained inside one previous block, it inherits that previous block's alpha.
- If the new block is not contained in any previous block, it gets alpha `0.0`.

Examples:

```text
old: [a1, a2]

split old block 1 into two children:
new: [a1, a1, a2]

insert a new block between old blocks:
new: [a1, 0.0, a2]
```

This handles same-sign splits naturally. If a split changes the actual feasible schedule semantics, the validation step below rejects the inherited sample.

## Validation Rule

A previous result is reused only if:

1. The previous record has `alphas` and `profit`.
2. The augmented alpha vector passes the current df `make_B_DA`.
3. The old and new day-ahead schedules are identical within tolerance:

```python
np.allclose(B_DA_old, B_DA_new, atol=1e-10, rtol=0.0)
```

This prevents training the current df GP on a mislabeled point. The old profit is only valid if the augmented current-df alpha vector produces the same `B_DA` schedule.

The tolerance can be overridden in `pipeline_config.json` with:

```json
"warm_start_bda_atol": 1e-10
```

## Stored Records

Inherited records are appended to the current df `results_index.pkl` with metadata:

```python
{
    "alphas": [...],
    "alpha_repr": [...],
    "profit": ...,
    "policy_maps_file": "... or None",
    "inherited": True,
    "source_df": previous_df,
    "source_id": previous_record_id,
    "source_alphas": [...]
}
```

Inherited records do not copy policy maps from the previous df. They are profit-only warm-start records for GP fitting.

The current df gets its own policy map from the final posterior-mean ShadowGP evaluation. The split is determined from that current-df final policy.

## BO Behavior

Generated `bo_Xd.py` loads all records from `results_index.pkl` that have:

- `alphas` of the current dimension
- `profit`

It does not require a policy map. Therefore inherited samples participate in GP fitting.

## Duplicate Handling

Inheritance skips a sample if the current df index already has the same alpha vector within `1e-12`.

Generated `run_shadowgp_Trainer.py` also checks for an existing matching alpha vector. If it evaluates a point already present as an inherited record, it updates that record and attaches a current policy map instead of creating a duplicate record.

## Propagation Across Multiple dfs

Only the immediate previous df is read:

```text
3df inherits from 2df
4df inherits from 3df
5df inherits from 4df
```

Older samples propagate forward because they are already present in the previous df `results_index.pkl` if they remained valid through earlier transitions.
