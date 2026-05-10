#!/usr/bin/env python3
import argparse
import importlib.util
import inspect
import json
import os
import pickle
import shutil
import sys
from typing import List, Dict, Tuple

import numpy as np

RESULTS_INDEX_FILE = "results_index.pkl"
ZERO_TOL = 1e-12


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def find_asset_source(base_dir: str, fname: str, prev_scripts: str = "") -> str:
    script_auto_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if prev_scripts:
        candidates.append(os.path.join(prev_scripts, fname))
    candidates.append(os.path.join(base_dir, "auto", fname))
    candidates.append(os.path.join(base_dir, fname))
    candidates.append(os.path.join(script_auto_dir, fname))
    for src in candidates:
        if src and os.path.exists(src):
            return src
    return ""


def encode_tag(x: float) -> str:
    s = f"{x:.16e}"
    return s.replace(".", "p").replace("-", "m").replace("+", "")


def decode_tag(s: str) -> float:
    s = s.replace("em", "e-")
    s = s.replace("p", ".")
    s = s.replace("m", "-")
    return float(s)


def parse_alphas_from_filename(fname: str, k: int) -> List[float]:
    name = os.path.splitext(fname)[0]
    alphas = []
    for i in range(1, k + 1):
        token = f"alpha{i}"
        idx = name.find(token)
        if idx == -1:
            return []
        start = idx + len(token)
        if i < k:
            next_token = f"_alpha{i+1}"
            end = name.find(next_token)
            if end == -1:
                return []
        else:
            end = len(name)
        tag = name[start:end]
        alphas.append(decode_tag(tag))
    return alphas


def load_results_index(outdir: str) -> dict:
    path = os.path.join(outdir, RESULTS_INDEX_FILE)
    if not os.path.exists(path):
        return {"records": {}}
    with open(path, "rb") as f:
        index = pickle.load(f)
    if not isinstance(index, dict):
        raise RuntimeError(f"Invalid results index: {path}")
    records = index.get("records", {})
    if not isinstance(records, dict):
        raise RuntimeError(f"Invalid results index records: {path}")
    return index


def save_results_index(outdir: str, index: dict) -> None:
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, RESULTS_INDEX_FILE), "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def normalize_block(block: Dict) -> Dict:
    out = {"start": block["start"], "end": block["end"]}
    if "sign" in block:
        out["sign"] = block["sign"]
    if block.get("fixed_zero"):
        out["fixed_zero"] = True
    return out


def normalize_blocks(blocks: List[Dict]) -> List[Dict]:
    return [normalize_block(b) for b in blocks]


def decision_block_indices(blocks: List[Dict]) -> List[int]:
    return [i for i, b in enumerate(blocks) if not b.get("fixed_zero")]


def decision_blocks(blocks: List[Dict]) -> List[Dict]:
    return [blocks[i] for i in decision_block_indices(blocks)]


def full_to_decision_alphas(blocks: List[Dict], full_alphas: List[float]) -> List[float]:
    return [float(full_alphas[i]) for i in decision_block_indices(blocks)]


def decision_to_full_alphas(blocks: List[Dict], decision_alphas: List[float]) -> List[float]:
    out = []
    free_idx = 0
    for block in blocks:
        if block.get("fixed_zero"):
            out.append(0.0)
        else:
            out.append(float(decision_alphas[free_idx]))
            free_idx += 1
    if free_idx != len(decision_alphas):
        raise ValueError("Decision alpha length does not match non-fixed blocks.")
    return out


def get_final_postmean_record(outdir: str, k: int) -> dict:
    index = load_results_index(outdir)
    records = index.get("records", {})
    final_id = index.get("final_postmean_id")
    if final_id is None or str(final_id) not in records:
        raise RuntimeError(f"Missing final_postmean_id in {os.path.join(outdir, RESULTS_INDEX_FILE)}")
    rec = records[str(final_id)]
    alphas = rec.get("alphas")
    if not isinstance(alphas, list) or len(alphas) != k:
        raise RuntimeError(f"Invalid final_postmean alphas in {os.path.join(outdir, RESULTS_INDEX_FILE)}")
    return rec


def mark_zero_fixed_blocks(blocks: List[Dict], alphas: List[float], zero_tol: float) -> List[Dict]:
    if len(blocks) != len(alphas):
        raise ValueError("Block count and alpha count do not match for zero-drop marking.")
    marked = []
    for block, alpha in zip(blocks, alphas):
        out = dict(block)
        if block.get("fixed_zero") or abs(float(alpha)) <= zero_tol:
            out["fixed_zero"] = True
        marked.append(out)
    return normalize_blocks(marked)


def mark_zero_fixed_blocks_except_split(
    blocks: List[Dict],
    alphas: List[float],
    zero_tol: float,
    split_info: Dict,
) -> List[Dict]:
    return mark_zero_fixed_blocks_except_splits(blocks, alphas, zero_tol, [split_info])


def mark_zero_fixed_blocks_except_splits(
    blocks: List[Dict],
    alphas: List[float],
    zero_tol: float,
    split_infos: List[Dict],
) -> List[Dict]:
    if len(blocks) != len(alphas):
        raise ValueError("Block count and alpha count do not match for zero-drop marking.")

    split_block_keys = {
        (info["segment_start"], info["segment_end"])
        for info in split_infos
        if info["segment_type"] == "block"
    }

    marked = []
    for block, alpha in zip(blocks, alphas):
        out = dict(block)
        is_split_block = (block["start"], block["end"]) in split_block_keys

        if is_split_block:
            out.pop("fixed_zero", None)
        elif block.get("fixed_zero") or abs(float(alpha)) <= zero_tol:
            out["fixed_zero"] = True
        marked.append(out)
    return normalize_blocks(marked)


def next_record_id(records: dict) -> str:
    existing_ids = [int(rid) for rid in records.keys() if str(rid).isdigit()]
    return f"{(max(existing_ids) if existing_ids else 0) + 1:06d}"


def alphas_match(a: List[float], b: List[float], tol: float = 1e-12) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def import_bda_from_scripts(scripts_dir: str, module_name: str):
    path = os.path.join(scripts_dir, "B_DA.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load B_DA module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def call_make_bda(make_bda, alphas: List[float], cfg: dict, blocks: List[Dict] | None = None) -> np.ndarray:
    sig = inspect.signature(make_bda)
    alpha_params = [name for name in sig.parameters if name.startswith("alpha")]
    alpha_values = [float(a) for a in alphas]

    if len(alpha_values) != len(alpha_params):
        if blocks is None:
            raise ValueError(
                f"Alpha length {len(alpha_values)} does not match B_DA signature "
                f"expecting {len(alpha_params)} alpha args."
            )
        if len(alpha_values) != len(blocks):
            raise ValueError(
                f"Alpha length {len(alpha_values)} does not match block count {len(blocks)}."
            )
        alpha_values = full_to_decision_alphas(blocks, alpha_values)
        if len(alpha_values) != len(alpha_params):
            raise ValueError(
                f"Decision alpha length {len(alpha_values)} still does not match "
                f"B_DA signature expecting {len(alpha_params)} alpha args."
            )

    return make_bda(
        *alpha_values,
        i0=cfg["i0"],
        imax=cfg["imax"],
        bmax=cfg["bmax"],
        charging_eff=cfg["charging_eff"],
    )


def augment_alphas_from_blocks(old_blocks: List[Dict], new_blocks: List[Dict], old_alphas: List[float]) -> List[float]:
    old_sorted = sorted(old_blocks, key=lambda b: b["start"])
    new_alphas = []
    for new_block in sorted(new_blocks, key=lambda b: b["start"]):
        parent_idx = None
        for i, old_block in enumerate(old_sorted):
            if old_block["start"] <= new_block["start"] and new_block["end"] <= old_block["end"]:
                parent_idx = i
                break
        if parent_idx is None:
            new_alphas.append(0.0)
        else:
            new_alphas.append(float(old_alphas[parent_idx]))
    return new_alphas


def inherit_previous_results(
    base_dir: str,
    prev_df: int,
    curr_df: int,
    prev_blocks: List[Dict],
    curr_blocks: List[Dict],
    cfg: dict,
    prev_bda_blocks: List[Dict] | None = None,
) -> Tuple[int, int, int, int]:
    prev_outdir = os.path.join(base_dir, f"{prev_df}df", "outputs", f"ndec{cfg['ndecisions']}nstep{cfg['nstep']}_{prev_df}d")
    curr_outdir = os.path.join(base_dir, f"{curr_df}df", "outputs", f"ndec{cfg['ndecisions']}nstep{cfg['nstep']}_{curr_df}d")
    prev_index_path = os.path.join(prev_outdir, RESULTS_INDEX_FILE)
    if not os.path.exists(prev_index_path):
        return 0, 0, 0, 0

    os.makedirs(curr_outdir, exist_ok=True)
    prev_index = load_results_index(prev_outdir)
    curr_index = load_results_index(curr_outdir)
    curr_records = curr_index.setdefault("records", {})
    prev_records = prev_index.get("records", {})

    prev_bda = import_bda_from_scripts(os.path.join(base_dir, f"{prev_df}df", "scripts"), f"B_DA_{prev_df}d_inherit")
    curr_bda = import_bda_from_scripts(os.path.join(base_dir, f"{curr_df}df", "scripts"), f"B_DA_{curr_df}d_inherit")
    prev_bda_blocks = prev_blocks if prev_bda_blocks is None else prev_bda_blocks

    bda_atol = float(cfg.get("warm_start_bda_atol", 1e-10))
    existing_alpha_rows = []
    for rec in curr_records.values():
        alphas = rec.get("alphas")
        if isinstance(alphas, list) and len(alphas) == len(curr_blocks):
            existing_alpha_rows.append([float(a) for a in alphas])

    added = 0
    skipped_duplicate = 0
    skipped_invalid = 0
    skipped_inherited = 0
    for source_id, source_rec in sorted(prev_records.items(), key=lambda item: str(item[0])):
        if source_rec.get("inherited"):
            skipped_inherited += 1
            continue

        old_alphas = source_rec.get("alphas")
        if not isinstance(old_alphas, list) or len(old_alphas) != len(prev_blocks):
            skipped_invalid += 1
            continue
        if "profit" not in source_rec:
            skipped_invalid += 1
            continue

        new_alphas = augment_alphas_from_blocks(prev_blocks, curr_blocks, old_alphas)
        if any(alphas_match(row, new_alphas) for row in existing_alpha_rows):
            skipped_duplicate += 1
            continue

        try:
            old_bda = call_make_bda(prev_bda.make_B_DA, old_alphas, cfg, prev_bda_blocks)
            new_bda = call_make_bda(curr_bda.make_B_DA, new_alphas, cfg, curr_blocks)
        except (AssertionError, ValueError):
            skipped_invalid += 1
            continue

        if not np.allclose(old_bda, new_bda, atol=bda_atol, rtol=0.0):
            skipped_invalid += 1
            continue

        policy_file = None
        source_policy = source_rec.get("policy_maps_file")
        record_id = next_record_id(curr_records)

        inherited = {
            "id": record_id,
            "alphas": [float(a) for a in new_alphas],
            "alpha_repr": [format(float(a), ".17g") for a in new_alphas],
            "profit": float(source_rec["profit"]),
            "policy_maps_file": policy_file,
            "inherited": True,
            "source_df": int(prev_df),
            "source_id": str(source_rec.get("id", source_id)),
            "source_alphas": [float(a) for a in old_alphas],
            "source_alpha_repr": [format(float(a), ".17g") for a in old_alphas],
            "source_policy_maps_file": source_policy,
            "bda_reuse_atol": bda_atol,
        }
        if "dart_trading_profit" in source_rec:
            inherited["dart_trading_profit"] = float(source_rec["dart_trading_profit"])
        curr_records[record_id] = inherited
        existing_alpha_rows.append([float(a) for a in new_alphas])
        added += 1

    if added:
        history = curr_index.setdefault("warm_start_history", [])
        history.append({
            "from_df": int(prev_df),
            "to_df": int(curr_df),
            "added": int(added),
            "skipped_duplicate": int(skipped_duplicate),
            "skipped_invalid": int(skipped_invalid),
            "skipped_inherited": int(skipped_inherited),
            "source_records": int(len(prev_records)),
        })
        save_results_index(curr_outdir, curr_index)
    return added, skipped_duplicate, skipped_invalid, skipped_inherited


def load_best_profit(outdir: str, k: int) -> Tuple[List[float], float]:
    index = load_results_index(outdir)
    records = index.get("records", {})
    if records:
        final_id = index.get("final_postmean_id")
        if final_id is not None and str(final_id) in records:
            rec = records[str(final_id)]
            alphas = rec.get("alphas")
            policy_file = rec.get("policy_maps_file")
            has_policy = bool(policy_file) and os.path.exists(os.path.join(outdir, policy_file))
            if isinstance(alphas, list) and len(alphas) == k and "profit" in rec and has_policy:
                return [float(a) for a in alphas], float(rec["profit"])

        valid_records = []
        for rec in records.values():
            alphas = rec.get("alphas")
            if not isinstance(alphas, list) or len(alphas) != k:
                continue
            if "profit" not in rec:
                continue
            policy_file = rec.get("policy_maps_file")
            has_policy = bool(policy_file) and os.path.exists(os.path.join(outdir, policy_file))
            valid_records.append((float(rec["profit"]), [float(a) for a in alphas], has_policy))
        with_policy = [row for row in valid_records if row[2]]
        candidates = with_policy if with_policy else valid_records
        if candidates:
            best_profit, best_alphas, _ = max(candidates, key=lambda row: row[0])
            return best_alphas, float(best_profit)

    files = [f for f in os.listdir(outdir) if f.startswith("profit_alpha1") and f.endswith(".npy")]
    best_profit = None
    best_alphas = None
    for f in files:
        alphas = parse_alphas_from_filename(f, k)
        if not alphas:
            continue
        data = np.load(os.path.join(outdir, f), allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            item = data.item()
            profit = float(item["Total_profit"]) if isinstance(item, dict) else float(item)
        else:
            profit = float(np.array(data).squeeze())
        if best_profit is None or profit > best_profit:
            best_profit = profit
            best_alphas = alphas
    if best_alphas is None:
        raise RuntimeError(f"No profit files found in {outdir} for k={k}")
    return best_alphas, float(best_profit)


def load_policy_maps(outdir: str, alphas: List[float]) -> list:
    index = load_results_index(outdir)
    for rec in index.get("records", {}).values():
        rec_alphas = rec.get("alphas")
        if not isinstance(rec_alphas, list) or len(rec_alphas) != len(alphas):
            continue
        if all(abs(float(a) - float(b)) <= 1e-15 for a, b in zip(rec_alphas, alphas)):
            fname = rec.get("policy_maps_file")
            if not fname:
                break
            path = os.path.join(outdir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Policy maps not found: {path}")
            with open(path, "rb") as f:
                return pickle.load(f)

    tags = [encode_tag(a) for a in alphas]
    parts = [f"alpha{i+1}{tags[i]}" for i in range(len(tags))]
    fname = "policy_maps_" + "_".join(parts) + ".pkl"
    path = os.path.join(outdir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Policy maps not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_main_and_bda(scripts_dir: str, auto_dir: str):
    if auto_dir not in sys.path:
        sys.path.insert(0, auto_dir)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import importlib
    main_mod = importlib.import_module("main")
    importlib.reload(main_mod)
    bda_mod = importlib.import_module("B_DA")
    importlib.reload(bda_mod)
    return main_mod, bda_mod


def simulate_policy_decisions(policy_maps,
                              dt_opt,
                              decision_idx,
                              B_DA_vec,
                              process,
                              I0,
                              Imax,
                              charging_eff,
                              running_cost,
                              final_cost,
                              Bmin_scalar,
                              Bmax_scalar,
                              rt_adjustment_pen):
    X_sims = process.sim_trajectories
    nsim = X_sims.shape[0]
    ndec = len(policy_maps)

    Is = np.zeros((nsim, ndec + 1))
    Is[:, 0] = I0
    B_rt = np.zeros((nsim, ndec))
    P_rt = np.zeros((nsim, ndec))
    running_profit = np.zeros((nsim, ndec))
    Price_end = np.zeros((nsim, ndec))

    Ilb, Iub = 0.0, Imax
    rt_total_pen = np.zeros(nsim)
    rt_total_profit = np.zeros(nsim)
    for k in range(ndec):
        I_k = Is[:, k]
        idx_end = decision_idx[k]
        Price_k = X_sims[:, idx_end]
        Price_end[:, k] = Price_k

        B_DA_t = B_DA_vec[k]

        LB = np.maximum(
            Bmin_scalar - B_DA_t,
            (charging_eff * (Ilb - I_k) / dt_opt) - B_DA_t
        )
        UB = np.minimum(
            Bmax_scalar - B_DA_t,
            ((Iub - I_k) / (charging_eff * dt_opt)) - B_DA_t
        )

        inp = np.column_stack((Price_k, I_k))
        B_k = policy_maps[k].predict(inp)[0].flatten()
        B_k = np.maximum(LB, np.minimum(B_k, UB))

        P_k = B_DA_t + B_k
        eff = charging_eff * (P_k > 0) + (1.0 / charging_eff) * (P_k < 0)

        Is[:, k + 1] = np.clip(I_k + P_k * eff * dt_opt, Ilb, Iub)
        running_profit[:, k] = -running_cost.cost(B_k, Price_k, B_DA_t) * dt_opt

        rt_total_pen += 0.5 * rt_adjustment_pen * B_k**2 * dt_opt
        rt_total_profit += -Price_k * B_k * dt_opt

        B_rt[:, k] = B_k
        P_rt[:, k] = P_k

    RT_value = np.mean(np.sum(running_profit, axis=1))
    return {
        "Is": Is,
        "B_rt": B_rt,
        "P_rt": P_rt,
        "Price_end": Price_end,
        "RT_profit": np.mean(rt_total_profit),
        "RT_reg": np.mean(rt_total_pen),
        "RT_value": np.mean(RT_value),
        "t_dec": np.arange(ndec) * dt_opt,
    }


def build_segments(blocks: List[Dict]) -> List[Tuple[int, int, str]]:
    blocks_sorted = sorted(blocks, key=lambda b: b["start"])
    segments = []
    cur = 0
    for b in blocks_sorted:
        if b["start"] > cur:
            segments.append((cur, b["start"], "gap"))
        segments.append((b["start"], b["end"], "block"))
        cur = b["end"]
    if cur < 24:
        segments.append((cur, 24, "gap"))
    return segments


def choose_split(B_96: np.ndarray, blocks: List[Dict]) -> Dict:
    return choose_splits(B_96, blocks, 1)[0]


def choose_splits(B_96: np.ndarray, blocks: List[Dict], split_count: int) -> List[Dict]:
    if split_count <= 0:
        raise ValueError("split_count must be positive.")

    metrics = compute_split_metrics(B_96, blocks)
    selected = []
    used_segments = set()
    for row in metrics:
        key = (row["segment_start"], row["segment_end"], row["segment_type"])
        if key in used_segments:
            continue
        selected.append(row)
        used_segments.add(key)
        if len(selected) == split_count:
            break

    if not selected:
        raise RuntimeError("No valid split candidates found.")
    return selected

def compute_split_metrics(B_96: np.ndarray, blocks: List[Dict]) -> List[Dict]:
    rows = []
    segments = build_segments(blocks)
    for start, end, kind in segments:
        split_points = list(range(start + 1, end))
        # Allow creating a full 1-hour block from a 1-hour gap (e.g., 23-24 via k=24).
        if kind == "gap" and (end - start) == 1:
            split_points = [end]
        if not split_points:
            continue
        for k in split_points:
            lhs = B_96[:, start * 4 : k * 4].sum(axis=1) * 0.25
            rhs = B_96[:, k * 4 : end * 4].sum(axis=1) * 0.25
            mean_diff = lhs.mean() - rhs.mean()
            rows.append(
                {
                    "segment": f"{start}-{end}({kind})",
                    "split_at": k,
                    "segment_start": int(start),
                    "segment_end": int(end),
                    "segment_type": kind,
                    "lhs_mean": float(lhs.mean()),
                    "rhs_mean": float(rhs.mean()),
                    "lhs_std": float(lhs.std(ddof=1)),
                    "rhs_std": float(rhs.std(ddof=1)),
                    "mean_diff": float(mean_diff),
                    "abs_mean_diff": float(abs(mean_diff)),
                    "abs_mean_diff_round": float(round(abs(mean_diff), 3)),
                    "left_len": int(k - start),
                    "right_len": int(end - k),
                    "longest_block": int(max(k - start, end - k)),
                }
            )
    rows.sort(
        key=lambda r: (
            r["abs_mean_diff_round"],
            r["longest_block"],
            r["left_len"],
            r["abs_mean_diff"],
        ),
        reverse=True,
    )
    return rows


def apply_split(blocks: List[Dict], split_info: Dict) -> List[Dict]:
    split_at = split_info["split_at"]
    seg_start = split_info["segment_start"]
    seg_end = split_info["segment_end"]
    seg_type = split_info["segment_type"]

    blocks_sorted = sorted(blocks, key=lambda b: b["start"])

    if seg_type == "block":
        new_blocks = []
        for b in blocks_sorted:
            if b["start"] <= split_at < b["end"]:
                if split_at == b["start"] or split_at == b["end"]:
                    raise RuntimeError("Split at block boundary produces zero-length block.")
                # split block: allow sign to be re-inferred per side (keep parent for zero-mean)
                parent = {"parent_sign": b.get("sign")} if b.get("sign") else {}
                if b.get("fixed_zero"):
                    parent["fixed_zero"] = True
                new_blocks.append({"start": b["start"], "end": split_at, "split_child": True, **parent})
                new_blocks.append({"start": split_at, "end": b["end"], "split_child": True, **parent})
            else:
                # preserve sign for untouched blocks
                base = {"sign": b.get("sign")} if b.get("sign") else {}
                if b.get("fixed_zero"):
                    base["fixed_zero"] = True
                new_blocks.append({"start": b["start"], "end": b["end"], **base})
        return sorted(new_blocks, key=lambda b: b["start"])

    # gap split -> create one block based on lhs/rhs mean values
    if split_at <= seg_start:
        raise RuntimeError("Split at gap boundary produces zero-length block.")
    if "lhs_mean" not in split_info or "rhs_mean" not in split_info:
        raise RuntimeError("Gap split requires lhs_mean and rhs_mean in split_info.")
    lhs_mean = float(split_info["lhs_mean"])
    rhs_mean = float(split_info["rhs_mean"])
    if split_at == seg_end:
        choose_left = True
    elif split_at == seg_start:
        choose_left = False
    else:
        choose_left = abs(lhs_mean) > abs(rhs_mean)
    if choose_left:
        new_block = {"start": seg_start, "end": split_at}
    else:
        new_block = {"start": split_at, "end": seg_end}
    if new_block["start"] == new_block["end"]:
        raise RuntimeError("Gap split produced zero-length block.")
    new_blocks = blocks_sorted + [new_block]
    return sorted(new_blocks, key=lambda b: b["start"])


def apply_splits(blocks: List[Dict], split_infos: List[Dict]) -> List[Dict]:
    new_blocks = normalize_blocks(blocks)
    for split_info in split_infos:
        new_blocks = apply_split(new_blocks, split_info)
    return normalize_blocks(new_blocks)


def block_mean(B_96: np.ndarray, start: int, end: int) -> float:
    return float((B_96[:, start * 4 : end * 4].sum(axis=1) * 0.25).mean())


def assign_block_signs(B_96: np.ndarray, B_DA_vec_96: np.ndarray, blocks: List[Dict], cfg: dict) -> List[Dict]:
    signed = []
    for b in sorted(blocks, key=lambda x: x["start"]):
        if b.get("sign") in ("pos", "neg"):
            sign = b["sign"]
        elif cfg.get("rt_adjustment_pen", 0.0) == 0.0 and b.get("parent_sign") in ("pos", "neg"):
            sign = b["parent_sign"]
        else:
            if b.get("split_child"):
                s = b["start"] * 4
                e = b["end"] * 4
                mean_val = float((B_DA_vec_96[s:e] + B_96[:, s:e].mean(axis=0)).mean())
            else:
                mean_val = block_mean(B_96, b["start"], b["end"])
            mean_round = round(mean_val, 3)
            if mean_round == 0.0 and b.get("parent_sign") in ("pos", "neg"):
                sign = b["parent_sign"]
            else:
                sign = "pos" if mean_val >= 0 else "neg"
        out = {"start": b["start"], "end": b["end"], "sign": sign}
        if b.get("fixed_zero"):
            out["fixed_zero"] = True
        signed.append(out)
    return signed


def write_blocks_json(path: str, blocks: List[Dict]) -> None:
    with open(path, "w") as f:
        json.dump(blocks, f, indent=2)


def save_split_report(base_dir: str, df: int, metrics: List[Dict], split_info: Dict, split_infos: List[Dict] | None = None) -> Tuple[str, str]:
    split_infos = [split_info] if split_infos is None else split_infos
    df_dir = os.path.join(base_dir, f"{df}df")
    os.makedirs(df_dir, exist_ok=True)

    json_path = os.path.join(df_dir, "split_metrics.json")
    txt_path = os.path.join(df_dir, "split_metrics.txt")

    with open(json_path, "w") as f:
        json.dump(
            {
                "df": df,
                "selected_split": split_info,
                "selected_splits": split_infos,
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    with open(txt_path, "w") as f:
        f.write(f"[{df}d] Split metrics (sorted by abs_mean_diff):\n")
        for row in metrics:
            f.write(
                f"  seg={row['segment']} split={row['split_at']} "
                f"lhs_mean={row['lhs_mean']:.6f} rhs_mean={row['rhs_mean']:.6f} "
                f"lhs_std={row['lhs_std']:.6f} rhs_std={row['rhs_std']:.6f} "
                f"mean_diff={row['mean_diff']:.6f} abs_diff={row['abs_mean_diff']:.6f}\n"
            )
        for i, info in enumerate(split_infos, start=1):
            f.write(
                f"[{df}d] Split #{i} at hour {info['split_at']} "
                f"(segment {info['segment_start']}-{info['segment_end']}, {info['segment_type']})\n"
            )

    return json_path, txt_path


def cleanup_outputs_keep_best(outdir: str) -> Tuple[str, float]:
    index = load_results_index(outdir)
    records = index.get("records", {})
    if records:
        best_id = ""
        best_rec = None
        best_profit = None
        for rid, rec in records.items():
            if "profit" not in rec:
                continue
            profit = float(rec["profit"])
            if best_profit is None or profit > best_profit:
                best_profit = profit
                best_id = str(rid)
                best_rec = rec
        if best_rec is None or best_profit is None:
            raise RuntimeError(f"No valid profit records found in {os.path.join(outdir, RESULTS_INDEX_FILE)}")

        keep_policy = ""
        final_id = index.get("final_postmean_id")
        if final_id is not None and str(final_id) in records:
            final_policy = records[str(final_id)].get("policy_maps_file", "")
            if final_policy and os.path.exists(os.path.join(outdir, final_policy)):
                keep_policy = final_policy

        if not keep_policy:
            keep_policy = best_rec.get("policy_maps_file", "")
        if not keep_policy or not os.path.exists(os.path.join(outdir, keep_policy)):
            best_policy_profit = None
            best_policy_file = ""
            for rec in records.values():
                policy_file = rec.get("policy_maps_file", "")
                if not policy_file or "profit" not in rec:
                    continue
                if not os.path.exists(os.path.join(outdir, policy_file)):
                    continue
                profit = float(rec["profit"])
                if best_policy_profit is None or profit > best_policy_profit:
                    best_policy_profit = profit
                    best_policy_file = policy_file
            keep_policy = best_policy_file
        for fname in os.listdir(outdir):
            is_policy_maps = fname.startswith("policy_maps_") and fname.endswith(".pkl")
            is_policy_map = fname.startswith("policy_map_") and fname.endswith(".pkl")
            is_legacy_profit = fname.startswith("profit_") and fname.endswith(".npy")
            if (is_policy_maps or is_policy_map or is_legacy_profit) and fname != keep_policy:
                os.remove(os.path.join(outdir, fname))

        for rid, rec in records.items():
            policy_file = rec.get("policy_maps_file", "")
            if policy_file and policy_file == keep_policy:
                rec["policy_maps_kept"] = True
            else:
                rec["policy_maps_kept"] = False
                if policy_file and policy_file != keep_policy:
                    rec["policy_maps_file_removed"] = policy_file
                    rec["policy_maps_file"] = None
        index["best_id"] = best_id
        with open(os.path.join(outdir, RESULTS_INDEX_FILE), "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        return best_id, float(best_profit)

    profit_files = [f for f in os.listdir(outdir) if f.startswith("profit_") and f.endswith(".npy")]
    if not profit_files:
        raise RuntimeError(f"No profit files found in {outdir}")

    best_file = ""
    best_profit = None
    for fname in sorted(profit_files):
        path = os.path.join(outdir, fname)
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.shape == ():
            item = data.item()
            if isinstance(item, dict) and "Total_profit" in item:
                profit = float(item["Total_profit"])
            else:
                profit = float(item)
        else:
            profit = float(np.array(data).squeeze())
        if best_profit is None or profit > best_profit:
            best_profit = profit
            best_file = fname

    best_tag = best_file[len("profit_"):-len(".npy")]
    keep = {best_file, f"policy_maps_{best_tag}.pkl", f"policy_map_{best_tag}.pkl"}

    for fname in os.listdir(outdir):
        is_profit = fname.startswith("profit_") and fname.endswith(".npy")
        is_policy_maps = fname.startswith("policy_maps_") and fname.endswith(".pkl")
        is_policy_map = fname.startswith("policy_map_") and fname.endswith(".pkl")
        if (is_profit or is_policy_maps or is_policy_map) and fname not in keep:
            os.remove(os.path.join(outdir, fname))

    assert best_profit is not None
    return best_file, float(best_profit)


def render_bda_py(blocks: List[Dict], k: int, cfg: dict) -> str:
    decision_idx = decision_block_indices(blocks)
    if not decision_idx:
        return """import numpy as np\n\n""" + \
            f"""def make_B_DA(\n    i0: float = {cfg['i0']},\n    imax: float = {cfg['imax']},\n    bmax: float = {cfg['bmax']},\n    charging_eff: float = {cfg['charging_eff']},\n) -> np.ndarray:\n    B = np.zeros(24)\n\n    assert charging_eff > 0.0, \"charging_eff must be positive\"\n\n    # 1) power constraint\n    assert np.all(B >= -bmax) and np.all(B <= bmax), \\\n        \"B values exceed power limits [-bmax, bmax]\"\n\n    # 2) SoC constraints with charging/discharging efficiency\n    I = i0\n    for t in range(24):\n        B_t = B[t]\n        eff = charging_eff if B_t >= 0 else (1.0 / charging_eff)\n        I += B_t * eff\n        assert (0.0 - 1e-3) <= I <= (imax + 1e-3), \\\n            f\"SoC out of bounds at hour {{t+1}}: {{I}}\"\n\n    return B\n"""
    block_defs = []
    for i, b in enumerate(blocks):
        hours = list(range(b["start"], b["end"]))
        block_defs.append(f"    block{i} = {hours}")
    assigns = []
    arg_defs = []
    free_pos = 0
    for i, b in enumerate(blocks):
        if b.get("fixed_zero"):
            assigns.append(f"    B[block{i}] = 0.0")
        else:
            free_pos += 1
            assigns.append(f"    B[block{i}] = alpha{free_pos}")
            arg_defs.append(f"alpha{free_pos}: float")

    args = ",\n    ".join(arg_defs)

    return """import numpy as np\n\n""" + \
        f"""def make_B_DA(\n    {args},\n    i0: float = {cfg['i0']},\n    imax: float = {cfg['imax']},\n    bmax: float = {cfg['bmax']},\n    charging_eff: float = {cfg['charging_eff']},\n) -> np.ndarray:\n    B = np.zeros(24)\n\n    assert charging_eff > 0.0, \"charging_eff must be positive\"\n\n""" + \
        "\n".join(block_defs) + "\n\n" + \
        "\n".join(assigns) + "\n\n" + \
        """    # 1) power constraint\n    assert np.all(B >= -bmax) and np.all(B <= bmax), \
        "B values exceed power limits [-bmax, bmax]"\n\n    # 2) SoC constraints with charging/discharging efficiency\n    I = i0\n    for t in range(24):\n        B_t = B[t]\n        eff = charging_eff if B_t >= 0 else (1.0 / charging_eff)\n        I += B_t * eff\n        assert (0.0 - 1e-3) <= I <= (imax + 1e-3), \
            f"SoC out of bounds at hour {t+1}: {I}"\n\n    return B\n"""


def render_sampler_py(blocks: List[Dict], cfg: dict) -> str:
    blocks_free = decision_blocks(blocks)
    k = len(blocks_free)
    lens = [b["end"] - b["start"] for b in blocks_free]
    bounds_lo = []
    bounds_hi = []
    for b in blocks_free:
        if b.get("sign", "pos") == "pos":
            bounds_lo.append(0.0)
            bounds_hi.append(cfg["bmax"])
        else:
            bounds_lo.append(-cfg["bmax"])
            bounds_hi.append(0.0)

    bounds_lo_s = ", ".join([f"{x}" for x in bounds_lo])
    bounds_hi_s = ", ".join([f"{x}" for x in bounds_hi])

    lines = []
    lines.append("import torch")
    lines.append("import numpy as np")
    lines.append("from scipy.stats import qmc")
    lines.append("")
    if k == 0:
        lines.append("samples = np.zeros((1, 0), dtype=float)")
        lines.append("for row in samples:")
        lines.append("    print(\" \".join([f\"{v:.6f}\" for v in row.tolist()]))")
        return "\n".join(lines) + "\n"
    lines.append("")
    lines.append("def build_soc_inequality_constraints(i0: float, imax: float, charging_eff: float):")
    lines.append("    ineq = []")
    lines.append("    if charging_eff <= 0.0:")
    lines.append("        raise ValueError(\"charging_eff must be positive\")")
    lines.append("")
    lines.append(f"    lengths = {lens}")
    lines.append(f"    signs = {[b.get('sign', 'pos') for b in blocks_free]}")
    lines.append("    rhs_lo = -float(i0)")
    lines.append("    rhs_hi = -(float(imax) - float(i0))")
    lines.append("    for i in range(len(lengths)):")
    lines.append("        idx = torch.tensor(list(range(i + 1)), dtype=torch.long)")
    lines.append("        coeff = []")
    lines.append("        for j in range(i + 1):")
    lines.append("            eff = float(charging_eff) if signs[j] == \"pos\" else (1.0 / float(charging_eff))")
    lines.append("            coeff.append(float(lengths[j]) * eff)")
    lines.append("        coeff_t = torch.tensor(coeff, dtype=torch.double)")
    lines.append("        ineq.append((idx, coeff_t, rhs_lo))")
    lines.append("        ineq.append((idx, -coeff_t, rhs_hi))")
    lines.append("")
    lines.append("    return ineq")
    lines.append("")
    lines.append("")
    lines.append("def satisfies_constraints(x: np.ndarray, ineq, tol: float = 1e-12) -> bool:")
    lines.append("    # Constraint format: sum(coeff * x[idx]) >= rhs")
    lines.append("    for idx, coeff, rhs in ineq:")
    lines.append("        lhs = np.dot(coeff.numpy(), x[idx.numpy()])")
    lines.append("        if lhs < float(rhs) - tol:")
    lines.append("            return False")
    lines.append("    return True")
    lines.append("")
    lines.append("")
    lines.append(f"seed = {cfg['design_seed']}")
    lines.append("torch.manual_seed(seed)")
    lines.append("np.random.seed(seed)")
    lines.append("")
    lines.append(f"bmax = {cfg['bmax']}")
    lines.append(f"i0 = {cfg['i0']}")
    lines.append(f"imax = {cfg['imax']}")
    lines.append(f"charging_eff = {cfg['charging_eff']}")
    lines.append("")
    lines.append("bounds = torch.tensor(")
    lines.append(f"    [[{bounds_lo_s}],")
    lines.append(f"     [{bounds_hi_s}]],")
    lines.append("    dtype=torch.double")
    lines.append(")")
    lines.append("")
    lines.append("ineq = build_soc_inequality_constraints(i0=i0, imax=imax, charging_eff=charging_eff)")
    lines.append("")
    lines.append("# LHS over the bounding box, then accept feasible points")
    lines.append("n_candidates = 1024")
    lines.append("d = bounds.shape[1]")
    lines.append("n_keep = int(np.floor(6 * np.sqrt(d)))")
    lines.append("")
    lines.append("lb = bounds[0].numpy()")
    lines.append("ub = bounds[1].numpy()")
    lines.append("")
    lines.append("sampler = qmc.LatinHypercube(d=d, seed=seed)")
    lines.append("X_unit = sampler.random(n=n_candidates)")
    lines.append("X_lhs = qmc.scale(X_unit, lb, ub)")
    lines.append("")
    lines.append("feasible = []")
    lines.append("for x in X_lhs:")
    lines.append("    if satisfies_constraints(x, ineq):")
    lines.append("        feasible.append(x)")
    lines.append("    if len(feasible) == n_keep:")
    lines.append("        break")
    lines.append("")
    lines.append("if len(feasible) < n_keep:")
    lines.append("    raise RuntimeError(")
    lines.append("        f\"Only found {len(feasible)} feasible points out of {n_candidates} LHS samples.\"")
    lines.append("    )")
    lines.append("")
    lines.append("samples = torch.tensor(np.array(feasible), dtype=torch.double)")
    lines.append("")
    lines.append("for row in samples:")
    lines.append("    vals = row.tolist()")
    lines.append("    print(\" \".join([f\"{v:.6f}\" for v in vals]))")

    return "\n".join(lines) + "\n"


def render_bo_py(blocks: List[Dict], df: int, cfg: dict) -> str:
    k = len(blocks)
    free_blocks = decision_blocks(blocks)
    free_k = len(free_blocks)
    bounds_lo = []
    bounds_hi = []
    for b in free_blocks:
        if b.get("sign", "pos") == "pos":
            bounds_lo.append(0.0)
            bounds_hi.append(cfg["bmax"])
        else:
            bounds_lo.append(-cfg["bmax"])
            bounds_hi.append(0.0)

    bounds_lo_s = ", ".join([f"{x}" for x in bounds_lo])
    bounds_hi_s = ", ".join([f"{x}" for x in bounds_hi])

    parse_fn_name = "parse_" + "_".join([f"alpha{i+1}" for i in range(free_k)])
    next_x_path = "next_" + "_".join([f"alpha{i+1}" for i in range(free_k)]) + f"_{df}d.txt"
    postmean_x_path = "postmean_" + "_".join([f"alpha{i+1}" for i in range(free_k)]) + f"_{df}d.txt"
    lengths_s = ", ".join([str(b["end"] - b["start"]) for b in free_blocks])
    signs_s = ", ".join([repr(b.get("sign", "pos")) for b in free_blocks])
    fixed_mask_s = ", ".join(["True" if b.get("fixed_zero") else "False" for b in blocks])

    lines = []
    lines.append("import os")
    lines.append("import fnmatch")
    lines.append("import argparse")
    lines.append("import numpy as np")
    lines.append("import torch")
    lines.append("import pickle")
    lines.append("")
    lines.append("from botorch.models import SingleTaskGP")
    lines.append("from botorch.fit import fit_gpytorch_mll")
    lines.append("from botorch.acquisition import UpperConfidenceBound")
    lines.append("from botorch.acquisition.analytic import PosteriorMean")
    lines.append("from botorch.optim.optimize import optimize_acqf")
    lines.append("from gpytorch.mlls import ExactMarginalLogLikelihood")
    lines.append("from gpytorch.kernels import MaternKernel, ScaleKernel")
    lines.append("from botorch.models.transforms.input import Normalize")
    lines.append("from botorch.models.transforms.outcome import Standardize")
    lines.append("")
    lines.append("from B_DA import make_B_DA")
    lines.append("")
    lines.append("torch.set_default_dtype(torch.double)")
    lines.append("")
    lines.append(f"NEXT_X_PATH = \"{next_x_path}\"")
    lines.append(f"POSTMEAN_X_PATH = \"{postmean_x_path}\"")
    bo_acq_samples = cfg.get("bo_acq_samples", 512)
    lines.append("RESULTS_INDEX_FILE = \"results_index.pkl\"")
    lines.append(f"BO_ACQ_SAMPLES = {bo_acq_samples}")
    lines.append("REPEAT_STOP_WINDOW = 5")
    lines.append("REPEAT_STOP_DECIMALS = 4")
    lines.append("")
    lines.append(f"TOTAL_BLOCKS = {k}")
    lines.append(f"FIXED_ZERO_MASK = [{fixed_mask_s}]")
    lines.append(f"LENGTHS = [{lengths_s}]")
    lines.append(f"SIGNS = [{signs_s}]")
    lines.append("")
    lines.append("")
    lines.append("def beta(iter_num, num_params=None, delta=0.1):")
    lines.append("    if num_params is None:")
    lines.append("        num_params = len(LENGTHS)")
    lines.append("    return 2.0 * np.log(num_params * iter_num**2 * np.pi**2 / (6.0 * delta))")
    lines.append("")
    lines.append("")
    lines.append("def decode_float(s: str) -> float:")
    lines.append("    s = s.replace(\"p\", \".\")")
    lines.append("    s = s.replace(\"em\", \"e-\")")
    lines.append("    s = s.replace(\"m\", \"-\")")
    lines.append("    return float(s)")
    lines.append("")
    lines.append("def decision_to_full_alphas(decision_alphas):")
    lines.append("    out = []")
    lines.append("    free_idx = 0")
    lines.append("    for is_fixed in FIXED_ZERO_MASK:")
    lines.append("        if is_fixed:")
    lines.append("            out.append(0.0)")
    lines.append("        else:")
    lines.append("            out.append(float(decision_alphas[free_idx]))")
    lines.append("            free_idx += 1")
    lines.append("    if free_idx != len(decision_alphas):")
    lines.append("        raise ValueError(\"Decision alpha length mismatch.\")")
    lines.append("    return out")
    lines.append("")
    lines.append("def full_to_decision_alphas(full_alphas):")
    lines.append("    if len(full_alphas) != TOTAL_BLOCKS:")
    lines.append("        raise ValueError(\"Full alpha length mismatch.\")")
    lines.append("    return [float(a) for a, is_fixed in zip(full_alphas, FIXED_ZERO_MASK) if not is_fixed]")
    lines.append("")
    lines.append(f"def {parse_fn_name}(filename: str):")
    lines.append("    name = os.path.splitext(filename)[0]")
    for i in range(free_k):
        token = f"alpha{i+1}"
        if i < free_k - 1:
            lines.append(f"    a{i+1}_str = name.split(\"{token}\")[1].split(\"_alpha{i+2}\")[0]")
        else:
            lines.append(f"    a{i+1}_str = name.split(\"{token}\")[1]")
    lines.append("    return " + ", ".join([f"decode_float(a{i+1}_str)" for i in range(free_k)]))
    lines.append("")
    lines.append("def load_existing_results(outdir: str):")
    lines.append("    dict_x = {}")
    lines.append("    index_path = os.path.join(outdir, RESULTS_INDEX_FILE)")
    lines.append("    if os.path.exists(index_path):")
    lines.append("        with open(index_path, \"rb\") as f:")
    lines.append("            index = pickle.load(f)")
    lines.append("        for rec in index.get(\"records\", {}).values():")
    lines.append("            alphas = rec.get(\"alphas\")")
    lines.append("            if not isinstance(alphas, list):")
    lines.append("                continue")
    lines.append(f"            if len(alphas) != {k}:")
    lines.append("                continue")
    lines.append("            if \"profit\" not in rec:")
    lines.append("                continue")
    lines.append("            decision_alphas = full_to_decision_alphas(alphas)")
    lines.append("            dict_x[tuple(decision_alphas)] = {\"Total_profit\": float(rec[\"profit\"])}")
    lines.append("        if dict_x:")
    lines.append("            print(f\"Loaded {len(dict_x)} samples from results index.\")")
    lines.append("            return dict_x")
    lines.append("")
    lines.append("    files = [")
    lines.append("        f for f in os.listdir(outdir)")
    pattern = "profit_" + "_".join([f"alpha{i+1}*" for i in range(free_k)]) + ".npy"
    lines.append(f"        if fnmatch.fnmatch(f, \"{pattern}\")")
    lines.append("    ]")
    lines.append("")
    lines.append("    if not files:")
    lines.append("        raise RuntimeError(f\"No profit files found in {outdir}\")")
    lines.append("")
    lines.append("    for f in files:")
    lines.append("        data = np.load(os.path.join(outdir, f), allow_pickle=True)")
    lines.append("        if isinstance(data, np.ndarray) and data.shape == ():")
    lines.append("            item = data.item()")
    lines.append("            profit = float(item[\"Total_profit\"]) if isinstance(item, dict) else float(item)")
    lines.append("        else:")
    lines.append("            profit = float(np.array(data).squeeze())")
    lines.append("")
    lines.append(f"        alphas = list({parse_fn_name}(f))")
    lines.append("        dict_x[tuple(alphas)] = {\"Total_profit\": profit}")
    lines.append("")
    lines.append("    print(f\"Loaded {len(dict_x)} samples.\")")
    lines.append("    return dict_x")
    lines.append("")
    lines.append("def is_duplicate(x: torch.Tensor, train_x: torch.Tensor, tol: float = 1e-2) -> bool:")
    lines.append("    diffs = torch.abs(train_x - x)")
    lines.append("    return bool(torch.any(diffs.max(dim=1).values < tol))")
    lines.append("")
    lines.append("def is_soc_feasible(decision_alphas, i0: float, imax: float, bmax: float, charging_eff: float) -> bool:")
    lines.append("    try:")
    lines.append("        alphas = decision_to_full_alphas(decision_alphas)")
    call_args = ", ".join([f"alpha{i+1}=float(decision_alphas[{i}])" for i in range(free_k)])
    lines.append(f"        _ = make_B_DA({call_args}, i0=i0, imax=imax, bmax=bmax, charging_eff=charging_eff)")
    lines.append("        return True")
    lines.append("    except AssertionError:")
    lines.append("        return False")
    lines.append("")
    lines.append("def build_soc_inequality_constraints(i0: float, imax: float, charging_eff: float):")
    lines.append("    ineq = []")
    lines.append("    if charging_eff <= 0.0:")
    lines.append("        raise ValueError(\"charging_eff must be positive\")")
    lines.append("    lengths = LENGTHS")
    lines.append("    signs = SIGNS")
    lines.append("    rhs_lo = -float(i0)")
    lines.append("    rhs_hi = -(float(imax) - float(i0))")
    lines.append("    for i in range(len(lengths)):")
    lines.append("        idx = torch.tensor(list(range(i + 1)), dtype=torch.long)")
    lines.append("        coeff = []")
    lines.append("        for j in range(i + 1):")
    lines.append("            eff = float(charging_eff) if signs[j] == \"pos\" else (1.0 / float(charging_eff))")
    lines.append("            coeff.append(float(lengths[j]) * eff)")
    lines.append("        coeff_t = torch.tensor(coeff, dtype=torch.double)")
    lines.append("        ineq.append((idx, coeff_t, rhs_lo))")
    lines.append("        ineq.append((idx, -coeff_t, rhs_hi))")
    lines.append("    return ineq")
    lines.append("")
    lines.append("def fit_and_suggest(dict_x, i0: float, imax: float, bmax: float, charging_eff: float):")
    lines.append("    keys = sorted(dict_x.keys())")
    lines.append("    X = np.array(keys)")
    lines.append("    Y = np.array([dict_x[k][\"Total_profit\"] for k in keys]).reshape(-1, 1)")
    lines.append("    train_x = torch.tensor(X, dtype=torch.double)")
    lines.append("    train_y = torch.tensor(Y, dtype=torch.double)")
    lines.append("    best_y = float(train_y.max().item())")
    lines.append("")
    lines.append("    model = SingleTaskGP(")
    lines.append("        train_x,")
    lines.append("        train_y,")
    lines.append("        covar_module=ScaleKernel(MaternKernel(nu=2.5)),")
    lines.append(f"        input_transform=Normalize(d={free_k}),")
    lines.append("        outcome_transform=Standardize(m=1),")
    lines.append("    )")
    lines.append("    mll = ExactMarginalLogLikelihood(model.likelihood, model)")
    lines.append("    fit_gpytorch_mll(mll)")
    lines.append("")
    lines.append("    bounds = torch.tensor(")
    lines.append(f"        [[{bounds_lo_s}],")
    lines.append(f"         [{bounds_hi_s}]],")
    lines.append("        dtype=torch.double,")
    lines.append("    )")
    lines.append("    inequality_constraints = build_soc_inequality_constraints(i0=i0, imax=imax, charging_eff=charging_eff)")
    lines.append("")
    lines.append("    iter_num = train_x.shape[0]")
    lines.append("    num_params = train_x.shape[1]")
    lines.append("    beta_iter = beta(iter_num=iter_num, num_params=num_params, delta=0.1)")
    lines.append("    sqrt_beta_iter = np.sqrt(beta_iter)")
    lines.append("    print(f\"beta_iter: {beta_iter}\")")
    lines.append("")
    lines.append("    # UCB optimizer: used for the next BO evaluation point")
    lines.append("    ucb = UpperConfidenceBound(model=model, beta=beta_iter)")
    lines.append("    ucb_x, ucb_val = optimize_acqf(")
    lines.append("        ucb,")
    lines.append("        bounds=bounds,")
    lines.append("        q=1,")
    lines.append("        num_restarts=32,")
    lines.append("        raw_samples=BO_ACQ_SAMPLES,")
    lines.append("        inequality_constraints=inequality_constraints,")
    lines.append("    )")
    lines.append("    ucb_global_max = float(ucb_val.reshape(-1).max().item())")
    lines.append("")
    lines.append("    posterior = model.posterior(train_x)")
    lines.append("    mu_x = posterior.mean.reshape(-1)")
    lines.append("    sigma_x = posterior.variance.clamp_min(0.0).sqrt().reshape(-1)")
    lines.append("    lcb_x_vals = mu_x - sqrt_beta_iter * sigma_x")
    lines.append("    lcb_local_idx = int(torch.argmax(lcb_x_vals).item())")
    lines.append("    lcb_local_max = float(lcb_x_vals[lcb_local_idx].item())")
    lines.append("    gap = float(ucb_global_max - lcb_local_max)")
    lines.append("")
    lines.append("    # Posterior mean optimizer: used only for final recommended point")
    lines.append("    post_mean = PosteriorMean(model=model)")
    lines.append("    post_x, post_val = optimize_acqf(")
    lines.append("        post_mean,")
    lines.append("        bounds=bounds,")
    lines.append("        q=1,")
    lines.append("        num_restarts=32,")
    lines.append("        raw_samples=BO_ACQ_SAMPLES,")
    lines.append("        inequality_constraints=inequality_constraints,")
    lines.append("    )")
    lines.append("    post_x = post_x.reshape(-1, post_x.shape[-1])[0]")
    lines.append("    post_mean_max = float(post_val.reshape(-1).max().item())")
    lines.append("")
    lines.append("    # Candidate for next BO evaluation.")
    lines.append("    cand = ucb_x.reshape(-1, ucb_x.shape[-1])[0]")
    lines.append("    if is_duplicate(cand, train_x):")
    lines.append("        print(\"Selected candidate was duplicate. Searching for nearby non-duplicate candidate.\")")
    lines.append("        base = cand.detach().cpu().numpy()")
    lines.append("        lo = bounds[0].detach().cpu().numpy()")
    lines.append("        hi = bounds[1].detach().cpu().numpy()")
    lines.append("        found = False")
    lines.append("        for _ in range(1024):")
    lines.append("            jitter = np.random.standard_normal(size=base.shape) * 1e-2")
    lines.append("            trial = base + jitter")
    lines.append("            trial = np.minimum(np.maximum(trial, lo), hi)")
    lines.append("            trial_t = torch.tensor(trial, dtype=torch.double)")
    lines.append("            if (not is_duplicate(trial_t, train_x)) and is_soc_feasible(trial.tolist(), i0=i0, imax=imax, bmax=bmax, charging_eff=charging_eff):")
    lines.append("                cand = trial_t")
    lines.append("                found = True")
    lines.append("                break")
    lines.append("        if not found:")
    lines.append("            print(\"Warning: selected candidate was duplicate and jitter search did not find a replacement.\")")
    lines.append("")
    lines.append("    next_vals = cand.detach().cpu().numpy().tolist()")
    lines.append("    post_vals = post_x.detach().cpu().numpy().tolist()")
    lines.append("    return (")
    lines.append("        next_vals,")
    lines.append("        post_vals,")
    lines.append("        X,")
    lines.append("        Y,")
    lines.append("        best_y,")
    lines.append("        beta_iter,")
    lines.append("        ucb_global_max,")
    lines.append("        ucb_x.detach().cpu().numpy().reshape(-1).tolist(),")
    lines.append("        lcb_local_max,")
    lines.append("        X[lcb_local_idx].tolist(),")
    lines.append("        gap,")
    lines.append("        post_mean_max,")
    lines.append("    )")
    lines.append("")
    lines.append("def main():")
    lines.append("    parser = argparse.ArgumentParser()")
    lines.append("    parser.add_argument(\"--outdir\", required=True)")
    lines.append(f"    parser.add_argument(\"--i0\", type=float, default={cfg['i0']})")
    lines.append(f"    parser.add_argument(\"--imax\", type=float, default={cfg['imax']})")
    lines.append(f"    parser.add_argument(\"--bmax\", type=float, default={cfg['bmax']})")
    lines.append(f"    parser.add_argument(\"--charging_eff\", type=float, default={cfg['charging_eff']})")
    lines.append("    args = parser.parse_args()")
    lines.append("")
    lines.append("    outdir = args.outdir")
    lines.append(f"    history_path = os.path.join(outdir, \"bo_history_{'_'.join([f'alpha{i+1}' for i in range(free_k)])}_{df}d.pkl\")")
    lines.append("")
    lines.append("    (")
    lines.append("        alphas,")
    lines.append("        post_alphas,")
    lines.append("        X_arr,")
    lines.append("        y_arr,")
    lines.append("        best_y,")
    lines.append("        beta_iter,")
    lines.append("        ucb_global_max,")
    lines.append("        ucb_argmax,")
    lines.append("        lcb_local_max,")
    lines.append("        lcb_local_argmax,")
    lines.append("        gap,")
    lines.append("        post_mean_max,")
    lines.append("    ) = fit_and_suggest(")
    lines.append("        load_existing_results(outdir),")
    lines.append("        i0=args.i0,")
    lines.append("        imax=args.imax,")
    lines.append("        bmax=args.bmax,")
    lines.append("        charging_eff=args.charging_eff,")
    lines.append("    )")
    lines.append("    if alphas is None:")
    lines.append("        print(\"ERROR: no feasible non-duplicate candidate found.\")")
    lines.append("        return 1")
    lines.append("")
    lines.append("    np.savetxt(NEXT_X_PATH, alphas)")
    lines.append("    print(f\"Next alphas → {NEXT_X_PATH}\")")
    alpha_print_expr = ", ".join([f"alpha{i+1}={{alphas[{i}]}}" for i in range(free_k)])
    lines.append(f"    print(f\"Selected next alphas: {alpha_print_expr}\")")
    post_alpha_print_expr = ", ".join([f"alpha{i+1}={{post_alphas[{i}]}}" for i in range(free_k)])
    lines.append("")
    lines.append("    np.savetxt(POSTMEAN_X_PATH, post_alphas)")
    lines.append("    print(f\"Posterior-mean maximizer → {POSTMEAN_X_PATH}\")")
    lines.append(f"    print(f\"Posterior-mean alphas: {post_alpha_print_expr}\")")
    lines.append("")
    lines.append("    print(f\"ucb_global_max: {ucb_global_max}\")")
    lines.append("    print(f\"lcb_local_max: {lcb_local_max}\")")
    lines.append("    print(f\"ucb_minus_lcb: {gap}\")")
    lines.append("    print(f\"post_mean_max: {post_mean_max}\")")

    lines.append(f"    gap_path = os.path.join(outdir, \"last_ucb_lcb_gap_{'_'.join([f'alpha{i+1}' for i in range(free_k)])}_{df}d.txt\")")
    lines.append("    with open(gap_path, \"w\") as f:")
    lines.append("        f.write(f\"{gap:.16e}\\n\")")
    lines.append("    print(f\"Wrote ucb_minus_lcb={gap:.16e} → {gap_path}\")")
    lines.append("")
    lines.append(f"    post_path = os.path.join(outdir, \"last_postmeanmax_{'_'.join([f'alpha{i+1}' for i in range(free_k)])}_{df}d.txt\")")
    lines.append("    with open(post_path, \"w\") as f:")
    lines.append("        f.write(f\"{post_mean_max} {best_y}\\n\")")
    lines.append("    print(f\"Wrote postmean_max={post_mean_max}, best_y={best_y} → {post_path}\")")
    lines.append("")
    lines.append("    if os.path.exists(history_path):")
    lines.append("        hist = pickle.load(open(history_path, \"rb\"))")
    lines.append("    else:")
    lines.append("        hist = []")
    lines.append("")
    lines.append("    hist.append({")
    for i in range(free_k):
        lines.append(f"        \"alpha{i+1}\": alphas[{i}],")
    for i in range(free_k):
        lines.append(f"        \"post_alpha{i+1}\": post_alphas[{i}],")
    lines.append("        \"X_arr\": X_arr,")
    lines.append("        \"y_arr\": y_arr,")
    lines.append("        \"best_y\": best_y,")
    lines.append("        \"n_obs\": int(X_arr.shape[0]),")
    lines.append("        \"beta_iter\": beta_iter,")
    lines.append("        \"ucb_global_max\": ucb_global_max,")
    lines.append("        \"ucb_argmax\": ucb_argmax,")
    lines.append("        \"lcb_local_max\": lcb_local_max,")
    lines.append("        \"lcb_local_argmax\": lcb_local_argmax,")
    lines.append("        \"ucb_minus_lcb\": gap,")
    lines.append("        \"post_mean_max\": post_mean_max,")
    lines.append("    })")
    lines.append("")
    lines.append(f"    repeat_path = os.path.join(outdir, \"last_repeat_stop_{'_'.join([f'alpha{i+1}' for i in range(free_k)])}_{df}d.txt\")")
    lines.append("    repeat_stop = False")
    lines.append("    if len(hist) >= REPEAT_STOP_WINDOW:")
    lines.append("        recent = hist[-REPEAT_STOP_WINDOW:]")
    lines.append("        rounded_recent = [")
    rounded_tuple = ", ".join([f"round(float(row[\"alpha{i+1}\"]), REPEAT_STOP_DECIMALS)" for i in range(free_k)])
    lines.append(f"            ({rounded_tuple})")
    lines.append("            for row in recent")
    lines.append("        ]")
    lines.append("        repeat_stop = len(set(rounded_recent)) == 1")
    lines.append("    with open(repeat_path, \"w\") as f:")
    lines.append("        f.write(\"yes\\n\" if repeat_stop else \"no\\n\")")
    lines.append("    print(f\"Wrote repeat_stop={repeat_stop} → {repeat_path}\")")
    lines.append("")
    lines.append("    pickle.dump(hist, open(history_path, \"wb\"))")
    lines.append("    print(f\"History updated → {history_path}\")")
    lines.append("")
    lines.append("if __name__ == \"__main__\":")
    lines.append("    raise SystemExit(main())")

    return "\n".join(lines) + "\n"


def render_trainer_py(blocks: List[Dict], df: int, cfg: dict) -> str:
    k = len(blocks)
    free_idx = decision_block_indices(blocks)
    free_k = len(free_idx)
    if free_k == 0:
        alpha_args = ""
        alpha_prints = ""
        bda_args = ""
        tag_lines = []
        tag_join = "rt_only"
    else:
        alpha_args = "\n".join([f"    p.add_argument(\"--alpha{i+1}\", type=float, required=True)" for i in range(free_k)])
        alpha_prints = "\n".join([f"    print(f\"alpha{i+1}                = {{args.alpha{i+1}}}\")" for i in range(free_k)])
        bda_args = ",\n        ".join([f"alpha{i+1}=args.alpha{i+1}" for i in range(free_k)])
        tag_lines = [f"    tag_alpha{i+1} = tag(args.alpha{i+1})" for i in range(free_k)]
        tag_join = "_".join(["alpha{}{{tag_alpha{}}}".format(i + 1, i + 1) for i in range(free_k)])

    lines = []
    lines.append("#!/usr/bin/env python3")
    lines.append("import argparse, os, pickle, numpy as np, sys")
    lines.append("")
    lines.append("SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))")
    lines.append("AUTO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, \"..\", \"..\", \"auto_drop_zeros\"))")
    lines.append("if AUTO_DIR not in sys.path:")
    lines.append("    sys.path.insert(0, AUTO_DIR)")
    lines.append("")
    lines.append("from main import OU_w_derivative, RTRunningCost, softSoC_Constraint, ShadowGPTrainer")
    lines.append("from B_DA import make_B_DA")
    lines.append("")
    lines.append(f"FIXED_ZERO_MASK = {[bool(b.get('fixed_zero')) for b in blocks]}")
    lines.append("")
    lines.append("def tag(x: float) -> str:")
    lines.append("    s = f\"{x:.16e}\"")
    lines.append("    return s.replace(\".\", \"p\").replace(\"-\", \"m\").replace(\"+\", \"\")")
    lines.append("")
    lines.append("def build_full_alpha_values(args):")
    lines.append("    out = []")
    lines.append("    free_idx = 0")
    lines.append("    for is_fixed in FIXED_ZERO_MASK:")
    lines.append("        if is_fixed:")
    lines.append("            out.append(0.0)")
    lines.append("        else:")
    lines.append("            out.append(float(getattr(args, f\"alpha{free_idx + 1}\")))")
    lines.append("            free_idx += 1")
    lines.append("    return out")
    lines.append("")
    lines.append("def main():")
    lines.append("    p = argparse.ArgumentParser()")
    lines.append("")
    lines.append("    # decision variables")
    if alpha_args:
        lines.append(alpha_args)
    lines.append("")
    lines.append("    # OU process")
    lines.append(f"    p.add_argument(\"--nstep\", type=int, default={cfg['nstep']})")
    lines.append(f"    p.add_argument(\"--nsim\", type=int, default={cfg['nsim']})")
    lines.append(f"    p.add_argument(\"--maturity\", type=float, default={cfg['maturity']})")
    lines.append(f"    p.add_argument(\"--ou_alpha\", type=float, default={cfg['ou_alpha']})")
    lines.append(f"    p.add_argument(\"--ou_sigma\", type=float, default={cfg['ou_sigma']})")
    lines.append(f"    p.add_argument(\"--ou_sigma0\", type=float, default={cfg['ou_sigma0']})")
    lines.append("")
    lines.append("    # BESS parameters")
    lines.append(f"    p.add_argument(\"--Bmax\", type=float, default={cfg['bmax']})")
    lines.append(f"    p.add_argument(\"--Bmin\", type=float, default={-cfg['bmax']})")
    lines.append(f"    p.add_argument(\"--SoCmax\", type=float, default={cfg['imax']})")
    lines.append(f"    p.add_argument(\"--charging_eff\", type=float, default={cfg['charging_eff']})")
    lines.append("")
    lines.append("    # penalties")
    lines.append(f"    p.add_argument(\"--initialSoC\", type=float, default={cfg['initialSoC']})")
    lines.append(f"    p.add_argument(\"--terminal_SoCtarget\", type=float, default={cfg['terminal_SoCtarget']})")
    lines.append(f"    p.add_argument(\"--rt_adjustment_pen\", type=float, default={cfg['rt_adjustment_pen']})")
    lines.append(f"    p.add_argument(\"--da_pen\", type=float, default={cfg['da_pen']})")
    lines.append(f"    p.add_argument(\"--final_SoC_penalty\", type=float, default={cfg['final_SoC_penalty']})")
    lines.append("")
    lines.append("    # seeds")
    lines.append(f"    p.add_argument(\"--x0_seed\", type=int, default={cfg['x0_seed']})")
    lines.append(f"    p.add_argument(\"--ou_seed\", type=int, default={cfg['ou_seed']})")
    lines.append(f"    p.add_argument(\"--one_step_seed\", type=int, default={cfg['one_step_seed']})")
    lines.append(f"    p.add_argument(\"--design_seed\", type=int, default={cfg['design_seed']})")
    lines.append("")
    lines.append("    # simulation design")
    lines.append(f"    p.add_argument(\"--nsim_design\", type=int, default={cfg['nsim_design']})")
    lines.append(f"    p.add_argument(\"--batch_size\", type=int, default={cfg['batch_size']})")
    lines.append(f"    p.add_argument(\"--fence\", type=int, default={cfg['fence']})")
    lines.append("")
    lines.append(f"    p.add_argument(\"--ndecisions\", type=int, default={cfg['ndecisions']})")
    lines.append("    p.add_argument(\"--outdir\", type=str, default=\"outputs\")")
    lines.append("    p.add_argument(\"--record_role\", type=str, default=\"bo_eval\")")
    lines.append("")
    lines.append("    args = p.parse_args()")
    lines.append("    if args.ndecisions is None:")
    lines.append("        args.ndecisions = args.nstep")
    lines.append("")
    lines.append("    print(\"========== BESS PARAMETERS ==========\")")
    if alpha_prints:
        lines.append(alpha_prints)
    lines.append("    print(f\"Bmax                  = {args.Bmax}\")")
    lines.append("    print(f\"Bmin                  = {args.Bmin}\")")
    lines.append("    print(f\"SoCmax                = {args.SoCmax}\")")
    lines.append("    print(f\"charging_eff          = {args.charging_eff}\")")
    lines.append("    print(f\"initialSoC            = {args.initialSoC}\")")
    lines.append("    print(f\"terminal_SoCtarget    = {args.terminal_SoCtarget}\")")
    lines.append("    print(f\"rt_adjustment_pen     = {args.rt_adjustment_pen}\")")
    lines.append("    print(f\"da pen                = {args.da_pen}\")")
    lines.append("    print(f\"final_SoC_penalty     = {args.final_SoC_penalty}\")")
    lines.append("    print(f\"nstep                 = {args.nstep}\")")
    lines.append("    print(f\"ndecisions            = {args.ndecisions}\")")
    lines.append("    print(\"=====================================\")")
    lines.append("")
    lines.append("    os.makedirs(args.outdir, exist_ok=True)")
    lines.append("    alpha_values = build_full_alpha_values(args)")
    lines.append("")
    lines.append("    # tags for filenames")
    lines.extend(tag_lines)
    lines.append("")
    lines.append("    # load spline")
    lines.append("    spline_path = os.path.join(AUTO_DIR, 'cubic_spline.pkl')")
    lines.append("    with open(spline_path, 'rb') as f:")
    lines.append("        spline = pickle.load(f)")
    lines.append("    def RT_M_func(t):  return spline['cs'](t)  / 20")
    lines.append("    def RT_dM_func(t): return spline['csd'](t) / 20")
    lines.append("    def DA_M_func(t):  return spline['cs'](t)  / 20")
    lines.append("    assert DA_M_func(0) == RT_M_func(0)")
    lines.append("")
    lines.append("    np.random.seed(args.x0_seed)")
    lines.append("    X0 = np.random.normal(RT_M_func(0), args.ou_sigma0, size=args.nsim)")
    lines.append("")
    lines.append("    process = OU_w_derivative(")
    lines.append("        X0=X0,")
    lines.append("        nstep=args.nstep,")
    lines.append("        nsim=args.nsim,")
    lines.append("        maturity=args.maturity,")
    lines.append("        alpha=args.ou_alpha,")
    lines.append("        meanRevRate_func=RT_M_func,")
    lines.append("        dmeanRevRate_func=RT_dM_func,")
    lines.append("        sigma=args.ou_sigma,")
    lines.append("        noises=None,")
    lines.append("        seed=args.ou_seed")
    lines.append("    )")
    lines.append("")
    lines.append("    running_cost = RTRunningCost(penalty=args.rt_adjustment_pen)")
    lines.append("    final_cost = softSoC_Constraint(")
    lines.append("        charging_efficiency=args.charging_eff,")
    lines.append("        SoC_max=args.SoCmax,")
    lines.append("        SoC_target=args.terminal_SoCtarget,")
    lines.append("        penalty_coeff=args.final_SoC_penalty")
    lines.append("    )")
    lines.append("")
    lines.append("    B_DA_24 = make_B_DA(")
    if bda_args:
        lines.append(f"        {bda_args},")
    lines.append("        charging_eff=args.charging_eff,")
    lines.append("    )")
    lines.append("    assert args.ndecisions % args.maturity == 0")
    lines.append("    rep_factor = int(args.ndecisions // args.maturity)")
    lines.append("    B_DA_vec = np.repeat(B_DA_24, rep_factor)")
    lines.append("")
    lines.append("    trainer = ShadowGPTrainer(")
    lines.append("        process=process,")
    lines.append("        BESSparameters=(args.Bmax, args.Bmin, B_DA_vec, args.SoCmax, args.charging_eff),")
    lines.append("        running_cost=running_cost,")
    lines.append("        final_cost=final_cost,")
    lines.append("        nsim_design=args.nsim_design,")
    lines.append("        batch_size=args.batch_size,")
    lines.append("        ndecisions=args.ndecisions")
    lines.append("    )")
    lines.append("")
    lines.append("    trainer.fit(")
    lines.append("        design_seed=args.design_seed,")
    lines.append("        one_step_seed=args.one_step_seed,")
    lines.append("        fence=args.fence")
    lines.append("    )")
    lines.append("")
    lines.append("    index_path = os.path.join(args.outdir, \"results_index.pkl\")")
    lines.append("    if os.path.exists(index_path):")
    lines.append("        with open(index_path, \"rb\") as f:")
    lines.append("            results_index = pickle.load(f)")
    lines.append("    else:")
    lines.append("        results_index = {\"records\": {}}")
    lines.append("    records = results_index.setdefault(\"records\", {})")
    lines.append("    record_id = None")
    lines.append("    for rid, rec in records.items():")
    lines.append("        rec_alphas = rec.get(\"alphas\")")
    lines.append("        if not isinstance(rec_alphas, list) or len(rec_alphas) != len(alpha_values):")
    lines.append("            continue")
    lines.append("        if all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(rec_alphas, alpha_values)):")
    lines.append("            record_id = str(rid)")
    lines.append("            break")
    lines.append("    existing_ids = [int(rid) for rid in records.keys() if str(rid).isdigit()]")
    lines.append("    if record_id is None:")
    lines.append("        record_id = f\"{(max(existing_ids) if existing_ids else 0) + 1:06d}\"")
    lines.append("    maps_fname = f\"policy_maps_{record_id}.pkl\"")
    lines.append("    maps_path = os.path.join(args.outdir, maps_fname)")
    lines.append("    with open(maps_path, \"wb\") as f:")
    lines.append("        pickle.dump(trainer.policy_maps, f, protocol=pickle.HIGHEST_PROTOCOL)")
    lines.append("")
    lines.append("    def compute_total_profit(tr):")
    lines.append("        I0 = args.initialSoC")
    lines.append("        X_sims = tr.process.sim_trajectories")
    lines.append("        nsim = X_sims.shape[0]")
    lines.append("        ndec = tr.ndecisions")
    lines.append("        dt_opt = tr.dt_opt")
    lines.append("        decision_idx = tr.decision_idx")
    lines.append("        B_DA = tr.B_DA")
    lines.append("")
    lines.append("        Is = np.zeros((nsim, ndec + 1))")
    lines.append("        Is[:, 0] = I0")
    lines.append("        running_profit = np.zeros((nsim, ndec))")
    lines.append("        rc = running_cost")
    lines.append("        for k in range(ndec):")
    lines.append("            Ilb, Iub = 0.0, tr.Imax")
    lines.append("            I_k = Is[:, k]")
    lines.append("            idx_end = decision_idx[k]")
    lines.append("            Price_end = X_sims[:, idx_end]")
    lines.append("            B_DA_k = B_DA[k]")
    lines.append("            LB = np.maximum(")
    lines.append("                tr.Bmin_scalar - B_DA_k,")
    lines.append("                (tr.charging_eff * (Ilb - I_k) / dt_opt) - B_DA_k")
    lines.append("            )")
    lines.append("            UB = np.minimum(")
    lines.append("                tr.Bmax_scalar - B_DA_k,")
    lines.append("                ((Iub - I_k) / (tr.charging_eff * dt_opt)) - B_DA_k")
    lines.append("            )")
    lines.append("            inp = np.column_stack((Price_end, I_k))")
    lines.append("            B_rt = tr.policy_maps[k].predict(inp)[0].flatten()")
    lines.append("            B_rt = np.minimum(np.maximum(B_rt, LB), UB)")
    lines.append("            P_rt = B_DA_k + B_rt")
    lines.append("            eff = tr.charging_eff * (P_rt > 0) + (1/tr.charging_eff) * (P_rt < 0)")
    lines.append("            Is[:, k+1] = np.clip(I_k + P_rt * eff * dt_opt, Ilb, Iub)")
    lines.append("            running_profit[:, k] = -rc.cost(B_rt, 20 * Price_end, B_DA_k) * dt_opt")
    lines.append("        RT_value = np.mean(np.sum(running_profit, axis=1) - final_cost.cost(Is[:, -1]))")
    lines.append("        hours = np.linspace(0, 23, 24)")
    lines.append("        DA_prices = 20 * DA_M_func(hours)")
    lines.append("        B24 = B_DA_24")
    lines.append("        DA_profit_energy = -np.sum(B24 * DA_prices)")
    lines.append("        DA_value = DA_profit_energy - np.abs((B24[1:] - B24[:-1])).sum() * args.da_pen")
    lines.append("        return float(DA_value + RT_value)")
    lines.append("")
    lines.append("    def compute_dart_trading_profit(tr):")
    lines.append("        I0 = args.initialSoC")
    lines.append("        X_sims = tr.process.sim_trajectories")
    lines.append("        nsim = X_sims.shape[0]")
    lines.append("        ndec = tr.ndecisions")
    lines.append("        dt_opt = tr.dt_opt")
    lines.append("        decision_idx = tr.decision_idx")
    lines.append("        B_DA = tr.B_DA")
    lines.append("")
    lines.append("        Is = np.zeros((nsim, ndec + 1))")
    lines.append("        Is[:, 0] = I0")
    lines.append("        rt_trading_profit = np.zeros((nsim, ndec))")
    lines.append("        for k in range(ndec):")
    lines.append("            Ilb, Iub = 0.0, tr.Imax")
    lines.append("            I_k = Is[:, k]")
    lines.append("            idx_end = decision_idx[k]")
    lines.append("            Price_end = X_sims[:, idx_end]")
    lines.append("            B_DA_k = B_DA[k]")
    lines.append("            LB = np.maximum(")
    lines.append("                tr.Bmin_scalar - B_DA_k,")
    lines.append("                (tr.charging_eff * (Ilb - I_k) / dt_opt) - B_DA_k")
    lines.append("            )")
    lines.append("            UB = np.minimum(")
    lines.append("                tr.Bmax_scalar - B_DA_k,")
    lines.append("                ((Iub - I_k) / (tr.charging_eff * dt_opt)) - B_DA_k")
    lines.append("            )")
    lines.append("            inp = np.column_stack((Price_end, I_k))")
    lines.append("            B_rt = tr.policy_maps[k].predict(inp)[0].flatten()")
    lines.append("            B_rt = np.minimum(np.maximum(B_rt, LB), UB)")
    lines.append("            P_rt = B_DA_k + B_rt")
    lines.append("            eff = tr.charging_eff * (P_rt > 0) + (1/tr.charging_eff) * (P_rt < 0)")
    lines.append("            Is[:, k+1] = np.clip(I_k + P_rt * eff * dt_opt, Ilb, Iub)")
    lines.append("            rt_trading_profit[:, k] = -(20 * Price_end) * B_rt * dt_opt")
    lines.append("        RT_trading_value = np.mean(np.sum(rt_trading_profit, axis=1))")
    lines.append("        hours = np.linspace(0, 23, 24)")
    lines.append("        DA_prices = 20 * DA_M_func(hours)")
    lines.append("        B24 = B_DA_24")
    lines.append("        DA_trading_value = -np.sum(B24 * DA_prices)")
    lines.append("        return float(DA_trading_value + RT_trading_value)")
    lines.append("")
    lines.append("    total_profit = compute_total_profit(trainer)")
    lines.append("    dart_trading_profit = compute_dart_trading_profit(trainer)")
    lines.append("")
    lines.append("    rec = dict(records.get(record_id, {}))")
    lines.append("    rec.update({")
    lines.append("        \"id\": record_id,")
    lines.append("        \"alphas\": alpha_values,")
    lines.append("        \"alpha_repr\": [format(a, \".17g\") for a in alpha_values],")
    lines.append("        \"profit\": float(total_profit),")
    lines.append("        \"dart_trading_profit\": float(dart_trading_profit),")
    lines.append("        \"policy_maps_file\": maps_fname,")
    lines.append("        \"record_role\": args.record_role,")
    lines.append("        \"evaluated\": True,")
    lines.append("    })")
    lines.append("    records[record_id] = rec")
    lines.append("    if args.record_role == \"final_postmean\":")
    lines.append("        results_index[\"final_postmean_id\"] = record_id")
    lines.append("    with open(index_path, \"wb\") as f:")
    lines.append("        pickle.dump(results_index, f, protocol=pickle.HIGHEST_PROTOCOL)")
    lines.append("    print(f\"[SAVED] maps  → {maps_path}\")")
    lines.append("    print(f\"[SAVED] index → {index_path} (id={record_id}, profit={total_profit:.6f})\")")
    lines.append("    print(f\"[INFO] pure DART trading profit → {dart_trading_profit:.6f}\")")
    lines.append("")
    lines.append("if __name__ == \"__main__\":")
    lines.append("    main()")

    return "\n".join(lines) + "\n"


def render_print_profits_py(k: int) -> str:
    lines = []
    lines.append("import os")
    lines.append("import fnmatch")
    lines.append("import numpy as np")
    lines.append("import re")
    lines.append("import pickle")
    lines.append("")
    lines.append("def decode_float(s: str) -> float:")
    lines.append("    s = s.replace(\"p\", \".\")")
    lines.append("    s = s.replace(\"em\", \"e-\")")
    lines.append("    s = s.replace(\"m\", \"-\")")
    lines.append("    return float(s)")
    lines.append("")
    lines.append("def parse_alphas(filename: str):")
    lines.append("    name = os.path.splitext(os.path.basename(filename))[0]")
    lines.append("    parts = re.findall(r\"alpha\\d+([^_]+)\", name)")
    lines.append("    if not parts:")
    lines.append("        return []")
    lines.append("    return [decode_float(tag) for tag in parts]")
    lines.append("")
    lines.append("def main():")
    lines.append("    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\", \"outputs\"))")
    lines.append("    if not os.path.isdir(outdir):")
    lines.append("        print(f\"Missing outputs dir: {outdir}\")")
    lines.append("        return 1")
    lines.append("    rows = []")
    lines.append("    for root, _, fnames in os.walk(outdir):")
    lines.append("        if \"results_index.pkl\" in fnames:")
    lines.append("            index_path = os.path.join(root, \"results_index.pkl\")")
    lines.append("            with open(index_path, \"rb\") as f:")
    lines.append("                index = pickle.load(f)")
    lines.append("            for rec in index.get(\"records\", {}).values():")
    lines.append("                if \"profit\" not in rec or \"alphas\" not in rec:")
    lines.append("                    continue")
    lines.append("                rows.append((float(rec[\"profit\"]), [float(a) for a in rec[\"alphas\"]], rec.get(\"id\", \"\")))")
    lines.append("    if rows:")
    lines.append("        rows.sort(key=lambda r: r[0], reverse=True)")
    lines.append("        for profit, alphas, rid in rows:")
    lines.append("            alpha_str = \", \".join([f\"a{i+1}={alphas[i]:.16g}\" for i in range(len(alphas))])")
    lines.append("            rid_str = f\"id={rid}  \" if rid else \"\"")
    lines.append("            print(f\"profit={profit:.6f}  {rid_str}{alpha_str}\")")
    lines.append("        return 0")
    lines.append("")
    lines.append("    files = []")
    lines.append("    for root, _, fnames in os.walk(outdir):")
    lines.append("        for f in fnames:")
    lines.append("            if fnmatch.fnmatch(f, \"profit_alpha1*.npy\"):")
    lines.append("                files.append(os.path.join(root, f))")
    lines.append("    for f in files:")
    lines.append("        data = np.load(f, allow_pickle=True)")
    lines.append("        if isinstance(data, np.ndarray) and data.shape == ():")
    lines.append("            item = data.item()")
    lines.append("            profit = float(item[\"Total_profit\"]) if isinstance(item, dict) else float(item)")
    lines.append("        else:")
    lines.append("            profit = float(np.array(data).squeeze())")
    lines.append("        alphas = parse_alphas(f)")
    lines.append("        if not alphas:")
    lines.append("            continue")
    lines.append("        rows.append((profit, alphas))")
    lines.append("")
    lines.append("    rows.sort(key=lambda r: r[0], reverse=True)")
    lines.append("    for profit, alphas in rows:")
    lines.append("        alpha_str = \", \".join([f\"a{i+1}={alphas[i]:.6g}\" for i in range(len(alphas))])")
    lines.append("        print(f\"profit={profit:.6f}  {alpha_str}\")")
    lines.append("    return 0")
    lines.append("")
    lines.append("if __name__ == \"__main__\":")
    lines.append("    raise SystemExit(main())")
    return "\n".join(lines) + "\n"


def conda_source_lines(indent: str = "") -> list[str]:
    return [
        f"{indent}if [[ -f \"$HOME/miniforge3/etc/profile.d/conda.sh\" ]]; then",
        f"{indent}    source \"$HOME/miniforge3/etc/profile.d/conda.sh\"",
        f"{indent}elif [[ -f \"$HOME/miniconda3/etc/profile.d/conda.sh\" ]]; then",
        f"{indent}    source \"$HOME/miniconda3/etc/profile.d/conda.sh\"",
        f"{indent}else",
        f"{indent}    echo \"ERROR: conda.sh not found under ~/miniforge3 or ~/miniconda3.\"",
        f"{indent}    exit 1",
        f"{indent}fi",
    ]


def render_run_trainer_sh(df: int, cfg: dict, k: int) -> str:
    free_k = k
    if free_k == 0:
        outdir = f"../outputs/ndec{cfg['ndecisions']}nstep{cfg['nstep']}_{df}d"
        lines = []
        lines.append("#!/usr/bin/env bash")
        lines.append("set -e")
        lines.append("")
        lines.append("SCRIPT_TRAIN=\"./run_shadowgp_Trainer.py\"")
        lines.append(f"OUTDIR=\"{outdir}\"")
        lines.append("mkdir -p \"$OUTDIR\"")
        lines.append("")
        lines.append("echo \">>> Running ShadowGPTrainer (RT-only, B_DA=0)\"")
        lines.extend(conda_source_lines())
        lines.append("conda activate test")
        lines.append("")
        cmd = "python3 \"$SCRIPT_TRAIN\" \\\n"
        cmd += f"    --rt_adjustment_pen {cfg['rt_adjustment_pen']} \\\n"
        cmd += f"    --da_pen {cfg['da_pen']} \\\n"
        cmd += f"    --final_SoC_penalty {cfg['final_SoC_penalty']} \\\n"
        cmd += f"    --initialSoC {cfg['initialSoC']} \\\n"
        cmd += f"    --terminal_SoCtarget {cfg['terminal_SoCtarget']} \\\n"
        cmd += f"    --charging_eff {cfg['charging_eff']} \\\n"
        cmd += f"    --outdir \"$OUTDIR\""
        lines.append(cmd)
        lines.append("")
        lines.append("echo \">>> Done. Artifacts saved to: $OUTDIR\"")
        return "\n".join(lines) + "\n"

    outdir = f"../outputs/ndec{cfg['ndecisions']}nstep{cfg['nstep']}_{df}d"
    alpha_labels = " ".join([f"alpha{i+1}" for i in range(free_k)])
    alpha_read = " ".join([f"alpha{i+1}" for i in range(free_k)])
    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -e")
    lines.append("")
    lines.append("SCRIPT_TRAIN=\"./run_shadowgp_Trainer.py\"")
    lines.append("SCRIPT_SAMPLE=\"./sampler.py\"")
    lines.append(f"OUTDIR=\"{outdir}\"")
    lines.append("mkdir -p \"$OUTDIR\"")
    lines.append("")
    lines.append("echo \">>> Activating gpytorchCPU & running sampler.py\"")
    lines.extend(conda_source_lines())
    lines.append("conda activate gpytorchCPU")
    lines.append("")
    lines.append("SAMPLE_FILE=\"$(mktemp)\"")
    lines.append("python3 \"$SCRIPT_SAMPLE\" > \"$SAMPLE_FILE\"")
    lines.append("")
    lines.append(f"echo \">>> Generated ({alpha_labels}) samples:\"")
    lines.append("cat \"$SAMPLE_FILE\"")
    lines.append("")
    lines.append("i=0")
    lines.append(f"while read -r {alpha_read}; do")
    lines.append(f"    [[ -z \"${'{'}alpha1{'}'}\" ]] && continue")
    lines.append("    i=$((i + 1))")
    lines.append("    echo \"\" ")
    lines.append(f"    echo \">>> [$i] Running ShadowGPTrainer for ({alpha_labels})\"")
    lines.append("")
    lines.append("    conda deactivate")
    lines.append("    conda activate test")
    lines.append("")
    cmd = "    python3 \"$SCRIPT_TRAIN\" \\\n"
    for i in range(free_k):
        cmd += f"        --alpha{i+1} \"${{{'alpha'+str(i+1)}}}\" \\\n"
    cmd += f"        --rt_adjustment_pen {cfg['rt_adjustment_pen']} \\\n"
    cmd += f"        --da_pen {cfg['da_pen']} \\\n"
    cmd += f"        --final_SoC_penalty {cfg['final_SoC_penalty']} \\\n"
    cmd += f"        --initialSoC {cfg['initialSoC']} \\\n"
    cmd += f"        --terminal_SoCtarget {cfg['terminal_SoCtarget']} \\\n"
    cmd += f"        --charging_eff {cfg['charging_eff']} \\\n"
    cmd += f"        --outdir \"$OUTDIR\""
    lines.append(cmd)
    lines.append("")
    lines.append("    conda deactivate")
    lines.append("    conda activate gpytorchCPU")
    lines.append("")
    lines.append("done < \"$SAMPLE_FILE\"")
    lines.append("")
    lines.append("echo \"\" ")
    lines.append("echo \">>> All runs completed successfully.\"")
    lines.append("echo \">>> Artifacts saved to: $OUTDIR\"")
    return "\n".join(lines) + "\n"


def render_bo_sh(df: int, cfg: dict, k: int) -> str:
    free_k = k
    outdir = f"../outputs/ndec{cfg['ndecisions']}nstep{cfg['nstep']}_{df}d"
    next_file = "next_" + "_".join([f"alpha{i+1}" for i in range(free_k)]) + f"_{df}d.txt"
    postmean_file = "postmean_" + "_".join([f"alpha{i+1}" for i in range(free_k)]) + f"_{df}d.txt"
    gap_file = "last_ucb_lcb_gap_" + "_".join([f"alpha{i+1}" for i in range(free_k)]) + f"_{df}d.txt"
    repeat_file = "last_repeat_stop_" + "_".join([f"alpha{i+1}" for i in range(free_k)]) + f"_{df}d.txt"
    lines = []
    lines.append("#!/bin/bash")
    lines.extend(conda_source_lines())
    lines.append("")
    lines.append(f"OUTDIR=\"{outdir}\"")
    lines.append("mkdir -p \"$OUTDIR\"")
    lines.append("")
    lines.append(f"THRESH={cfg['thresh']}")
    lines.append(f"MAX_ITER={cfg.get('max_iter', 30)}")
    lines.append("")
    lines.append("FINAL_RUN_DONE=0")
    lines.append("")
    lines.append("run_final_postmean_eval () {")
    lines.append("    echo \"[Final Step] Running one final ShadowGP evaluation at posterior-mean maximizer...\"")
    lines.append("")
    lines.append(f"    POST_FILE=\"{postmean_file}\"")
    lines.append("    if [[ ! -f \"$POST_FILE\" ]]; then")
    lines.append("        echo \"ERROR: $POST_FILE missing.\"")
    lines.append("        exit 1")
    lines.append("    fi")
    lines.append("")
    for i in range(free_k):
        lines.append(f"    POST_ALPHA{i+1}=$(awk 'NR=={i+1} {{gsub(/\\r/,\"\"); gsub(/^[ \\t]+|[ \\t]+$/,\"\"); print; exit}}' \"$POST_FILE\")")
    lines.append("")
    for i in range(free_k):
        lines.append(f"    echo \"  posterior-mean alpha{i+1} = $POST_ALPHA{i+1}\"")
    lines.append("")
    lines.append("    conda deactivate")
    lines.append("    conda activate test")
    lines.append("")
    final_cmd = "    python3 run_shadowgp_Trainer.py \\\n"
    for i in range(free_k):
        final_cmd += f"        --alpha{i+1}=$POST_ALPHA{i+1} \\\n"
    final_cmd += f"        --rt_adjustment_pen={cfg['rt_adjustment_pen']} \\\n"
    final_cmd += f"        --da_pen={cfg['da_pen']} \\\n"
    final_cmd += f"        --final_SoC_penalty={cfg['final_SoC_penalty']} \\\n"
    final_cmd += f"        --initialSoC={cfg['initialSoC']} \\\n"
    final_cmd += f"        --terminal_SoCtarget={cfg['terminal_SoCtarget']} \\\n"
    final_cmd += f"        --charging_eff={cfg['charging_eff']} \\\n"
    final_cmd += "        --record_role=final_postmean \\\n"
    final_cmd += f"        --outdir=\"$OUTDIR\""
    lines.append(final_cmd)
    lines.append("")
    lines.append("    FINAL_RUN_DONE=1")
    lines.append("}")
    lines.append("")
    lines.append("ITER=0")
    lines.append("while [[ $ITER -lt $MAX_ITER ]]; do")
    lines.append("    ITER=$((ITER + 1))")
    lines.append("    echo \"========== BO/TRAIN ITERATION $ITER =========\"")
    lines.append("")
    lines.append(f"    echo \"[Step 1] Running {df}D BO (UCB) in env gpytorchCPU...\"")
    lines.append("    conda activate gpytorchCPU")
    lines.append("")
    lines.append(f"    python3 bo_{df}d.py --outdir \"$OUTDIR\"")
    lines.append("")
    lines.append(f"    echo \"[Step 2] Reading BO next alphas...\"")
    lines.append(f"    NEXT_FILE=\"{next_file}\"")
    lines.append("    if [[ ! -f \"$NEXT_FILE\" ]]; then")
    lines.append("        echo \"ERROR: $NEXT_FILE missing.\"")
    lines.append("        exit 1")
    lines.append("    fi")
    lines.append("")
    for i in range(free_k):
        lines.append(f"    ALPHA{i+1}_VAL=$(awk 'NR=={i+1} {{gsub(/\\r/,\"\"); gsub(/^[ \\t]+|[ \\t]+$/,\"\"); print; exit}}' \"$NEXT_FILE\")")
    for i in range(free_k):
        lines.append(f"    echo \"  alpha{i+1} = $ALPHA{i+1}_VAL\"")
    lines.append("")
    lines.append(f"    echo \"[Step 3] Reading stopping info from $OUTDIR/{gap_file}...\"")
    lines.append(f"    GAP_FILE=\"$OUTDIR/{gap_file}\"")
    lines.append("    if [[ ! -f \"$GAP_FILE\" ]]; then")
    lines.append("        echo \"ERROR: $GAP_FILE not found.\"")
    lines.append("        exit 1")
    lines.append("    fi")
    lines.append("    GAP=$(awk 'NR==1 {gsub(/\\r/,\"\"); gsub(/^[ \\t]+|[ \\t]+$/,\"\"); print; exit}' \"$GAP_FILE\")")
    lines.append("    echo \"  ucb_minus_lcb = $GAP\"")
    lines.append("    echo \"  threshold     = $THRESH\"")
    lines.append("")
    lines.append(f"    REPEAT_FILE=\"$OUTDIR/{repeat_file}\"")
    lines.append("    if [[ ! -f \"$REPEAT_FILE\" ]]; then")
    lines.append("        echo \"ERROR: $REPEAT_FILE not found.\"")
    lines.append("        exit 1")
    lines.append("    fi")
    lines.append("    REPEAT_STOP=$(awk 'NR==1 {gsub(/\\r/,\"\"); gsub(/^[ \\t]+|[ \\t]+$/,\"\"); print; exit}' \"$REPEAT_FILE\")")
    lines.append("    echo \"  repeat_stop   = $REPEAT_STOP\"")
    lines.append("")
    lines.append("    below=$(awk -v g=\"$GAP\" -v thr=\"$THRESH\" 'BEGIN { if (g < thr) print \"yes\"; else print \"no\"; }')")
    lines.append("    if [[ \"$below\" == \"yes\" ]]; then")
    lines.append("        echo \"Stopping criterion met: ucb_minus_lcb = $GAP < $THRESH\"")
    lines.append("        run_final_postmean_eval")
    lines.append("        break")
    lines.append("    fi")
    lines.append("    if [[ \"$REPEAT_STOP\" == \"yes\" ]]; then")
    lines.append("        echo \"Stopping criterion met: last repeated BO candidates match after rounding.\"")
    lines.append("        run_final_postmean_eval")
    lines.append("        break")
    lines.append("    fi")
    lines.append("")
    lines.append("    echo \"[Step 4] Running ShadowGP Trainer at BO-selected point...\"")
    lines.append("    conda deactivate")
    lines.append("    conda activate test")
    lines.append("")
    cmd = "    python3 run_shadowgp_Trainer.py \\\n"
    for i in range(free_k):
        cmd += f"        --alpha{i+1}=$ALPHA{i+1}_VAL \\\n"
    cmd += f"        --rt_adjustment_pen={cfg['rt_adjustment_pen']} \\\n"
    cmd += f"        --da_pen={cfg['da_pen']} \\\n"
    cmd += f"        --final_SoC_penalty={cfg['final_SoC_penalty']} \\\n"
    cmd += f"        --initialSoC={cfg['initialSoC']} \\\n"
    cmd += f"        --terminal_SoCtarget={cfg['terminal_SoCtarget']} \\\n"
    cmd += f"        --charging_eff={cfg['charging_eff']} \\\n"
    cmd += f"        --outdir=\"$OUTDIR\""
    lines.append(cmd)
    lines.append("")
    lines.append("    echo \"========== END ITERATION $ITER =========\"")
    lines.append("    echo")
    lines.append("done")
    lines.append("")
    lines.append("if [[ $ITER -ge $MAX_ITER && $FINAL_RUN_DONE -eq 0 ]]; then")
    lines.append("    echo \"Reached MAX_ITER=$MAX_ITER.\"")
    lines.append("    echo \"Running final evaluation at posterior-mean maximizer.\"")
    lines.append("    run_final_postmean_eval")
    lines.append("fi")
    lines.append("")
    lines.append("echo \"BO loop stopped successfully.\"")
    return "\n".join(lines) + "\n"


def generate_df_folder(base_dir: str, df: int, prev_df: int, blocks: List[Dict], cfg: dict) -> None:
    df_dir = os.path.join(base_dir, f"{df}df")
    scripts_dir = os.path.join(df_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    prev_scripts = os.path.join(base_dir, f"{prev_df}df", "scripts")
    for fname in ["main.py", "cubic_spline.pkl"]:
        src = find_asset_source(base_dir, fname, prev_scripts=prev_scripts)
        dst = os.path.join(scripts_dir, fname)
        if src and not os.path.exists(dst):
            shutil.copy2(src, dst)

    blocks = normalize_blocks(blocks)
    k = len(blocks)
    free_k = len(decision_block_indices(blocks))

    with open(os.path.join(df_dir, "blocks.json"), "w") as f:
        json.dump(blocks, f, indent=2)

    with open(os.path.join(scripts_dir, "B_DA.py"), "w") as f:
        f.write(render_bda_py(blocks, k, cfg))

    if free_k > 0:
        with open(os.path.join(scripts_dir, "sampler.py"), "w") as f:
            f.write(render_sampler_py(blocks, cfg))

        with open(os.path.join(scripts_dir, f"bo_{df}d.py"), "w") as f:
            f.write(render_bo_py(blocks, df, cfg))

    with open(os.path.join(scripts_dir, "run_shadowgp_Trainer.py"), "w") as f:
        f.write(render_trainer_py(blocks, df, cfg))

    with open(os.path.join(scripts_dir, "run_shadowgp_trainer.sh"), "w") as f:
        f.write(render_run_trainer_sh(df, cfg, free_k))

    if free_k > 0:
        with open(os.path.join(scripts_dir, f"BO_{df}d.sh"), "w") as f:
            f.write(render_bo_sh(df, cfg, free_k))

    with open(os.path.join(scripts_dir, "print_profits.py"), "w") as f:
        f.write(render_print_profits_py(k))

    if free_k > 0:
        next_path = os.path.join(scripts_dir, "next_" + "_".join([f"alpha{i+1}" for i in range(free_k)]) + f"_{df}d.txt")
        if not os.path.exists(next_path):
            with open(next_path, "w") as f:
                f.write("\n")


def ensure_zero_df(base_dir: str, cfg: dict) -> None:
    df = 0
    df_dir = os.path.join(base_dir, f"{df}df")
    scripts_dir = os.path.join(df_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)

    for fname in ["main.py", "cubic_spline.pkl"]:
        src = find_asset_source(base_dir, fname)
        dst = os.path.join(scripts_dir, fname)
        if src and not os.path.exists(dst):
            shutil.copy2(src, dst)

    blocks: List[Dict] = []
    with open(os.path.join(df_dir, "blocks.json"), "w") as f:
        json.dump(blocks, f, indent=2)

    with open(os.path.join(scripts_dir, "B_DA.py"), "w") as f:
        f.write(render_bda_py(blocks, 0, cfg))

    with open(os.path.join(scripts_dir, "run_shadowgp_Trainer.py"), "w") as f:
        f.write(render_trainer_py(blocks, df, cfg))

    with open(os.path.join(scripts_dir, "run_shadowgp_trainer.sh"), "w") as f:
        f.write(render_run_trainer_sh(df, cfg, 0))

    print(f"[OK] Initialized 0df (RT-only, B_DA=0) in {df_dir}")


def ensure_auto_assets(base_dir: str) -> None:
    auto_dir = os.path.join(base_dir, "auto_drop_zeros")
    os.makedirs(auto_dir, exist_ok=True)
    for fname in ["main.py", "cubic_spline.pkl"]:
        src = find_asset_source(base_dir, fname)
        dst = os.path.join(auto_dir, fname)
        if src and not os.path.exists(dst):
            shutil.copy2(src, dst)


def ensure_print_profits(scripts_dir: str, k: int) -> None:
    os.makedirs(scripts_dir, exist_ok=True)
    path = os.path.join(scripts_dir, "print_profits.py")
    with open(path, "w") as f:
        f.write(render_print_profits_py(k))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base", default="Nesting_start_v2")
    parser.add_argument("--start-df", type=int, default=None)
    parser.add_argument("--max-df", type=int, default=None)
    parser.add_argument("--init", action="store_true", help="Initialize start df folder only, no split.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    base_dir = args.base
    start_df = args.start_df if args.start_df is not None else cfg["start_df"]
    max_df = args.max_df if args.max_df is not None else cfg["max_df"]
    ensure_auto_assets(base_dir)
    if cfg.get("include_0df", False):
        ensure_zero_df(base_dir, cfg)

    # initial blocks for start_df
    blocks = normalize_blocks([
        {"start": b["start"], "end": b["end"], **({"sign": b["sign"]} if "sign" in b else {})}
        for b in cfg["initial_blocks"]
    ])

    blocks_path = os.path.join(base_dir, f"{start_df}df", "blocks.json")
    if os.path.exists(blocks_path):
        with open(blocks_path, "r") as f:
            blocks = normalize_blocks(json.load(f))

    if args.init:
        df_dir = os.path.join(base_dir, f"{start_df}df")
        if os.path.exists(df_dir):
            scripts_dir = os.path.join(df_dir, "scripts")
            ensure_print_profits(scripts_dir, len(blocks))
            print(f"[OK] Updated print_profits.py in {scripts_dir}.")
            return
        generate_df_folder(base_dir, start_df, start_df, blocks, cfg)
        print(f"[OK] Initialized {start_df}df.")
        next_script = "run_shadowgp_trainer.sh" if not decision_block_indices(blocks) else "run_shadowgp_trainer.sh"
        print(f"Next step: (cd {os.path.join(base_dir, f'{start_df}df', 'scripts')} && bash {next_script})")
        return

    df = start_df
    while df < max_df:
        df_dir = os.path.join(base_dir, f"{df}df")
        scripts_dir = os.path.join(df_dir, "scripts")
        outdir = os.path.join(df_dir, "outputs", f"ndec{cfg['ndecisions']}nstep{cfg['nstep']}_{df}d")

        if not os.path.isdir(outdir):
            print(f"[STOP] Missing outputs for {df}d: {outdir}")
            script_name = f"BO_{df}d.sh" if decision_block_indices(blocks) else "run_shadowgp_trainer.sh"
            print(f"Run: (cd {scripts_dir} && bash {script_name})")
            return

        # load best policy
        k = len(blocks)
        best_alphas, best_profit = load_best_profit(outdir, k)
        print(f"[{df}d] Best profit: {best_profit:.6f} alphas={best_alphas}")
        final_postmean_rec = get_final_postmean_record(outdir, k)
        zero_tol = float(cfg.get("drop_zero_tol", ZERO_TOL))

        policy_maps = load_policy_maps(outdir, best_alphas)
        auto_dir = os.path.join(base_dir, "auto_drop_zeros")
        main_mod, bda_mod = load_main_and_bda(scripts_dir, auto_dir)

        # spline
        spline_path = os.path.join(auto_dir, "cubic_spline.pkl")
        with open(spline_path, "rb") as f:
            spline = pickle.load(f)
        def RT_M_func(t):  return spline['cs'](t) / 20
        def RT_dM_func(t): return spline['csd'](t) / 20
        def DA_M_func(t):  return spline['cs'](t) / 20

        np.random.seed(cfg["x0_seed"])
        X0 = np.random.normal(RT_M_func(0), cfg["ou_sigma0"], size=cfg["nsim"])

        process = main_mod.OU_w_derivative(
            X0=X0,
            nstep=cfg["nstep"],
            nsim=cfg["nsim"],
            maturity=cfg["maturity"],
            alpha=cfg["ou_alpha"],
            meanRevRate_func=RT_M_func,
            dmeanRevRate_func=RT_dM_func,
            sigma=cfg["ou_sigma"],
            noises=None,
            seed=cfg["ou_seed"],
        )

        running_cost = main_mod.RTRunningCost(penalty=cfg["rt_adjustment_pen"])
        final_cost = main_mod.softSoC_Constraint(
            charging_efficiency=cfg["charging_eff"],
            SoC_max=cfg["imax"],
            SoC_target=cfg["terminal_SoCtarget"],
            penalty_coeff=cfg["final_SoC_penalty"],
        )

        make_bda = bda_mod.make_B_DA
        B_DA_hourly = call_make_bda(make_bda, best_alphas, cfg, blocks)

        assert cfg["ndecisions"] % cfg["maturity"] == 0
        rep_factor = int(cfg["ndecisions"] // cfg["maturity"])
        B_DA_vec = np.repeat(B_DA_hourly, rep_factor)

        assert cfg["nstep"] % cfg["ndecisions"] == 0
        step_factor = cfg["nstep"] // cfg["ndecisions"]
        decision_idx = np.arange(0, cfg["nstep"], step_factor)
        dt_opt = cfg["maturity"] / cfg["ndecisions"]

        res = simulate_policy_decisions(
            policy_maps=policy_maps,
            dt_opt=dt_opt,
            decision_idx=decision_idx,
            B_DA_vec=B_DA_vec,
            process=process,
            I0=cfg["initialSoC"],
            Imax=cfg["imax"],
            charging_eff=cfg["charging_eff"],
            running_cost=running_cost,
            final_cost=final_cost,
            Bmin_scalar=-cfg["bmax"],
            Bmax_scalar=cfg["bmax"],
            rt_adjustment_pen=cfg["rt_adjustment_pen"],
        )

        B_96 = res["B_rt"]

        metrics = compute_split_metrics(B_96, blocks)
        print(f"[{df}d] Split metrics (sorted by abs_mean_diff):")
        for row in metrics:
            print(
                f"  seg={row['segment']} split={row['split_at']} "
                f"lhs_mean={row['lhs_mean']:.6f} rhs_mean={row['rhs_mean']:.6f} "
                f"lhs_std={row['lhs_std']:.6f} rhs_std={row['rhs_std']:.6f} "
                f"mean_diff={row['mean_diff']:.6f} abs_diff={row['abs_mean_diff']:.6f}"
            )

        split_count = int(cfg.get("splits_per_round", 2))
        split_infos = choose_splits(B_96, blocks, split_count)
        split_info = split_infos[0]
        for i, info in enumerate(split_infos, start=1):
            print(
                f"[{df}d] Split #{i} at hour {info['split_at']} "
                f"(segment {info['segment_start']}-{info['segment_end']}, {info['segment_type']})"
            )
        report_json, report_txt = save_split_report(base_dir, df, metrics, split_info, split_infos)
        print(f"[{df}d] Saved split metrics: {report_txt}")
        print(f"[{df}d] Saved split metrics (json): {report_json}")

        blocks_for_next = mark_zero_fixed_blocks_except_splits(
            blocks,
            final_postmean_rec["alphas"],
            zero_tol,
            split_infos,
        )
        zero_fixed_count = sum(1 for b in blocks_for_next if b.get("fixed_zero"))
        print(f"[{df}d] Zero-fixed untouched blocks for next dfs: {zero_fixed_count}/{len(blocks_for_next)}")

        new_blocks = apply_splits(blocks_for_next, split_infos)
        new_blocks_signed = assign_block_signs(B_96, B_DA_vec, new_blocks, cfg)
        new_blocks_signed = normalize_blocks(new_blocks_signed)

        next_df = len(new_blocks_signed)
        if next_df <= df:
            raise RuntimeError(f"Expected split refinement to increase df, got {df} -> {next_df}.")
        next_df_dir = os.path.join(base_dir, f"{next_df}df")
        if os.path.exists(next_df_dir):
            existing_blocks_path = os.path.join(next_df_dir, "blocks.json")
            if os.path.exists(existing_blocks_path):
                with open(existing_blocks_path, "r") as f:
                    existing_blocks = normalize_blocks(json.load(f))
                if existing_blocks != new_blocks_signed:
                    raise RuntimeError(
                        f"Existing {next_df}df blocks differ from the two-split refinement. "
                        f"Use a fresh --base directory or remove/regenerate {next_df_dir}."
                    )
            next_scripts = os.path.join(next_df_dir, "scripts")
            ensure_print_profits(next_scripts, len(new_blocks_signed))
            print(f"[OK] Updated print_profits.py in {next_scripts}.")
        else:
            generate_df_folder(base_dir, next_df, df, new_blocks_signed, cfg)

        added, skipped_duplicate, skipped_invalid, skipped_inherited = inherit_previous_results(
            base_dir=base_dir,
            prev_df=df,
            curr_df=next_df,
            prev_blocks=blocks_for_next,
            curr_blocks=new_blocks_signed,
            cfg=cfg,
            prev_bda_blocks=blocks,
        )
        print(
            f"[{next_df}d] Warm-start inheritance: "
            f"added={added}, skipped_duplicate={skipped_duplicate}, "
            f"skipped_invalid={skipped_invalid}, skipped_inherited={skipped_inherited}"
        )

        blocks = normalize_blocks(new_blocks_signed)

        print(f"[OK] Generated {next_df}df with {len(blocks)} blocks.")
        next_script = "BO_{}d.sh".format(next_df) if decision_block_indices(blocks) else "run_shadowgp_trainer.sh"
        print(f"Next step: (cd {os.path.join(base_dir, f'{next_df}df', 'scripts')} && bash {next_script})")
        kept_profit_file, kept_profit_val = cleanup_outputs_keep_best(outdir)
        print(f"[{df}d] Cleanup complete: kept {kept_profit_file} (profit={kept_profit_val:.6f})")
        df = next_df


if __name__ == "__main__":
    main()
