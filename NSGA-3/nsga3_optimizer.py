import os, json, argparse
import numpy as np
import pandas as pd
from itertools import product
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

# ── Files ────────────────────────────────────────────────────────────────────
CACHE_FILE   = "eval_cache.json"
RESULTS_FILE = "pareto_results/iter-1.csv"
SUGGEST_FILE = "suggested_configs.csv"

# ── Your exact parameter space ───────────────────────────────────────────────
PARAMS = {
    "xN_model":          ["Large-v3", "Medium", "Small", "Tiny",
                          "Large-v3 Distil", "Medium Distil", "Small Distil"],
    "xT_input_length":   ["1500 Frames", "750 Frames"],
    "xV_resolution":     [1, 2],
    "xR_lora_rank":      ["No Lora", "r=8", "r=16", "r=32", "r=64"],
    "xQ_quantization":   ["QAT-FP32", "QAT-FP16", "QAT-INT8", "QAT-INT4"],
    "xP_pruning":        ["dense", "2:4", "1:4"],
}

# ── CSV column aliases  (what your spreadsheet actually says → internal key) ─
COL_ALIASES = {
    "xN (Model)":           "xN_model",
    "xT(Input length)":     "xT_input_length",
    "xV(Resolution/Stride)":"xV_resolution",
    "xRLora Rank":          "xR_lora_rank",
    "xQ(Quantization)":     "xQ_quantization",
    "xP(Pruning)":          "xP_pruning",
    "WER":                  "wer",
    "FLOPS":                "flops",
    "Space Memory":         "memory",
}

PARAM_NAMES = list(PARAMS.keys())
PARAM_GRID  = list(PARAMS.values())
OBJ_NAMES   = ["wer", "flops", "memory"]
N_OBJ       = 3

# ── Ordinal encoding for NSGA-III (maps each value → integer index) ──────────
ENCODERS = {p: {v: i for i, v in enumerate(vals)} for p, vals in PARAMS.items()}
DECODERS = {p: {i: v for i, v in enumerate(vals)} for p, vals in PARAMS.items()}

# Build a string→canonical lookup so CSV strings always resolve correctly:
#   e.g.  "2" → 2  (int),  "1" → 1  (int),  "r =8" → "r=8" etc.
def _build_str_lookup():
    lookup = {}
    for p, vals in PARAMS.items():
        lookup[p] = {}
        for v in vals:
            # map the canonical value itself
            lookup[p][str(v).strip()] = v
            # also map without spaces around = and :
            lookup[p][str(v).replace(" ", "")] = v
    return lookup

STR_LOOKUP = _build_str_lookup()

def canonicalise(param, raw_value):
    """Coerce a raw CSV string to the canonical Python value defined in PARAMS."""
    s = str(raw_value).strip()
    if s in STR_LOOKUP[param]:
        return STR_LOOKUP[param][s]
    # fallback: try removing all spaces
    s2 = s.replace(" ", "")
    if s2 in STR_LOOKUP[param]:
        return STR_LOOKUP[param][s2]
    raise ValueError(
        f"Unknown value {raw_value!r} for parameter '{param}'. "
        f"Expected one of: {list(PARAMS[param])}"
    )

def encode(cfg):
    return np.array([ENCODERS[p][cfg[p]] for p in PARAM_NAMES], dtype=float)

def decode(x):
    return {p: DECODERS[p][int(round(x[i])) % len(PARAMS[p])]
            for i, p in enumerate(PARAM_NAMES)}

def config_key(cfg):
    return json.dumps({k: cfg[k] for k in PARAM_NAMES}, sort_keys=True)

def build_search_space():
    return [dict(zip(PARAM_NAMES, c)) for c in product(*PARAM_GRID)]

# ── Cache I/O ────────────────────────────────────────────────────────────────
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# ── CSV ingestion (handles both alias and canonical column names) ─────────────
def normalise_df(df):
    df = df.rename(columns={k: v for k, v in COL_ALIASES.items() if k in df.columns})
    # also accept lower-case / stripped variants
    df.columns = [c.strip() for c in df.columns]
    return df

def load_csv_into_cache(csv_path, cache):
    df = normalise_df(pd.read_csv(csv_path))

    # Rows with all three objectives filled = evaluated
    obj_mask = df[OBJ_NAMES].notna().all(axis=1)
    eval_df  = df[obj_mask].copy()
    pend_df  = df[~obj_mask].copy()

    added = 0
    for _, row in eval_df.iterrows():
        try:
            cfg = {p: canonicalise(p, row[p]) for p in PARAM_NAMES}
        except ValueError as e:
            print(f"  Skipping row: {e}")
            continue
        objs = [float(row[o]) for o in OBJ_NAMES]
        key  = config_key(cfg)
        if key not in cache:
            cache[key] = {"config": cfg, "objectives": objs}
            added += 1

    save_cache(cache)
    print(f"Loaded {added} new evaluated rows from {csv_path}  "
          f"({len(pend_df)} rows pending objectives)")
    return cache, pend_df

# ── NSGA-III problem wrapper ─────────────────────────────────────────────────
class WhisperProblem(Problem):
    def __init__(self, search_space, cache):
        self.search_space = search_space
        self.cache        = cache
        xl = np.array([0] * len(PARAM_NAMES), dtype=float)
        xu = np.array([len(v) - 1 for v in PARAM_GRID], dtype=float)
        super().__init__(n_var=len(PARAM_NAMES), n_obj=N_OBJ, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for x in X:
            cfg  = decode(x)
            key  = config_key(cfg)
            objs = self.cache.get(key, {}).get("objectives", [1e9, 1e9, 1e9])
            F.append(objs)
        out["F"] = np.array(F, dtype=float)

def seed_pop(cache, search_space):
    idx_map = {config_key(c): i for i, c in enumerate(search_space)}
    X, F = [], []
    for key, entry in cache.items():
        if entry["objectives"][0] >= 1e8:
            continue
        try:
            cfg = {p: canonicalise(p, entry["config"][p]) for p in PARAM_NAMES}
        except ValueError:
            continue
        norm_key = config_key(cfg)
        if norm_key in idx_map:
            X.append(encode(cfg))
            F.append(entry["objectives"])
    return (np.array(X), np.array(F)) if X else (None, None)

def run_nsga3(cache, search_space, n_gen, pop_size):
    ref_dirs = get_reference_directions("das-dennis", N_OBJ, n_partitions=12)
    pop_size = max(pop_size, len(ref_dirs))
    problem  = WhisperProblem(search_space, cache)

    seed_X, seed_F = seed_pop(cache, search_space)
    if seed_X is not None and len(seed_X) >= 2:
        sampling = Population.new(X=seed_X, F=seed_F)
        print(f"Seeding NSGA-III with {len(seed_X)} cached evaluations")
    else:
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        sampling = FloatRandomSampling()

    algo   = NSGA3(ref_dirs=ref_dirs, pop_size=pop_size,
                   sampling=sampling, eliminate_duplicates=True)
    result = minimize(problem, algo,
                      termination=get_termination("n_gen", n_gen),
                      seed=42, verbose=True)
    return result

# ── Suggestion logic (exploration-aware) ────────────────────────────────────
def suggest_next(cache, search_space, n_suggest, rng_seed=42):
    evaluated_keys = set(cache.keys())
    valid_objs = [cache[k]["objectives"] for k in evaluated_keys
                  if cache[k]["objectives"][0] < 1e8]

    def safe_encode(cfg_raw):
        try:
            return encode({p: canonicalise(p, cfg_raw[p]) for p in PARAM_NAMES})
        except ValueError:
            return None

    unevaluated = [c for c in search_space if config_key(c) not in evaluated_keys]
    if not unevaluated:
        print("All configurations have been evaluated.")
        return pd.DataFrame()

    rng = np.random.default_rng(rng_seed)

    if not valid_objs:
        # cold start — spread across parameter diversity
        chosen = rng.choice(len(unevaluated), min(n_suggest, len(unevaluated)), replace=False)
        suggestions = [unevaluated[i] for i in chosen]
    else:
        # run NSGA-III on cache to find promising directions
        result = run_nsga3(cache, search_space, n_gen=20, pop_size=92)
        nsga_cfgs = [decode(x) for x in result.X]
        nsga_keys = {config_key(c) for c in nsga_cfgs}

        # Priority 1: NSGA-III suggested & unevaluated
        priority = [c for c in nsga_cfgs if config_key(c) not in evaluated_keys]
        print(f"Priority configurations: {len(priority)}")

        # Priority 2: fill remainder with diversity sampling from unevaluated
        if len(priority) < n_suggest:
            objs_arr  = np.array(valid_objs)
            nd_mask   = find_non_dominated(objs_arr)
            pareto_y  = objs_arr[nd_mask]
            ref_mean  = pareto_y.mean(axis=0)

            remainder = [c for c in unevaluated if config_key(c) not in nsga_keys]
            sample    = remainder if len(remainder) <= 300 else [
                remainder[i] for i in rng.choice(len(remainder), 300, replace=False)]

            scores = []
            for cfg in sample:
                x_enc  = encode(cfg) / np.array([max(len(v)-1,1) for v in PARAM_GRID])
                p_encs = [safe_encode(cache[k]["config"]) for k in evaluated_keys
                          if cache[k]["objectives"][0] < 1e8]
                p_enc  = np.array([e for e in p_encs if e is not None]) / np.array([max(len(v)-1,1) for v in PARAM_GRID])
                dist   = np.min(np.linalg.norm(p_enc - x_enc, axis=1)) if len(p_enc) else 1.0
                scores.append(dist + rng.normal(0, 0.001))

            fill = [sample[i] for i in np.argsort(scores)[-(n_suggest - len(priority)):]]
            priority += fill

        suggestions = priority[:n_suggest]

    df = pd.DataFrame(suggestions)[PARAM_NAMES]

    # Add empty objective columns so user can fill them in
    for o in OBJ_NAMES:
        df[o] = ""

    df.to_csv(SUGGEST_FILE, index=False)
    print(f"\n{len(df)} suggested configs → {SUGGEST_FILE}")
    print(df[PARAM_NAMES].to_string(index=False))
    return df

# ── Register completed results ───────────────────────────────────────────────
def register_results(csv_path):
    cache = load_cache()
    cache, _ = load_csv_into_cache(csv_path, cache)
    n_valid = sum(1 for v in cache.values() if v["objectives"][0] < 1e8)
    print(f"Cache now holds {n_valid} valid evaluated configurations.")

# ── Pareto extraction & summary ──────────────────────────────────────────────
def pareto_summary(cache):
    valid = []
    for v in cache.values():
        if v["objectives"][0] >= 1e8:
            continue
        try:
            cfg = {p: canonicalise(p, v["config"][p]) for p in PARAM_NAMES}
        except ValueError:
            continue
        valid.append((cfg, v["objectives"]))
    if len(valid) < 2:
        print("Need at least 2 evaluated configs. Register more results first.")
        return

    cfgs, objs = zip(*valid)
    objs_arr   = np.array(objs)
    # BUG FIX: find_non_dominated() returns an array of INDICES of the
    # non-dominated points directly, not a same-length boolean mask.
    # np.where() on that index array corrupts it. Use the returned
    # indices directly (same as suggest_next() already correctly does
    # via plain fancy-indexing).
    nd_idx = find_non_dominated(objs_arr)

    rows = [{**cfgs[i], **dict(zip(OBJ_NAMES, objs_arr[i]))} for i in nd_idx]
    df   = pd.DataFrame(rows)
    df.to_csv(RESULTS_FILE, index=False)

    print(f"\n{'='*72}")
    print(f"PARETO FRONT  —  {len(df)} non-dominated configurations")
    print(f"{'='*72}")
    print(df[PARAM_NAMES + OBJ_NAMES].to_string(index=False))

    # Knee point
    obj_df   = df[OBJ_NAMES].astype(float)
    lo, hi   = obj_df.min(), obj_df.max()
    norm     = (obj_df - lo) / (hi - lo + 1e-8)
    knee_idx = (norm ** 2).sum(axis=1).idxmin()
    print(f"\n{'─'*72}")
    print("KNEE POINT  (best balanced trade-off across all 3 objectives)")
    print(f"{'─'*72}")
    print(df.loc[knee_idx][PARAM_NAMES + OBJ_NAMES].to_string())

    print(f"\nFull Pareto CSV saved → {RESULTS_FILE}")
    return df

# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="NSGA-III Semi-Manual Optimizer")
    sub    = parser.add_subparsers(dest="mode", required=True)

    p_sug = sub.add_parser("suggest", help="Suggest next configs to evaluate")
    p_sug.add_argument("--csv",       default=None, help="Optional: load a results CSV first")
    p_sug.add_argument("--n",         type=int, default=10, help="How many configs to suggest")

    p_reg = sub.add_parser("register", help="Register completed results into cache")
    p_reg.add_argument("csv", help="CSV with param columns + wer/flops/memory filled in")

    sub.add_parser("pareto", help="Compute and display Pareto front from all cached results")

    args = parser.parse_args()

    search_space = build_search_space()
    print(f"Search space: {len(search_space)} total combinations\n")

    if args.mode == "suggest":
        cache = load_cache()
        if args.csv:
            cache, _ = load_csv_into_cache(args.csv, cache)
        suggest_next(cache, search_space, args.n)

    elif args.mode == "register":
        register_results(args.csv)

    elif args.mode == "pareto":
        pareto_summary(load_cache())

if __name__ == "__main__":
    main()