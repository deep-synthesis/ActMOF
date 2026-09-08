"""
================================================================================
Seed-fixed rerun of the two HYBRID_GP_Matern52_{EI,PI}_ALT_gpt-5.5 methods
================================================================================

These are the paper's headline result (#1 ranked method, best_q=215,832 for
the PI variant). Odd iterations use a plain sklearn GP (Matern5/2 kernel,
n_restarts=1) with EI or PI acquisition over 15,000 randomly-sampled
candidates; even iterations use gpt-5.5 with the same 45-experiment history
window as the rest of the benchmark. Matches Benchmark v3.ipynb's
"hybrid_alternating" logic exactly (iteration % 2 == 0 -> LLM, else -> GP).

Fixes applied (same two real, fixable bugs as gpt5mini_seed_rerun.py):
  - seed is now passed to every gpt-5.5 call (confirmed accepted)
  - reasoning_effort="minimal" is set explicitly (gpt-5.5 defaults to
    "medium" and otherwise silently burns the whole token budget - the same
    failure mode DeepSeek and the unfixed gpt-5.6-luna/gpt-5-mini calls hit)
  - temperature is NOT attempted - gpt-5.5 rejects it server-side (confirmed
    by verify_original_models.py), a hard API constraint, not something this
    script can control. Disclosed rather than silently worked around.

The GP half of each iteration is already fully deterministic given a seed
(scikit-learn, no API involved) - only the LLM half benefits from seed/
reasoning_effort control.

Protocol: window=45, BUDGET_TOTAL_EXPERIMENTS=150 (49 iterations after the
initial 3), 5 runs, same seed formula as Benchmark v3.ipynb
(RANDOM_STATE + 10_000*run + stable_int(method)).
================================================================================
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

# ==============================================================================
# 0. CONFIG
# ==============================================================================

OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_API_KEY_FALLBACK = "PASTE_YOUR_OPENAI_API_KEY_HERE"

# Observed 2026-09-08: CONFIRMED via a real API call (response.model read
# back directly) to resolve to gpt-5.5-2026-04-23. Used as-is below, not
# pinned to this dated snapshot -- see ../README.md's "Resolved model
# snapshots" section.
OPENAI_MODEL = "gpt-5.5"
# gpt-5.5's reasoning_effort vocabulary is {none, low, medium, high, xhigh} -
# NOT {minimal, low, medium, high} like gpt-5-mini/gpt-5-nano (confirmed by
# probing the live 400 error text). Using the wrong word would just get
# silently dropped by the kwarg-unwind logic below, defaulting to "medium" -
# exactly the failure mode this rerun exists to avoid - so get it right here
# instead of relying on that fallback.
REASONING_EFFORT = "none"
TRY_TEMPERATURE = False   # gpt-5.5 rejects temperature server-side (confirmed)
LOW_TEMPERATURE = 0.2

WINDOW = 45
RANDOM_STATE = 42
INITIAL_EXPERIMENTS = 3
BATCH_SIZE = 3
BUDGET_TOTAL_EXPERIMENTS = 150
N_ITERATIONS = (BUDGET_TOTAL_EXPERIMENTS - INITIAL_EXPERIMENTS) // BATCH_SIZE  # 49
N_RUNS = 5
LLM_MAX_REPAIR_ATTEMPTS = 3

ACQ_CANDIDATES_PER_ITER = 15_000   # GP half uses the FAST candidate pool size,
                                     # not the *_SLOW=4000 one - matches the
                                     # original notebook's hybrid branch exactly.
DIVERSITY_LAMBDA = 0.03
MIN_STD = 1e-9
USE_LOG1P_TARGET = True
GP_N_RESTARTS = 1

VARIANTS = [
    {"name": "HYBRID_GP_Matern52_EI_ALT_gpt-5.5", "gp_acq": "ei"},
    {"name": "HYBRID_GP_Matern52_PI_ALT_gpt-5.5", "gp_acq": "pi"},
]

SMOKE_TEST = False   # 1 variant x 1 run x 3 iterations first

# Portability note (only change vs original hybrid_seed_rerun.py): POOL_DIR/
# OUTPUT_DIR now default to ../../data and ../../outputs inside this repo,
# overridable via ACTMOF_POOL_DIR / ACTMOF_OUTPUT_DIR. No GP, prompt, seed,
# or acquisition logic below was touched.
_REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = Path(os.environ.get("ACTMOF_POOL_DIR", str(_REPO_ROOT / "data")))
POOL_GLOB = "design_grid_part*.csv"
FEATURES = ["metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature"]
TARGET = "q_final"
FEATURE_RANGES = {
    "metal_amount": (5, 75), "modulator": (5, 15), "add_solvent": (0, 30),
    "reaction_time": (1, 12), "reaction_temperature": (10, 30),
}
OUTPUT_DIR = Path(os.environ.get("ACTMOF_OUTPUT_DIR", str(_REPO_ROOT / "outputs" / "hybrid_gp_llm")))

PRICE_INPUT_PER_M = 5.00    # gpt-5.5, last known pricing (verify before a big run)
PRICE_OUTPUT_PER_M = 30.00

BASELINE = {
    "HYBRID_GP_Matern52_EI_ALT_gpt-5.5": (160198.8, 18481.6),
    "HYBRID_GP_Matern52_PI_ALT_gpt-5.5": (215831.8, 42825.3),
}


def stable_int(text: str, modulo: int = 2**31 - 1) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16) % modulo


# ==============================================================================
# 1. Candidate pool oracle
# ==============================================================================

def build_oracle_lookup() -> np.ndarray:
    shape = tuple(hi - lo + 1 for lo, hi in FEATURE_RANGES.values())
    oracle = np.full(shape, np.nan, dtype=np.float32)
    files = sorted(glob.glob(str(POOL_DIR / POOL_GLOB)))
    if not files:
        raise FileNotFoundError(f"No files matched {POOL_DIR / POOL_GLOB}")
    usecols = FEATURES + [TARGET]
    mins = [lo for lo, _ in FEATURE_RANGES.values()]
    print(f"[pool] loading {len(files)} file(s)")
    for fp in files:
        df = pd.read_csv(fp, usecols=usecols)
        idx = tuple((df[c].to_numpy(dtype=np.int64) - m) for c, m in zip(FEATURES, mins))
        oracle[idx] = df[TARGET].to_numpy(dtype=np.float32)
    print(f"[pool] filled: {oracle.size - int(np.isnan(oracle).sum()):,}/{oracle.size:,}")
    return oracle


def q_of(oracle: np.ndarray, cond: Dict[str, int]) -> float:
    idx = tuple(cond[c] - FEATURE_RANGES[c][0] for c in FEATURES)
    val = oracle[idx]
    return 0.0 if np.isnan(val) else float(val)


def in_range(cond: Dict[str, int]) -> bool:
    return all(
        isinstance(cond.get(c), (int, np.integer)) and FEATURE_RANGES[c][0] <= cond[c] <= FEATURE_RANGES[c][1]
        for c in FEATURES
    )


def scale_row(cond: Dict[str, int]) -> np.ndarray:
    return np.array([(cond[c] - lo) / (hi - lo) for c, (lo, hi) in FEATURE_RANGES.items()], dtype=float)


def sample_untried_candidates(rng: np.random.Generator, tried: set, n: int) -> List[Dict[str, int]]:
    out, seen_local = [], set()
    while len(out) < n:
        cond = {c: int(rng.integers(lo, hi + 1)) for c, (lo, hi) in FEATURE_RANGES.items()}
        t = tuple(cond[c] for c in FEATURES)
        if t in tried or t in seen_local:
            continue
        seen_local.add(t)
        out.append(cond)
    return out


# ==============================================================================
# 2. GP half (identical to Benchmark v3.ipynb's SklearnGPSurrogate, matern52)
# ==============================================================================

def y_to_model_scale(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.log1p(np.maximum(y, 0.0)) if USE_LOG1P_TARGET else y.copy()


def fit_gp(X: np.ndarray, y: np.ndarray, random_state: int) -> GaussianProcessRegressor:
    p = X.shape[1]
    kernel = (C(1.0, (1e-3, 1e3))
              * Matern(length_scale=np.ones(p) * 0.5, length_scale_bounds=(1e-2, 10.0), nu=2.5)
              + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0)))
    model = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True,
                                      n_restarts_optimizer=GP_N_RESTARTS, random_state=random_state)
    model.fit(X, y)
    return model


def gaussian_ei(mu, sd, best, xi=0.01):
    sd = np.maximum(sd, MIN_STD)
    imp = mu - best - xi
    z = imp / sd
    return imp * norm.cdf(z) + sd * norm.pdf(z)


def gaussian_pi(mu, sd, best, xi=0.01):
    sd = np.maximum(sd, MIN_STD)
    return norm.cdf((mu - best - xi) / sd)


def greedy_diverse_batch(cand_local_idx, acq, X_scaled_cand, k, rng, diversity_lambda):
    if len(cand_local_idx) <= k:
        return cand_local_idx
    acq = np.where(np.isfinite(acq), acq, -np.inf) + rng.normal(0.0, 1e-12, size=len(acq))
    acq_z = (acq - np.nanmean(acq)) / (np.nanstd(acq) + 1e-12)
    chosen_local: List[int] = []
    remaining = np.arange(len(cand_local_idx))
    for _ in range(k):
        if not chosen_local or diversity_lambda <= 0:
            best_local = remaining[np.argmax(acq_z[remaining])]
        else:
            X_chosen = X_scaled_cand[np.asarray(chosen_local)]
            X_rem = X_scaled_cand[remaining]
            dist = np.sqrt(((X_rem[:, None, :] - X_chosen[None, :, :]) ** 2).sum(axis=2))
            score = acq_z[remaining] + diversity_lambda * dist.min(axis=1)
            best_local = remaining[np.argmax(score)]
        chosen_local.append(int(best_local))
        remaining = remaining[remaining != best_local]
        if len(remaining) == 0:
            break
    return cand_local_idx[np.asarray(chosen_local, dtype=int)]


def gp_select_batch(history_conditions: List[Dict[str, int]], history_y_raw: List[float],
                     tried: set, rng: np.random.Generator, gp_acq: str, seed: int) -> List[Dict[str, int]]:
    X_train = np.array([scale_row(c) for c in history_conditions])
    y_train_model = y_to_model_scale(np.array(history_y_raw))
    model = fit_gp(X_train, y_train_model, random_state=seed)

    candidates = sample_untried_candidates(rng, tried, ACQ_CANDIDATES_PER_ITER)
    X_cand = np.array([scale_row(c) for c in candidates])
    mu, sd = model.predict(X_cand, return_std=True)
    sd = np.maximum(sd, MIN_STD)

    y_best_model = float(np.max(y_train_model))
    acq = gaussian_ei(mu, sd, y_best_model) if gp_acq == "ei" else gaussian_pi(mu, sd, y_best_model)

    chosen_local = greedy_diverse_batch(np.arange(len(candidates)), acq, X_cand, BATCH_SIZE, rng, DIVERSITY_LAMBDA)
    return [candidates[i] for i in chosen_local]


# ==============================================================================
# 3. LLM half (identical machinery to gpt5mini_seed_rerun.py)
# ==============================================================================

class MOFConditionChoice(BaseModel):
    metal_amount: int = Field(..., description="Integer 5 to 75.")
    modulator: int = Field(..., description="Integer 5 to 15.")
    add_solvent: int = Field(..., description="Integer 0 to 30.")
    reaction_time: int = Field(..., description="Integer 1 to 12.")
    reaction_temperature: int = Field(..., description="Integer 10 to 30.")
    rationale: str = Field(..., description="Brief chemical and active-learning rationale.")


class MOFNextBatch(BaseModel):
    strategy_summary: str = Field(..., description="Brief strategy for the three proposed experiments.")
    next_conditions: List[MOFConditionChoice] = Field(..., min_length=3, max_length=3)


def llm_system_prompt() -> str:
    return """
You are a metal-organic framework synthetic chemist who choose the next microwave synthesis experiments for aluminum dicarboxylate MOFs.

The task:
Choose exactly 3 new experiments from candidate_pool.
The goal is to maximize q_final = Intensity_final / FWHM_final.
Higher q_final is better because it means stronger diffraction intensity and sharper peaks.

Reaction background:
Each experiment uses the same amount of organic linker: 1 mmol.
The linker is a linear or V-shaped dicarboxylate linker.
The metal precursor is AlCl3.6H2O stock solution (100 mg/mL) in DI water.
The modulator/base is NaOH stock solution (100 mg/mL) in DI water.
The reaction is run in a 35 mL Pyrex microwave pressure vessel.

Very important:
The five input variables are integer codes, not direct mmol/mL/minute/Celsius values.

1. metal_amount: integer 5 to 75 (encodes AlCl3.6H2O stock volume).
2. modulator: integer 5 to 15 (encodes NaOH stock volume).
3. add_solvent: integer 0 to 30 (encodes extra DI water).
4. reaction_time: integer 1 to 12 (encodes microwave heating time).
5. reaction_temperature: integer 10 to 30 (encodes microwave heating temperature).

Hard rules:
- Do not invent conditions outside the stated integer ranges.
- Do not repeat previously suggested conditions.
- Output ONLY the five encoded integer variables (plus a short rationale) for
  each of the three new experiments - never decoded mL/mmol/min/Celsius values.
"""


def build_user_payload(history_records: List[Dict[str, Any]], repair_context: str = "") -> Dict[str, Any]:
    return {
        "job": "Suggest the next three MOF synthesis experiments for active learning.",
        "objective": "Maximize q_final = Intensity_final / FWHM_final.",
        "important_instruction": "Generate new synthesis parameter sets directly.",
        "previous_experiments": history_records[-WINDOW:],
        "allowed_ranges": {
            "metal_amount": "integer 5 to 75", "modulator": "integer 5 to 15",
            "add_solvent": "integer 0 to 30", "reaction_time": "integer 1 to 12",
            "reaction_temperature": "integer 10 to 30",
        },
        "required_behavior": [
            "Propose exactly three new experiments.",
            "Return only the five encoded integer variables for each experiment.",
            "Do not repeat previous experiments.",
            "Prefer high expected q_final, but include diversity to learn the landscape.",
            "Use one exploitation condition, one local perturbation, and one exploration condition.",
        ],
        "repair_context": repair_context,
    }


def get_openai_client():
    from openai import OpenAI

    key = os.environ.get(OPENAI_API_KEY_ENV_VAR, "").strip()
    if not key or key == "PASTE_YOUR_OPENAI_API_KEY_HERE":
        key = OPENAI_API_KEY_FALLBACK.strip()
    if not key or key == "PASTE_YOUR_OPENAI_API_KEY_HERE":
        raise RuntimeError(f"No OpenAI API key found. Set {OPENAI_API_KEY_ENV_VAR}.")
    return OpenAI(api_key=key)


_COST_STATE = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "calls": 0}
_WARNED_DROPPED_KWARGS: set = set()
_DROPPABLE_KWARGS = ("reasoning_effort", "seed", "temperature", "max_completion_tokens")


def _track_cost(response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    _COST_STATE["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
    _COST_STATE["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    _COST_STATE["calls"] += 1
    details = getattr(usage, "completion_tokens_details", None)
    _COST_STATE["reasoning_tokens"] += (getattr(details, "reasoning_tokens", 0) or 0) if details else 0


def current_cost_estimate() -> float:
    return (_COST_STATE["input_tokens"] / 1e6 * PRICE_INPUT_PER_M
            + _COST_STATE["output_tokens"] / 1e6 * PRICE_OUTPUT_PER_M)


def _extract_offending_kwarg(exc: Exception) -> Optional[str]:
    msg = str(exc)
    m = re.search(r"unexpected keyword argument '(\w+)'", msg)
    if m and m.group(1) in _DROPPABLE_KWARGS:
        return m.group(1)
    for k in _DROPPABLE_KWARGS:
        if f"'{k}'" in msg:
            return k
    return None


def _extract_rename(exc: Exception):
    m = re.search(r"'(\w+)' is not supported.*[Uu]se '(\w+)' instead", str(exc))
    return (m.group(1), m.group(2)) if m else None


def call_llm_for_batch(client, history_records: List[Dict[str, Any]], tried: set, seed: int
                        ) -> Tuple[List[Dict[str, int]], Dict[str, Any]]:
    repair_context = ""
    attempts_log = []

    for attempt in range(1, LLM_MAX_REPAIR_ATTEMPTS + 1):
        payload = build_user_payload(history_records, repair_context)
        kwargs = dict(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": llm_system_prompt()},
                {"role": "user", "content": json.dumps(payload, default=float)},
            ],
            response_format=MOFNextBatch,
            reasoning_effort=REASONING_EFFORT,
            seed=seed,
            max_completion_tokens=1500,
        )
        if TRY_TEMPERATURE:
            kwargs["temperature"] = LOW_TEMPERATURE

        try:
            response = None
            for _ in range(len(_DROPPABLE_KWARGS) + 1):
                try:
                    response = client.chat.completions.parse(**kwargs)
                    break
                except Exception as e:
                    rename = _extract_rename(e)
                    if rename and rename[0] in kwargs:
                        old, new = rename
                        kwargs[new] = kwargs.pop(old)
                        continue
                    offending = _extract_offending_kwarg(e)
                    if offending and offending in kwargs:
                        if offending not in _WARNED_DROPPED_KWARGS:
                            print(f"[warn] model={OPENAI_MODEL} rejected `{offending}` ({str(e)[:150]}); dropping.")
                            _WARNED_DROPPED_KWARGS.add(offending)
                        kwargs.pop(offending, None)
                        continue
                    raise
            if response is None:
                response = client.chat.completions.parse(**kwargs)

            _track_cost(response)
            choice = response.choices[0]
            parsed = choice.message.parsed
            if parsed is None:
                raise ValueError(f"message.parsed is None (finish_reason={getattr(choice, 'finish_reason', '?')})")

            conditions = [c.model_dump() for c in parsed.next_conditions]
            errors = []
            seen = set()
            for c in conditions:
                t = tuple(int(c[f]) for f in FEATURES)
                if not in_range(c):
                    errors.append(f"Out of range: {c}")
                elif t in tried:
                    errors.append(f"Repeats an already-tried condition: {c}")
                elif t in seen:
                    errors.append(f"Duplicate within this batch: {c}")
                seen.add(t)

            attempts_log.append({"attempt": attempt, "valid": not errors, "errors": errors})
            if not errors and len(conditions) == 3:
                return conditions, {"valid": True, "attempt_count": attempt,
                                     "strategy_summary": parsed.strategy_summary,
                                     "attempts": attempts_log, "fallback_used": False}
            repair_context = "Your previous answer was invalid: " + "; ".join(errors)

        except Exception as e:
            attempts_log.append({"attempt": attempt, "valid": False, "error": str(e)[:300]})
            repair_context = f"Your previous answer could not be used ({str(e)[:200]}). Try again."

    fallback = []
    while len(fallback) < 3:
        c = {col: random.randint(*FEATURE_RANGES[col]) for col in FEATURES}
        t = tuple(c[col] for col in FEATURES)
        if t not in tried and t not in {tuple(f[col] for col in FEATURES) for f in fallback}:
            c["rationale"] = "random_fallback_after_repair_attempts_exhausted"
            fallback.append(c)
    return fallback, {"valid": False, "attempt_count": LLM_MAX_REPAIR_ATTEMPTS, "strategy_summary": "",
                       "attempts": attempts_log, "fallback_used": True}


def run_canary_check(client) -> None:
    print(f"[canary] sending 1 real test call to {OPENAI_MODEL} (reasoning_effort={REASONING_EFFORT!r}) ...")
    dummy_history = [{"metal_amount": 25, "modulator": 10, "add_solvent": 15, "reaction_time": 6,
                       "reaction_temperature": 24, "q_final": 12000.0}]
    conditions, meta = call_llm_for_batch(client, dummy_history, set(), seed=RANDOM_STATE)
    if not meta["valid"]:
        raise RuntimeError(f"[CANARY FAILED] no valid structured response after 3 attempts. "
                            f"Attempts: {json.dumps(meta['attempts'], default=str)[:1500]}")
    reasoning_tok = _COST_STATE["reasoning_tokens"]
    if reasoning_tok > 200:
        raise RuntimeError(f"[CANARY FAILED] reasoning_tokens={reasoning_tok} despite "
                            f"reasoning_effort={REASONING_EFFORT!r} - refusing to proceed.")
    if "seed" in _WARNED_DROPPED_KWARGS:
        raise RuntimeError("[CANARY FAILED] seed was silently dropped - refusing to proceed uncontrolled.")
    if "reasoning_effort" in _WARNED_DROPPED_KWARGS:
        raise RuntimeError(f"[CANARY FAILED] reasoning_effort={REASONING_EFFORT!r} was rejected and silently "
                            f"dropped - the model would fall back to its default effort (likely 'medium'), "
                            f"risking the same empty-content failure mode this rerun exists to avoid. "
                            f"Check the current valid reasoning_effort values for {OPENAI_MODEL}.")
    print(f"[canary] OK - reasoning_tokens={reasoning_tok}, seed accepted={'seed' not in _WARNED_DROPPED_KWARGS}")
    print(f"[canary] sample conditions: {conditions}")
    print(f"[canary] cost so far: ${current_cost_estimate():.5f}")


# ==============================================================================
# 4. One full alternating run for one HYBRID variant
# ==============================================================================

def run_single_run(client, oracle: np.ndarray, method: str, gp_acq: str, run_idx: int, history_writer
                    ) -> List[Dict[str, Any]]:
    seed_base = (RANDOM_STATE + 10_000 * run_idx + stable_int(method)) % (2**31 - 1)
    rng = np.random.default_rng(seed_base)

    def sample_cond() -> Dict[str, int]:
        return {c: int(rng.integers(lo, hi + 1)) for c, (lo, hi) in FEATURE_RANGES.items()}

    tried: set = set()
    history_conditions: List[Dict[str, int]] = []
    history_y_raw: List[float] = []
    best_q = 0.0
    rows = []

    def add_experiment(cond: Dict[str, int]) -> float:
        nonlocal best_q
        t = tuple(cond[c] for c in FEATURES)
        tried.add(t)
        q = q_of(oracle, cond)
        history_conditions.append(cond)
        history_y_raw.append(q)
        best_q = max(best_q, q)
        return q

    for _ in range(INITIAL_EXPERIMENTS):
        add_experiment(sample_cond())
    rows.append({"method": method, "run": run_idx, "iteration": 0, "n_experiments": len(history_conditions),
                 "best_q": best_q, "status": "initial_random"})

    for iteration in range(1, N_ITERATIONS + 1):
        if iteration % 2 == 0:
            call_seed = seed_base + iteration
            history_records = [{**c, TARGET: y} for c, y in zip(history_conditions, history_y_raw)]
            conditions, meta = call_llm_for_batch(client, history_records, tried, seed=call_seed)
            status = "llm_ok" if meta["valid"] else "llm_fallback_random"
        else:
            conditions = gp_select_batch(history_conditions, history_y_raw, tried, rng, gp_acq, seed=seed_base + iteration)
            status = f"gp_iteration_acq={gp_acq}"

        for c in conditions:
            add_experiment(c)

        rows.append({"method": method, "run": run_idx, "iteration": iteration, "n_experiments": len(history_conditions),
                     "best_q": best_q, "status": status})

        history_writer.write(json.dumps({
            "method": method, "run": run_idx, "iteration": iteration, "status": status,
            "best_q_after": best_q, "running_cost_usd": round(current_cost_estimate(), 4),
        }, default=float) + "\n")
        history_writer.flush()

        print(f"[{method} run={run_idx}] iter {iteration:>2}/{N_ITERATIONS} n={len(history_conditions):>3} "
              f"best_q={best_q:>10,.0f} status={status:<22} cost=${current_cost_estimate():.4f}")

    return rows


# ==============================================================================
# 5. Main
# ==============================================================================

def main() -> None:
    global N_RUNS, N_ITERATIONS, VARIANTS

    client = get_openai_client()
    run_canary_check(client)

    if SMOKE_TEST:
        print("=== SMOKE_TEST True: 1 variant x 1 run x 3 iterations only ===")
        VARIANTS = VARIANTS[:1]
        N_RUNS = 1
        N_ITERATIONS = 3

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "hybrid_iteration_metrics.csv"
    history_path = OUTPUT_DIR / "hybrid_history.jsonl"

    done_pairs = set()
    prev = pd.DataFrame()
    if metrics_path.exists():
        prev = pd.read_csv(metrics_path)
        for (m, r), g in prev.groupby(["method", "run"]):
            if g["iteration"].max() >= N_ITERATIONS:
                done_pairs.add((m, r))
        print(f"[resume] {len(done_pairs)} (method, run) pair(s) already complete")

    oracle = build_oracle_lookup()
    all_rows: List[Dict[str, Any]] = prev.to_dict("records") if not prev.empty else []

    with open(history_path, "a", encoding="utf-8") as history_writer:
        for run_idx in range(N_RUNS):
            for variant in VARIANTS:
                if (variant["name"], run_idx) in done_pairs:
                    print(f"[skip] {variant['name']} run={run_idx} already done")
                    continue
                rows = run_single_run(client, oracle, variant["name"], variant["gp_acq"], run_idx, history_writer)
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(metrics_path, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(metrics_path, index=False)

    print("\n=== DONE ===")
    print(f"Total LLM calls: {_COST_STATE['calls']}, cost: ${current_cost_estimate():.4f}")
    if not SMOKE_TEST:
        final = df[df["iteration"] == N_ITERATIONS]
        for method, (bmean, bstd) in BASELINE.items():
            sub = final[final["method"] == method]
            if not sub.empty:
                print(f"{method}: new mean={sub['best_q'].mean():,.0f} std={sub['best_q'].std():,.0f}  "
                      f"(baseline mean={bmean:,.0f} std={bstd:,.0f})")


if __name__ == "__main__":
    main()
