"""
================================================================================
Seed-fixed rerun of LLM_PURE_gpt-5-mini (window=45, full budget)
================================================================================

Validation-scale rerun of the cheapest of the 3 original benchmark models,
with the two real, fixable bugs corrected:
  - seed IS now passed (gpt-5-mini confirmed to accept it - see
    verify_original_models.py results)
  - reasoning_effort="minimal" is set explicitly (gpt-5-mini defaults to
    "medium" reasoning effort, which - confirmed by the same probe - silently
    burns the whole token budget and returns empty content otherwise; this is
    the same failure mode DeepSeek hit)
  - temperature is NOT passed - gpt-5-mini rejects it server-side (confirmed);
    this is a hard API limitation for GPT-5-family models, not something this
    script can fix, and is disclosed rather than silently worked around.

Protocol matches Benchmark v3.ipynb exactly for this one method: window=45,
BUDGET_TOTAL_EXPERIMENTS=150 (49 iterations after the initial 3), 5 runs,
same seed formula (RANDOM_STATE + 10_000*run + stable_int(method)).

Uses the real design_grid_part*.csv full-factorial pool for ground truth
q_final (same O(1) lookup array as the other standalone scripts in this
project), and client.chat.completions.parse (not client.responses.parse -
that endpoint has no seed parameter, see openai_window_ablation.py).

Compare the result against the existing LLM_PURE_gpt-5-mini row in
bo_benchmark_final_summary.csv (mean_best_q=74,206, std=29,098, 5 runs) to
see whether fixing seed materially changes the reported numbers.
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

# ==============================================================================
# 0. CONFIG
# ==============================================================================

OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"
OPENAI_API_KEY_FALLBACK = "PASTE_YOUR_OPENAI_API_KEY_HERE"

# Observed 2026-09-08: CONFIRMED via a real API call (response.model read
# back directly) to resolve to gpt-5-mini-2025-08-07. Used as-is below, not
# pinned to this dated snapshot -- see ../README.md's "Resolved model
# snapshots" section.
OPENAI_MODEL = "gpt-5-mini"
REASONING_EFFORT = "minimal"   # confirmed necessary - see module docstring
TRY_TEMPERATURE = False        # gpt-5-mini rejects temperature server-side;
                                 # kept as a flag rather than silently omitted
                                 # so this fact is visible in the code.
LOW_TEMPERATURE = 0.2

METHOD_NAME = "LLM_PURE_gpt-5-mini"   # must match the original registry name
                                        # exactly - it feeds stable_int() below
WINDOW = 45
RANDOM_STATE = 42
INITIAL_EXPERIMENTS = 3
BATCH_SIZE = 3
BUDGET_TOTAL_EXPERIMENTS = 150
N_ITERATIONS = (BUDGET_TOTAL_EXPERIMENTS - INITIAL_EXPERIMENTS) // BATCH_SIZE  # 49
N_RUNS = 5
LLM_MAX_REPAIR_ATTEMPTS = 3

SMOKE_TEST = False   # 1 run x 3 iterations first; flip to False for the real thing

# Portability note (only change vs original gpt5mini_seed_rerun.py): POOL_DIR/
# OUTPUT_DIR now default to ../../data and ../../outputs inside this repo,
# overridable via ACTMOF_POOL_DIR / ACTMOF_OUTPUT_DIR. No prompt, model,
# seed, or acquisition logic below was touched.
_REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = Path(os.environ.get("ACTMOF_POOL_DIR", str(_REPO_ROOT / "data")))
POOL_GLOB = "design_grid_part*.csv"
FEATURES = ["metal_amount", "modulator", "add_solvent", "reaction_time", "reaction_temperature"]
TARGET = "q_final"
FEATURE_RANGES = {
    "metal_amount": (5, 75), "modulator": (5, 15), "add_solvent": (0, 30),
    "reaction_time": (1, 12), "reaction_temperature": (10, 30),
}
OUTPUT_DIR = Path(os.environ.get("ACTMOF_OUTPUT_DIR", str(_REPO_ROOT / "outputs" / "llm_pure_gpt5mini")))

PRICE_INPUT_PER_M = 0.125   # gpt-5-mini, last known pricing - verify before a
PRICE_OUTPUT_PER_M = 1.00   # big run if it's been a while (see verify_original_models.py)

# Baseline to compare against (bo_benchmark_final_summary.csv, LLM_PURE_gpt-5-mini)
BASELINE_MEAN = 74206.0
BASELINE_STD = 29098.0


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


# ==============================================================================
# 2. Schema + prompt (identical to Benchmark v3.ipynb)
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


# ==============================================================================
# 3. OpenAI call - chat.completions.parse, defensive kwarg unwind
# ==============================================================================

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
                return conditions, {
                    "valid": True, "attempt_count": attempt,
                    "strategy_summary": parsed.strategy_summary,
                    "attempts": attempts_log, "fallback_used": False,
                }
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


# ==============================================================================
# 4. Canary check
# ==============================================================================

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
        raise RuntimeError("[CANARY FAILED] seed was silently dropped - the whole point of this "
                            "rerun is to control seed, refusing to proceed uncontrolled.")
    print(f"[canary] OK - reasoning_tokens={reasoning_tok}, seed accepted={'seed' not in _WARNED_DROPPED_KWARGS}, "
          f"temperature accepted={'temperature' not in _WARNED_DROPPED_KWARGS if TRY_TEMPERATURE else 'not attempted'}")
    print(f"[canary] sample conditions: {conditions}")
    print(f"[canary] cost so far: ${current_cost_estimate():.5f}")


# ==============================================================================
# 5. One full run (49 iterations)
# ==============================================================================

def run_single_run(client, oracle: np.ndarray, run_idx: int, history_writer) -> List[Dict[str, Any]]:
    seed_base = (RANDOM_STATE + 10_000 * run_idx + stable_int(METHOD_NAME)) % (2**31 - 1)
    rng = np.random.default_rng(seed_base)

    def sample_cond() -> Dict[str, int]:
        return {c: int(rng.integers(lo, hi + 1)) for c, (lo, hi) in FEATURE_RANGES.items()}

    tried: set = set()
    history_records: List[Dict[str, Any]] = []
    best_q = 0.0
    rows = []

    def add_experiment(cond: Dict[str, int]) -> float:
        nonlocal best_q
        t = tuple(cond[c] for c in FEATURES)
        tried.add(t)
        q = q_of(oracle, cond)
        rec = {c: cond[c] for c in FEATURES}
        rec[TARGET] = q
        history_records.append(rec)
        best_q = max(best_q, q)
        return q

    for _ in range(INITIAL_EXPERIMENTS):
        add_experiment(sample_cond())
    rows.append({"run": run_idx, "iteration": 0, "n_experiments": len(history_records),
                 "best_q": best_q, "status": "initial_random"})

    for iteration in range(1, N_ITERATIONS + 1):
        call_seed = seed_base + iteration
        conditions, meta = call_llm_for_batch(client, history_records, tried, seed=call_seed)
        for c in conditions:
            add_experiment(c)
        status = "llm_ok" if meta["valid"] else "llm_fallback_random"
        rows.append({"run": run_idx, "iteration": iteration, "n_experiments": len(history_records),
                     "best_q": best_q, "status": status})

        history_writer.write(json.dumps({
            "run": run_idx, "iteration": iteration, "seed": call_seed,
            "llm_metadata": meta, "selected_conditions": conditions,
            "best_q_after": best_q, "running_cost_usd": round(current_cost_estimate(), 4),
        }, default=float) + "\n")
        history_writer.flush()

        print(f"[run={run_idx}] iter {iteration:>2}/{N_ITERATIONS} n={len(history_records):>3} "
              f"best_q={best_q:>10,.0f} status={status:<18} cost=${current_cost_estimate():.4f}")

    return rows


# ==============================================================================
# 6. Main
# ==============================================================================

def main() -> None:
    global N_RUNS, N_ITERATIONS

    client = get_openai_client()
    run_canary_check(client)

    if SMOKE_TEST:
        print("=== SMOKE_TEST True: 1 run x 3 iterations only ===")
        N_RUNS = 1
        N_ITERATIONS = 3

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "iteration_metrics_gpt5mini.csv"
    history_path = OUTPUT_DIR / "llm_history_gpt5mini.jsonl"

    done_runs = set()
    prev = pd.DataFrame()
    if metrics_path.exists():
        prev = pd.read_csv(metrics_path)
        for r, g in prev.groupby("run"):
            if g["iteration"].max() >= N_ITERATIONS:
                done_runs.add(r)
        print(f"[resume] {len(done_runs)} run(s) already complete")

    oracle = build_oracle_lookup()
    all_rows: List[Dict[str, Any]] = prev.to_dict("records") if not prev.empty else []

    with open(history_path, "a", encoding="utf-8") as history_writer:
        for run_idx in range(N_RUNS):
            if run_idx in done_runs:
                print(f"[skip] run={run_idx} already done")
                continue
            rows = run_single_run(client, oracle, run_idx, history_writer)
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(metrics_path, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(metrics_path, index=False)

    print("\n=== DONE ===")
    print(f"Total calls: {_COST_STATE['calls']}, cost: ${current_cost_estimate():.4f}")
    final = df[df["iteration"] == N_ITERATIONS]
    if not final.empty and not SMOKE_TEST:
        mean, std = final["best_q"].mean(), final["best_q"].std()
        print(f"New mean_best_q={mean:,.0f}  std={std:,.0f}  (n={len(final)} runs)")
        print(f"Baseline (no seed control) mean_best_q={BASELINE_MEAN:,.0f}  std={BASELINE_STD:,.0f}")


if __name__ == "__main__":
    main()
