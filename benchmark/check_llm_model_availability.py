"""
================================================================================
One-shot availability + capability check for the 3 original benchmark models
================================================================================

Purpose: before committing to the ~$17-38 rerun of Benchmark v3.ipynb's LLM
methods with temperature/seed fixed, confirm (for gpt-4o, gpt-5.5, gpt-5-mini):
  1. the model ID is still callable at all (OpenAI delisted all three from the
     public pricing page - "delisted" is not proof they're retired, but it's
     not proof they still work either)
  2. `temperature` and `seed` are both actually accepted (we've now seen two
     different failure shapes for this: gpt-5.6-luna rejected temperature
     entirely because it's in the reasoning-effort family; gpt-4.1-nano's
     `responses.parse` bound to a method with no `seed` kwarg at all - see
     openai_window_ablation.py for the full story)
  3. real token usage for one representative call, so the $17-38 estimate in
     chat can be sanity-checked against real numbers instead of assumptions

This makes exactly ONE call per model (3 calls total, a few cents at most)
and prints a summary table. It does not loop, retry heavily, or write any
output files - it's a throwaway diagnostic, not part of the ablation pipeline.

Key handling
------------
Reads the key ONLY from the OPENAI_API_KEY environment variable - there is no
fallback constant in this file on purpose, so it can never accidentally reuse
the key hardcoded in Benchmark v3.ipynb / v3_update.ipynb
(OPENAI_API_KEY_PLACEHOLDER), which is a different, advisor-owned key.
================================================================================
"""

from __future__ import annotations

import os
import re
import sys

MODELS = ["gpt-4o", "gpt-5.5", "gpt-5-mini"]
TEMPERATURE = 0.2
SEED = 42

TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Reply in one short sentence."},
    {"role": "user", "content": "Say 'ok' and name the exact model you are running as, in one sentence."},
]


def get_client():
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        print(
            "No OPENAI_API_KEY found in the environment. This script will NOT "
            "fall back to any key hardcoded elsewhere (e.g. the "
            "OPENAI_API_KEY_PLACEHOLDER in Benchmark v3.ipynb, which is a "
            "different key). Set it and try again:\n"
            '  $env:OPENAI_API_KEY = "your key"'
        )
        sys.exit(1)
    return OpenAI(api_key=key)


def _extract_offending_kwarg(exc: Exception, droppable=("temperature", "seed")):
    msg = str(exc)
    m = re.search(r"unexpected keyword argument '(\w+)'", msg)
    if m and m.group(1) in droppable:
        return m.group(1)
    for k in droppable:
        if f"'{k}'" in msg:
            return k
    return None


def _extract_rename(exc: Exception):
    """GPT-5-family models reject `max_tokens` and want `max_completion_tokens`
    instead - this is a rename, not a drop, so it gets handled separately."""
    m = re.search(r"'(\w+)' is not supported.*[Uu]se '(\w+)' instead", str(exc))
    return (m.group(1), m.group(2)) if m else None


def probe_model(client, model: str) -> dict:
    result = {
        "model": model,
        "callable": False,
        "temperature_accepted": True,
        "seed_accepted": True,
        "resolved_model": None,
        "reply": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
    }

    kwargs = dict(model=model, messages=TEST_MESSAGES, temperature=TEMPERATURE, seed=SEED, max_tokens=60)

    for _ in range(4):  # at most: rename max_tokens, drop temperature, drop seed, bail
        try:
            response = client.chat.completions.create(**kwargs)
            result["callable"] = True
            result["resolved_model"] = response.model
            result["reply"] = response.choices[0].message.content
            usage = response.usage
            if usage is not None:
                result["prompt_tokens"] = usage.prompt_tokens
                result["completion_tokens"] = usage.completion_tokens
            return result
        except Exception as e:
            rename = _extract_rename(e)
            if rename and rename[0] in kwargs:
                old, new = rename
                kwargs[new] = kwargs.pop(old)
                continue
            offending = _extract_offending_kwarg(e)
            if offending and offending in kwargs:
                result[f"{offending}_accepted"] = False
                kwargs.pop(offending)
                continue
            result["error"] = str(e)[:300]
            return result

    result["error"] = "gave up after renaming/dropping parameters"
    return result


def main() -> None:
    client = get_client()
    results = [probe_model(client, m) for m in MODELS]

    print("\n" + "=" * 100)
    print(f"{'model':<14} {'callable':<9} {'temp_ok':<8} {'seed_ok':<8} {'resolved_model':<28} {'prompt_tok':<11} {'completion_tok'}")
    print("-" * 100)
    for r in results:
        print(f"{r['model']:<14} {str(r['callable']):<9} {str(r['temperature_accepted']):<8} "
              f"{str(r['seed_accepted']):<8} {str(r['resolved_model'] or '-'): <28} "
              f"{str(r['prompt_tokens'] or '-'):<11} {str(r['completion_tokens'] or '-')}")
    print("=" * 100)

    for r in results:
        if not r["callable"]:
            print(f"\n[{r['model']}] FAILED: {r['error']}")
        else:
            print(f"\n[{r['model']}] reply: {r['reply']!r}")


if __name__ == "__main__":
    main()
