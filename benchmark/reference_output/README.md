# reference_output

Results of the runs reported in the paper, so that figures and tables can be checked without rerunning the benchmark.
All files are cut at iteration 39 (120 experiments = 3 starting experiments + 39 iterations x 3).

| File | Content |
|---|---|
| `all_methods_PERRUN_curves_UNIFIED.csv` | best-so-far `best_q` of all 39 methods, 5 runs each, iterations 0-39 (7,800 rows; columns `method`, `family`, `run`, `iteration`, `best_q`) |
| `leading_strategies_n20_PERRUN_curves.csv` | the same for the six leading strategies and random search with 20 runs each (5,600 rows) |
| `logs/core29_selected_conditions.jsonl` | every condition selected by the 29 methods of `families/core29_gp_ml_tree_rl_mof.py`, with its `q_final` |
| `logs/lagp_history.jsonl` | batches selected by the two laGP methods |
| `logs/llm_pure_gpt4o_gpt55_history.jsonl`, `logs/llm_pure_gpt5mini_history.jsonl` | one line per LLM iteration: accepted conditions with the model's rationales, strategy summary, every attempt with its error, whether the random fallback was used, best-so-far, cost |
| `logs/hybrid_gp_llm_history.jsonl` | one line per iteration of the two GP-LLM hybrids: who chose the batch (GP or LLM) and the best-so-far |

The user message of every LLM request is a deterministic function of the history of the run, so the requests can be
rebuilt from these logs, and the best-so-far curves can be recomputed from the logged conditions without calling the API.
