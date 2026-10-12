# benchmark

The 39 search strategies of the ActMOF benchmark and the results of the runs reported in the paper.

```
benchmark/
├── run_all_methods.py     single entry point: runs the 7 scripts below, then merges their outputs
├── merge_outputs.py       combines the outputs into all_methods_PERRUN_curves_UNIFIED.csv
├── reference_output/      results and logs of the runs reported in the paper
└── families/
    ├── core29_gp_ml_tree_rl_mof.py   29 strategies: GPs, tree and other ML surrogates, approximate and
    │                                 linear models, bandits, random search
    ├── rvm_like.py                    1 strategy:   ARD regression on random features
    ├── lagp.py                        2 strategies: local approximate GPs              (needs R)
    ├── rgasp_bnn.py                   2 strategies: RobustGaSP, MLP ensemble           (needs R)
    ├── llm_pure_gpt4o_gpt55.py        2 strategies: gpt-4o, gpt-5.5                    (needs an OpenAI key)
    ├── llm_pure_gpt5mini.py           1 strategy:   gpt-5-mini                         (needs an OpenAI key)
    └── hybrid_gp_llm.py               2 strategies: GP (EI or PI) alternating with gpt-5.5 (needs an OpenAI key)
```

`../docs/method_registry.csv` lists every strategy with its name in the paper and the script that runs it.

## Protocol

Every run starts from 3 conditions drawn at random and then proposes 3 conditions per iteration for 39 iterations:
120 experiments. The landscape in `../data/` returns the crystallinity index `q` of every proposed condition. In each
iteration, 15,000 untried conditions are drawn at random and scored (4,000 for laGP; none for the LLM strategies,
which write their conditions themselves). Each strategy is run 5 times; the seed of run `r` of method `m` is
`42 + 10000 r + stable_int(m)`.

## Running

```bash
pip install -r ../requirements.txt

python run_all_methods.py                                           # all 39 strategies
python run_all_methods.py --only core29_gp_ml_tree_rl_mof,rvm_like  # the 30 that need neither R nor an API key
python run_all_methods.py --skip lagp,rgasp_bnn                     # without the R-backed strategies
python run_all_methods.py --list                                    # show the families
python run_all_methods.py --dry-run                                 # show the commands without running them
python run_all_methods.py --merge-only                              # only merge existing outputs
```

Each script writes its results to `../outputs/<family>/` (best-so-far of every run and iteration, a log of the
selected conditions) and can be run on its own. Most scripts have a `SMOKE_TEST` switch near the top for a short trial
run. For the 20 runs of the six leading strategies, set `ACTMOF_N_RUNS=20` before running
`families/core29_gp_ml_tree_rl_mof.py`.

## R-backed strategies

`lagp.py` and `rgasp_bnn.py` call R through `rpy2` and need R with the packages `laGP` and `RobustGaSP`. The R
start-up in these two scripts is written for Windows; set the environment variable `ACTMOF_R_HOME` to the folder of
the R installation.

## LLM strategies

The three LLM scripts need an OpenAI API key in the environment variable `OPENAI_API_KEY`; the calls are billed
(a few US dollars for all five strategies).

- Models: `gpt-4o`, `gpt-5.5` and `gpt-5-mini`; the snapshots at the time of the runs (August 2026) were
  `gpt-4o-2024-08-06`, `gpt-5.5-2026-04-23` and `gpt-5-mini-2025-08-07`.
- Settings: temperature 0.2 for `gpt-4o`; the GPT-5 models accept no temperature and are run with the lowest reasoning
  effort (`none` for `gpt-5.5`, `minimal` for `gpt-5-mini`). A fixed seed (seed of the run + iteration) is sent with
  every call.
- Before the runs, each script sends one test call and stops if the API has dropped the seed, the temperature or the
  reasoning effort.
- Every answer is checked (range, no repeat within the run, no duplicate in the batch). An invalid answer is returned
  to the model with the error, up to three attempts; after that, three random untried conditions are used.
- Every iteration is logged. The logs of the reported runs are in `reference_output/logs/`.

The answers of a language model are not exactly reproducible, even with a fixed seed. A rerun gives similar but not
necessarily identical proposals; the numbers of the paper come from the logged runs.
