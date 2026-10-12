# analyses

Code and results of the analyses of the Supporting Information that go beyond the 39-strategy benchmark.
All runs use the benchmark loop of `../benchmark/families/core29_gp_ml_tree_rl_mof.py` and a budget of 120 experiments
(3 starting experiments + 39 iterations x 3).

| Folder | Section of the SI | Content |
|---|---|---|
| `landscape_reliability/` | S2.5 | leave-one-out prediction of the 95 anchors; distance to the nearest anchor and vote of the five forests for every condition of the landscape |
| `landscape_robustness/` | S4.4 | 41 landscapes rebuilt with other construction choices; six leading strategies and random search on each (n = 20) |
| `pi_vs_ei/` | S4.5 | Matern-5/2 PI and EI from shared starts, with and without 30% measurement noise (n = 20) |
| `warm_start/` | S4.6 | six leading strategies and random search started from the prior conditions or from the top-1%-0.1% band (n = 20) |
| `no_product_region/` | S4.7 | two policies that model "product / no product" separately, against Matern-5/2 EI (n = 20) |
