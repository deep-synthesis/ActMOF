# pi_vs_ei (SI Section S4.5)

`replay.py` re-executes one run of the benchmark loop; with `paired=True` the seed depends on the run index only, so that
all strategies of a run start from the same three experiments, and with `noise_cv > 0` every queried `q` is multiplied
by log-normal noise (mean 1) before the strategy sees it. `run_noise_benchmark.py` runs Matern-5/2 PI and EI with shared
starts, with and without 30% noise (20 runs, 39 iterations). `noise_benchmark.csv` holds the results: `best_true_q`
is the best noise-free `q` among the conditions chosen so far.
