# warm_start (SI Section S4.6)

`run_warmstart.py` runs the six leading strategies and random search from two warm starts, 20 runs of 39 iterations each:
`faithful` (the three conditions proposed by the MOF-321 prior, identical in all runs) and `same_quantile` (three conditions
drawn from the band between the top-1% and top-0.1% thresholds, one draw per run; `prepare_percentile_band.py` builds the band).
`results/metrics_*.csv` hold the best-so-far of every run, `results/start_points_*.csv` the starting conditions.
