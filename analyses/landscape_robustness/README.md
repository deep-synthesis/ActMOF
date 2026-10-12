# landscape_robustness (SI Section S4.4)

The landscape is rebuilt 41 times, each time with one construction choice changed, and the six leading strategies and
random search are run on every landscape (20 runs, 39 iterations).

```bash
python build_maps.py          # writes maps/<variant>.npy (42 arrays of 6.1M values; not shipped) and variants.csv
python landscape_metrics.py   # landscape_metrics.csv
python bench_runner.py ...    # benchmark on every map (see the header of the script)
python analyze_bench.py       # strategy_by_variant.csv, ranking_summary.csv
python temporal_analysis.py   # prediction of later anchors from the first N (Table S9)
```

`bench_PERRUN_curves.csv` holds the best-so-far of every run (235,200 rows: 42 landscapes x 7 strategies x 20 runs x
40 iterations); `V0` is the original landscape. `pseudolab_common.py` holds the 96 recorded syntheses in recording order.
