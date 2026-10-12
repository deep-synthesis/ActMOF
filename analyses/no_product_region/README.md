# no_product_region (SI Section S4.7)

`feasibility.py` implements the policies (`B0`, Matern-5/2 EI; `M1`, probability of a product x EI of the products;
`M3`, experiments without product entered with the lowest value observed so far). `hurdle_ei.md` shows that the score of
`M1` is the expected improvement of a model with a separate no-product branch; `test_hurdle_ei.py` checks it numerically.

```bash
python run_feasibility_paired.py --methods B0 M1 M3 --runs 0-19 --tag main --workers 8
python analyze_feasibility.py --tag main
```

`results/` holds the runs reported in the paper (20 runs, 39 iterations, shared starts).
