# M2 — EI of a hurdle model equals p · EI₊

Setting: y = log(1+q) ≥ 0, incumbent y⁺ ≥ 0, ξ > 0. Predictive distribution of a hurdle model at x:

- with probability 1 − p(x): y = 0 (no product);
- with probability p(x): y ~ N(μ₊(x), σ₊(x)²).

Improvement I = max(y − y⁺ − ξ, 0). Then

EI_hurdle(x) = E[I] = (1 − p)·E[I | y = 0] + p·E[I | y ~ N(μ₊, σ₊²)].

**Zero branch.** y = 0 ≤ y⁺ < y⁺ + ξ, so I = max(0 − y⁺ − ξ, 0) = 0. It contributes nothing.

**Positive branch.** For a normal variable, with z = (μ₊ − y⁺ − ξ)/σ₊,
E[max(y − y⁺ − ξ, 0)] = (μ₊ − y⁺ − ξ)Φ(z) + σ₊φ(z) = EI₊(x).

Hence **EI_hurdle(x) = p(x) · EI₊(x)**, exactly. This is the acquisition of M1.

Caveat: the identity needs y⁺ ≥ 0 (true by construction, y⁺ = 0 when no product yet, and log(1+q) ≥ 0), and ξ > 0. It does not need the two branches to be independent of anything else.

Verified numerically in `tests/test_hurdle_ei.py` (1,000 random cases, 200,000 Monte-Carlo samples each).

**Exploratory ablation.** a(x) = p(x)^γ · EI₊(x), γ ∈ {0.5, 2}. For γ ≠ 1 this is no longer the exact hurdle EI. Exploratory only; not used to pick a γ.
