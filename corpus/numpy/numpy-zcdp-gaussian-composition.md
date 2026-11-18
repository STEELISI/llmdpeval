---
id: numpy-zcdp-gaussian-composition
title: Composing several Gaussian releases with zCDP and converting to (epsilon, delta) in NumPy
framework: numpy
framework_version: "any"
mechanism: gaussian
task_type: composition
dp_notion: zcdp
needs_bounds: true
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator.normal
  - numpy.clip
  - numpy.sqrt
  - numpy.log
source_url: https://arxiv.org/abs/1605.02065
source_license: original code (algorithm from cited paper)
---

## Goal
Answer several queries with Gaussian noise under one overall (ε, δ) target, with tighter accounting than adding (ε_i, δ_i). Zero-concentrated DP, Bun & Steinke, TCC 2016: the Gaussian mechanism with L2 sensitivity Δ and noise σ is ρ = Δ²/(2σ²)-zCDP (Prop. 1.6); ρ's add up (Lemma 2.3); ρ-zCDP implies (ρ + 2√(ρ·ln(1/δ)), δ)-DP for every δ > 0 (Prop. 1.3). Neighbouring relation: replace-one (n public).

## Steps
1. Fix the target (ε, δ) and convert to a total ρ by solving ε = ρ + 2√(ρ·L) with L = ln(1/δ): √ρ = √(L + ε) − √L.
2. Split ρ over the k queries (ρ_i, summing to ρ). Equal split is fine; give more to the query that needs accuracy.
3. For each query: its L2 sensitivity Δ_i under replace-one (a clamped sum: U − L; a count of rows meeting a condition: 1; add/remove alternative for the sum: max(|L|, |U|)), then σ_i = Δ_i / √(2ρ_i).
4. `answer_i <- exact_i + rng.normal(loc=0, scale=σ_i)` (σ is a standard deviation).
5. After all releases: ρ_spent = Σ Δ_i²/(2σ_i²); ε_spent = ρ_spent + 2√(ρ_spent·L).

## Privacy accounting
Replace-one Δ_i throughout. Account in ρ, convert once at the end. For k equal queries the noise grows like √k, compared with k for basic (ε, δ) composition of Gaussian mechanisms. Assert ε_spent ≤ ε in code and print ρ, δ and each σ.

## Pitfalls
- Converting each query to (ε_i, δ_i) and then adding: much looser.
- Using the classical σ = Δ√(2 ln(1.25/δ))/ε formula per query and then zCDP-composing as if ρ were ε: different units.
- Mixing pure-ε Laplace queries in: a pure ε-DP mechanism is (ε²/2)-zCDP, so it can be added to the ρ ledger.
- δ ≥ 1/N gives a meaningless guarantee.
