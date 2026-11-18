---
id: numpy-rng-seeding
title: Random number generation and seeding for DP noise in NumPy
framework: numpy
framework_version: "any"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - numpy.random.default_rng
  - numpy.random.Generator
  - numpy.random.Generator.laplace
  - numpy.random.Generator.normal
  - numpy.random.Generator.gumbel
  - numpy.random.Generator.choice
  - numpy.random.SeedSequence
source_url: https://numpy.org/doc/stable/reference/random/generator.html
source_license: "NumPy documentation, BSD-3-Clause"
---

## Goal
Draw DP noise from an unpredictable source while still allowing reproducible tests. DP guarantees assume the attacker does not know the noise; a known seed turns any mechanism into a deterministic function of the data. Applies under any neighbouring relation; the corpus default is replace-one (n public).

## Steps
1. One generator per program run: `rng <- numpy.random.default_rng()`. With no argument NumPy pulls fresh entropy from the operating system (NumPy docs, `default_rng`).
2. Pass `rng` into every function that adds noise, instead of creating generators inside loops or calling the legacy global functions (`numpy.random.laplace`, `numpy.random.seed`).
3. Draw with Generator methods: `rng.laplace(loc, scale, size)`, `rng.normal(loc, scale, size)` (scale = standard deviation), `rng.gumbel(loc, scale, size)`, `rng.choice(K, p=p)`.
4. Tests only: `rng <- numpy.random.default_rng(SEED)`, with SEED passed in from the test harness (e.g. a seed argument whose default is None). A test can then check that outputs vary across seeds and that the empirical noise spread matches the calibrated scale.
5. Several independent streams (parallel workers): spawn them from one `numpy.random.SeedSequence()` rather than using seeds 0, 1, 2.

## Privacy accounting
Seeding does not change ε on paper, but the guarantee holds only if the seed is secret and unpredictable. Treat a release made with a hard-coded seed as not private. Print whether a seed was used.

## Pitfalls
- A fixed seed left in production code: anyone with the code can regenerate the noise and subtract it.
- Seeding from the data (a hash of the dataset, its row count, a timestamp derived from it).
- Re-creating a seeded generator inside a function, so every call returns the same noise; two queries then share noise and their difference is exact.
- NumPy's documentation warns its generators (default PCG64) are for statistical modelling and simulation, not security or cryptographic use. For high-stakes releases use a vetted DP library with a secure sampler; this outline covers research and teaching code.
- NumPy noise is floating-point; see `numpy-geometric-floating-point` for the precision attack.
- Floating-point Laplace noise from NumPy is not hardened against the Mironov 2012 attack (fine for experiments); for integer outputs use `numpy-geometric-floating-point`.
