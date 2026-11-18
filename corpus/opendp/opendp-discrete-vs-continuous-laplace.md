---
id: opendp-discrete-vs-continuous-laplace
title: Discrete vs continuous Laplace, scalar vs vector, in make_laplace
framework: opendp
framework_version: "0.16.0"
mechanism: laplace
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - opendp.measurements.make_laplace
  - opendp.measurements.then_laplace
  - opendp.domains.atom_domain
  - opendp.domains.vector_domain
  - opendp.metrics.absolute_distance
  - opendp.metrics.l1_distance
  - opendp.measurements.make_geometric
source_url: https://github.com/opendp/opendp/blob/v0.16.0/docs/source/api/user-guide/measurements/additive-noise-mechanisms.ipynb
source_license: MIT
---

## Goal
Choose the right Laplace variant. In OpenDP one constructor, `make_laplace`, covers all of them; the input space decides which. Neighbouring relation: replace-one (n public): the row-level domain is sized and maps are read at `d_in=2`.

## Steps
1. `enable_features("contrib")`.
2. Pick the input space by the type and shape of the aggregate you are protecting:
   - integer scalar (a count): `atom_domain(T=int), absolute_distance(T=int)` -> discrete Laplace (geometric) noise, integer output.
   - float scalar (a sum or mean): `atom_domain(T=float, nan=False), absolute_distance(T=float)` -> continuous Laplace, float output.
   - vector of counts/sums: `vector_domain(atom_domain(T=?)), l1_distance(T=?)` -> independent noise per entry, sensitivity measured in L1.
3. Usually you do not build the space yourself (row-level space under replace-one: `vector_domain(atom_domain(T=float, nan=False), size=N), symmetric_distance()`, searched at `d_in=2`): chain `>> then_laplace(scale)` after a transformation and the space comes from its output (count -> int, float sum -> float, `count_by_categories` -> int vector with L1).
4. Stand-alone use, when you already have an aggregate with known sensitivity: `meas <- make_laplace(*SPACE, scale=?)`; `eps <- meas.map(d_in=SENS)`.

Signature for reference: `make_laplace(input_domain, input_metric, scale, k=None, MO='MaxDivergence')`.

## Privacy accounting
The map takes the sensitivity in the space's metric (absolute or L1) and returns epsilon. Check `meas.map(SENS) <= EPS`; search the scale with `binary_search_param`. For a full chain from rows, replace-one is `chain.map(2)` (a sized clamped sum is charged U - L). Alternative, add/remove-one (n private): unsized domain and `chain.map(1)` (a clamped sum is then charged max(|L|, |U|, U - L), a count 1).

## Pitfalls
- Why discrete exists: naive floating-point Laplace sampling leaks through the low bits (Mironov 2012). OpenDP's "continuous" Laplace adds discrete noise to an exact rational form of the float, so it is safe; `np.random.laplace` is not, and it is not a framework call.
- `k` sets the noise granularity 2^k. Coarser k is faster but adds a penalty to the map; leave it at None unless speed matters.
- `make_geometric` is the integer-only variant (optional `bounds` for constant-time sampling); `make_laplace` on an int space gives the same discrete Laplace noise and is the usual choice.
- Vector Laplace needs L1 sensitivity; vector Gaussian needs L2. Do not swap the metrics.
- Integer data can be cast to float to get float noise, but the count's sensitivity type must still match the space.
