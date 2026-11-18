---
id: diffprivlib-transforms-int-to-string
title: Wrapping a mechanism with diffprivlib.mechanisms.transforms (IntToString, StringToInt)
framework: diffprivlib
framework_version: "0.6.6"
mechanism: other
task_type: other
dp_notion: pure
needs_bounds: false
api_calls:
  - diffprivlib.mechanisms.transforms.IntToString
  - diffprivlib.mechanisms.transforms.StringToInt
  - diffprivlib.mechanisms.transforms.DPTransformer
  - diffprivlib.mechanisms.Binary
  - diffprivlib.mechanisms.Geometric
source_url: https://github.com/IBM/differential-privacy-library/blob/0.6.6/tests/mechanisms/transforms/test_IntToString.py
source_license: MIT
---

## Goal
Use a mechanism whose input type does not match your column: string-only mechanisms (`Binary`, `ExponentialCategorical`) on integer codes, or integer mechanisms (`Geometric`) on numbers stored as strings. The transform converts on the way in and back on the way out; privacy is that of the wrapped mechanism. Local DP per record, which implies replace-one (bounded DP, n public) central DP at the same epsilon.

Signature: `IntToString(parent)`, `StringToInt(parent)`; `parent` is a mechanism instance.

## Steps
1. Build the inner mechanism with labels in ITS type:
   - `inner <- mechanisms.Binary(epsilon=?, value0="0", value1="1")` (string labels).
2. Wrap it: `mech <- transforms.IntToString(inner)`.
3. `mech.randomise(int_value)` converts the int to a string, randomises with the inner mechanism, and converts the result back to int.
4. All transforms subclass `transforms.DPTransformer` (a mechanism wrapper). Reverse direction: `transforms.StringToInt(mechanisms.Geometric(epsilon=?, sensitivity=1)).randomise("12")` returns a string.
5. Apply per record (local DP) or to one aggregate (central DP), as the inner mechanism requires.
6. Output the noisy values and the epsilon of the inner mechanism.

## Privacy accounting
Replace-one (bounded DP, n public): each record is randomised on its own, so replacing one record changes one randomised value; the guarantee is epsilon under replace-one and under add/remove. Transforms are deterministic re-encodings, so they add no privacy cost and no protection: the cost is the inner mechanism's epsilon per randomise call on the same person or statistic. Record it on a `BudgetAccountant` yourself; mechanisms and transforms never touch an accountant.

## Pitfalls
- The import path is `diffprivlib.mechanisms.transforms`, not `diffprivlib.transforms`.
- The inner mechanism's labels must be the string form the transform produces (`"0"`, `"1"`), or `randomise` raises a ValueError.
- The transform takes the mechanism object as its only positional argument; it does not take `epsilon` itself.
- `RoundedInteger` is the transform for rounding a real-valued mechanism's output to int (separate outline).
