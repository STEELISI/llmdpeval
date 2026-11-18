---
id: theory-side-channels
title: Leaks around the mechanism - sort order, exceptions, unverified filters and printed exact values
topic: implementation
task_types: [other]
sources:
  - "Dwork & Roth 2014, Proposition 2.1 (post-processing applies to functions of the private output only)"
  - "Mironov 2012, CCS (floating-point Laplace leaks through its low-order bits)"
  - "Haeberlen, Pierce & Narayan 2011, USENIX Security (timing, state and budget side channels around DP queries)"
---
## Idea
A DP guarantee covers what the mechanism outputs. Post-processing is free only when it uses the private output and public facts. Code around the mechanism can still leak through other channels.

- Order. If candidates are listed in the order of their true counts and ties in the noisy counts keep that order, a tie reveals the true ranking. Integer noise makes exact ties possible.
- Exceptions. An assertion or error inside the private function that fires depending on the raw data is an observable output. A check on a public fact, such as a public label vocabulary, is fine.
- Unverified steps. A filter written in plain dataframe code before the framework call is not checked by the framework. If it is wrong, for example counting the whole column instead of the matching rows, nothing will object.
- Extra prints. The exact value, the exact group sizes, or a data-derived bound printed next to the private answer is a release with no noise.

## Why it matters when writing DP code
Keep the private function's inputs to data plus public constants. Its only outputs should be the noisy result and the budget spent. Validate the data against public facts outside that function.

## Common mistakes
- Category lists in frequency order.
- Raising on rare values inside the private function.
- Printing exact answers without labelling them as non-private.
- Hand-rolled float noise.
