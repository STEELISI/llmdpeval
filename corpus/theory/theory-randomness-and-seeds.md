---
id: theory-randomness-and-seeds
title: Randomness, seeds and reusing noise
topic: implementation
task_types: [other]
sources:
  - "Dwork & Roth 2014, Definition 2.4 (probability over the mechanism's coin flips)"
  - "Dwork & Roth 2014, Theorem 3.16 (composition)"
---
## Idea
The DP guarantee holds over the mechanism's own random coin flips. That only protects anyone if the attacker cannot know or control those coins. If the noise is predictable, the release is effectively the exact value.

A fixed seed written in the code makes the noise a known constant. Anyone who reads the code can re-create it and subtract it. The same holds if the same noise draw is reused for two releases: subtracting the two outputs cancels the noise and exposes the exact difference.

## Why it matters when writing DP code
- Do not set a fixed seed for the noise in a DP release. Let the library draw fresh randomness, ideally from the operating system's secure source, for every release.
- Draw independent noise for every query and every coordinate.
- Run the mechanism once per release. Each extra run spends more budget under sequential composition.
- A fixed seed is acceptable only in a test that checks code paths. Such output is not a private release and must not be reported as one.

## Common mistakes
- Fixing the seed "for reproducibility" in code that produces the reported DP result.
- Adding one noise draw to several related statistics.
- Running the mechanism many times and averaging the results, which spends the budget once per run.
- Re-running until the output looks close to the truth, which is a non-private choice.
