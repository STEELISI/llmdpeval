---
id: theory-floating-point-discrete
title: Floating-point attacks and discrete noise
topic: implementation
task_types: [other]
sources:
  - "Dwork & Roth 2014, Remark 2.1 (floating-point implementation)"
  - "Mironov, CCS 2012, On significance of the least significant bits for differential privacy"
---
## Idea
The proofs for Laplace and Gaussian noise assume exact real numbers. Computers use floating-point numbers, which are a finite, unevenly spaced set. A textbook sampler that turns a uniform random number into Laplace noise can reach only some of those numbers. Which ones it can reach depends on the true value it adds noise to. Mironov showed that the low-order bits of such outputs can reveal the true value, and he proposed a fix, the snapping mechanism. Dwork and Roth note the same risk: because of rounding, an output possible on one dataset can be impossible on a neighbouring one, and then DP fails.

## Why it matters when writing DP code
- For integer results such as counts and histogram cells, use noise defined on the integers. Discrete Laplace, also called two-sided geometric noise, avoids the floating-point problem.
- For real-valued results, use a sampler from an established DP library that documents protection against this attack.
- Do not write your own noise from a uniform draw and a logarithm.
- Rounding a noisy real output does not fix the leak if the noise was sampled naively; the leak happens during sampling.

## Common mistakes
- Hand-rolled Laplace sampling with general numeric routines, then calling the result DP.
- Assuming that a correct scale is enough. The sampler matters too.
- Adding continuous noise to an integer count when a discrete mechanism is available.
