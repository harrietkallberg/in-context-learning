# Mission: Understand and defend our in-context option-pricing research

> Draft written 2026-10-06 from the repo's state; not yet confirmed by Harriet.

## Why
Harriet is reproducing Garg et al. (2022) on a fork of the authors' code and extending it with a Black–Scholes task (a Transformer prices options by calibrating volatility in context). She needs to understand every change, the theory behind both halves, and what the results do and don't show, well enough to write the paper (NeurIPS 2026 template in the repo) and answer a reviewer's questions about it.

## Success looks like
- Explain, without notes, how a prompt is built in the paper and in our BS tasks, and what the model must infer in each.
- Derive C/S = Φ(d₁) − e^(m−rT)·Φ(d₂) from the Black–Scholes formula and say why S drops out.
- Read our result plots and state which curve is the evidence for in-context learning (the gap below the BS prior mean) and why.
- List the repo's changes from upstream commit by commit and justify each.
- Name the open gaps (posterior-mean baseline, calibration precision, seeds, the scale-m=3 failure, the unwritten paper) and say which ones a reviewer would raise first.

## Constraints
- Has the code, trained models and metrics locally; can run Python in the `in-context-learning` conda env.
- Prefers concrete, code-linked explanations over abstract theory.

## Out of scope
- Re-deriving Black–Scholes from stochastic calculus (Itô, PDE): we use the formula, not its proof.
- Transformer internals and mechanistic interpretability, unless a reviewer question needs them.
- Extensions A/B (mixed pricing models, risk-neutral density) until the current results are written up.
