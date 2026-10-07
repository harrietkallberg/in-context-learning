# Teaching notes

- Workspace lives in `learn/` inside the research repo (untracked, not committed) so it sits next to the code it teaches.
- Harriet asks many questions at once, mixing "what did we do" with "why". Answer with the project map first, then one focused lesson.
- Earlier in session 1 she asked what n_dims, x₀/x₁ and the b/n axes are, so tensor shapes and the input encoding were not obvious. Use concrete numbers and shapes.
- Ground everything in real numbers from `models/*/metrics.json`; she has the results locally.
- Writes in Swedish and English (has a `harriet-skrivstil` skill); lessons in English to match the paper.

## Facts established while preparing (2026-10-06)
- BS smile calibration doesn't converge to ~0 error with noise-free data: about 1e-4 at k=10 and 7e-6 at k=40. Flat calibration plateaus at about 1.2e-6. Both are limits of the grid-search optimizer, not of the information in the examples.
- The Transformer beats smile calibration for k ≤ 3 (0.0107 vs 0.0211 at k=1), as expected, because least squares is underdetermined below 3 examples.
- scale-m=3: the Transformer is worse than the no-context prior mean (flat 0.082 vs 0.073 at k=40). Moneyness extrapolation fails.

## Session 2 (2026-10-07)
- Built Lesson 2 (Black–Scholes in four lines) and the reusable `assets/bs.js` (mirrors `src/tasks.py`; checked against the Python to 6 digits) and `assets/calibrate.js` (k = 1 calibration game). Added `reference/black-scholes.html`.
- Mission draft still unconfirmed: Harriet hasn't said whether this is a thesis/course project, or given a deadline.
- No learning records yet: no evidence of learning has come back (quiz scores, explanations). Ask how lessons 1–2 went before writing any.
