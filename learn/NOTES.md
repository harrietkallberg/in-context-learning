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

## Session 3 (2026-10-07)
- Lesson 3 (reading our results) plus `assets/posterior.js`, a k = 1 posterior-vs-one-fit widget with an SVG smile fan and 1,000-stock scoring.
- Verified by simulation (4,000 stocks): at k = 1 on bs_smile the posterior mean gets 0.0099 and a random exact fit 0.0196 (ratio 1.98; theory 2). The trained models get Transformer 0.0107 and smile calibration 0.0211. So the Transformer ≈ Bayes-optimal at k = 1. This is a strong, paper-worthy point.
- Harriet asked "what is ndtr?" between lessons: torch's standard normal CDF wasn't obvious. Name library functions explicitly in lessons.

## Session 4 (2026-10-07)
- "exercise 4" read as Lesson 4: a hands-on build of BSPosteriorMeanModel in src/models.py (she writes it; I didn't touch src/). Checker: `learn/exercises/check_posterior.py`, validated (reference 10/10, leaky version 5/10).
- Prototype findings (640 prompts): on clean smile at k=1–3 the posterior with τ=0.01 matches the Transformer (0.0109/0.0012 vs 0.0107/0.0012). bs_flat k=1: the Transformer is ~30× above calibration (a precision limit). Noisy: the Transformer sits above the τ=0.05 posterior at small k (trained clean). ESS collapses on clean smile beyond k≈5.
- Earlier she didn't recognise "§" ("swirly 3"). Avoid § in lessons, or write "section 3".
- Still no quiz results reported, so no learning records yet.
