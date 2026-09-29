# Experimental setup: reproduction of Garg et al. (2022)

Reproduction of the four main models from *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes* (Garg, Tsipras, Liang, Valiant), using the authors' code (fork of `dtsip/in-context-learning`, branch `harriet`).

Paper references are to **arXiv:2208.01066v3 (11 Aug 2023)**, read in full (31 pages). Sections: §3–5 main results, §6 capacity/curriculum/data, App. A setup, App. B additional results.

## What we run

One Transformer per function class, trained from scratch with the repository's own configs and `src/train.py`. Runs started 2026-09-30 around 00:00 CEST, in parallel.

| Function class | Config | wandb run | Run ID |
|---|---|---|---|
| Linear regression | `conf/linear_regression.yaml` | `u7ewhglt` | `52e59405-04a5-4e41-8caa-6bc1e02d8304` |
| Sparse linear regression | `conf/sparse_linear_regression.yaml` | `6fhhfz2p` | `56bb1fd5-0882-411f-92ac-7e2109f3f4d0` |
| Decision tree | `conf/decision_tree.yaml` | `3tqj7h53` | `53417d4c-9a6c-4c1a-ae09-3df8c0af2676` |
| 2-layer ReLU network | `conf/relu_2nn_regression.yaml` | `9noyd1yu` | `0c3f3424-0387-4099-bff0-004968547236` |

- **wandb project:** `harriet-kallberg02-kth-royal-institute-of-technology/in-context-training`
- **Outputs:** Modal volume `harriet-icl-models` (environment `in-context-learning`, workspace `kthaisociety-dev`), at `<task>/<run id>/`. Each output folder holds `config.yaml`, `state.pt` (saved every 1000 steps), `model_100000.pt` and so on (every 100k steps), and `metrics.json` (written when training ends).

These correspond to the paper's Fig. 2 (linear), Fig. 4/8/9 (linear, out-of-distribution) and Fig. 5a–d (sparse, tree, ReLU net).

## Shared with the paper (unchanged)

| | Setting | Paper | Code |
|---|---|---|---|
| Model | GPT-2 "Standard": 12 layers, 8 heads, 256-dim embedding (9.5M parameters), dropout 0 | §2, App. A.1 | `conf/models/standard.yaml`, `src/models.py` |
| Input/output mapping | x padded; f(x) padded with zeros to dimension d; linear read-in; read-out by dot product; predictions read at the x positions | App. A.1 | `src/models.py` |
| Optimisation | Adam, lr 1e-4, batch 64, 500,000 steps, squared error averaged over all prompt prefixes | §2, App. A.2 | `conf/base.yaml`, `src/train.py` |
| Data | Fresh prompts every step (≈32M functions over training); x ~ N(0, I_d), d = 20 | §3, §6, App. A.2 | `src/samplers.py` |
| Curriculum (linear, sparse) | d_cur 5→20 (+1), k_cur 11→41 (+2), every 2000 steps | §6, App. A.2 | task configs |
| Curriculum (tree, ReLU net) | d_cur 5→20 (+1), k_cur 26→101 (+5), every 2000 steps | App. A.2 | task configs |
| Linear functions | w ~ N(0, I_d) | §3, App. A.2 | `src/tasks.py` |
| Sparse linear | k = 3 nonzero coordinates chosen from the first d_cur | §5, App. A.2 | `sparsity: 3` |
| Decision trees | Depth 4, 16 leaves; node coordinates uniform on {1..d}; branch on sign; leaf values ~ N(0, 1) | §5, App. A.2 | `depth: 4` |
| ReLU networks | r = 100; α_i ~ N(0, 2/r), w_i ~ N(0, I_d) | §5, App. A.2 | `hidden_layer_size: 100` |
| Evaluation | Error normalised by the zero estimator (d, or k for sparse, 1 for trees); 1280 prompts; 90% bootstrap CIs over 1000 resamples | Fig. 2, Fig. 5 | `src/eval.py`, `src/plot_utils.py` |
| Out-of-distribution prompts | Skewed covariance, d/2 subspace, input/weight scaling {1/3, 1/2, 2, 3}, noisy outputs, different orthants, orthogonal query, query matches an example | §4, App. B.2 | `src/eval.py` (`build_evals`) |
| Numerical precision | fp32; both GPUs are Ampere, so PyTorch 1.11's default TF32 matmuls apply in both setups | | |

## Differences from the paper's setup

### 1. Number of runs (seeds)
**What the paper states:**
- Multiple seeds are stated explicitly only for studies on **linear regression with other settings**:
  - Fig. 6 and Fig. 10 (§6, App. B.3): model capacity × dimension d ∈ {10, 20, 30, 40, 50}. "We train 3 models in each case with different random seeds", with the median shown and min–max shaded.
  - Fig. 11 (App. B.4): training variance, with 3 seeds for each d.
  - Fig. 12 (App. B.5): curriculum vs. no curriculum, with 3 seeds each.
- For the main results (Fig. 2, 4, 5, 8, 9) the paper **does not state** how many trained models were used. The captions describe only the evaluation (1280 prompts, bootstrap over prompts).
- The authors released **one checkpoint per function class** (`models.zip`, the `pretrained` runs).

**Us:** one run per function class. This corresponds to the released checkpoints. Our error bars, like the paper's Fig. 2 and Fig. 5, reflect variation over evaluation prompts only, not over training seeds.

**Seeding:** neither setup fixes a random seed. `train.py` sets no global seed, and `cudnn.benchmark = True` makes GPU kernels non-deterministic. A rerun reproduces behaviour, not exact curves.

### 2. Hardware and training time
| | Paper (App. A.2, Acknowledgements) | Us |
|---|---|---|
| GPU | 1× NVIDIA GeForce RTX 3090 per run, on the Stanford NLP cluster | 1× NVIDIA A100 (40 GB; Modal may give 80 GB) per run, 4 runs in parallel |
| Platform | University cluster | Modal serverless containers (Debian slim) |
| Linear regression, d = 20 | ~7 h | ~5 h (26–29 steps/s) |
| Decision tree | ~17 h | ~6.5 h estimated in the first minute; will increase as prompts lengthen |
| Pre-emption | n/a | Modal may pre-empt a GPU container. It restarts automatically (up to 10 retries) and resumes from the last `state.pt` committed to the volume. Commits happen every 5 min and checkpoints every 1000 steps, so up to ~5 min + 1000 steps (≈ 9000 steps at 26 steps/s) may be repeated, and wandb then receives repeated step numbers. The four reproduction runs use the original non-atomic `torch.save`, so a commit taken mid-save could leave a truncated `state.pt`. Later runs write to a temp file and rename it. |

### 3. Software environment
| Package | `environment.yml` (authors) | Modal image (us) |
|---|---|---|
| Python | 3.8.12 | **3.10** (Modal's image builder no longer supports 3.8/3.9) |
| PyTorch | 1.11.0 | 1.11.0 + CUDA 11.3 |
| transformers / tokenizers | 4.17.0 / 0.12.1 | same |
| numpy, pandas, scikit-learn, xgboost, matplotlib, seaborn, tqdm, quinine, protobuf | pinned | same pins |
| wandb | 0.12.11 | **0.24.2** (0.12.11 rejects current wandb API keys) |
| PyYAML (pulled in by quinine) | 5.4 | 5.4, built with Cython < 3 (no py3.10 wheel) |

### 4. Model config: `n_positions`
The repo's `conf/base.yaml` sets `n_positions: 101` for all tasks. The released pretrained configs use:

| Task | Released model | Our run |
|---|---|---|
| Linear regression | 101 | 101 |
| Sparse linear regression | 200 | 101 |
| Decision tree | 201 | 101 |
| ReLU network | 201 | 101 |

This only sets the size of the positional-embedding table (the maximum prompt length). Training uses at most 41 or 101 examples, so training is not affected. Our sparse, tree and ReLU models, however, **cannot be evaluated on prompts longer than 101 examples**, whereas the released models can take up to 200/201.

### 5. Baselines: public code ≠ paper
The end-of-training `metrics.json` is computed with the baselines in `src/models.py` (`get_relevant_baselines`). Several differ from App. A.3. This is a difference between the authors' public code and their paper, not one we introduced.

| Baseline | Paper (App. A.3) | Public code (what our runs compute) |
|---|---|---|
| Least squares, 3-NN, averaging | as described | same |
| Lasso (sparse) | α ∈ {1, 1e-1, …, 1e-4}; α = 1e-2 reported | same grid; all five computed |
| 2-layer NN with GD (ReLU task) | Adam, batch 10, **5000 steps**, lr 5e-3 (5e-2 for the linear-function evaluation), r = 100 | Adam, **batch 100, 100 steps**, lr 5e-3 for both evaluations |
| Greedy tree learning (tree task) | scikit-learn tree, **max_depth 2**; also a sign-preprocessed variant with depth 4 | max_depth **4** and **unbounded**; no sign-preprocessed variant. Both map to the same plot label "Greedy Tree Learning", so one overwrites the other in plots. |
| XGBoost (tree task) | **50 estimators, max depth 4, lr 0.1**; also a sign-preprocessed variant | `XGBRegressor()` with library defaults; no sign-preprocessed variant |
| 2-layer NN with GD (tree task) | not in Fig. 5b | **not computed** by current code. The released `pretrained/metrics.json` does contain it (100 and 200 steps), so the authors computed it with a different version of the code. |

Consequences:
- Transformer errors are comparable to the paper. Baseline curves for the tree and ReLU tasks are **not** directly comparable to Fig. 5b–d.
- `src/plot_utils.py` lists "2-layer NN, GD" for `decision_tree`, which our tree run's `metrics.json` will not contain. Plotting our tree run in `eval.ipynb` with the default model list will therefore raise a `KeyError`, unless that name is dropped from the list.

### 6. Code changes relative to upstream
None of these change the training computation on a GPU.
- `src/train.py`, `src/eval.py`, `src/models.py`: `.cuda()` replaced by `.to(device)`, with `device = "cuda" if available else "cpu"`, and `torch.load(..., map_location=...)`. On a GPU this behaves identically. The purpose is to allow local CPU runs.
- `src/eval.py` (`get_run_metrics`): when models are not loaded (`skip_model_load=True`), cached metrics are no longer recomputed. Upstream, that path could overwrite `metrics.json` with empty results. This does not affect the end-of-training metrics, which load the model.
- `src/conf/wandb.yaml`: our wandb entity. Logging destination only.
- `modal_train.py` (new): a launcher only. It builds the image, uploads `src/`, runs `python train.py --config <config> --training.resume_id <id>`, and commits the output volume every 5 min.

### 7. Other notes
- **ReLU width:** upstream commit `3383371` changed the ReLU-network hidden width from 4 to 100 in the config and in `tasks.py`. r = 100 matches the paper (§5, App. A.2) and the released pretrained config.
- **Noise level in the noisy-regression evaluation:** stated as ε ~ N(0, 1) in §4 and as N(0, d/20) in App. B.2. These are identical for d = 20.

## Extension: in-context option pricing (Black–Scholes)

A new function class in the same framework. **Each prompt is one underlying with a hidden volatility.** The in-context examples are other options on that underlying (moneyness, maturity) → price, and the query is a new option. To price it, the model must implicitly calibrate the volatility from the examples.

| | `bs_flat` | `bs_smile` |
|---|---|---|
| Hidden per prompt | σ ~ U[0.1, 0.5] | σ(m) = σ₀ + b·m + c·m², clipped at 0.05; σ₀ ~ U[0.15, 0.4], b ~ U[−0.4, 0], c ~ U[0, 1] |
| Inputs x ∈ ℝ² (Gaussian, as in the paper) | mapped to log-moneyness m = ln(K/S) = 0.2·x₀ and maturity T = 0.1 + 1.9·Φ(x₁) ∈ [0.1, 2] years | same |
| Target | Black–Scholes call price C/S with r = 2%, standardised by the prior's mean and std (fixed-seed Monte Carlo), so the trivial estimator has error ≈ 1 | same |
| What it tests | Implied-volatility inversion in context. One example determines σ in principle. | Learning the smile shape. Needs ≥ 3 examples. |

- **Model and training:** identical to the paper's Standard model, lr, batch size and steps; configs `conf/bs_flat.yaml` and `conf/bs_smile.yaml` inherit `base.yaml`. Points curriculum 11→41 (+2 every 2000 steps), as for linear regression. No dimension curriculum, since n_dims = 2.
- **Baselines:** least squares on x and 3-NN (from the paper), plus **BS calibration**. BS calibration fits the vol parameters to the examples by least squares (a global grid, then 5 rounds of local refinement) and prices the query. On `bs_smile` it is computed both with the correct smile model (the optimal estimator, error ~1e-5 at 40 examples) and with a flat-vol model (misspecified, error ~1e-2). This shows whether the Transformer uses the smile. With no examples (k = 0), calibration falls back to the prior-mean price below.
- **No-context baseline (BS prior mean):** prices each query at E[C/S | m, T], averaged over 1024 fixed-seed draws of the vol parameters from the prior, and ignores the examples. The price surface in (m, T) is the same in every prompt, so a Transformer trained from scratch can learn it in its weights; only the volatility must come from the context. This baseline has error ≈ 0.14 (`bs_flat`) and ≈ 0.07 (`bs_smile`) at every k, versus 1 for the trivial estimator. Only Transformer error below it is evidence of in-context calibration.
- **Implementation:** `src/tasks.py` (`BlackScholesFlat`, `BlackScholesSmile`), `src/models.py` (`BSCalibrationModel`, `BSPriorMeanModel`), `src/schema.py`, `src/eval.py`, `src/plot_utils.py`. In wandb, `excess_loss` for these tasks equals the standardised MSE.
- **Training is noise-free**, as in the paper.
- **Evaluation:**
  - `standard`: clean prices.
  - `noisy`: standardised prices + ε, ε ~ N(0, 0.05²), on examples and query. That is 0.005·S, about 5% of an at-the-money 1-year option's price, a bid–ask-like level. The error floor on noisy targets is 0.05² = 0.0025.
  - This mirrors the paper's noisy-linear-regression test (§4, Fig. 4b), where the model is trained clean and tested on noisy outputs. The paper runs that test only for linear regression; we run it for the BS models. Under Gaussian noise, least-squares calibration is the maximum-likelihood estimator, so it remains the reference.
  - Distribution shift, the counterpart of the paper's §4, on the trained models without retraining. The vol prior is unchanged, so BS calibration stays exact and remains the reference.
    - `otm_to_itm`: examples are out-of-the-money calls (m > 0), the query is in the money (m < 0).
    - `short_to_long`: examples have T below the median (~1.05 years), the query above it.
    - `scale-m=2`, `scale-m=3`: moneyness scaled by 2 or 3, so |m| reaches values rarely seen in training; maturity unchanged.
    - The split evals use the paper's train/test prompt mechanism (`xs_p`), as for `opposite_quadrants`. These shifts change the no-context floor (e.g. deep in- or out-of-the-money prices depend little on σ), so compare the Transformer against the BS prior mean of the same eval, not against 0.14 / 0.07.
    - Not included, because they need changes to the task and baselines: σ outside the prior (the calibration grid only covers the prior), a different r (not an input; it changes the pricing rule, closer to proposal A below), and maturity outside [0.1, 2] (impossible, since T = 0.1 + 1.9·Φ(x₁) maps every x₁ into that range).

## Extended proposal: letting the pricing rule vary in context
Not implemented. In `bs_flat` / `bs_smile` the pricing formula is the same in every prompt, so a trained model can store it in its weights and only the vol has to come from the context. These extensions make more of the task in-context.

**A. Mixed pricing models.** Each prompt draws a model (Black–Scholes, Bachelier/normal, Merton jump-diffusion) and its parameters; the model must infer which rule generated the examples.
- Fits the current framework: same (m, T) inputs, one new task class.
- Baselines: calibration under each model, plus model selection by best fit (the Bayes-style reference), and the no-context prior mean over the mixture.
- Effort: one task, one set of baselines, one training run. Closest to "one more task".

**B. Learning the risk-neutral distribution.** Each prompt has a hidden risk-neutral distribution at one maturity (e.g. a mixture of lognormals with mean equal to the forward, so prices are arbitrage-free and closed-form). Examples are prices of payoffs across strikes; the query can be a different payoff (digital, put spread). By Breeden–Litzenberger, call prices across strikes determine the distribution, so the model must learn the pricing measure in context, not a parameter.
- Needs design beyond a new task class:
  - Input encoding: strike plus a payoff-type feature (n_dims changes).
  - Training mixes payoff types within prompts, because the framework trains a prediction at every position; a model trained on calls only never learns to output a digital. The key test is then an eval-only split: calls in the context, a digital as the query.
  - Baselines: mixture fit by least squares (optimal when the family is known) and a model-free Breeden–Litzenberger estimate (smooth fit of C(K), digital = −∂C/∂K).
- Effort: a written spec first, then the task, baselines, new evals and training. Probably also a smaller model size (Small, 1.2M) to see where in-context learning breaks down.

## Not reproduced
- Model capacity (Tiny 0.2M, Small 1.2M) × dimension d ∈ {10, 30, 40, 50}, 3 seeds each (§6, Fig. 6, 10, 11)
- Training without curriculum (§6, App. B.5, Fig. 12–13)
- Limited number of distinct prompts n_p / weight vectors n_w (§6, App. B.6, Fig. 14; trained without curriculum for n_p)
- Memorisation analysis (App. B.7). This needs no training and could be recomputed.
- Query-scale robustness (App. B.1, Fig. 7). This needs no training, only evaluation of our linear model.
