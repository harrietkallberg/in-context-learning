# Handoff: after training finishes

Written 2026-09-30 ~01:30 CEST. Read `experimental_setup.md` first: it is the source of truth for the setup, the differences from the paper, and what is not reproduced. This file is only the next steps.

## 1. What is training, and when it finishes

Six runs on Modal, workspace `kthaisociety-dev`, **environment `in-context-learning`** (not `main`: every `modal` command needs `-e in-context-learning`). Volume `harriet-icl-models`. wandb project `harriet-kallberg02-kth-royal-institute-of-technology/in-context-training`.

| Task | Run id (volume folder `<task>/<id>`) | wandb name | Code it runs | Expected finish (CEST, 30 Sep) |
|---|---|---|---|---|
| linear_regression | `52e59405-04a5-4e41-8caa-6bc1e02d8304` | `linear_regression_standard` | `00178ea` | ~05:00 |
| sparse_linear_regression | `56bb1fd5-0882-411f-92ac-7e2109f3f4d0` | `sparse_regression_standard` | `00178ea` | ~05:00 |
| decision_tree | `53417d4c-9a6c-4c1a-ae09-3df8c0af2676` | `decision_tree_standard` | `00178ea` | ~12:00 |
| relu_2nn_regression | `0c3f3424-0387-4099-bff0-004968547236` | `relu_2nn_regression_standard` | `00178ea` | ~11:00 |
| bs_flat | `dc2e5022-dd53-436f-89fb-8c0a6b4fa3ff` | `bs_flat_standard` | `e61880c` | ~06:00 |
| bs_smile | `61af878c-034c-459f-b647-78b71b2bcfdf` | `bs_smile_standard` | `e61880c` | ~06:00 |

Times assume no preemption; `modal_train.py` retries and resumes, which adds delay.

Two earlier BS runs were cancelled at ~18k steps and relaunched (to pick up the no-context baseline); their volume folders and wandb runs have been deleted, so each BS task folder holds only the run above.

**A run is finished when** its app shows `stopped` in `modal app list -e in-context-learning`, the wandb run is `finished`, and its volume folder contains `metrics.json` (written by `train.py` right after the last step):

```
modal volume ls -e in-context-learning harriet-icl-models linear_regression/52e59405-04a5-4e41-8caa-6bc1e02d8304
```

If an app stopped but there is no `metrics.json`, the run crashed: check `modal app logs -e in-context-learning <app id>` and resume with `modal run --detach -e in-context-learning modal_train.py --config conf/<task>.yaml --resume-id <run id>`.

Interim sanity check (wandb, ~02:30): all runs were learning. Raw `overall_loss` rises during the first 30k steps because the curriculum grows d and k; that is expected, not divergence. Linear regression sat at `excess_loss` ≈ 1.0, i.e. roughly least-squares level.

## 2. Download

Each folder has `config.yaml`, `state.pt` (final weights), `model_100000.pt` … (every 100k steps), `metrics.json`, `wandb/`. From the repo root:

```
modal volume get -e in-context-learning harriet-icl-models linear_regression/52e59405-04a5-4e41-8caa-6bc1e02d8304 models/linear_regression/52e59405-04a5-4e41-8caa-6bc1e02d8304
```

Repeat per row of the table. Check afterwards that `models/<task>/<id>/config.yaml` exists at that depth (not nested one level deeper). `models/` is git-ignored.

**Already in local `models/`:**
- `models/<task>/pretrained/`: the authors' released checkpoints and `metrics.json` for the four paper tasks. This is the numeric reference for "our model vs the paper's model".
- Nothing else. Two old local toy runs (5k steps, both named `linear_regression_toy`) were moved to `models_toy/linear_regression/` (git-ignored), because `read_run_dir` asserts unique wandb names across all runs under `models/`.

## 3. Evals to compute

Local env: conda env `in-context-learning` (`C:\Users\harri\anaconda3\envs\in-context-learning`). From Git Bash, put `$ENV:$ENV/Library/bin` on `PATH` or HTTPS/SSL (wandb) fails. Eval runs on CPU if there is no GPU.

**Paper tasks: nothing to compute.** Their end-of-training `metrics.json` already has every eval (`standard` plus the §4 out-of-distribution set from `build_evals`) and the code's baselines.

**BS tasks: add the shift evals.** The BS runs launched from `e61880c`, so their `metrics.json` has `standard`, `noisy`, the no-context baseline and the k = 0 calibration fix, but not the four shift evals added later in `32d227a` (`otm_to_itm`, `short_to_long`, `scale-m=2`, `scale-m=3`). `compute_evals` caches per eval and per model, so this computes only what is missing:

```
cd src
python -c "from eval import get_run_metrics; get_run_metrics('../models/bs_flat/dc2e5022-dd53-436f-89fb-8c0a6b4fa3ff')"
python -c "from eval import get_run_metrics; get_run_metrics('../models/bs_smile/61af878c-034c-459f-b647-78b71b2bcfdf')"
```

If the run folder's mtime is newer than `metrics.json` (possible after download), it recomputes everything instead: still correct, ~15–20 min per task on CPU, mostly the smile calibration.

## 4. Plots

Use `src/eval.ipynb` (`run_dir = "../models"`): set `task` and `run_id`, then it plots `standard` and every OOD eval with `plot_utils.basic_plot`, using `relevant_model_names[task]`. Known issues to fix in the notebook or in `plot_utils.py`:

- **decision_tree:** `relevant_model_names` lists "2-layer NN, GD", which our tree run does not compute → `KeyError`. Drop it from the list (see `experimental_setup.md` §5).
- **BS tasks, trivial line:** the OOD cell sets `trivial = scale²` for names containing `scale` and `1 + 1/n_dims` for `noisy`. Both are for linear regression. For BS the targets are standardised and not scaled, so the trivial line is 1 for every eval (noisy: 1 + 0.05²). More important than the trivial line is the **"BS prior mean (no context)"** curve, which is already in `relevant_model_names`: only Transformer error below it shows in-context calibration. Under the shift evals this floor changes, so compare within each eval.
- **Ours vs the paper's model:** the notebook plots one run at a time. For the comparison, plot our Transformer and the `pretrained` Transformer for the same task and eval on one axis (both `metrics.json` use the key `gpt2_embd=256_layer=12_head=8` and the same eval names).

## 5. What is reproduced, and where the numbers are

| Paper result | Our run | Numbers | Comparable? |
|---|---|---|---|
| Fig. 2: linear regression, Transformer vs least squares / 3-NN / averaging | linear_regression | `metrics.json` → `standard` | Yes |
| Fig. 4, 8, 9: linear regression under distribution shift (skewed covariance, half subspace, x/y scaling, noisy outputs, quadrants, orthogonal, overlapping query) | linear_regression | `metrics.json` → the other eval names | Yes |
| Fig. 5a: sparse linear regression (with Lasso) | sparse_linear_regression | `metrics.json` → `standard` | Yes |
| Fig. 5b: decision trees | decision_tree | `metrics.json` → `standard` | Transformer yes; baselines no (public code ≠ paper, §5) |
| Fig. 5c–d: 2-layer ReLU networks (also evaluated on linear functions) | relu_2nn_regression | `metrics.json` → `standard`, `linear_regression` | Transformer yes; GD baseline no (§5) |

- The paper does not tabulate its curves. The numeric reference is the authors' `models/<task>/pretrained/metrics.json` (same eval names), plus the figures in arXiv:2208.01066v3.
- One seed per task; the paper also shows single runs for these figures (§1 of `experimental_setup.md`).
- **Not reproduced** (list and reasons in `experimental_setup.md` → "Not reproduced"): capacity × dimension sweeps, training without curriculum, limited n_p / n_w, memorisation analysis, query-scale robustness. The last two need no training and could be added from the downloaded linear model.

**Extension (no paper numbers):** `bs_flat`, `bs_smile`. References are BS calibration (exact under the true vol model) and the no-context floor (≈ 0.14 flat, ≈ 0.07 smile on `standard`). Setup, evals and interpretation are in `experimental_setup.md` → "Extension: in-context option pricing". Continued work (not implemented) is in "Extended proposal": A (mixed pricing models) and B (learning the risk-neutral distribution, the planned continuation).

## 6. Checklist

1. Wait for all six runs; check `metrics.json` exists for each.
2. Download the six folders into `models/<task>/<id>`.
3. Run `get_run_metrics` for the two BS runs (section 3).
4. Make the plots (section 4): the paper tasks next to `pretrained`, and the BS tasks with the prior-mean floor.
5. Write the results into `experimental_setup.md` (or a results section): for each figure, whether our curve matches the paper's and where it differs.
