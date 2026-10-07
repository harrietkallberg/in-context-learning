"""Checker for Lesson 4: BSPosteriorMeanModel in src/models.py.

Run from the repo root, in the in-context-learning conda env:

    python learn/exercises/check_posterior.py

Each check prints PASS or FAIL with a hint. Takes about a minute on CPU.
"""
import os
import sys

import torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from models import BSPriorMeanModel  # noqa: E402
from samplers import get_data_sampler  # noqa: E402
from tasks import get_task_sampler  # noqa: E402

results = []


def check(name, ok, hint):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"       hint: {hint}")


def prompts(task, noise, batches, seed):
    torch.manual_seed(seed)
    sampler = get_data_sampler("gaussian", n_dims=2)
    for _ in range(batches):
        xs = sampler.sample_xs(41, 64)
        ys = get_task_sampler(task, 2, 64, noise_std=noise)().evaluate(xs)
        yield xs, ys


def error_by_k(model, task, noise, batches=6, seed=0):
    total = 0
    for xs, ys in prompts(task, noise, batches, seed):
        total = total + (model(xs, ys) - ys).square().mean(0)
    return total / batches


def run_checks(cls):
    # 1. construction and name
    try:
        smile = cls("bs_smile", tau=0.01)
        flat = cls("bs_flat", tau=0.05)
    except Exception as e:  # noqa: BLE001
        check("constructs with (task_name, tau=...)", False, f"__init__ raised {e!r}")
        return
    check("constructs with (task_name, tau=...)", True, "")
    check(
        "name starts with 'bs_posterior_mean'",
        getattr(smile, "name", "").startswith("bs_posterior_mean") and smile.name != flat.name,
        "set self.name = f'bs_posterior_mean_tau={tau}' so the two instances get different names",
    )

    xs, ys = next(prompts("bs_smile", 0.0, 1, seed=1))

    # 2. shapes and inds
    full = smile(xs, ys)
    check("returns shape (b, n)", tuple(full.shape) == (64, 41), f"got {tuple(full.shape)}; sum the weighted prices over the prior axis")
    try:
        one = smile(xs, ys, inds=[3])
        ok = tuple(one.shape) == (64, 1) and torch.allclose(one[:, 0], full[:, 3], atol=1e-5)
    except Exception as e:  # noqa: BLE001
        ok, one = False, e
    check("inds=[i] returns only column i", ok, "eval.py calls model(xs, ys, inds=[i]) for the shift evals; return pred[:, inds]")

    # 3. no leakage: the prediction at position i must not use ys[:, i]
    ys2 = ys.clone()
    ys2[:, 5] += 10.0
    pred2 = smile(xs, ys2)
    check(
        "prediction at i ignores y_i (no leakage)",
        torch.allclose(pred2[:, 5], full[:, 5], atol=1e-5),
        "the SSE for position i must use examples 0..i-1 only: shift the cumsum by one (prepend zeros, drop the last)",
    )
    check(
        "prediction at i+1 does use y_i",
        not torch.allclose(pred2[:, 6], full[:, 6], atol=1e-3),
        "weights should depend on the earlier examples' squared errors",
    )

    # 4. k = 0 is the prior mean
    prior = BSPriorMeanModel("bs_smile")(xs, ys)
    gap = (full[:, 0] - prior[:, 0]).abs().mean().item()
    check(
        f"k = 0 matches the BS prior mean (mean gap {gap:.4f})",
        gap < 0.02,
        "with no examples all weights are equal, so the prediction is the plain average over prior samples",
    )

    # 5. numbers against the reference implementation
    e = error_by_k(smile, "bs_smile", 0.0)
    check(
        f"bs_smile clean, tau=0.01: k=1 error {e[1]:.4f} (reference ~0.0105, Transformer 0.0107)",
        0.007 < e[1] < 0.015,
        "too low means leakage; too high, check the weights use exp(-SSE / (2 tau^2)) via softmax over the prior axis",
    )
    check(
        f"bs_smile clean, tau=0.01: k=3 error {e[3]:.5f} (reference ~0.0012, Transformer 0.0012)",
        0.0005 < e[3] < 0.0025,
        "check the standardisation: compare prices from self.data_cls.standardize(...) with ys",
    )
    e = error_by_k(flat, "bs_flat", 0.05)
    check(
        f"bs_flat noisy, tau=0.05: k=1 error {e[1]:.4f} (reference ~0.0117, Transformer 0.0167)",
        0.007 < e[1] < 0.02,
        "with tau equal to the true noise this is the exact Bayes-optimal predictor",
    )

    print(f"\n{sum(results)}/{len(results)} checks passed.")
    if all(results):
        print("Now register it (Lesson 4, step 6) and recompute the BS metrics.")


if __name__ == "__main__":
    try:
        from models import BSPosteriorMeanModel
    except ImportError:
        print("[FAIL] src/models.py has no BSPosteriorMeanModel yet. Start with Lesson 4, step 1.")
        sys.exit(1)
    run_checks(BSPosteriorMeanModel)
