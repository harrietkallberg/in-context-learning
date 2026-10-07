"""LaTeX tables for the paper, generated from each run's metrics.json.

Only the tabular is written; the caption and label stay in the paper, so a rerun
updates the numbers without overwriting text. Include with \\input{tables/<name>.tex}.
"""

import json
import math
import os

TRANSFORMER = "gpt2_embd=256_layer=12_head=8"

# (task label, task, [(row label, model key in metrics.json)])
BS_RESULTS_ROWS = [
    (
        "Flat",
        "bs_flat",
        [
            ("Transformer", TRANSFORMER),
            ("Calibration", "bs_calibration_vol=bs_flat"),
            ("Prior mean (no context)", "bs_prior_mean"),
        ],
    ),
    (
        "Smile",
        "bs_smile",
        [
            ("Transformer", TRANSFORMER),
            ("Calibration (smile)", "bs_calibration_vol=bs_smile"),
            ("Calibration (flat)", "bs_calibration_vol=bs_flat"),
            ("Prior mean (no context)", "bs_prior_mean"),
        ],
    ),
]

# calibration with the volatility model that generated the data; the Transformer
# is bold where it beats this
CORRECT_CALIBRATION = {
    "bs_flat": "bs_calibration_vol=bs_flat",
    "bs_smile": "bs_calibration_vol=bs_smile",
}


def load_metrics(run_dir, task, run_id):
    with open(os.path.join(run_dir, task, run_id, "metrics.json")) as f:
        return json.load(f)


def format_error(value, bold=False):
    """Three decimals down to 0.05, then two significant digits as a·10^b."""
    if value >= 0.05:
        text = f"{value:.3f}"
    else:
        exponent = math.floor(math.log10(value))
        mantissa = value / 10**exponent
        if round(mantissa, 1) >= 10:  # e.g. 9.97e-3 rounds to 10.0
            mantissa, exponent = mantissa / 10, exponent + 1
        text = f"{mantissa:.1f}\\cdot10^{{{exponent}}}"
    return f"$\\mathbf{{{text}}}$" if bold else f"${text}$"


def bs_results_table(run_dir, run_ids, eval_name="standard", ks=(0, 1, 3, 40)):
    """Error after k in-context examples for each method, flat and smile tasks."""
    lines = [
        "\\begin{tabular}{ll" + "c" * len(ks) + "}",
        "\\toprule",
        "Task & Method & " + " & ".join(f"$k={k}$" for k in ks) + " \\\\",
    ]
    for task_label, task, rows in BS_RESULTS_ROWS:
        metrics = load_metrics(run_dir, task, run_ids[task])[eval_name]
        reference = metrics[CORRECT_CALIBRATION[task]]["mean"]
        lines.append("\\midrule")
        for i, (row_label, key) in enumerate(rows):
            mean = metrics[key]["mean"]
            cells = [
                # at k = 0 calibration is the prior mean, so no comparison
                format_error(
                    mean[k], bold=key == TRANSFORMER and k > 0 and mean[k] < reference[k]
                )
                for k in ks
            ]
            label = task_label if i == 0 else ""
            lines.append(f"{label} & {row_label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_tables(run_dir, run_ids, table_dir):
    os.makedirs(table_dir, exist_ok=True)
    path = os.path.join(table_dir, "bs_results.tex")
    with open(path, "w") as f:
        f.write(bs_results_table(run_dir, run_ids))
    return [path]
