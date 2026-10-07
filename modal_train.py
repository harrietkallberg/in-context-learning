"""Train on a Modal GPU.

    modal run modal_train.py --test                   # 100-step smoke test, prints it/s
    modal run --detach modal_train.py                 # full run of conf/linear_regression.yaml
    modal run --detach modal_train.py --resume-id ID  # continue an interrupted run

Checkpoints and metrics are written to the Modal volume "harriet-icl-models", which
mirrors the local models/ folder. Fetch a finished run with:

    modal volume get harriet-icl-models linear_regression/ID models/linear_regression/ID
"""

import subprocess
import time
import uuid

import modal

GPU = "A100"  # torch 1.11/cu113 has no kernels for H100 or L4

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==1.11.0+cu113",
        extra_index_url="https://download.pytorch.org/whl/cu113",
    )
    # quinine pins pyyaml==5.4, which has no py3.10 wheel and fails to build with Cython 3
    .run_commands(
        "pip install 'cython<3' wheel",
        "pip install --no-build-isolation pyyaml==5.4",
    )
    # Same pins as environment.yml
    .pip_install(
        "matplotlib==3.5.2",
        "numpy==1.22.3",
        "pandas==1.4.2",
        "quinine==0.3.0",
        "scikit-learn==1.0.2",
        "seaborn==0.11.2",
        "tqdm==4.64.0",
        "transformers==4.17.0",
        "tokenizers==0.12.1",
        "wandb==0.24.2",
        "xgboost==1.6.1",
        "protobuf==3.20.1",
    )
    .add_local_dir("src", "/root/src", ignore=["__pycache__", "*.ipynb"])
)

volume = modal.Volume.from_name("harriet-icl-models", create_if_missing=True)
app = modal.App("harriet-in-context-learning", image=image)


@app.function(
    gpu=GPU,
    timeout=24 * 60 * 60,
    # Restart after preemption; train.py resumes from state.pt in the same run folder
    retries=modal.Retries(initial_delay=0.0, max_retries=10),
    volumes={"/root/models": volume},  # conf out_dir is ../models/<task> from src/
    secrets=[modal.Secret.from_name("harriet-wandb")],
)
def train(config: str, test: bool, run_id: str):
    import torch

    # train.py falls back to CPU silently; on Modal that would burn hours unnoticed
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU visible in the container")

    cmd = ["python", "train.py", "--config", config]
    if test:
        cmd += ["--test_run", "True"]
    else:
        cmd += ["--training.resume_id", run_id]

    proc = subprocess.Popen(cmd, cwd="/root/src")
    # Commit checkpoints regularly so they survive a crash or timeout
    last_commit = time.time()
    while proc.poll() is None:
        time.sleep(10)
        if time.time() - last_commit > 300:
            volume.commit()
            last_commit = time.time()
    volume.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"train.py exited with code {proc.returncode}")


@app.local_entrypoint()
def main(
    config: str = "conf/linear_regression.yaml",
    test: bool = False,
    resume_id: str = "",
):
    # Pick the run id here so a restarted container writes to the same folder
    run_id = resume_id or str(uuid.uuid4())
    print(f"Run id: {run_id}")
    train.spawn(config, test, run_id).get()
