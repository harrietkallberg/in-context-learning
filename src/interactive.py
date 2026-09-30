import os

import matplotlib.pyplot as plt
import torch

from eval import get_model_from_run
from models import LeastSquaresModel
from samplers import get_data_sampler
from tasks import get_task_sampler

# Load the model and measure its in-context learning ability on a batch of random
# inputs. (In the paper we average over multiple such batches for better estimates.)

src_dir = os.path.dirname(os.path.abspath(__file__))
run_dir = os.path.join(src_dir, "..", "models")
task_name = "linear_regression"
# task_name = "sparse_linear_regression"
# task_name = "decision_tree"
# task_name = "relu_2nn_regression"

# if you train more models, replace with the run_id from the models dir
run_id = "52e59405-04a5-4e41-8caa-6bc1e02d8304"
run_path = os.path.join(run_dir, task_name, run_id)

model, conf = get_model_from_run(run_path)
model.eval()

n_dims = conf.model.n_dims
batch_size = 100  # evaluation batch size (training used conf.training.batch_size)

data_sampler = get_data_sampler(conf.training.data, n_dims)
task_sampler = get_task_sampler(
    conf.training.task, n_dims, batch_size, **conf.training.task_kwargs
)

task = task_sampler()
xs = data_sampler.sample_xs(
    b_size=batch_size, n_points=conf.training.curriculum.points.end
)
ys = task.evaluate(xs)

with torch.no_grad():
    pred = model(xs, ys)

# Least squares fit on the same in-context examples (min-norm solution when
# there are fewer examples than dimensions).
with torch.no_grad():
    pred_ls = LeastSquaresModel()(xs, ys)

metric = task.get_metric()
loss = metric(pred, ys).numpy()
loss_ls = metric(pred_ls, ys).numpy()

sparsity = (
    conf.training.task_kwargs.sparsity
    if "sparsity" in conf.training.task_kwargs
    else None
)
baseline = {
    "linear_regression": n_dims,
    "sparse_linear_regression": sparsity,
    "relu_2nn_regression": n_dims,
    "decision_tree": 1,
}[conf.training.task]

plt.figure()
plt.plot(loss.mean(axis=0), lw=2, label="Transformer")
plt.plot(loss_ls.mean(axis=0), lw=2, label="Least squares")
plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
plt.xlabel("# in-context examples")
plt.ylabel("squared error")
plt.legend()
plt.show()
