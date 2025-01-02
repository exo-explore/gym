# Simulators

Simulators are at the core of exo gym. They are responsible for generating the environment for a virtual distributed system. The main sim that's implemented right now is 'dilco-sim' which is a simulator for the [DiLoCo algorithm](https://arxiv.org/abs/2311.08105). Credit for implementation of the 1st version Dilco-sim goes to [Matthew Reed](https://github.com/matttreed/diloco-sim) funded by exo. We used a fork of the original implementation from [Jianbo27](https://github.com/jianbo27/diloco-sim) as it contained bug fixes and improvements over the original repository.


## Types of simulators
To start the main simulator provided is diloco-sim, the main simulator added. It will be generalized for other types of data parallelism methods. The second main sim to be added will be pipeline-sim to simulate the pipeline parallelism methods.

## diloco-sim
diloco-sim (original implementation by matttreed and improved by Jianbo27) is a simulator for the [DiLoCo algorithm](https://arxiv.org/abs/2311.08105), which is a distributed training algorithm that synchronizes models every n steps instead of every step.

diloco-sim merely simulates this distributed network, but the workers may or may not be running on the same machine, depending on how many devices are available.

Example usage can be found in the `examples` directory.

The minimal arguments for training are shown below:

```python

from diloco_sim import DilocoSimulator
from models import ModelArchitecture
import torch.nn.functional as F
from data import train_dataset, test_dataset

simulator = DilocoSimulator(
    model_cls=CNNModel,
    model_kwargs={"num_classes": 100},
    train_dataset=train_dataset,
    loss_fn=F.cross_entropy,
)

simulator.train()

```

The full list of available arguments is shown below:

| **Argument**         | **Type**                | **Default**                  | **Description**                                                                 |
|-----------------------|-------------------------|------------------------------|---------------------------------------------------------------------------------|
| `model_cls`          | `Type[Module]` | **Required**                | The model class to be instantiated and trained. Must be a subclass of `torch.nn.Module`. |
| `loss_fn`            | `(x,y) -> loss` | **Required**                | The loss function used during training. Example: `torch.nn.functional.cross_entropy`. Must be of form (x,y) => loss |
| `train_dataset`      | `Dataset` | **Required**                | The dataset for training. Should be a subclass of `torch.utils.data.Dataset`.                   |
| `model_kwargs`       | `dict`                 | `{}`                        | Keyword arguments to initialize the model. Example: `{"num_classes": 100, ...}`.     |
| `num_nodes`          | `int`                  | `4`                         | Number of nodes (simulated workers) in the distributed system.                  |
| `optimizer_kwargs`   | `dict`                 | `{"lr": 0.001}`             | Keyword arguments for the inner optimizer. Example: `{"lr": 0.001}`.                 |
| `diloco_interval`    | `int`                  | `500`                       | Number of local steps before synchronizing the models.                          |
| `batch_size`         | `int`                  | `16`                        | Batch size for training and evaluation.                                         |
| `eval_dataset`       | `Optional[Dataset]` | `None`                       | The dataset for evaluation. Optional. Should be a subclass of `torch.utils.data.Dataset`.                                       |
| `optimizer_cls`      | `Type[Optimizer]` | `torch.optim.AdamW`         | Inner Optimizer class for training. `AdamW` is the default per recommendation of DiLiCo.                                            |
| `ckpt_interval`      | `Optional[int]`        | `None`                      | Number of outer steps between model checkpoints. Default is `None`.             |
| `eval_iters`         | `int`                  | `50`                        | Number of iterations to use for evaluation. Loss is approximated by `eval_iters * batch_size` samples. Default is `50`.                   |
| `save_dir`           | `Optional[str]`        | `None`                      | Directory to save model checkpoints. Default is `None`.                        |
| `outer_optimizer_cls` | `Type[Optimizer]` | `torch.optim.SGD`           | Optimizer class for outer training. Default is `SGD` per recommnedation of DiLoCo.                |
| `outer_optimizer_kwargs` | `dict`                 | `{"lr": 0.7, "nesterov": True, "momentum": 0.9}` | Keyword arguments for the outer optimizer. Nesterov momentum is default per recommendation of DiLoCo.      |
| `max_local_step`     | `Optional[int]`        | `None`                      | Maximum number of local steps to train. Default is `None`. If specified, training will stop after this many local steps if it occurs before the end of `num_epochs` epochs. |
| `num_epochs`         | `int`                  | `1`                        | Total number of training epochs.                                                |
| `cosine_anneal`      | `bool`                 | `False`                     | Whether to use cosine annealing for learning rate scheduling. Default is `False`. |
| `train_loss_hook`    | `(TrainStats) -> None` | `None`                      | Function to call after each local step. Default is `None`. `TrainStats` is a dataclass defined below.                   |
| `eval_loss_hook`     | `(EvalStats) -> None` | `None`                      | Function to call after each evaluation. Default is `None`. `EvalStats` is a dataclass defined below.                    |













