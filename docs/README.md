# EXO Gym Technical Details

## Codebase Structure

- `Trainer`: Builds simulation environment. `Trainer` will spawn multiple `TrainNode` instances using PyTorch Distributed. The local instances are connected together with `_build_connection`, and `TrainNode.train()` is executed on each rank.
- `TrainNode`: A single node (rank) running its own training loop. At each train step, instead of calling `optim.step()`, it calls `strategy.step()`.
- `Strategy`: Abstract class for an optimization strategy, which both defines **how the nodes communicate** with each other and **how model weights are updated**. Typically, a gradient strategy will include an optimizer as well as a communication step. Sometimes (eg. DeMo), the optimizer step is comingled with the communication.

## Technical Details

EXO Gym uses pytorch multiprocessing to spawn a subprocess per-node, which are able to communicate with each other using regular operations such as `all_reduce`.

### Model

The model is expected in a form that takes a `batch` (the same format as `dataset` outputs), and **returns a scalar loss** over the entire batch. This ensures the model is agnostic to the format of the data (eg. masked LM training doesn't have a clear `x`/`y` split).

### Dataset

Recall that when we call `trainer.fit()`, $K$ subprocesses are spawned to handle each of the virtual workers. There are two options for creating dataset:

#### PyTorch `Dataset`

Instantiate a single `Dataset`. The `dataset` object is passed to every subprocess, and a `DistributedSampler` will be used to select which datapoints are sampled per-node (to ensure each datapoint is only used once by each node). If the dataset is entirely loaded into memory, this memory will be duplicated per-node - be careful not to run out of memory! If the dataset is larger, it should be lazily loaded.

#### `dataset_factory` function

In place of the dataset object, pass a function with the following signature:

```python
def dataset_factory(rank: int, num_nodes: int, train_dataset: bool) -> torch.utils.data.Dataset
```

This will be called within each rank to build the dataset. Instead of each node storing the whole dataset and subsampling datapoints, each node only loads the necessary datapoints.

