<div align="center">

# EXO Gym

<img src="docs/imagination.png" alt="EXO Gym" width="50%">

<!-- New image ideas: A macbook with a loss curve on it, with a 'thinking bubbles' coming out of the macbook, and in the bubble there is a stack of 4 H100 GPUs. It's like the laptop is imagining the cluster.  -->


##### EXO Gym: Simulate distributed training on any hardware configuration, at any scale.

Simulate a GPU cluster with just your laptop! For example:

</div>

- Simulate training with SPARTA with a cluster of 4 Mac Studios connected over Ethernet
- Simulate training with DiLoCo on a cluster of 16 H100's connected over the internet

## What Does EXO Gym Do?

- Simulate distributed training without setting up distributed clusters; avoid Kubernetes, GPU hosting, etc.
- Fast iteration: implementing a new distributed training algo from scratch takes as little as 5 lines
- Scale up number of nodes by changing a single parameter

<!-- Bullet this -->
<!-- Forget about high GPU bills 💸 and painful Kubernetes setup 🤯. 
Want to scale up from 4 to 8 nodes? Just change a single parameter 🔧 
Implementing a new algo from scratch takes as little at 5 lines 🚀 -->

## Supported Algorithms

- AllReduce (Equivalent to PyTorch [DDP](https://arxiv.org/abs/2006.15704))
- [FedAvg](https://arxiv.org/abs/2311.08105)
- [DiLoCo](https://arxiv.org/abs/2311.08105)
- [SPARTA](https://openreview.net/forum?id=stFPf3gzq1)
- [DeMo](https://arxiv.org/abs/2411.19870)

... and anything else you can imagine! Implementing new algorithms with EXO Gym is very simple - see <a href='#custom-algorithms'>Custom Algorithms</a>.


## Installation

### Dependencies

- `python>=3.10`

### Installation

To install:
```bash
git clone https://github.com/exo-explore/gym.git exogym
cd exogym
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Pip Installation
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ exogym
```

## Usage

### Example Scripts

| Example | Result |
| ------- | ------ |
| **MNIST Comparison** <br><br> Compare DDP, DiLoCo, SPARTA on MNIST dataset. Runs in <2 mins on a M4 Mac Mini. <br><br> ```bash<br>python example/mnist.py<br>``` | <img src="docs/mnist_compare.png" alt="MNIST Training Comparison" width="100%"> |
| **NanoGPT OpenWebText** <br><br> Train a NanoGPT-style transformer on the OpenWebText dataset. <br><br> ```bash<br>python example/nanogpt_train.py --dataset owt --strategy diloco<br>``` | <img src="docs/OWT%20DiLoCo%20N=4.png" alt="OWT DiLoCo N=4" width="100%"> |
| **Shakespeare DiLoCo Scaling K** <br><br> How does DiLoCo compare for different device count (K)? This script compares DiLoCo for different device counts, normalized by FLOPs. <br><br> ```bash<br>python example/diloco_scaling.py --dataset shakespeare<br>``` <br><br> We can generate text with the model that has just been trained as so: <br><br> ```bash<br>python example/nanogpt/shakespeare_inference.py<br>``` | <img src="docs/diloco-batchsize.png" alt="Shakespeare Training Results" width="100%"> |

### Custom Training

Strategies (eg. DiLoCo, SPARTA) are portable across domains. A custom dataset and model can be trained with a distributed algorithm like so:

```python
from exogym import Trainer
from exogym.strategy.diloco import DiLoCoStrategy

train_dataset, val_dataset = ...
model = ... # model.forward() expects a batch, and returns a scalar loss

trainer = Trainer(model, train_dataset, val_dataset)

# Strategy for optimization & communication
strategy = DiLoCoStrategy(
  inner_optim='adam',
  H=100
)

trainer.fit(
  strategy=strategy,
  num_nodes=4,
  device='mps'
)
```

### Custom Algorithms

`example/playground.py` is a minimal starting-point for writing new algorithms. For example, to implement gradient quantization from scratch:

```python
class QuantizationStrategy(Strategy):
    def __init__(self, optim_spec, quantization_level: Literal['int8']):
        super().__init__()
        self.optim_spec = optim_spec
        self.scale = 0.024
        self.zero_point = 0
        self.qdtype = torch.uint8

    def step(self):
        for param in self.model.parameters():
            if param.grad is not None:
                quantized = torch.round(param.grad / self.scale + self.zero_point).clamp(0, 255).to(self.qdtype)
                
                q_wide = quantized.to(torch.int32)
                all_reduce(q_wide)
                
                param.grad = (q_wide.to(torch.float32) * self.scale) / self.num_nodes

        self.optim.step()
        super().step()
```



## Supported Devices

- CPU
- CUDA
- MPS (CPU-bound for copy operations, see [here](https://github.com/pytorch/pytorch/issues/141287))


## Technical Details

For further details on how EXO Gym works under-the-hood, please see [docs/](docs/README.md).

## Citation

If you use EXO Gym in your research, please cite:

```bibtex
@software{exogym2025,
  title={EXO Gym},
  author={Matt Beton, Mohamed Baioumy, Matt Reed, Seth Howes, Alex Cheema},
  year={2025},
  url={https://github.com/exo-explore/gym}
}
```



