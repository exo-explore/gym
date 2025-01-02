# exo gym: Accelerating Distributed AI Research

exo gym is an open-source Python toolkit that facilitates distributed AI research.
It contains an evolving collection of simulators that run on a single machine but simulate a distributed setup. For example, diloco-sim simulates running the DiLoCo algorithm by training on N nodes, but it can all run locally on a single machine. The goal is to abstract away the complexity of maintaining a real distributed system and allow researchers to focus on algorithmic innovation. 

## Getting Started

To get started, clone the repo, then head to the sims directory and choose one to run. The first implementation is a data-parallel sim called diloco-sim.

```bash
git clone https://github.com/exo-explore/gym.git
pip install -r requirements.txt
```

Choose a sim to run, for example, diloco-sim:
```bash
cd sims/diloco-sim
```

Then get started with one of the examples available in the examples folder. Here is an example of the minimal arguments needed to train a CNN with diloco-sim:


```python
simulator = DilocoSimulator(
    model_cls=CNNModel,
    model_kwargs={"num_classes": 100, "input_channels": 3, "input_height": 32, "input_width": 32},
    optimizer_kwargs={"lr": 0.001},
    num_nodes = 2,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss_fn=F.cross_entropy,
    num_epochs=10,
)
simulator.train()
```


# Resources
For a short introduction to understanding distributed training check out [the following blog](https://blog.exolabs.net/day-5/) written by the EXO team. 
For more details about EXO gym & related competitions, [click here](https://blog.exolabs.net/day-5/).


