# exo gym: Accelerating Distributed AI Research

exo gym is an open-source Python toolkit that facilitates distributed AI research.
We aim to do this by providing a suite of tools to simulate, test, and scale distributed training methods without requiring distributed infrastructure. Just on your laptop, you can train a model simulated on a 4-node cluster and submit it for evaluation. The best methods are then routinely scaled to a real cluster for evaluation.

## Getting Started

To get started, clone the repo, then head to the sims directory and choose one to run. The first implementation is a data parallel sim called diloco-sim.

```bash
git clone https://github.com/exo-explore/gym.git
```

Choose a sim to run, for example diloco-sim:
```bash
cd sims/diloco-sim
```

Here  
```bash
pip install -r requirements.txt
pip install -e .
```

Then get started with one of the examples available in the examples folder. Here is an example of the minimal arguments needed to train a nanoGPT with diloco-sim:


```python
simulator = DilocoSimulator(
    model_cls=CNNModel,
    model_kwargs={"num_classes": 100},
    train_dataset=train_dataset,
    loss_fn=F.cross_entropy,
)

simulator.train()
```


# Documentation




