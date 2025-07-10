# mnist_compare_strategies_big.py  (2-space indent preserved ✨)
from exogym.trainer import Trainer
from exogym.strategy.diloco import DiLoCoStrategy
from exogym.strategy.sparta import SPARTAStrategy
from exogym.strategy.strategy import SimpleReduceStrategy
from exogym.strategy.optim import OptimSpec

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split


# ── 1. Dataset ───────────────────────────────────────────────────────────────
def get_mnist_splits(root="data", train_frac=1.0):
    tfm = transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    full = datasets.MNIST(root, True, download=True, transform=tfm)
    if train_frac < 1.0:
        n_train = int(len(full) * train_frac)
        n_val = len(full) - n_train
        return random_split(full, [n_train, n_val])
    return full, None  # val set handled separately


# ── 2. Stronger CNN ───────────────────────────────────────────────────────────
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 28×28 → 14×14
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            # Block 2: 14×14 → 7×7
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── 3. Wrapper (returns logits, loss) ─────────────────────────────────────────
class ModelWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch):
        imgs, labels = batch
        logits = self.backbone(imgs)
        return F.cross_entropy(logits, labels)


# ── 4. Training sweep ─────────────────────────────────────────────────────────
def run_sweep():
    train_ds, _ = get_mnist_splits()
    val_ds = datasets.MNIST(
        "data",
        False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    optim_spec = OptimSpec(torch.optim.AdamW, lr=3e-4, weight_decay=1e-4)

    for name, Strat in [
        ("diloco", DiLoCoStrategy),
        ("sparta", SPARTAStrategy),
        ("simplereduce", SimpleReduceStrategy),
    ]:
        model = ModelWrapper(CNN())
        trainer = Trainer(model, train_ds, val_ds)

        strategy = Strat(
            optim_spec=optim_spec,
            H=10,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs={"warmup_steps": 100, "cosine_anneal": True},
        )

        print(f"\n=== {name.upper()} ===")
        trainer.fit(
            num_epochs=5,
            strategy=strategy,
            num_nodes=4,
            device=device,
            batch_size=256,  # larger batch is fine with this model
            minibatch_size=256,
            val_size=len(val_ds),  # evaluate on the full 10 000 test set
            val_interval=10,
            # wandb_project="mnist-compare",
            # run_name=f"{name}",
        )


if __name__ == "__main__":
    run_sweep()
