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
import argparse
import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime


# ── 0. Plotting function ─────────────────────────────────────────────────────
def plot_validation_results_with_run_names(strategy_names, run_names):
    """Read CSV validation losses and plot results for all strategies."""
    val_losses_dict = {}
    
    # Debug: Print what we're looking for
    print(f"\nLooking for validation data for strategies: {strategy_names}")
    print(f"Using run names: {run_names}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Read validation losses from CSV files for each strategy
    for name, run_name in zip(strategy_names, run_names):
        val_csv_path = os.path.join('logs', run_name, 'validation.csv')
        print(f"Checking path: {val_csv_path}")
        
        if os.path.exists(val_csv_path):
            print(f"Found CSV file for {name}")
            steps = []
            losses = []
            with open(val_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                print(f"CSV headers: {headers}")
                
                for row in reader:
                    if row.get('global_loss') and row['global_loss'].strip():  # Use global loss if available
                        steps.append(int(row['step']))
                        losses.append(float(row['global_loss']))
                    elif row.get('local_loss') and row['local_loss'].strip():  # Fallback to local loss
                        steps.append(int(row['step']))
                        losses.append(float(row['local_loss']))
            
            if steps and losses:
                val_losses_dict[name] = (steps, losses)
                print(f"Loaded {len(steps)} validation points for {name.upper()}")
            else:
                print(f"No valid loss data found for {name}")
        else:
            print(f"CSV file not found: {val_csv_path}")
            # Check if directory exists
            dir_path = os.path.join('logs', run_name)
            if os.path.exists(dir_path):
                print(f"Directory exists, contents: {os.listdir(dir_path)}")
            else:
                print(f"Directory doesn't exist: {dir_path}")
    
    # List what's actually in the logs directory
    if os.path.exists('logs'):
        print(f"Available log directories: {os.listdir('logs')}")
    else:
        print("Logs directory doesn't exist")
    
    if not val_losses_dict:
        print("No validation data to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    
    colors = {'ddp': 'blue', 'diloco': 'green', 'sparta': 'red'}
    final_losses = {}
    
    for name, (steps, losses) in val_losses_dict.items():
        plt.plot(steps, losses, label=name.upper(), color=colors.get(name, 'black'), linewidth=2)
        final_losses[name] = losses[-1] if losses else float('inf')
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title('MNIST Validation Loss Comparison', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Print final validation loss summary
    print("\n" + "="*50)
    print("##### FINAL VALIDATION LOSS SUMMARY #####")
    print("="*50)
    
    # Sort strategies by final validation loss (best to worst)
    sorted_strategies = sorted(final_losses.items(), key=lambda x: x[1])
    
    for i, (name, final_loss) in enumerate(sorted_strategies):
        rank = f"#{i+1}"
        print(f"{rank:>3} {name.upper():>8}: {final_loss:.4f}")
    
    print("="*50)

    # Display the plot
    plt.show()
    

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
def run_sweep(use_wandb=False):
    # Create unique timestamp for this run to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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

    strategy_names = []
    run_names = []  # Store the actual run names used
    for name, Strat in [
        ("ddp", SimpleReduceStrategy),
        ("diloco", DiLoCoStrategy),
        ("sparta", SPARTAStrategy),
    ]:
        strategy_names.append(name)
        run_name = f"mnist_{name}_{timestamp}"
        run_names.append(run_name)
        
        model = ModelWrapper(CNN())
        trainer = Trainer(model, train_ds, val_ds)

        strategy = Strat(
            optim_spec=optim_spec,
            H=20,
            p=0.05,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs={"warmup_steps": 100, "cosine_anneal": True},
        )

        print(f"\n\n=== Training {name.upper()} ===")
        
        trainer.fit(
            max_steps=150,
            num_epochs=5,
            strategy=strategy,
            num_nodes=4,
            device=device,
            batch_size=256,  # larger batch is fine with this model
            val_size=len(val_ds),  # evaluate on the full 10 000 test set
            val_interval=10,
            wandb_project="mnist-compare" if use_wandb else None,
            run_name=run_name,
        )

    # Plot validation losses if using CSV logging
    if not use_wandb:
        plot_validation_results_with_run_names(strategy_names, run_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST training sweep")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    args = parser.parse_args()
    
    run_sweep(use_wandb=args.wandb)
