import torch
import argparse

from exogym.trainer import Trainer
from exogym.strategy.optim import OptimSpec
from exogym.aux.utils import get_device

from nanogpt import GPT, GPTConfig, get_dataset

NUM_NODES = 4

### PLAYGROUND
### This is a minimal configuration for training a nanogpt model with a given strategy.
### The strategy can be swapped out for custom logic by writing a new strategy class.

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="shakespeare")
    args = arg_parser.parse_args()
    dataset = args.dataset

    # Get datasets - this will take a while the first time, as the dataset has to be imported and processed.
    train_dataset, vocab_size = get_dataset(
        dataset,
        block_size=1024,
        device="cpu",
        start_pc=0.0,
        end_pc=0.005 * NUM_NODES if dataset == "owt" else 0.99,
    )
    val_dataset, vocab_size = get_dataset(
        dataset, 
        block_size=1024, 
        device="cpu",
        start_pc=0.99, 
        end_pc=1.0
    )

    device = get_device()

    # Create model
    if dataset == "shakespeare":
        gpt_config = GPTConfig.gpt2_small()
        gpt_config.dropout = 0.2
    elif dataset == "owt" and device == 'mps':
        gpt_config = GPTConfig.gpt_sbase()
    elif dataset == "owt" and device == 'cuda':
        gpt_config = GPTConfig.gpt2_base()
    else:
        raise ValueError(f"Invalid dataset: {dataset} on device: {device}")

    gpt_config.vocab_size = vocab_size
    model = GPT(gpt_config)

    # Create trainer
    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
    )

    ## STRATEGY - This is where we define custom logic

    from exogym.strategy.diloco import DiLoCoStrategy

    strategy = DiLoCoStrategy(
        optim_spec=OptimSpec(torch.optim.AdamW, lr=0.0004),
        lr_scheduler="lambda_cosine",
        lr_scheduler_kwargs={
            "warmup_steps": 1000,
            "cosine_anneal": True,
        },
        max_norm=1.0,
        H=200,
    )

    # Train it!
    trainer.fit(
        num_epochs=1,
        max_steps=5000,
        strategy=strategy,
        num_nodes=NUM_NODES,
        device=device,
        batch_size=16,
        minibatch_size=8, # Gradient accumulation to ensure we can fit in memory. Make this even lower for smaller devices.
        shuffle=False,
        val_size=256,
        val_interval=100,
        # wandb_project='exo-gym',
        # run_name=f'diloco-{dataset}-{NUM_NODES}nodes'
    )


if __name__ == "__main__":
    main()
