from exogym.trainer import Trainer
from nanogpt import GPT, GPTConfig, get_dataset
from exogym.strategy.optim import OptimSpec
import torch

NUM_NODES = 4
# NUM_NODES = 2

### PLAYGROUND
### This is a minimal configuration for training a nanogpt model with a given strategy.
### The strategy can be swapped out for custom logic by writing a new strategy class.


def main():
    # Get datasets - this will take a while the first time, as the dataset has to be imported and processed.
    train_dataset, vocab_size = get_dataset(
        "owt",
        block_size=1024,
        device="cpu",
        start_pc=0.0,
        end_pc=0.005 * NUM_NODES,
    )
    val_dataset, vocab_size = get_dataset(
        "owt", block_size=1024, device="cpu", start_pc=0.99, end_pc=1.0
    )

    # Create model
    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=1024,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.0,
    )
    model = GPT(gpt_config)

    # Create trainer
    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
        # port=12355 # Modify this if we get port conflict errors
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
        device="mps",
        batch_size=16,
        minibatch_size=8,  # Gradient accumulation to ensure we can fit in memory for a 96GB machine. Make this even lower for smaller devices.
        shuffle=False,
        val_size=256,
        val_interval=100,
        # wandb_project='50M-GPT',
        # run_name='diloco-H100'
    )


if __name__ == "__main__":
    main()
