from exogym.trainer import LocalTrainer
from exogym.strategy.optim import OptimSpec
from exogym.strategy.diloco import DiLoCoStrategy
from exogym.strategy.strategy import SimpleReduceStrategy

from nanogpt import GPT, GPTConfig, get_dataset

import torch

MAX_NODES = 4
H = 30
TOTAL_TOKENS = (2**15) * (2**13)  # 1024 steps for smallest GBS
# TOTAL_TOKENS = (2**15) * 10  # 1024 steps for smallest GBS
SEQ_LEN = 2**10
BASE_BATCH_SIZE = 2**16

def main():
    # Get datasets - this will take a while the first time, as the dataset has to be imported and processed.
    train_dataset, vocab_size = get_dataset(
        "owt",
        block_size=1024,
        device="cpu",
        start_pc=0.0,
        end_pc=0.005 * MAX_NODES,
    )
    val_dataset, vocab_size = get_dataset(
        "owt", 
        block_size=1024, 
        device="cpu", 
        start_pc=0.99, 
        end_pc=1.0
    )
    # train_dataset, vocab_size = get_dataset(
    #     "shakespeare",
    #     block_size=SEQ_LEN,
    #     device="cpu",
    #     start_pc=0.0,
    #     end_pc=0.9
    # )
    # val_dataset, vocab_size = get_dataset(
    #     "shakespeare", 
    #     block_size=SEQ_LEN, 
    #     device="cpu", 
    #     start_pc=0.9,
    #     end_pc=1.0
    # )

    # Create model
    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=1024,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.0,
    )
    # gpt_config = GPTConfig.gpt2_small()
    # gpt_config.dropout = 0.2
    gpt_config.vocab_size = vocab_size

    model = GPT(gpt_config)
    trainer = LocalTrainer(
        model,
        train_dataset,
        val_dataset,
        start_port=12355
    )


    batch_size_multiplier_list = [1, 2, 4, 8]

    for batch_size_multiplier in batch_size_multiplier_list:
        global_batch = batch_size_multiplier * BASE_BATCH_SIZE
        strategy = SimpleReduceStrategy(
            optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001 * batch_size_multiplier),
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs={
                "warmup_steps": 1024 // batch_size_multiplier,
                "cosine_anneal": True,
            },
            max_norm=1.0,
        )

        trainer.fit(
            num_epochs=1,
            max_steps=TOTAL_TOKENS // global_batch,
            strategy=strategy,
            num_nodes=1,
            device="mps",
            batch_size=global_batch // SEQ_LEN,
            shuffle=True,
            val_size=512,
            val_interval=100,
            wandb_project="DiLoCo-Batchsize-Scaling",
            run_name=f"ddp-batchsize{global_batch}",
        )

        for K in [1, 2, 4]:
            strategy = DiLoCoStrategy(
                optim_spec=OptimSpec(torch.optim.AdamW, lr=0.001 * batch_size_multiplier),
                lr_scheduler="lambda_cosine",
                lr_scheduler_kwargs={
                    "warmup_steps": 1024 // batch_size_multiplier,
                    "cosine_anneal": True,
                },
                max_norm=1.0,
                H=H,
            )

            # Train it!

            trainer.fit(
                num_epochs=1,
                max_steps=TOTAL_TOKENS // global_batch,
                strategy=strategy,
                num_nodes=K,
                device="mps",
                batch_size=global_batch // SEQ_LEN // K,
                minibatch_size=32 // K,  # Gradient accumulation to ensure we can fit in memory for a 96GB machine. Make this even lower for smaller devices.
                shuffle=True,
                val_size=256,
                val_interval=128 // batch_size_multiplier,
                wandb_project="DiLoCo-Batchsize-Scaling",
                run_name=f"diloco-K{K}-batchsize{global_batch}",
            )


if __name__ == "__main__":
    main()
