import torch
import argparse

from exogym.trainer import Trainer
from exogym.strategy.optim import OptimSpec
from exogym.strategy.diloco import DiLoCoStrategy
from exogym.strategy.strategy import SimpleReduceStrategy
from exogym.aux.utils import get_device

from nanogpt import GPT, GPTConfig, get_dataset

MAX_NODES = 4
H = 30
TOTAL_TOKENS = (2**15) * (2**13)  # 1024 steps for smallest GBS
# TOTAL_TOKENS = (2**14) * 32
SEQ_LEN = 2**10
BASE_BATCH_SIZE = 2**10

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="shakespeare")
    args = arg_parser.parse_args()
    dataset = args.dataset

    # Get datasets - this will take a while the first time, as the dataset has to be imported and processed.
    train_dataset, vocab_size = get_dataset(
        dataset,
        block_size=SEQ_LEN,
        device="cpu",
        start_pc=0.0,
        end_pc=0.005 * MAX_NODES if dataset == "owt" else 0.99,
    )
    val_dataset, vocab_size = get_dataset(
        dataset, 
        block_size=SEQ_LEN, 
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
    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
    )


    batch_size_multiplier_list = [1, 2, 4, 8, 16, 32]
    # K_list = [0, 1, 2, 4]  # K=0 now represents SimpleReduceStrategy
    K_list = [4]  # K=0 now represents SimpleReduceStrategy

    for K in K_list:
        for batch_size_multiplier in batch_size_multiplier_list:
            global_batch = batch_size_multiplier * BASE_BATCH_SIZE
            
            if K == 0:
                # K=0 corresponds to SimpleReduceStrategy
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
                    log_x_axis="examples",
                    kwargs={'dataset_name':dataset},
                )
            else:
                # K > 0 corresponds to DiLoCoStrategy with K nodes
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

                local_batch_size = global_batch // SEQ_LEN // K
                local_minibatch_size = min(32 // K, local_batch_size)  # Ensure minibatch_size <= batch_size

                trainer.fit(
                    num_epochs=1,
                    max_steps=TOTAL_TOKENS // global_batch,
                    strategy=strategy,
                    num_nodes=K,
                    device="mps",
                    batch_size=local_batch_size,
                    minibatch_size=local_minibatch_size,
                    shuffle=True,
                    val_size=256,
                    val_interval=128 // batch_size_multiplier,
                    wandb_project="DiLoCo-Batchsize-Scaling",
                    run_name=f"diloco-K{K}-batchsize{global_batch}",
                    log_x_axis="examples",
                    kwargs={'dataset_name':dataset},
                )


if __name__ == "__main__":
    main()
