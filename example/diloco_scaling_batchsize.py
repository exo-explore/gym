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
BASE_BATCH_SIZE = 2**15
WARMUP_TOKENS = BASE_BATCH_SIZE * 256  # Same as 1024 steps for smallest batch size

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="shakespeare")
    arg_parser.add_argument("--port", type=int, default=12355)
    arg_parser.add_argument("--only_run", type=int, default=None, help="Only run the i-th training run (0-indexed)")
    arg_parser.add_argument('--base_lr', type=float, default=0.004)
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
        start_port=args.port + args.only_run if args.only_run else 0,
    )


    batch_size_multiplier_list = [1, 2, 4, 8, 16, 32]
    # K_list = [0, 1, 2, 4]  # K=0 now represents SimpleReduceStrategy
    K_list = [4]  # K=0 now represents SimpleReduceStrategy

    # Calculate total number of training runs
    total_runs = len(K_list) * len(batch_size_multiplier_list)
    
    # Validate only_run argument if provided
    if args.only_run is not None:
        if args.only_run < 0 or args.only_run >= total_runs:
            raise ValueError(f"only_run must be between 0 and {total_runs-1}, got {args.only_run}")
        print(f"Running only training run {args.only_run} out of {total_runs} total runs")

    run_index = 0
    for K in K_list:
        for batch_size_multiplier in batch_size_multiplier_list:
            # Check if we should run this training run
            if args.only_run is not None and run_index != args.only_run:
                run_index += 1
                continue
                
            global_batch = batch_size_multiplier * BASE_BATCH_SIZE
            warmup_steps = WARMUP_TOKENS // global_batch
            
            if K == 0:
                # K=0 corresponds to SimpleReduceStrategy
                strategy = SimpleReduceStrategy(
                    optim_spec=OptimSpec(torch.optim.AdamW, lr=args.base_lr * batch_size_multiplier),
                    lr_scheduler="lambda_cosine",
                    lr_scheduler_kwargs={
                        "warmup_steps": warmup_steps,
                        "cosine_anneal": True,
                    },
                    max_norm=1.0,
                )

                trainer.fit(
                    num_epochs=1,
                    max_steps=TOTAL_TOKENS // global_batch,
                    strategy=strategy,
                    num_nodes=1,
                    device=device,
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
                    optim_spec=OptimSpec(torch.optim.AdamW, lr=args.base_lr * batch_size_multiplier),
                    lr_scheduler="lambda_cosine",
                    lr_scheduler_kwargs={
                        "warmup_steps": warmup_steps,
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
                    device=device,
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
            
            run_index += 1


if __name__ == "__main__":
    main()
