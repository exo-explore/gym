from exogym.trainer import LocalTrainer
from nanogpt import GPT, GPTConfig, get_dataset
from exogym.strategy.optim import OptimSpec

import argparse
import torch


def gen_run_name(args, strategy):
    """Generate wandb name based on strategy and arguments."""
    base_name = f"bs{args.batch_size}_lr{args.lr:.0e}"

    if strategy == "base":
        return f"{base_name}_warm{args.warmup_steps}_max{args.max_steps}"
    elif strategy == "ddp":
        return f"ddp_{base_name}_n{args.num_nodes}"
    elif strategy == "fedavg":
        return f"{base_name}_H{args.H}_n{args.num_nodes}"
    elif strategy == "sparta":
        return f"p{args.p_sparta}_n{args.num_nodes}_lr{args.lr:.0e}"
    elif strategy == "diloco":
        return f"{base_name}_outer{args.outer_lr:.0e}_H{args.diloco_interval}"
    elif strategy == "demo":
        return f"{base_name}_topk{args.compression_topk}_decay{args.compression_decay}"
    elif strategy == "diloco_sparta":
        return f"{base_name}_outer{args.outer_lr:.0e}_H{args.diloco_interval}_p{args.p_sparta}"
    else:
        return base_name


def arg_parse():
    """Create parser with all arguments for all strategies."""
    parser = argparse.ArgumentParser(conflict_handler="resolve")

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        help="which dataset to use (shakespeare, wikitext, code, owt)",
    )
    parser.add_argument("--start_pc", type=float, default=0.0)
    parser.add_argument("--end_pc", type=float, default=0.9)
    parser.add_argument("--val_start_pc", type=float, default=0.9)
    parser.add_argument("--val_end_pc", type=float, default=1.0)
    parser.add_argument("--block_size", type=int, default=1024)

    # Training arguments
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "base", "medium", "large", "xl"],
    )
    parser.add_argument("--dropout", type=float, default=None)

    # Optimization arguments
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--cosine_anneal", action="store_true")

    # Logging and reproducibility
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--val_interval", type=int, default=100)

    # Strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        default="base",
        choices=["base", "ddp", "fedavg", "sparta", "diloco", "demo", "diloco_sparta"],
        help="Training strategy to use",
    )

    # FedAvg-specific arguments
    parser.add_argument(
        "--H", type=int, default=100, help="FedAvg communication interval"
    )
    parser.add_argument(
        "--island_size", type=int, default=None, help="FedAvg island size"
    )

    # SPARTA-specific arguments
    parser.add_argument(
        "--p_sparta", type=float, default=0.005, help="SPARTA sparsity parameter"
    )
    parser.add_argument(
        "--async_sparta_delay", type=int, default=0, help="SPARTA async delay"
    )
    parser.add_argument(
        "--sparta_interval", type=int, default=1, help="SPARTA communication interval"
    )

    # DiLoCo-specific arguments
    parser.add_argument(
        "--diloco_interval", type=int, default=100, help="DiLoCo communication interval"
    )
    parser.add_argument(
        "--outer_lr", type=float, default=0.7, help="DiLoCo outer learning rate"
    )
    parser.add_argument(
        "--nesterov", type=bool, default=True, help="DiLoCo Nesterov momentum"
    )
    parser.add_argument(
        "--outer_momentum", type=float, default=0.9, help="DiLoCo outer momentum"
    )

    # DeMo-specific arguments
    parser.add_argument(
        "--compression_decay",
        type=float,
        default=0.999,
        help="DeMo gradient error feedback decay",
    )
    parser.add_argument(
        "--compression_topk", type=int, default=32, help="DeMo top-k compression"
    )
    parser.add_argument(
        "--compression_chunk", type=int, default=64, help="DeMo DCT chunk size"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay factor"
    )

    return parser


def create_strategy(args):
    """Create strategy based on args.strategy selection."""

    # Common lr scheduler config
    lr_scheduler_kwargs = {
        "warmup_steps": args.warmup_steps,
        "cosine_anneal": args.cosine_anneal,
    }

    if args.strategy == "ddp" or args.strategy == "base" or args.strategy == "":
        from exogym.strategy.strategy import SimpleReduceStrategy

        optim = OptimSpec(torch.optim.AdamW, lr=args.lr)
        return SimpleReduceStrategy(
            optim_spec=optim,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    elif args.strategy == "fedavg":
        from exogym.strategy.federated_averaging import FedAvgStrategy

        if args.island_size is None:
            args.island_size = args.num_nodes
        optim = OptimSpec(torch.optim.AdamW, lr=args.lr)
        return FedAvgStrategy(
            inner_optim_spec=optim,
            H=args.H,
            island_size=args.island_size,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    elif args.strategy == "sparta":
        from exogym.strategy.sparta import SPARTAStrategy

        optim = OptimSpec(torch.optim.AdamW, lr=args.lr)
        return SPARTAStrategy(
            optim_spec=optim,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
            p_sparta=args.p_sparta,
            async_sparta_delay=args.async_sparta_delay,
        )

    elif args.strategy == "diloco":
        from exogym.strategy.diloco import DiLoCoStrategy

        inner_optim = OptimSpec(torch.optim.AdamW, lr=args.lr)
        outer_optim = OptimSpec(
            torch.optim.SGD,
            lr=args.outer_lr,
            nesterov=args.nesterov,
            momentum=args.outer_momentum,
        )
        return DiLoCoStrategy(
            optim_spec=inner_optim,
            outer_optim_spec=outer_optim,
            H=args.diloco_interval,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    elif args.strategy == "demo":
        from exogym.strategy.demo import DeMoStrategy

        optim = OptimSpec(
            torch.optim.AdamW,
            lr=args.lr,
            compression_decay=args.compression_decay,
            compression_topk=args.compression_topk,
            compression_chunk=args.compression_chunk,
            weight_decay=args.weight_decay,
        )
        return DeMoStrategy(
            optim_spec=optim,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    elif args.strategy == "diloco_sparta":
        from exogym.strategy.sparta_diloco import SPARTADiLoCoStrategy

        inner_optim = OptimSpec(torch.optim.AdamW, lr=args.lr)
        outer_optim = OptimSpec(
            torch.optim.SGD,
            lr=args.outer_lr,
            nesterov=args.nesterov,
            momentum=args.outer_momentum,
        )
        return SPARTADiLoCoStrategy(
            inner_optim_spec=inner_optim,
            outer_optim_spec=outer_optim,
            H=args.diloco_interval,
            p_sparta=args.p_sparta,
            sparta_interval=args.sparta_interval,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


def main():
    parser = arg_parse()
    args = parser.parse_args()

    ## Example of dataset factory for OWT.
    if args.dataset == "owt" or False:

        def dataset_factory(
            rank: int, num_nodes: int, train_dataset: bool
        ) -> torch.utils.data.Dataset:
            if train_dataset:
                start_pc = (
                    rank / num_nodes * (args.end_pc - args.start_pc) + args.start_pc
                )
                end_pc = (rank + 1) / num_nodes * (
                    args.end_pc - args.start_pc
                ) + args.start_pc
            else:
                start_pc = args.val_start_pc
                end_pc = args.val_end_pc

            dataset, _ = get_dataset(
                args.dataset,
                block_size=args.block_size,
                device="cpu",
                start_pc=start_pc,
                end_pc=end_pc,
            )
            return dataset

        train_dataset = dataset_factory
        val_dataset = dataset_factory

        vocab_size = 50257

    else:
        # Get datasets
        train_dataset, vocab_size = get_dataset(
            args.dataset,
            block_size=args.block_size,
            device="cpu",
            start_pc=args.start_pc,
            end_pc=args.end_pc,
        )
        val_dataset, vocab_size = get_dataset(
            args.dataset,
            block_size=args.block_size,
            device="cpu",
            start_pc=args.val_start_pc,
            end_pc=args.val_end_pc,
        )

    # Create model
    gpt_config = GPTConfig.gpt2_size_map(args.model_size)
    if args.dropout is not None:
        gpt_config.dropout = args.dropout
    gpt_config.vocab_size = vocab_size
    model = GPT(gpt_config)

    # Create trainer
    trainer = LocalTrainer(
        model,
        train_dataset,
        val_dataset,
    )

    # Create strategy based on selection
    strategy = create_strategy(args)

    # Train
    trainer.fit(
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        strategy=strategy,
        num_nodes=args.num_nodes,
        device=args.device,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size or args.batch_size,
        shuffle=(args.dataset != "owt"),
        val_size=args.val_size,
        val_interval=args.val_interval,
        # wandb_project=args.wandb_project,
        # run_name=args.run_name or gen_run_name(args, args.strategy)
    )


if __name__ == "__main__":
    main()
