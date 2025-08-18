#!/usr/bin/env python3
"""
Image classification training using EXO Gym distributed training framework.
Supports ConvNeXt V2 Nano on ImageNet-100 with various distributed strategies.

Usage:
    python example/image_train.py --strategy diloco --num_nodes 4
    python example/image_train.py --strategy sparta --num_nodes 8 --p_sparta 0.01
"""

import argparse
import torch
import torch.nn as nn
import timm
import random
from functools import partial
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset

from exogym.trainer import Trainer
from exogym.strategy.optim import OptimSpec


# Adapter class must be top-level to be picklable
class HFDataset(torch.utils.data.Dataset):
    """Wrap a Hugging Face split to behave like a torchvision dataset."""

    def __init__(self, hf_ds, tf):
        self.ds = hf_ds
        self.tf = tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        img = ex["image"].convert("RGB")
        return self.tf(img), ex["label"]


def image_dataset_factory(subset: float, seed: int, batch_size: int, 
                         rank: int, num_nodes: int, train_dataset: bool) -> torch.utils.data.Dataset:
    """Dataset factory function for ImageNet-100 dataset."""
    
    if train_dataset:
        tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ds = load_dataset("clane9/imagenet-100", split="train")
    else:
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        ds = load_dataset("clane9/imagenet-100", split="validation")
    
    dataset = HFDataset(ds, tf)
    
    # Apply subset for training data
    if train_dataset and subset < 1.0:
        random.seed(seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = Subset(dataset, indices[:int(len(indices) * subset)])
    
    # For distributed training, split dataset across nodes
    if train_dataset and num_nodes > 1:
        total_samples = len(dataset)
        samples_per_node = total_samples // num_nodes
        start_idx = rank * samples_per_node
        end_idx = start_idx + samples_per_node if rank < num_nodes - 1 else total_samples
        
        if hasattr(dataset, 'indices'):  # Subset case
            node_indices = dataset.indices[start_idx:end_idx]
            dataset = Subset(dataset.dataset, node_indices)
        else:  # Full dataset case
            indices = list(range(start_idx, end_idx))
            dataset = Subset(dataset, indices)
    
    return dataset


def gen_run_name(args, strategy):
    """Generate wandb name based on strategy and arguments."""
    base_name = f"bs{args.batch_size}_lr{args.lr:.0e}_subset{args.subset}"

    if strategy == "ddp":
        return f"image_ddp_{base_name}_n{args.num_nodes}"
    elif strategy == "fedavg":
        return f"image_fedavg_{base_name}_H{args.H}_n{args.num_nodes}"
    elif strategy == "sparta":
        return f"image_sparta_p{args.p_sparta}_n{args.num_nodes}_lr{args.lr:.0e}"
    elif strategy == "diloco":
        return f"image_diloco_{base_name}_outer{args.outer_lr:.0e}_H{args.H}"
    elif strategy == "demo":
        return f"image_demo_{base_name}_topk{args.compression_topk}_decay{args.compression_decay}"
    elif strategy == "diloco_sparta":
        return f"image_diloco_sparta_{base_name}_outer{args.outer_lr:.0e}_H{args.H}_p{args.p_sparta}"
    elif strategy == "dgc":
        return f"image_dgc_{base_name}_sparsity{args.dgc_sparsity}_warmup{args.dgc_warmup_steps}"
    else:
        return f"image_{base_name}"


def arg_parse():
    """Create parser with all arguments for all strategies."""
    parser = argparse.ArgumentParser(conflict_handler="resolve")

    # Dataset arguments
    parser.add_argument("--subset", type=float, default=1.0, help="Train subset fraction (0 < s ≤ 1)")
    parser.add_argument("--seed", type=int, default=0)

    # Training arguments
    parser.add_argument("--num_nodes", type=int, default=4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="convnextv2_tiny", 
                       help="timm model name")

    # Optimization arguments
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--minibatch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--cosine_anneal", action="store_true", default=True)

    # Logging and reproducibility
    parser.add_argument("--wandb_project", type=str, default='image_exogym')
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--val_size", type=int, default=512)
    parser.add_argument("--val_interval", type=int, default=200)
    parser.add_argument("--correlation_interval", type=int, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    # Strategy selection
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        choices=["base", "ddp", "fedavg", "sparta", "diloco", "demo", "diloco_sparta", "dgc"],
        help="Training strategy to use",
    )

    # FedAvg-specific arguments
    parser.add_argument("--H", type=int, default=100, help="FedAvg communication interval")
    parser.add_argument("--island_size", type=int, default=None, help="FedAvg island size")

    # SPARTA-specific arguments
    parser.add_argument("--p_sparta", type=float, default=0.005, help="SPARTA sparsity parameter")
    parser.add_argument("--async_sparta_delay", type=int, default=0, help="SPARTA async delay")
    parser.add_argument("--sparta_interval", type=int, default=1, help="SPARTA communication interval")

    # DiLoCo-specific arguments
    parser.add_argument("--outer_lr", type=float, default=0.7, help="DiLoCo outer learning rate")
    parser.add_argument("--nesterov", type=bool, default=True, help="DiLoCo Nesterov momentum")
    parser.add_argument("--outer_momentum", type=float, default=0.9, help="DiLoCo outer momentum")

    # DeMo-specific arguments
    parser.add_argument("--compression_decay", type=float, default=0.999, 
                       help="DeMo gradient error feedback decay")
    parser.add_argument("--compression_topk", type=int, default=32, help="DeMo top-k compression")
    parser.add_argument("--compression_chunk", type=int, default=64, help="DeMo DCT chunk size")
    
    # DGC-specific arguments
    parser.add_argument("--dgc_sparsity", type=float, default=0.001, 
                       help="DGC target sparsity (0.001 = 0.1%)")
    parser.add_argument("--dgc_warmup_steps", type=int, default=500, 
                       help="DGC warmup steps before sparsification")
    parser.add_argument("--dgc_clip_threshold", type=float, default=1.0, 
                       help="DGC gradient clipping threshold")
    parser.add_argument("--dgc_momentum", type=float, default=0.9, help="DGC momentum factor")

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

        optim = OptimSpec(torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay)
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
        optim = OptimSpec(torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay)
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

        optim = OptimSpec(torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay)
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

        inner_optim = OptimSpec(torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay)
        outer_optim = OptimSpec(
            torch.optim.SGD,
            lr=args.outer_lr,
            nesterov=args.nesterov,
            momentum=args.outer_momentum,
        )
        return DiLoCoStrategy(
            optim_spec=inner_optim,
            outer_optim_spec=outer_optim,
            H=args.H,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    elif args.strategy == "demo":
        from exogym.strategy.demo import DeMoStrategy

        optim = OptimSpec(
            torch.optim.AdamW,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return DeMoStrategy(
            optim_spec=optim,
            compression_decay=args.compression_decay,
            compression_topk=args.compression_topk,
            compression_chunk=args.compression_chunk,
            weight_decay=args.weight_decay,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    elif args.strategy == "diloco_sparta":
        from exogym.strategy.sparta_diloco import SPARTADiLoCoStrategy

        inner_optim = OptimSpec(torch.optim.AdamW, lr=args.lr, weight_decay=args.weight_decay)
        outer_optim = OptimSpec(
            torch.optim.SGD,
            lr=args.outer_lr,
            nesterov=args.nesterov,
            momentum=args.outer_momentum,
        )
        return SPARTADiLoCoStrategy(
            inner_optim_spec=inner_optim,
            outer_optim_spec=outer_optim,
            H=args.H,
            p_sparta=args.p_sparta,
            sparta_interval=args.sparta_interval,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )
        
    elif args.strategy == "dgc":
        from exogym.strategy.dgc import DGCStrategy
        
        # DGC works best with SGD + momentum
        optim = OptimSpec(
            torch.optim.SGD, 
            lr=args.lr,
            momentum=args.dgc_momentum,
        )
        return DGCStrategy(
            optim_spec=optim,
            target_sparsity=args.dgc_sparsity,
            warmup_steps=args.dgc_warmup_steps,
            clip_threshold=args.dgc_clip_threshold,
            momentum=args.dgc_momentum,
            lr_scheduler="lambda_cosine",
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            max_norm=args.max_norm,
        )

    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


class ImageClassificationModel(nn.Module):
    """Wrapper around timm model to provide loss computation."""
    
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, batch):
        """Forward pass that returns loss for training."""
        x, y = batch
        logits = self.model(x)
        return self.loss_fn(logits, y)


def main():
    parser = arg_parse()
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Get number of classes from dataset
    ds_train = load_dataset("clane9/imagenet-100", split="train")
    num_classes = ds_train.features["label"].num_classes
    
    # Create dataset factory
    dataset_factory = partial(
        image_dataset_factory,
        args.subset,
        args.seed,
        args.batch_size
    )
    
    # Create model
    model = ImageClassificationModel(args.model_name, num_classes)
    
    # Create trainer
    trainer = Trainer(
        model,
        dataset_factory,  # train_dataset
        dataset_factory,  # val_dataset
    )
    
    # Create strategy based on selection
    strategy = create_strategy(args)
    
    # Define dataloader kwargs for better GPU utilization
    dataloader_kwargs = {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
    }
    
    # Train
    trainer.fit(
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        strategy=strategy,
        num_nodes=args.num_nodes,
        device=args.device,
        batch_size=args.batch_size,
        minibatch_size=args.minibatch_size or args.batch_size,
        shuffle=True,
        val_size=args.val_size,
        val_interval=args.val_interval,
        correlation_interval=args.correlation_interval,
        save_dir=args.save_dir,
        wandb_project=args.wandb_project,
        run_name=args.run_name or gen_run_name(args, args.strategy),
        dataloader_kwargs=dataloader_kwargs,
    )


if __name__ == "__main__":
    main()