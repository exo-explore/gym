from abc import abstractmethod
from typing import Optional
import torch
import argparse

from exogym.strategy.communicate import broadcast, all_reduce, all_gather
from exogym.strategy.strategy import SimpleReduceStrategy, Strategy
from exogym.trainer import Trainer
from exogym.strategy.optim import OptimSpec
from exogym.strategy.sparta import RandomIndexSelector
from exogym.aux.utils import get_device

from nanogpt import GPT, GPTConfig, get_dataset

NUM_NODES = 4

### PLAYGROUND
### This is a minimal configuration for training a nanogpt model with a given strategy.
### The strategy can be swapped out for custom logic by writing a new strategy class.

class IndexSelector:
    def __init__(self, p):
        self.state = {}
        self.p = p

    @abstractmethod
    def get_indices(self, param, iteration):
        ...


class RandomIndexSelector(IndexSelector):
    def get_indices(self, param, iteration):
        return torch.bernoulli(
            torch.full(param.shape, self.p, device=param.device)
        ).bool()

class SPARTAStrategy(Strategy):
    def __init__(
        self,
        optim_spec: Optional[str | OptimSpec] = None,
        p_sparta=0.005,
        **kwargs,
    ):

        index_selector = RandomIndexSelector(p_sparta)

        super().__init__(**kwargs)
        
        self.optim_spec = optim_spec if isinstance(optim_spec, OptimSpec) else OptimSpec.from_str(optim_spec)
        self.index_selector = index_selector

    def step(self, ):
        with torch.no_grad():
            for param in self.model.parameters():
                if not param.requires_grad or param.grad is None:
                    continue

                indices_mask = self.index_selector.get_indices(
                    param, self.local_step
                )

                broadcast(indices_mask, src=0)
                sparse_data = param.data[indices_mask]
                
                all_reduce(sparse_data, op=torch.distributed.ReduceOp.SUM)
                sparse_data /= self.num_nodes

                param.masked_scatter_(indices_mask, sparse_data)
    
        self.optim.step()
        super().step()

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, default="owt")
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

    gpt_config = GPTConfig.gpt2_sbase()
    gpt_config.vocab_size = vocab_size
    model = GPT(gpt_config)

    # Create trainer
    trainer = Trainer(
        model,
        train_dataset,
        val_dataset,
    )

    ## STRATEGY - This is where we define custom logic

    # to default back to data parallel training:
    # strategy = SimpleReduceStrategy(
    #     optim_spec=OptimSpec(torch.optim.AdamW, lr=0.0004),
    #     lr_scheduler="lambda_cosine",
    #     lr_scheduler_kwargs={
    #         "warmup_steps": 1000,
    #         "cosine_anneal": True,
    #     },
    #     max_norm=1.0,
    # )

    strategy = SPARTAStrategy(
        optim_spec=OptimSpec(torch.optim.AdamW, lr=0.0004),
        lr_scheduler="lambda_cosine",
        lr_scheduler_kwargs={
            "warmup_steps": 1000,
            "cosine_anneal": True,
        },
        max_norm=1.0,
        p=0.005,
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
        wandb_project='sparta',
        run_name=f'sparta-run'
    )


if __name__ == "__main__":
    main()
