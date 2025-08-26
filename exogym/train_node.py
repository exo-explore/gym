import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np

import copy
from typing import Union, Callable

from exogym.common import TrainConfig
from exogym.strategy.strategy import Strategy
from exogym.aux import Logger, WandbLogger, CSVLogger
from exogym.strategy.communicate import all_reduce, broadcast
from exogym.aux import LogModule, CheckpointMixin, CorrelationMixin

class TrainNode(LogModule, CheckpointMixin, CorrelationMixin):
    """
    Single node of distributed training process. Should be the same regardless of rank topology/architecture.
    """

    def __init__(self, config: TrainConfig):
        self.config = config

        seed = config.kwargs.get("seed", 42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model = config.model

        self.num_nodes = config.num_nodes
        self.rank = config.rank
        self.device = config.device

        self.strategy = config.strategy

        self.num_epochs = config.num_epochs
        self.max_steps = config.max_steps
        self.batch_size = config.batch_size
        self.minibatch_size = config.minibatch_size
        self.val_size = config.val_size
        self.val_interval = config.val_interval
        self.autocast = config.autocast
        self.checkpoint_interval = config.checkpoint_interval

        # if train_dataset is a pure dataset, we need a sampler.
        # Otherwise, it's a factory function - so no need :)
        if not callable(config.train_dataset):
            self.train_dataset = config.train_dataset
            self.train_sampler = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=self.num_nodes,
                rank=self.rank,
                shuffle=config.shuffle,
            )
        else:
            self.train_dataset = config.train_dataset(self.rank, self.num_nodes, train_dataset=True)
            self.train_sampler = None

        if not callable(config.val_dataset):
            self.val_dataset = config.val_dataset
        else:
            self.val_dataset = config.val_dataset(self.rank, self.num_nodes, train_dataset=False)

        self.kwargs = config.kwargs

        self.build_dataloaders()

        ## Ensure all process models share the same params
        if self.num_nodes > 1:
            for _, param in self.model.named_parameters():
                broadcast(param.data, src=0)

        self.local_step = 0
        self.epoch = 0

        # Attempt to load checkpoint before starting training
        # self._load_checkpoint()

    def build_dataloaders(self):
        """
        Builds dataloaders.
        """
        # For dataset factory case (when sampler is None), we can enable shuffling
        # For regular dataset case (when sampler is provided), shuffling is handled by the sampler
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.minibatch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            **self.config.dataloader_kwargs,
        )

        self.val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size=self.minibatch_size,
            shuffle=True,
            **self.config.dataloader_kwargs,
        )

        self.train_data_iter = iter(self.train_dataloader)
        self.val_data_iter = iter(self.val_dataloader)

    def _get_batch(self, eval=False):
        if not eval or self.val_data_iter is None:
            try:
                batch = next(self.train_data_iter)
            except StopIteration:
                self.epoch += 1
                self.train_data_iter = iter(self.train_dataloader)
                batch = next(self.train_data_iter)
        else:
            try:
                batch = next(self.val_data_iter)
            except StopIteration:
                self.val_data_iter = iter(self.val_dataloader)
                batch = next(self.val_data_iter)

        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = tuple(x.to(self.device) for x in batch)
        else:
            batch = batch.to(self.device)

        return batch

    def _train_step(self):
        self.strategy.zero_grad()

        grad_accumulation_steps = self.batch_size // self.minibatch_size
        assert grad_accumulation_steps >= 1, f"Gradient accumulation steps must be >= 1, but got batch_size={self.batch_size} // minibatch_size={self.minibatch_size} = {grad_accumulation_steps}"

        for i in range(grad_accumulation_steps):
            minibatch = self._get_batch()

            ## TODO: Do we want this?
            if self.autocast:
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    loss = self.model(minibatch)
            else:
                loss = self.model(minibatch)

            loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad /= grad_accumulation_steps

        self.strategy.step()

        if self.rank == 0:
            self.logger.log_examples_trained(examples=self.batch_size * self.num_nodes)
            self.logger.log_train(loss=loss.item())

        if self.checkpoint_interval and self.local_step % self.checkpoint_interval == 0:
            self._save_checkpoint()

    def _evaluate(self):
        if self.val_size == 0:
            return

        model_clone = copy.deepcopy(self.model)

        for name, param in model_clone.named_parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data = param.data / dist.get_world_size()

        if self.rank == 0:
            # For rank 0, we will calculate the local loss
            this_model = self.model

        if self.rank == 1:
            # For rank 1, we want to calculate the average model loss
            this_model = model_clone

        if self.rank == 0 or self.rank == 1:
            this_model.eval()

            loss_total = 0

            with torch.no_grad():
                for _ in range(int(self.val_size / self.batch_size)):

                    for i in range(self.batch_size // self.minibatch_size):
                        minibatch = self._get_batch(eval=True)

                        if self.autocast:
                            with torch.autocast(
                                device_type=self.device, dtype=torch.bfloat16
                            ):
                                ## TODO: Fix
                                loss = this_model(minibatch)
                        else:
                            loss = this_model(minibatch)

                        loss_total += loss.item() / (
                            self.batch_size // self.minibatch_size
                        )

        # Rank 0 logs the local evaluation.
        if self.rank == 0:
            self.logger.log_loss(
                loss=loss_total / int(self.val_size / self.batch_size), name="local"
            )

        # Broadcast the global loss from rank 1 to all ranks.
        if self.num_nodes > 1:
            # All ranks create a dummy tensor to participate.
            global_loss_tensor = torch.empty(
                1, device=next(self.model.parameters()).device
            )
            if self.rank == 1:
                global_loss_tensor[0] = loss_total / int(
                    self.val_size / self.batch_size
                )
            broadcast(global_loss_tensor, src=1)

            # Only rank 0 logs the global evaluation.
            if self.rank == 0:
                global_loss = global_loss_tensor.item()
                self.logger.log_loss(loss=global_loss, name="global")

        del model_clone


    def train(self):
        if self.max_steps is None:
            self.max_steps = (
                self.num_epochs
                * len(self.train_dataloader)
                / (self.batch_size // self.minibatch_size)
            )

        self.strategy.max_steps = self.max_steps

        if self.rank == 0:
            if self.kwargs.get("disable_logging", False):
                # Use base Logger for no-op logging during profiling
                self.logger = Logger(
                    model=self.model,
                    max_steps=self.max_steps,
                    strategy=self.strategy,
                    train_node=self,
                    init_tqdm=False,
                )
            elif self.kwargs.get("wandb_project", None) is not None:
                self.logger = WandbLogger(
                    model=self.model,
                    max_steps=self.max_steps,
                    strategy=self.strategy,
                    train_node=self,
                    wandb_project=self.kwargs.get("wandb_project", None),
                    run_name=self.kwargs.get("run_name", None),
                    x_axis=self.kwargs.get("log_x_axis", "step"),
                )
            else:
                self.logger = CSVLogger(
                    model=self.model,
                    max_steps=self.max_steps,
                    strategy=self.strategy,
                    train_node=self,
                    run_name=self.kwargs.get("run_name", None),
                )

        while self.local_step < self.max_steps:
            if self.local_step % self.val_interval == 0:
                self._evaluate()

            self._train_step()

            self.local_step += 1
            if self.rank == 0:
                self.logger.increment_step()

            # Calculate correlation if interval is set and it's time
            if self.config.correlation_interval and self.local_step > 0 and self.local_step % self.config.correlation_interval == 0:
                correlation_value = self._correlation_calculation()
                if self.rank == 0:
                    self.logger.log_info(correlation_value, 'correlation')

            dist.barrier()

        self._evaluate()

        # if self.config.checkpoint_interval is not None:
        #     self._save_checkpoint()

        # Return the final model state dict
        return self.model.state_dict()

    def __config__(self):
        remove_keys = ["model", "train_dataloader", "val_dataloader", "strategy"]

        config = super().__config__(remove_keys=remove_keys)

        return config
