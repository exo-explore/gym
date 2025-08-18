import torch.distributed as dist
from copy import deepcopy

from torch.nn import utils as nn_utils
import torch

from typing import Optional, Union

from .strategy import Strategy
from .communicate_optimize_strategy import CommunicateOptimizeStrategy, CommunicationModule
from .optim import OptimSpec, ensure_optim_spec
from .communicate import all_reduce, broadcast


class DiLoCoCommunicator(CommunicationModule):
    """
    Communication module for master-worker setup (like DiLoCo).
    """
    
    def __init__(self, 
                 H: int = 100, 
                 outer_optim_spec: Optional[Union[str, OptimSpec]] = None,
                 **kwargs):
        self.H = H
        self.outer_optim_spec = ensure_optim_spec(
            outer_optim_spec, OptimSpec(torch.optim.SGD, lr=0.7, nesterov=True, momentum=0.9)
        )
        self.strategy = None  # Will be set by CommunicateOptimizeStrategy
        self.master_model = None
        self.outer_optimizer = None
    
    def communicate(self, model, rank: int, num_nodes: int, local_step: int) -> None:
        """Perform master-worker communication."""
        if num_nodes > 1 and local_step % self.H == 0 and local_step > 0:
            # First average all models
            for param in model.parameters():
                all_reduce(param.data, op=dist.ReduceOp.SUM)
                param.data /= num_nodes

            # Master does outer optimization step
            if rank == 0 and self.master_model is not None:
                self.outer_optimizer.zero_grad()
                self._set_master_grad(model)
                self.outer_optimizer.step()
                self._synchronize_master_model(model)

            # Broadcast updated parameters
            for param in model.parameters():
                broadcast(param.data, src=0)
    
    def _init_node(self, model, rank: int, num_nodes: int) -> None:
        """Initialize master model for rank 0."""
        if rank == 0:
            self.master_model = deepcopy(model).to("cpu")
            for param in self.master_model.parameters():
                param.requires_grad = True
            self.outer_optimizer = self.outer_optim_spec.build(self.master_model)
    
    def _set_master_grad(self, model) -> None:
        """Set gradients on master model based on difference between master and worker models."""
        for name, param in self.master_model.named_parameters():
            param.grad = param.data - model.state_dict()[name].data.to("cpu")
    
    def _synchronize_master_model(self, model) -> None:
        """Synchronize worker model with master model parameters."""
        for name, param in model.named_parameters():
            param.data = self.master_model.state_dict()[name].data.to(param.device)


class DiLoCoStrategy(CommunicateOptimizeStrategy):
    def __init__(
        self,
        optim_spec: Optional[Union[str, OptimSpec]] = None, # inner optimizer is named optim_spec for consistency
        outer_optim_spec: Optional[Union[str, OptimSpec]] = None,
        H: int = 100,
        **kwargs,
    ):
        self.H = H
        
        # Ensure optim_spec is properly initialized
        optim_spec = ensure_optim_spec(
            optim_spec, OptimSpec(torch.optim.AdamW)
        )
        
        # Create the DiLoCo communicator
        self.diloco_comm = DiLoCoCommunicator(H=H, outer_optim_spec=outer_optim_spec)
        
        super().__init__(
            optim_spec=optim_spec,
            communication_modules=[self.diloco_comm],
            **kwargs
        )
