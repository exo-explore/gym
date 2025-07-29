from typing import Optional, Union

from .communicate_optimize_strategy import CommunicateOptimizeStrategy
from .optim import OptimSpec, ensure_optim_spec
from .sparta import SparseCommunicator, RandomIndexSelector
from .diloco import DiLoCoCommunicator
import torch


class SPARTADiLoCoStrategy(CommunicateOptimizeStrategy):
    """
    Strategy that combines SPARTA's sparse communication with DiLoCo's master-worker optimization.

    This strategy:
    1. Performs local optimization
    2. Applies sparse communication every sparta_interval steps (SPARTA)
    3. Applies master-worker optimization every H steps (DiLoCo)
    """

    def __init__(
        self,
        inner_optim_spec: Optional[Union[str, OptimSpec]] = None,
        outer_optim_spec: Optional[Union[str, OptimSpec]] = None,
        p_sparta: float = 0.005,
        sparta_interval: int = 1,
        H: int = 100,
        **kwargs,
    ):
        # Ensure optim_spec is properly initialized
        optim_spec = ensure_optim_spec(
            inner_optim_spec, OptimSpec(torch.optim.AdamW)
        )

        # Create both communication modules
        index_selector = RandomIndexSelector(p_sparta)
        self.sparse_comm = SparseCommunicator(index_selector)
        self.diloco_comm = DiLoCoCommunicator(H=H, outer_optim_spec=outer_optim_spec)
        
        # Store timing parameters
        self.sparta_interval = sparta_interval
        self.H = H

        super().__init__(
            optim_spec=optim_spec,
            communication_modules=[],  # We'll handle communication manually
            **kwargs,
        )

        self.index_selector = index_selector
    
    def _communicate(self):
        """Apply communication modules with different frequencies."""
        # SPARTA sparse communication every sparta_interval steps
        if self.local_step % self.sparta_interval == 0:
            self.sparse_comm.communicate(self.model, self.rank, self.num_nodes, self.local_step)
        
        # DiLoCo master-worker communication every H steps
        if self.local_step % self.H == 0 and self.local_step > 0:
            self.diloco_comm.communicate(self.model, self.rank, self.num_nodes, self.local_step)
    
    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)
        
        # Initialize communication modules
        self.sparse_comm._init_node(model, rank, num_nodes)
        self.diloco_comm._init_node(model, rank, num_nodes)
