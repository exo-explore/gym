import torch
import torch.distributed as dist
import torch.nn.utils as nn_utils
from typing import Optional, Union, Dict

from .strategy import Strategy
from .optim import OptimSpec, ensure_optim_spec
from .communicate import all_reduce


class DGCStrategy(Strategy):
    """
    Deep Gradient Compression strategy for distributed training.
    
    Reduces communication by sparsifying gradients while maintaining convergence
    through momentum correction and residual accumulation.
    """
    
    def __init__(
        self,
        optim_spec: Optional[Union[str, OptimSpec]] = None,
        target_sparsity: float = 0.001,  # 0.1% default
        warmup_steps: int = 500,
        clip_threshold: float = 1.0,
        momentum: float = 0.9,
        max_norm: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Initialize with SGD optimizer as recommended for DGC
        if optim_spec is None:
            optim_spec = OptimSpec(
                torch.optim.SGD,
                lr=0.1,
                momentum=momentum,
            )
        self.optim_spec = ensure_optim_spec(optim_spec)
        
        self.target_sparsity = target_sparsity
        self.warmup_steps = warmup_steps
        self.clip_threshold = clip_threshold
        self.momentum = momentum
        self.max_norm = max_norm
        
        # Persistent buffers for each parameter
        self.v_buffers = {}  # momentum buffers (separate from optimizer's momentum)
        self.r_buffers = {}  # residual buffers
        self.initialized = False

    def _init_node(self, model, rank, num_nodes):
        super()._init_node(model, rank, num_nodes)
        
        self.optim = self.optim_spec.build(model)
        self._setup_scheduler()
        
        # Initialize buffers for each parameter
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.v_buffers[name] = torch.zeros_like(param.data)
                self.r_buffers[name] = torch.zeros_like(param.data)
        self.initialized = True

    def step(self):
        """Perform DGC step with sparse gradient communication."""
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                
                # 1. Get gradient
                g = param.grad.data
                
                # 2. Gradient clipping
                grad_norm = g.norm()
                if grad_norm > self.clip_threshold:
                    g = g * (self.clip_threshold / grad_norm)
                
                # 3. Momentum update before sparsification
                v_t = self.v_buffers[name]
                v_t.mul_(self.momentum).add_(g)
                
                # 4. Add residuals from last round
                v_with_res = v_t + self.r_buffers[name]
                
                # 5. Choose k based on warmup
                numel = v_with_res.numel()
                if self.local_step < self.warmup_steps:
                    k = numel  # No sparsity during warmup
                else:
                    k = max(1, int(numel * self.target_sparsity))
                
                # 6. Top-k selection and communication
                if k < numel and self.num_nodes > 1:
                    # Get top-k values and indices
                    topk_vals, topk_idx = torch.topk(
                        v_with_res.abs().view(-1), k, sorted=False
                    )
                    
                    # Create mask
                    mask = torch.zeros_like(v_with_res, dtype=torch.bool).view(-1)
                    mask[topk_idx] = True
                    mask = mask.view(v_with_res.shape)
                    
                    # 7. Extract sparse values for communication
                    sparse_grad = v_with_res[mask].clone()
                    
                    # 8. Update residuals (keep dropped components)
                    self.r_buffers[name] = v_with_res.clone()
                    self.r_buffers[name][mask] = 0
                    
                    # 9. Momentum factor masking
                    v_t[~mask] = 0
                    
                    # 10. All-reduce sparse values
                    all_reduce(sparse_grad, op=dist.ReduceOp.SUM)
                    sparse_grad /= self.num_nodes
                    
                    # Set gradient to sparse communicated values
                    param.grad.data.zero_()
                    param.grad.data[mask] = sparse_grad
                    
                else:
                    # During warmup or if k >= numel, do dense communication
                    if self.num_nodes > 1:
                        all_reduce(v_with_res, op=dist.ReduceOp.SUM)
                        v_with_res /= self.num_nodes
                    param.grad.data = v_with_res
                    
                    # Clear residuals during dense communication
                    self.r_buffers[name].zero_()
        
        # Apply max norm clipping if specified
        if self.max_norm:
            nn_utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
        
        # Optimizer step with the sparse/compressed gradients
        self.optim.step()
        
        # Call parent step to handle scheduler and logging
        super().step()