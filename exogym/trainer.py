import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from exogym.train_node import TrainNode
from exogym.strategy import Strategy
from exogym.common import TrainConfig
from exogym.aux.utils import print_dataset_size, _average_model_states

import os
from abc import abstractmethod
import copy
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Union, Callable
from collections import OrderedDict


def _build_connection(config: TrainConfig):
    """
    This is the default callback for setting up pytorch distributed connections.
    All ranks are assumed to be on the same machine, and device is defaulted to cpu.
    In future, this can be swapped out assuming non-localhost connections, etc.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(config.port)

    if config.device == "" or config.device is None:
        if torch.cuda.is_available():
            config.device = "cuda"
        elif torch.backends.mps.is_available():
            config.device = "mps"
        else:
            config.device = "cpu"

    # initialize the process group
    if config.device == "cuda":
        # If we haven't specified devices, use all devices.
        if config.devices is None:
            config.devices = range(torch.cuda.device_count())

        dist.init_process_group(
            "nccl" if len(config.devices) == config.num_nodes else "gloo",
            rank=config.rank,
            world_size=config.num_nodes,
        )
        config.device = torch.device(
            f"cuda:{config.devices[config.rank % len(config.devices)]}"
        )
        torch.cuda.set_device(config.device)
    elif config.device == "cpu":
        dist.init_process_group("gloo", rank=config.rank, world_size=config.num_nodes)
        config.device = torch.device("cpu")
    elif config.device == "mps":
        dist.init_process_group("gloo", rank=config.rank, world_size=config.num_nodes)
        config.device = torch.device("mps")
    else:
        raise ValueError(f"Invalid device type: {config.device}")

    print(f"Rank {config.rank} using device {config.device}")

def _worker(rank: int, config: TrainConfig, result_queue: mp.Queue):
    """
    Entry point executed in every child process.
    This function is importable as exogym.trainer._worker, making it notebook-safe.
    """
    config.rank = rank    

    _build_connection(config)

    # TODO: Should these happen here or in TrainNode.__init__() ?
    config.model = copy.deepcopy(config.model).to(config.device)
    config.strategy = copy.deepcopy(config.strategy)
    config.strategy._init_node(config.model, config.rank, config.num_nodes)

    train_node = TrainNode(config=config)
    final_model_state = train_node.train()

    # Move tensors to CPU and detach to avoid CUDA serialization issues
    cpu_state_dict = OrderedDict()
    for key, tensor in final_model_state.items():
        cpu_state_dict[key] = tensor.detach().cpu()

    result_queue.put((rank, cpu_state_dict))

    dist.destroy_process_group()

class Trainer:
    """
    Trainer is used to train a model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Union[
            torch.utils.data.Dataset,
            Callable[[int, int, bool], torch.utils.data.Dataset],
        ],
        val_dataset: Union[
            torch.utils.data.Dataset,
            Callable[[int, int, bool], torch.utils.data.Dataset],
        ],
        start_port: Optional[int] = None,
    ):
        self.model_orig = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.port = start_port if start_port is not None else 12355

    def fit(
        self,
        num_epochs: int,
        strategy: Strategy,
        num_nodes: int,
        max_steps: int = None,
        device: str = None,
        devices: list[int] = None,
        batch_size: int = 16,
        minibatch_size: int = None,
        shuffle: bool = True,
        val_size: int = 64,
        val_interval: int = 100,
        autocast: bool = False,
        checkpoint_interval: Optional[int] = None,
        correlation_interval: Optional[int] = None,
        save_dir: str = "./checkpoints",
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # assert val_size // batch_size > 0, f"val_size must be geq batch_size: {val_size} // {batch_size}"
        assert batch_size > 0, 'local batch size needs to be nonzero'
        if minibatch_size is not None:
            assert minibatch_size <= batch_size, f'minibatch_size ({minibatch_size}) must be <= batch_size ({batch_size}) for gradient accumulation'

        # Move a *copy* of the model to CPU so that pickling for mp.spawn does not attempt to share GPU storage.
        cpu_model = copy.deepcopy(self.model_orig).cpu()

        self.config = TrainConfig(
            model=cpu_model,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            strategy=strategy,
            num_epochs=num_epochs,
            num_nodes=num_nodes,
            max_steps=max_steps,
            port=self.port,
            device=device,
            devices=devices,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            shuffle=shuffle,
            val_size=val_size,
            val_interval=val_interval,
            autocast=autocast,
            checkpoint_interval=checkpoint_interval,
            correlation_interval=correlation_interval,
            save_dir=save_dir,
            dataloader_kwargs=dataloader_kwargs or {},
            kwargs=kwargs,
        )

        # Auto-detect minibatch_size if not provided
        if minibatch_size is None:
            minibatch_size = self.find_minibatch_size(
                num_nodes,
                batch_size,
            )
            self.config.minibatch_size = minibatch_size

        self.port += 1

        
        manager = mp.Manager()
        result_queue = manager.Queue()

        mp.spawn(
            _worker,
            args=(self.config, result_queue),
            nprocs=self.config.num_nodes,
            start_method="spawn",
            join=True,
        )

        model_states = {}
        for _ in range(self.config.num_nodes):
            rank, state_dict = result_queue.get()
            model_states[rank] = state_dict

        averaged_state_dict = _average_model_states(model_states)

        final_model = copy.deepcopy(self.model_orig)
        final_model.load_state_dict(averaged_state_dict)
        return final_model

    def find_minibatch_size(self, num_nodes: int, batch_size: int):
        import psutil
        import gc

        print(f'Profiling system & training to find optimal minibatch size...')
        
        # Calculate available memory based on device type
        if torch.cuda.is_available():
            # For CUDA, get total memory across all available GPUs
            num_gpus = torch.cuda.device_count()
            if self.config.devices is not None:
                num_gpus = len(self.config.devices)
            
            # Get memory for a single GPU (assuming homogeneous)
            single_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = single_gpu_memory * num_gpus
            memory_type = "GPU"
            
            print(f"Detected {num_gpus} GPU(s) with {single_gpu_memory / (1024**3):.2f} GB each")
            print(f"Total GPU memory available: {available_memory / (1024**3):.2f} GB")
        else:
            # For CPU/MPS, use system memory
            available_memory = psutil.virtual_memory().available
            memory_type = "system"
            print(f"Using {memory_type} memory: {available_memory / (1024**3):.2f} GB available")
        
        search_minibatch = batch_size
        config = copy.deepcopy(self.config)
        config.num_nodes = 1
        config.rank = 0  # Single rank for profiling
        config.max_steps = 3
        
        # Set up device properly for profiling
        if config.device == "" or config.device is None:
            if torch.cuda.is_available():
                config.device = "cuda:0"
            elif torch.backends.mps.is_available():
                config.device = "mps"
            else:
                config.device = "cpu"
        
        # Move model to the correct device for profiling
        config.model = config.model.to(config.device)
        
        # Initialize strategy for the profiling node
        config.strategy = copy.deepcopy(config.strategy)
        config.strategy._init_node(config.model, config.rank, config.num_nodes)
        
        # Initialize a temporary process group for profiling
        need_cleanup = False
        if not dist.is_initialized():
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(self.port + 100)  # Use different port to avoid conflicts
            
            if "cuda" in str(config.device):
                backend = "nccl"
            else:
                backend = "gloo"
            
            dist.init_process_group(backend, rank=0, world_size=1)
            need_cleanup = True
        
        try:
            while search_minibatch > 0:
                config.minibatch_size = search_minibatch
                
                try:
                    # Clear cache and collect garbage before testing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Record memory before training
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        mem_before = torch.cuda.memory_allocated()
                    
                    # Create and run train node
                    train_node = TrainNode(config)
                    train_node.train()
                    
                    # Measure peak memory usage
                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated()
                        actual_usage = peak_memory - mem_before
                    else:
                        # For CPU/MPS, estimate based on process memory
                        process = psutil.Process()
                        actual_usage = process.memory_info().rss
                    
                    # Calculate total memory needed for all nodes
                    total_memory_needed = actual_usage * num_nodes
                    
                    # Check if this fits in available memory (with 10% safety margin)
                    if total_memory_needed < available_memory * 0.9:
                        print(f"Found suitable minibatch_size={search_minibatch}")
                        print(f"Estimated memory per node: {actual_usage / (1024**3):.2f} GB")
                        print(f"Total for {num_nodes} nodes: {total_memory_needed / (1024**3):.2f} GB")
                        print(f"Available {memory_type} memory: {available_memory / (1024**3):.2f} GB")
                        return search_minibatch
                    else:
                        print(f"minibatch_size={search_minibatch} would use {total_memory_needed / (1024**3):.2f} GB, reducing...")
                        
                except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                    # Handle OOM errors gracefully for CUDA, MPS, and CPU
                    if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                        print(f"OOM with minibatch_size={search_minibatch}: {str(e)}")
                        
                        # Clear cache to recover
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                            # MPS doesn't have explicit cache clearing yet, but gc helps
                            pass
                        gc.collect()
                    else:
                        # Re-raise if it's not a memory-related RuntimeError
                        raise
                    
                # Reduce minibatch size and try again
                search_minibatch = search_minibatch // 2
            
            raise Exception(f'Cannot find suitable minibatch size for device {config.device}')
            
        finally:
            # Clean up the temporary process group if we created one
            if need_cleanup:
                dist.destroy_process_group() 
