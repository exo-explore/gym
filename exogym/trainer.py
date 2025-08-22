import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from exogym.train_node import TrainNode
from exogym.strategy import Strategy
from exogym.common import TrainConfig
from exogym.aux.utils import print_dataset_size, _average_model_states

import os
import time
import threading
import psutil
import gc
from abc import abstractmethod
import copy
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Union, Callable
from collections import OrderedDict


def _get_mps_allocated_bytes():
    """Get MPS allocated memory in bytes, defensively handling API variations."""
    # Be defensive about PyTorch versions / API surface.
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        for name in ("current_allocated_memory", "allocated_memory", "memory_allocated"):
            fn = getattr(torch.mps, name, None)
            if callable(fn):
                try:
                    return int(fn())
                except Exception:
                    pass
    return 0


class _PeakMemSampler:
    """Background thread that samples RSS and MPS memory at high frequency to track peak usage."""
    
    def __init__(self, include_mps: bool):
        self.proc = psutil.Process(os.getpid())
        self.include_mps = include_mps and hasattr(torch, "mps") and torch.backends.mps.is_available()
        self._stop = threading.Event()
        self.peak_rss = self._rss_now()
        self.peak_mps = 0

    def _rss_now(self) -> int:
        """Get current RSS usage for parent + any dataloader workers, etc."""
        rss = self.proc.memory_info().rss
        for c in self.proc.children(recursive=True):
            try:
                rss += c.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return rss

    def start(self, period_s: float = 0.01):
        """Start the background sampling thread at ~100 Hz."""
        self._thr = threading.Thread(target=self._run, args=(period_s,), daemon=True)
        self._thr.start()

    def _run(self, period_s: float):
        """Background thread that continuously samples memory usage."""
        while not self._stop.is_set():
            rss = self._rss_now()
            if rss > self.peak_rss:
                self.peak_rss = rss
            if self.include_mps:
                try:
                    m = _get_mps_allocated_bytes()
                    if m > self.peak_mps:
                        self.peak_mps = m
                except Exception:
                    pass
            time.sleep(period_s)

    def stop(self):
        """Stop sampling and take final measurements after synchronization."""
        self._stop.set()
        # Final sync/samples to catch last kernels
        if self.include_mps:
            try:
                torch.mps.synchronize()
                m = _get_mps_allocated_bytes()
                if m > self.peak_mps:
                    self.peak_mps = m
            except Exception:
                pass
        # One last RSS sample
        rss = self._rss_now()
        if rss > self.peak_rss:
            self.peak_rss = rss
        self._thr.join()


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
        device: str = None,
        devices: list[int] = None,
    ):
        self.model_orig = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.port = start_port if start_port is not None else 12355
        self.device = device
        self.devices = devices
        self._minibatch_cache = {}

    def fit(
        self,
        num_epochs: int,
        strategy: Strategy,
        num_nodes: int,
        max_steps: int = None,
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
            device=self.device,
            devices=self.devices,
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
            force_recalculate = kwargs.get('force_minibatch_recalculate', False)
            minibatch_size = self.find_minibatch_size(
                num_nodes,
                batch_size,
                force_recalculate=force_recalculate,
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

    def clear_minibatch_cache(self):
        """Clear the cached minibatch size results."""
        self._minibatch_cache.clear()
        print("Minibatch size cache cleared.")
    
    def find_minibatch_size(self, num_nodes: int, batch_size: int, force_recalculate: bool = False):
        cache_key = (num_nodes, batch_size)
        
        if not force_recalculate and cache_key in self._minibatch_cache:
            cached_size = self._minibatch_cache[cache_key]
            print(f'Using cached minibatch_size={cached_size} for batch_size={batch_size}, num_nodes={num_nodes}')
            return cached_size
        
        print(f'Profiling system & training to find optimal minibatch size...')

        # -------- available memory & device selection ----------
        if torch.cuda.is_available():
            num_gpus = len(self.devices) if self.devices else torch.cuda.device_count()
            single_gpu_memory = torch.cuda.get_device_properties(0).total_memory
            available_memory = single_gpu_memory * num_gpus
            memory_type = "GPU"
            print(f"Detected {num_gpus} GPU(s) with {single_gpu_memory / (1024**3):.2f} GB each")
            print(f"Total GPU memory available: {available_memory / (1024**3):.2f} GB")
        else:
            available_memory = psutil.virtual_memory().available
            memory_type = "system"
            print(f"Using {memory_type} memory: {available_memory / (1024**3):.2f} GB available")

        search_minibatch = batch_size
        config = copy.deepcopy(self.config)
        config.num_nodes = 1
        config.rank = 0
        config.max_steps = 3
        config.kwargs = config.kwargs.copy() if getattr(config, "kwargs", None) else {}
        config.kwargs['disable_logging'] = True
        
        # Use Trainer's device settings
        config.device = self.device
        config.devices = self.devices
        
        if not config.device:
            if torch.cuda.is_available():
                config.device = "cuda:0"
            elif torch.backends.mps.is_available():
                config.device = "mps"
            else:
                config.device = "cpu"

        config.model = config.model.to(config.device)
        config.strategy = copy.deepcopy(config.strategy)
        config.strategy._init_node(config.model, config.rank, config.num_nodes)

        need_cleanup = False
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", str(self.port + 100))
            backend = "nccl" if "cuda" in str(config.device) else "gloo"
            dist.init_process_group(backend, rank=0, world_size=1)
            need_cleanup = True

        try:
            while search_minibatch > 0:
                config.minibatch_size = search_minibatch

                try:
                    # ---- clear caches ----
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                        try:
                            torch.mps.empty_cache()
                        except Exception:
                            pass
                    gc.collect()

                    # ---- start peak sampler for MPS/CPU ----
                    use_mps_or_cpu = (not torch.cuda.is_available())
                    sampler = _PeakMemSampler(include_mps=use_mps_or_cpu)
                    rss_before = sampler.proc.memory_info().rss if use_mps_or_cpu else 0
                    sampler.start(period_s=0.01)  # ~100 Hz

                    # ---- run training trial ----
                    train_node = TrainNode(config)
                    try:
                        train_node.train()
                    finally:
                        # Clean up TrainNode resources
                        if hasattr(train_node, 'train_dataloader'):
                            del train_node.train_dataloader
                        if hasattr(train_node, 'val_dataloader'):
                            del train_node.val_dataloader
                        if hasattr(train_node, 'train_data_iter'):
                            del train_node.train_data_iter
                        if hasattr(train_node, 'val_data_iter'):
                            del train_node.val_data_iter
                        if hasattr(train_node, 'model'):
                            del train_node.model
                        if hasattr(train_node, 'strategy'):
                            del train_node.strategy
                        del train_node
                        
                        # Force garbage collection
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                            try:
                                torch.mps.empty_cache()
                            except Exception:
                                pass
                        gc.collect()

                    # ---- stop sampler & read peaks ----
                    sampler.stop()

                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated()
                        actual_usage = peak_memory  # already a peak; no need to subtract
                    else:
                        # Peak delta over baseline. Also consider MPS counter if present.
                        peak_delta_rss = max(0, sampler.peak_rss - rss_before)
                        peak_mps_bytes = sampler.peak_mps
                        actual_usage = max(peak_delta_rss, peak_mps_bytes)
                        actual_usage = int(actual_usage * 1.3)  # ~30% safety margin for fragmentation

                    total_memory_needed = actual_usage * num_nodes

                    memory_threshold = available_memory * (0.9 if torch.cuda.is_available() else 0.5)
                    if total_memory_needed < memory_threshold:
                        print(f"Found suitable minibatch_size={search_minibatch}")
                        print(f"Estimated memory per node (peak): {actual_usage / (1024**3):.2f} GB")
                        print(f"Total for {num_nodes} nodes: {total_memory_needed / (1024**3):.2f} GB")
                        print(f"Available {memory_type} memory: {available_memory / (1024**3):.2f} GB")
                        self._minibatch_cache[cache_key] = search_minibatch
                        return search_minibatch
                    else:
                        print(f"minibatch_size={search_minibatch} would use {total_memory_needed / (1024**3):.2f} GB (peak), reducing...")

                except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                    if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                        print(f"OOM with minibatch_size={search_minibatch}: {str(e)}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                            try:
                                torch.mps.empty_cache()
                            except Exception:
                                pass
                        gc.collect()
                    else:
                        raise

                search_minibatch //= 2

            raise Exception(f'Cannot find suitable minibatch size for device {config.device}')

        finally:
            # Clean up config copies and model/strategy
            if hasattr(config, 'model'):
                del config.model
            if hasattr(config, 'strategy'):
                del config.strategy
            del config
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            gc.collect()
            
            if need_cleanup:
                dist.destroy_process_group() 
