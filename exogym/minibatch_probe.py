import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import multiprocessing
import os
import gc
import copy
import psutil
import time
import threading
from exogym.train_node import TrainNode
from exogym.common import TrainConfig


def _get_mps_allocated_bytes():
    """Get MPS allocated memory in bytes, defensively handling API variations."""
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


def _minibatch_probe_worker(config: TrainConfig, batch_size: int, num_nodes: int, result_queue: multiprocessing.Queue):
    """
    Runs minibatch size probe in an isolated subprocess.
    This ensures any CUDA/MPS contexts are fully released after probing.
    """
    import os, gc, torch, torch.distributed as dist
    from exogym.train_node import TrainNode

    # Ensure isolated process group
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(int(os.environ.get("MASTER_PORT", "29500")) + 199))

    # Single-rank probe configuration
    cfg = copy.deepcopy(config)
    cfg.num_nodes = 1
    cfg.rank = 0
    cfg.max_steps = 3
    cfg.kwargs = (cfg.kwargs or {}).copy()
    cfg.kwargs['disable_logging'] = True

    # Determine device for probe
    if not cfg.device:
        if torch.cuda.is_available():
            cfg.device = "cuda:0"
        elif torch.backends.mps.is_available():
            cfg.device = "mps"
        else:
            cfg.device = "cpu"

    # Initialize process group
    backend = "nccl" if ("cuda" in str(cfg.device)) else "gloo"
    dist.init_process_group(backend, rank=0, world_size=1)

    # Move model/strategy to device
    cfg.model = cfg.model.to(cfg.device)
    cfg.strategy = copy.deepcopy(cfg.strategy)
    cfg.strategy._init_node(cfg.model, cfg.rank, cfg.num_nodes)

    # Determine available memory
    if torch.cuda.is_available():
        num_gpus = len(cfg.devices) if cfg.devices else torch.cuda.device_count()
        single_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = single_gpu_memory * num_gpus
        memory_type = "GPU"
        memory_threshold = available_memory * 0.9
        print(f"Detected {num_gpus} GPU(s) with {single_gpu_memory / (1024**3):.2f} GB each")
        print(f"Total GPU memory available: {available_memory / (1024**3):.2f} GB")
    else:
        available_memory = psutil.virtual_memory().available
        memory_type = "system"
        memory_threshold = available_memory * 0.5
        print(f"Using {memory_type} memory: {available_memory / (1024**3):.2f} GB available")

    def try_minibatch(minib):
        """Try running with a specific minibatch size."""
        cfg.minibatch_size = minib
        
        try:
            # Clear caches before attempt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            gc.collect()

            # Setup memory sampling for non-CUDA devices
            sampler = None
            rss_before = 0
            if not torch.cuda.is_available():
                sampler = _PeakMemSampler(include_mps=True)
                rss_before = sampler.proc.memory_info().rss
                sampler.start(period_s=0.01)

            # Run training trial
            node = TrainNode(cfg)
            node.train()

            # Measure memory usage
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                actual_usage = peak_memory
            else:
                sampler.stop()
                peak_delta_rss = max(0, sampler.peak_rss - rss_before)
                peak_mps_bytes = sampler.peak_mps
                actual_usage = max(peak_delta_rss, peak_mps_bytes)
                actual_usage = int(actual_usage * 1.3)  # 30% safety margin

            total_memory_needed = actual_usage * num_nodes

            if total_memory_needed < memory_threshold:
                print(f"Found suitable minibatch_size={minib}")
                print(f"Estimated memory per node (peak): {actual_usage / (1024**3):.2f} GB")
                print(f"Total for {num_nodes} nodes: {total_memory_needed / (1024**3):.2f} GB")
                print(f"Available {memory_type} memory: {available_memory / (1024**3):.2f} GB")
                return True
            else:
                print(f"minibatch_size={minib} would use {total_memory_needed / (1024**3):.2f} GB (peak), reducing...")
                return False

        except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
            msg = str(e).lower()
            if "out of memory" in msg or isinstance(e, MemoryError):
                print(f"OOM with minibatch_size={minib}: {str(e)}")
                return False
            raise

    # Binary search for optimal minibatch size
    search = batch_size
    found = 0
    while search > 0:
        if try_minibatch(search):
            found = search
            break
        search //= 2

    # Clean up and return result
    result_queue.put(found)
    dist.destroy_process_group()


def find_minibatch_size_isolated(config: TrainConfig, num_nodes: int, batch_size: int, 
                                  devices=None, device=None, port=None):
    """
    Find optimal minibatch size using subprocess isolation.
    Always runs in a subprocess to ensure clean memory state.
    """
    print("Profiling minibatch size in an isolated subprocess...")
    
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()

    probe_cfg = copy.deepcopy(config)
    probe_cfg.devices = devices
    probe_cfg.device = device
    probe_cfg.port = port

    p = ctx.Process(
        target=_minibatch_probe_worker,
        args=(probe_cfg, batch_size, num_nodes, q),
    )
    p.start()
    
    try:
        found = q.get(timeout=600)  # 10 min timeout
    except Exception:
        found = 0
    
    p.join()

    if found <= 0:
        raise Exception(f'Cannot find suitable minibatch size with batch_size={batch_size}')

    print(f'Found suitable minibatch_size={found} (via isolated probe)')
    return found