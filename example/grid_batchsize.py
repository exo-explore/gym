#!/usr/bin/env python3
import argparse
import os
import sys
import time
import subprocess
from pathlib import Path
from collections import deque

def parse_args():
    p = argparse.ArgumentParser(description="Parallel batch size sweep for example/nanogpt_train.py")
    p.add_argument("--batch_sizes", type=int, nargs="+", required=True, help="Batch sizes to sweep")
    p.add_argument("--lr", type=float, required=True, help="Fixed learning rate to use")
    p.add_argument("--gpus", type=int, nargs="+", default=None, help="GPU IDs to use (default: all visible)")
    p.add_argument("--base_port", type=int, default=15000, help="Base port; each run uses base_port + 10*i")
    p.add_argument("--logs_dir", type=str, default="./logs", help="Directory for run logs")

    # forwarded training flags (keep in sync with example/nanogpt_train.py)
    p.add_argument("--dataset", type=str, default="owt")
    p.add_argument("--model_size", type=str, default="base", choices=["small","base","medium","large","xl"])
    p.add_argument("--strategy", type=str, default="base", choices=["base","ddp","fedavg","sparta","diloco","demo","diloco_sparta"])
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--minibatch_size", type=int, default=None)
    p.add_argument("--max_steps", type=int, default=30000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--cosine_anneal", action="store_true", default=False)
    p.add_argument("--val_size", type=int, default=256)
    p.add_argument("--val_interval", type=int, default=100)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--run_name_prefix", type=str, default="owt_bs")
    return p.parse_args()

def detect_gpus(user_list):
    if user_list:
        return user_list
    try:
        import torch
        n = torch.cuda.device_count()
        return list(range(n))
    except Exception:
        v = os.environ.get("CUDA_VISIBLE_DEVICES")
        if v:
            return [int(x) for x in v.split(",") if x.strip() != ""]
        return []

def build_cmd(args, batch_size, port):
    cmd = [
        sys.executable, "example/nanogpt_train.py",
        "--dataset", args.dataset,
        "--strategy", args.strategy,
        "--model_size", args.model_size,
        "--block_size", str(args.block_size),
        "--device", "cuda",
        "--num_nodes", "1",
        "--lr", str(args.lr),
        "--batch_size", str(batch_size),
        "--max_steps", str(args.max_steps),
        "--epochs", str(args.epochs),
        "--warmup_steps", str(args.warmup_steps),
        "--val_size", str(args.val_size),
        "--val_interval", str(args.val_interval),
        "--save_dir", args.save_dir,
        "--start_port", str(port),
        "--run_name", f"{args.run_name_prefix}{batch_size}",
    ]
    if args.minibatch_size is not None:
        cmd += ["--minibatch_size", str(args.minibatch_size)]
    if args.cosine_anneal:
        cmd += ["--cosine_anneal"]
    if args.wandb_project:
        cmd += ["--wandb_project", args.wandb_project]
    return cmd

def main():
    args = parse_args()
    gpus = detect_gpus(args.gpus)
    if not gpus:
        print("No GPUs detected. Set --gpus or CUDA_VISIBLE_DEVICES.")
        sys.exit(1)

    os.makedirs(args.logs_dir, exist_ok=True)
    pending = deque(enumerate(args.batch_sizes))  # (idx, batch_size)
    running = {}  # gpu_id -> (Popen, log_file_path, file_handle)

    def start_on(gpu_id, bs_idx, batch_size):
        port = args.base_port + 10 * bs_idx  # stable unique port per run
        cmd = build_cmd(args, batch_size, port)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("OMP_NUM_THREADS", "1")

        log_path = Path(args.logs_dir) / f"{args.run_name_prefix}{batch_size}_gpu{gpu_id}.log"
        fh = open(log_path, "w")
        print(f"[launch] gpu={gpu_id} batch_size={batch_size} lr={args.lr} port={port} -> {log_path}")
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        running[gpu_id] = (p, log_path, fh)

    # prime the pool
    for gpu_id in gpus:
        if pending:
            i, batch_size = pending.popleft()
            start_on(gpu_id, i, batch_size)

    # scheduler loop
    exit_code = 0
    while running:
        time.sleep(0.5)
        finished = []
        for gpu_id, (p, log_path, fh) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                fh.close()
                if rc != 0:
                    print(f"[error] run on gpu={gpu_id} exited {rc} (see {log_path})")
                    exit_code = rc
                else:
                    print(f"[done]  gpu={gpu_id} OK (log {log_path})")
                finished.append(gpu_id)
        for gpu_id in finished:
            running.pop(gpu_id, None)
            if pending:
                i, batch_size = pending.popleft()
                start_on(gpu_id, i, batch_size)

    sys.exit(exit_code)

if __name__ == "__main__":
    main()