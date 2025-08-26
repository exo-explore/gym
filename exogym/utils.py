import torch.distributed as dist
import os

def init_process_group_portsafe(backend: str, rank: int, world_size: int, retries: int = 10) -> None:
    port = int(os.environ.get("MASTER_PORT", "29500")) + 199

    os.environ.update({
        "MASTER_ADDR":"localhost",
    })
    if backend == 'gloo':
        os.environ.update({
            "GLOO_SOCKET_IFNAME":"lo0",
        })


    i = 0
    while i < retries:
        try:
            os.environ["MASTER_PORT"] = str(port)
            dist.init_process_group(backend, rank=rank, world_size=world_size)
            break
        except Exception as e:
            print(f'error when calling init_process_group: {e}')
            port += 1
            i += 1
