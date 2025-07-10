import torch
import torch.distributed as dist
import os

class LocalTrainNode(TrainNode):
    def __init__(self, config: TrainConfig):
        self.config = config

    def _build_connection(self):
        """
        This is the default callback for setting up pytorch distributed connections.
        All ranks are assumed to be on the same machine, and device is defaulted to cpu.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.port)

        if self.device == "" or self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # initialize the process group
        if self.device == "cuda":
            # If we haven't specified devices, use all devices.
            if self.devices is None:
                self.devices = range(torch.cuda.device_count())

            dist.init_process_group(
                "nccl" if len(self.devices) == self.num_nodes else "gloo",
                rank=self.rank,
                world_size=self.num_nodes,
            )
            self.device = torch.device(
                f"cuda:{self.devices[self.rank % len(self.devices)]}"
            )
            torch.cuda.set_device(self.device)
        elif self.device == "cpu":
            dist.init_process_group("gloo", rank=self.rank, world_size=self.num_nodes)
            self.device = torch.device("cpu")
        elif self.device == "mps":
            dist.init_process_group("gloo", rank=self.rank, world_size=self.num_nodes)
            self.device = torch.device("mps")
        else:
            raise ValueError(f"Invalid device type: {self.device}")

        print(f"Rank {self.rank} using device {self.device}")