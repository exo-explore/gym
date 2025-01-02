import torch
import torch.distributed as dist
from .config import DilocoSimulatorConfig
from .setup import DilocoSetup
import math


class SpartaInterpolator(DilocoSetup):

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        super().__init__(config)
        self.index_selector = PartitionedIndexSelector(self.config.p_sparta)

    def _interpolate_models(self):
        with torch.no_grad():
            for param in self.model.parameters():
                if not param.requires_grad:
                    continue
                indices = self.index_selector.get_indices(param)
                dist.broadcast(indices, src=0)
                sparse_data = param.data[indices]
                dist.all_reduce(sparse_data, op=dist.ReduceOp.SUM)
                sparse_data /= self.config.num_nodes
                param.masked_scatter_(indices, sparse_data)


class IndexSelector:
    def __init__(self, p):
        self.state = {}
        self.p = p

    def get_indices(self, param):
        return torch.ones(param.shape).bool()


class RandomIndexSelector(IndexSelector):
    def get_indices(self, param):
        return torch.bernoulli(torch.full(param.shape, self.p, device=param.device)).bool()


class PartitionedIndexSelector(IndexSelector):
    def __init__(self, p):
        super().__init__(p)

    def _set_partition(self, param):
        param_state = self.state[param]
        param_state["curr_partition"] = 0
        param_state["num_partitions"] = min(math.ceil(1 / self.p), param.numel())
        param_state["partitions"] = (
            torch.rand(param.numel(), device=param.device).argsort().view(param.shape) % param_state["num_partitions"]
        )

    def get_indices(self, param):
        if param not in self.state:
            self.state[param] = {}
            self._set_partition(param)
        elif self.state[param]["curr_partition"] >= self.state[param]["num_partitions"]:
            self._set_partition(param)

        indices = (self.state[param]["partitions"] == self.state[param]["curr_partition"]).bool()

        self.state[param]["curr_partition"] += 1

        return indices


# class TopKGradIndexSelector(IndexSelector):
#     def get_indices(self, param_index):
#         grads = torch.stack([list(model.parameters())[param_index].grad.abs() for model in self.distsim.models]).sum(
#             dim=0
#         )
#         sort = grads.view(-1).argsort(descending=True).view(grads.shape)
#         return sort < math.ceil(self.p * grads.numel())
