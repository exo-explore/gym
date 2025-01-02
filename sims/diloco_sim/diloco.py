import torch
import torch.distributed as dist
from tqdm import tqdm
from .config import DilocoSimulatorConfig
from .setup import DilocoSetup
from .eval import Evaluator
from .sparta import SpartaInterpolator
from dataclasses import dataclass
import wandb


@dataclass
class TrainStats:
    loss: float
    perplexity: float


class DilocoSimulator(Evaluator, SpartaInterpolator):

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        super().__init__(config)

    def _average_models(self) -> None:
        for param in self.model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= self.config.num_nodes

    def _broadcast_model_params(self) -> None:
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)

    def _set_master_grad(self) -> None:
        for name, param in self.model.named_parameters():
            param.grad = self.master_model.state_dict()[name].data.to(param.device) - param.data

    def _synchronize_master_model(self) -> None:
        for name, param in self.master_model.named_parameters():
            param.data = self.model.state_dict()[name].data.to("cpu")

    def _outer_step(self) -> None:
        self._average_models()

        if self.rank == 0:
            self.master_optimizer.zero_grad()
            self._set_master_grad()
            self.master_optimizer.step()
            self._synchronize_master_model()

        self._broadcast_model_params()

    def _train_step(self):
        x, y = self._get_batch()
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.config.loss_fn(output, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()

        if self.rank == 0:
            self._log_train(TrainStats(loss=loss.item(), perplexity=torch.exp(loss).item()))

        return loss.item()

    def _log_train(self, train_stats: TrainStats):
        lr = self.optimizer.param_groups[0]["lr"]
        self.pbar.update(1)
        self.pbar.set_postfix(
            {
                "loss": f"{train_stats.loss:.4f}",
                "perplexity": f"{train_stats.perplexity:.4f}",
                "lr": f"{lr:.4f}",
            }
        )

        if self.config.wandb_project is None:
            return
        wandb.log(
            {
                "step": self.local_step,
                "train_loss": train_stats.loss,
                "train_perplexity": train_stats.perplexity,
                "learning_rate": lr,
            }
        )

    def _train_loop(self):

        while self.local_step < self.max_local_step:

            if self.config.p_sparta > 0.0:
                self._interpolate_models()

            loss = self._train_step()

            if self.local_step % self.config.diloco_interval == 0:
                self._outer_step()
                if self.rank == 0:
                    self._evaluate()

            self.local_step += 1

    def _train(self, rank: int):
        try:
            self._setup(rank)
            self._train_loop()

            if self.rank == 0 and self.config.save_dir:
                self._save_checkpoint()
        finally:
            self._cleanup()

    def train(self):
        torch.multiprocessing.spawn(self._train, args=(), nprocs=self.config.num_nodes, join=True)
