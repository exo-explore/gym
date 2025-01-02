import torch
from .config import DilocoSimulatorConfig
from .setup import DilocoSetup
import math
from dataclasses import dataclass
import wandb


@dataclass
class EvalStats:
    loss: float
    perplexity: float


class Evaluator(DilocoSetup):

    def __init__(self, config: DilocoSimulatorConfig) -> None:
        super().__init__(config)

    def _evaluate(self):
        self.model.eval()

        losses = []
        num_batches = math.ceil(self.config.eval_iters / self.config.batch_size)
        with torch.no_grad():
            for _ in range(num_batches):
                x, y = self._get_batch(eval=True)
                output = self.model(x)
                loss = self.config.loss_fn(output, y)
                losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)

        print(f"Eval Loss: {avg_loss:.4f}, Eval Perplexity: {math.exp(avg_loss):.4f}")

        self._log_eval(EvalStats(loss=avg_loss, perplexity=math.exp(avg_loss)))

        self.model.train()

    def _log_eval(self, eval_stats: EvalStats):
        if self.config.wandb_project is None:
            return
        wandb.log(
            {
                "step": self.local_step,
                "eval_loss": eval_stats.loss,
                "eval_perplexity": eval_stats.perplexity,
            }
        )
