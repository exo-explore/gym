from tqdm import tqdm
import numpy as np
from torch import nn

from exogym.aux.utils import create_config

import json
import os
from datetime import datetime
import csv
from typing import Literal


class Logger:
    def __init__(self, model: nn.Module, max_steps: int):
        self.model = model
        self.max_steps = max_steps

        self.pbar = tqdm(total=self.max_steps, initial=0)

        tqdm.write("Logger initialized.")

        self.step = 0
        self.current_lr = 0
        self.examples_trained = 0

    def log(self, data: dict):
        pass

    def log_loss(self, loss: float, name: str):
        pass

    def log_examples_trained(self, examples: int):
        self.examples_trained += examples

    def log_train(self, loss: float):
        self.pbar.update(1)
        self.pbar.set_postfix(
            {
                "train_loss": f"{loss:.4f}",
                "lr": f"{self.current_lr:.6f}",
            }
        )

    def increment_step(self):
        self.step += 1

    def log_lr(self, lr: float):
        self.current_lr = lr


class WandbLogger(Logger):
    def __init__(
        self,
        model: nn.Module,
        max_steps: int,
        strategy=None,
        train_node=None,
        wandb_project: str = None,
        run_name: str = None,
        x_axis: Literal['step', 'examples'] = 'step',
    ):

        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Please install it using `pip install wandb`."
            )

        super().__init__(model, max_steps)

        self.wandb_project = wandb_project
        self.run_name = run_name or None
        self.x_axis = x_axis

        # Create wandb configuration using the utility function
        wandb_config = create_config(
            model=model,
            strategy=strategy,
            train_node=train_node,
            extra_config={
                "max_steps": max_steps,
            },
        )

        print(
            f'initialized wandb project with model size {wandb_config["model_parameters"]}'
        )

        init_kwargs = {
            "project": self.wandb_project,
            "name": self.run_name,
            "config": wandb_config,
            "resume": "allow",  # Allow resuming if possible, or create new
        }

        wandb.init(**init_kwargs)

        # Set the logger's step based on wandb's step for the run
        print(
            f"Started new wandb run '{self.run_name}' (ID: {wandb.run.id}). Starting at step {self.step}."
        )

        # Update tqdm progress bar
        self.pbar.n = self.step
        self.pbar.last_print_n = self.step
        self.pbar.refresh()

        strategy.lr_callbacks.append(self.log_lr)

    def log_loss(self, loss: float, name: str):
        import wandb

        if hasattr(self, "run_name"):
            data = {f"{name}_loss": loss, f"{name}_perplexity": float(np.exp(loss))}
            if self.x_axis == 'step':
                wandb.log(data, step=self.step)
            elif self.x_axis == 'examples':
                wandb.log(data, step=self.examples_trained)

    def log_train(self, loss: float):
        import wandb

        if hasattr(self, "run_name"):
            data = {
                "train_loss": loss,
                "train_perplexity": float(np.exp(loss)),
            }
            if self.current_lr:
                data["lr"] = self.current_lr

            if self.x_axis == 'step':
                wandb.log(data, step=self.step)
            elif self.x_axis == 'examples':
                wandb.log(data, step=self.examples_trained)

        self.pbar.update(1)
        self.pbar.set_postfix(
            {
                "train_loss": f"{loss:.4f}",
                "lr": f"{self.current_lr:.6f}",
            }
        )


class CSVLogger(Logger):
    def __init__(
        self,
        model: nn.Module,
        max_steps: int,
        strategy,
        train_node=None,
        log_dir: str = "logs",
        run_name: str = None,
    ):

        super().__init__(model, max_steps)

        # Generate run name based on datetime if not provided
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_name = run_name
        self.log_dir = log_dir

        # Create run directory
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # Create CSV file paths
        self.train_csv_path = os.path.join(self.run_dir, "train.csv")
        self.val_csv_path = os.path.join(self.run_dir, "validation.csv")

        # Create CSV files with headers
        self._init_csv_file(
            self.train_csv_path,
            ["step", "examples_trained", "train_loss", "train_perplexity", "lr"],
        )
        self._init_csv_file(
            self.val_csv_path,
            [
                "step",
                "local_loss",
                "local_perplexity",
                "global_loss",
                "global_perplexity",
            ],
        )

        # Create configuration using the utility function
        config = create_config(
            model=model,
            strategy=strategy,
            train_node=train_node,
            extra_config={
                "max_steps": max_steps,
                "run_name": run_name,
                "log_dir": log_dir,
            },
        )

        # Save config as JSON
        config_path = os.path.join(self.run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(
            f'CSV Logger initialized with model size {config["model_parameters"]}M parameters'
        )
        print(f"Logging to directory: {self.run_dir}")

        # Add learning rate callback
        strategy.lr_callbacks.append(self.log_lr)

    def _init_csv_file(self, filepath: str, headers: list):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def _write_csv_row(self, filepath: str, data: dict):
        """Append a row to CSV file"""
        file_exists = os.path.exists(filepath)
        with open(filepath, "a", newline="") as f:
            if file_exists:
                # Read the header to get field order
                with open(filepath, "r") as read_f:
                    reader = csv.reader(read_f)
                    headers = next(reader)

                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writerow(data)

    def log_loss(self, loss: float, name: str):
        """Log validation loss to CSV"""
        # Read existing row for this step if it exists
        existing_data = {}
        if os.path.exists(self.val_csv_path):
            with open(self.val_csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row["step"]) == self.step:
                        existing_data = row
                        break

        # Create or update the data for this step
        data = {
            "step": self.step,
            "local_loss": existing_data.get("local_loss", ""),
            "local_perplexity": existing_data.get("local_perplexity", ""),
            "global_loss": existing_data.get("global_loss", ""),
            "global_perplexity": existing_data.get("global_perplexity", ""),
        }

        # Update with the new loss data
        data[f"{name}_loss"] = loss
        data[f"{name}_perplexity"] = float(np.exp(loss))

        # If this step already exists, we need to update it
        if existing_data:
            # Rewrite the entire file with updated data
            temp_rows = []
            with open(self.val_csv_path, "r") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                for row in reader:
                    if int(row["step"]) == self.step:
                        temp_rows.append(data)
                    else:
                        temp_rows.append(row)

            with open(self.val_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(temp_rows)
        else:
            # Append new row
            self._write_csv_row(self.val_csv_path, data)

    def log_train(self, loss: float):
        """Log training loss to CSV"""
        data = {
            "step": self.step,
            "examples_trained": self.examples_trained,
            "train_loss": loss,
            "train_perplexity": float(np.exp(loss)),
        }
        if self.current_lr:
            data["lr"] = self.current_lr

        self._write_csv_row(self.train_csv_path, data)

        # Update progress bar
        self.pbar.update(1)
        self.pbar.set_postfix(
            {
                "train_loss": f"{loss:.4f}",
                "lr": f"{self.current_lr:.6f}",
            }
        )
