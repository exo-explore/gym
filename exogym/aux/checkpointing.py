import os
import torch
import zipfile

# TODO: Fix.
# How are we going to do config? 

class CheckpointMixin:
    def _save_checkpoint(self):
        save_path_dir = os.path.join(
            self.config.save_dir,
            self.config.wandb_project if self.config.wandb_project else "unnamed",
            self.config.run_name if self.config.run_name else "unnamed",
            str(self.rank),
        )
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir, exist_ok=True)

        # Use a fixed filename for single checkpoint
        filename = "checkpoint.pt"
        full_save_path = os.path.join(save_path_dir, filename)
        temp_save_path = os.path.join(save_path_dir, "checkpoint.tmp.pt")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.strategy.optim.state_dict(),
            "local_step": self.local_step,
            "epoch": self.epoch,
            "rng_state": torch.get_rng_state(),
        }
        if self.strategy.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.strategy.scheduler.state_dict()

        if self.device.type == "cuda":
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()

        try:
            # Save to temp file first to ensure atomic write
            torch.save(checkpoint, temp_save_path)
            # Move temp file to final location (atomic on most filesystems)
            os.replace(temp_save_path, full_save_path)
            print(
                f"Rank {self.rank} saved checkpoint to {full_save_path} at step {self.local_step}"
            )
        except OSError as e:
            print(
                f"Rank {self.rank}: Failed to save checkpoint {full_save_path} due to OSError: {e}"
            )
            # Clean up temp file if it exists
            if os.path.exists(temp_save_path):
                try:
                    os.remove(temp_save_path)
                except OSError:
                    pass
            raise e


    def _load_checkpoint(self):
        save_path_dir = os.path.join(
            self.config.save_dir,
            self.config.wandb_project if self.config.wandb_project else "unnamed",
            self.config.run_name if self.config.run_name else "unnamed",
            str(self.rank),
        )

        # Use fixed filename for single checkpoint
        checkpoint_path = os.path.join(save_path_dir, "checkpoint.pt")
        
        if not os.path.exists(checkpoint_path):
            print(
                f"Rank {self.rank}: No checkpoint found at {checkpoint_path}. Starting from scratch."
            )
            return False

        try:
            print(
                f"Rank {self.rank}: Loading checkpoint from {checkpoint_path}"
            )
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.strategy.optim.load_state_dict(checkpoint["optimizer_state_dict"])

            if (
                "scheduler_state_dict" in checkpoint
                and self.strategy.scheduler is not None
            ):
                self.strategy.scheduler.load_state_dict(
                    checkpoint["scheduler_state_dict"]
                )

            self.local_step = checkpoint["local_step"]
            self.epoch = checkpoint["epoch"]

            torch.set_rng_state(
                checkpoint["rng_state"].cpu()
            )  # Ensure RNG state is on CPU before loading
            if self.device.type == "cuda" and "cuda_rng_state" in checkpoint:
                if isinstance(checkpoint["cuda_rng_state"], torch.Tensor):
                    torch.cuda.set_rng_state(
                        checkpoint["cuda_rng_state"].cpu(), device=self.device
                    )
                else:
                    torch.cuda.set_rng_state(
                        checkpoint["cuda_rng_state"], device=self.device
                    )

            self.train_data_iter = iter(self.train_dataloader)
            self.val_data_iter = iter(self.val_dataloader)

            if len(self.train_dataloader) > 0:
                batches_to_skip = self.local_step % len(self.train_dataloader)
                print(
                    f"Rank {self.rank}: Restored to epoch {self.epoch}, step {self.local_step}. Skipping {batches_to_skip} batches."
                )
                for _ in range(batches_to_skip):
                    try:
                        next(self.train_data_iter)
                    except StopIteration:
                        print(
                            f"Rank {self.rank}: Warning - StopIteration while fast-forwarding train_data_iter."
                        )
                        break
            else:
                print(
                    f"Rank {self.rank}: Restored to epoch {self.epoch}, step {self.local_step}. Train dataloader empty."
                )

            if self.rank == 0 and hasattr(self.logger, "set_step"):
                self.logger.set_step(self.local_step)
            elif self.rank == 0:
                print(
                    f"Rank 0: Logger step will resume from loaded local_step: {self.local_step}"
                )

            print(
                f"Rank {self.rank}: Successfully loaded checkpoint. Resuming at epoch {self.epoch}, step {self.local_step}."
            )
            return True
            
        except Exception as e:
            print(
                f"Rank {self.rank}: Failed to load checkpoint from {checkpoint_path}: {e}. Starting from scratch."
            )
            # Reset relevant states if starting from scratch
            self.local_step = 0
            self.epoch = 0
            return False
