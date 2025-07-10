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

        filename = f"{self.local_step}.pt"
        full_save_path = os.path.join(save_path_dir, filename)

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
            torch.save(checkpoint, full_save_path)
            print(
                f"Rank {self.rank} saved checkpoint to {full_save_path} at step {self.local_step}"
            )
            self._delete_other_checkpoints(save_path_dir, full_save_path)
        except OSError as e:
            print(
                f"Rank {self.rank}: Failed to save checkpoint {full_save_path} due to OSError: {e}. Attempting to delete oldest checkpoint and retry."
            )

            oldest_step = float("inf")
            oldest_checkpoint_file = None
            # Ensure save_path_dir exists before listing its contents, though it should have been created.
            if os.path.exists(save_path_dir):
                for f_name in os.listdir(save_path_dir):
                    if f_name.endswith(".pt"):
                        try:
                            # Checkpoints are named as {step_num}.pt
                            step_num = int(f_name.split(".")[0])
                            if step_num < oldest_step:
                                oldest_step = step_num
                                oldest_checkpoint_file = f_name
                        except ValueError:
                            # Skip files not matching the expected N.pt pattern
                            continue

            if oldest_checkpoint_file:
                oldest_checkpoint_path = os.path.join(
                    save_path_dir, oldest_checkpoint_file
                )
                try:
                    os.remove(oldest_checkpoint_path)
                    print(
                        f"Rank {self.rank}: Deleted oldest checkpoint {oldest_checkpoint_path} to free space."
                    )

                    # Retry saving the current checkpoint
                    try:
                        torch.save(checkpoint, full_save_path)
                        print(
                            f"Rank {self.rank}: Successfully saved checkpoint {full_save_path} after deleting oldest."
                        )
                        self._delete_other_checkpoints(save_path_dir, full_save_path)
                    except OSError as e2:
                        print(
                            f"Rank {self.rank}: Still failed to save checkpoint {full_save_path} after deleting oldest: {e2}. Giving up."
                        )
                        raise  # Re-raise the second error, as we couldn't save even after cleanup
                except OSError as del_e:
                    print(
                        f"Rank {self.rank}: Failed to delete oldest checkpoint {oldest_checkpoint_path}: {del_e}. Original save error will be raised."
                    )
                    raise e  # Re-raise the original save error, as cleanup failed
            else:
                print(
                    f"Rank {self.rank}: No old checkpoints found to delete in {save_path_dir}. Original save error will be raised."
                )
                raise e  # Re-raise the original save error, as no space could be freed

    def _delete_other_checkpoints(
        self, save_path_dir: str, current_checkpoint_full_path: str
    ):
        if not os.path.exists(save_path_dir):
            return

        current_checkpoint_filename = os.path.basename(current_checkpoint_full_path)
        deleted_count = 0
        for f_name in os.listdir(save_path_dir):
            if f_name.endswith(".pt") and f_name != current_checkpoint_filename:
                try:
                    file_to_delete = os.path.join(save_path_dir, f_name)
                    os.remove(file_to_delete)
                    # print(f"Rank {self.rank}: Deleted old checkpoint {file_to_delete}")
                    deleted_count += 1
                except OSError as del_e:
                    print(
                        f"Rank {self.rank}: Warning - Failed to delete old checkpoint {file_to_delete}: {del_e}"
                    )
        if deleted_count > 0:
            print(
                f"Rank {self.rank}: Deleted {deleted_count} other checkpoint(s) in {save_path_dir}."
            )

    def _load_checkpoint(self):
        save_path_dir = os.path.join(
            self.config.save_dir,
            self.config.wandb_project if self.config.wandb_project else "unnamed",
            self.config.run_name if self.config.run_name else "unnamed",
            str(self.rank),
        )

        if not os.path.exists(save_path_dir):
            print(
                f"Rank {self.rank}: Checkpoint directory {save_path_dir} not found. Starting from scratch."
            )
            return False

        checkpoint_files = []
        for f_name in os.listdir(save_path_dir):
            if f_name.endswith(".pt"):
                try:
                    step_num = int(f_name.split(".")[0])
                    checkpoint_files.append((step_num, f_name))
                except ValueError:
                    # Not a valid checkpoint file name pattern
                    continue

        # Sort by step number in descending order (latest first)
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)

        loaded_successfully = False
        for step_num, f_name in checkpoint_files:
            full_checkpoint_path = os.path.join(save_path_dir, f_name)
            try:
                print(
                    f"Rank {self.rank}: Attempting to load checkpoint from {full_checkpoint_path}"
                )
                checkpoint = torch.load(full_checkpoint_path, map_location=self.device)

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
                    f"Rank {self.rank}: Successfully loaded checkpoint {f_name}. Resuming at epoch {self.epoch}, step {self.local_step}."
                )
                loaded_successfully = True
                break  # Exit loop once a checkpoint is successfully loaded
            except (
                RuntimeError,
                EOFError,
                zipfile.BadZipFile,
            ) as e:  # Catch specific errors related to corrupted files
                print(
                    f"Rank {self.rank}: Failed to load checkpoint {full_checkpoint_path}: {e}. Trying next available checkpoint."
                )
                # Optionally, delete the corrupted checkpoint file
                try:
                    os.remove(full_checkpoint_path)
                    print(
                        f"Rank {self.rank}: Deleted corrupted checkpoint {full_checkpoint_path}."
                    )
                except OSError as del_e:
                    print(
                        f"Rank {self.rank}: Warning - Failed to delete corrupted checkpoint {full_checkpoint_path}: {del_e}"
                    )
            except Exception as e:  # Catch any other unexpected error during loading
                print(
                    f"Rank {self.rank}: An unexpected error occurred while loading checkpoint {full_checkpoint_path}: {e}. Trying next."
                )
                # Optionally, delete or move the problematic checkpoint

        if not loaded_successfully:
            print(
                f"Rank {self.rank}: No valid checkpoint found in {save_path_dir} after trying all options. Starting from scratch."
            )
            # Reset relevant states if starting from scratch, though __init__ defaults should cover this.
            self.local_step = 0
            self.epoch = 0
            return False

        return True
