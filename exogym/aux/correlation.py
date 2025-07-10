import os
import torch
import numpy as np

class CorrelationMixin:
    def _correlation_calculation(self):
        if self.num_nodes < 2:
            raise Exception("Correlation calculation cannot be used with < 2 nodes")

        # Ensure correlation is only calculated if interval is set
        if not self.config.correlation_interval:
            return None

        # Create a temporary directory for this timestep's checkpoints
        tmp_dir = os.path.join(self.config.save_dir, f"tmp_corr_{self.local_step}")
        # Only rank 0 creates the directory to avoid race conditions
        if self.rank == 0:
            os.makedirs(tmp_dir, exist_ok=True)
        torch.distributed.barrier()  # Wait for rank 0 to create dir

        # Save model state dict for each rank
        checkpoint_path = os.path.join(tmp_dir, f"{self.rank}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

        # Wait for all processes to save their checkpoints
        torch.distributed.barrier()

        corr_value = None
        if self.rank == 0:
            # Load all models as vectors
            model_vectors = []
            for r in range(self.config.num_nodes):
                model_path = os.path.join(tmp_dir, f"{r}.pt")
                # Ensure the file exists before trying to load
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location="cpu")
                    vector_list = []
                    for key in sorted(checkpoint.keys()):
                        value = checkpoint[key]
                        if isinstance(value, torch.Tensor):
                            vector_list.append(value.cpu().numpy().ravel())
                    if vector_list:  # Check if we actually got any tensors
                        model_vectors.append(np.concatenate(vector_list))
                else:
                    print(
                        f"Warning: Checkpoint file {model_path} not found for rank {r}."
                    )

            if len(model_vectors) >= 2:  # Need at least two models to compare
                # Calculate correlations between all pairs
                correlations = []
                for i in range(len(model_vectors)):
                    for j in range(i + 1, len(model_vectors)):
                        corr = np.corrcoef(model_vectors[i], model_vectors[j])[0, 1]
                        correlations.append(corr)

                if correlations:  # Ensure correlations list is not empty
                    corr_value = np.mean(correlations)

                    # Log average correlation to wandb using the logger
                    if self.logger:
                        self.logger.log(data={"avg_model_correlation": corr_value})
                else:
                    print(
                        "Warning: Could not calculate correlation, not enough valid model pairs."
                    )
            else:
                print(
                    f"Warning: Not enough models loaded ({len(model_vectors)}) to calculate correlation."
                )

            # Clean up temporary directory
            import shutil

            shutil.rmtree(tmp_dir)

        # Wait for rank 0 to finish cleanup
        torch.distributed.barrier()

        return corr_value  # Only rank 0 returns a value, others return None
