import numpy as np
import os
from typing import Literal

from .build_dataset import build_dataset_small, build_dataset_owt
from .gpt_dataset import ContiguousGPTTrainDataset, LazyNonContiguousGPTTrainDataset


def load_chunk(chunk_id, s3_client):
    cache_location = "data/owt"
    if not os.path.exists(cache_location):
        os.makedirs(cache_location, exist_ok=True)

    cache_file = f"{cache_location}/chunk_{chunk_id}.npy"
    if os.path.exists(cache_file):
        return np.load(cache_file)
    else:
        raise Exception(f"Chunk {chunk_id} not found in {cache_file}")


def get_dataset(
    dataset_name: Literal["shakespeare", "wikitext", "owt"],
    block_size: int,
    device: str,
    start_pc=0.0,
    end_pc=1.0,
    max_workers=8,
    max_chunks_in_memory=None,
):
    if dataset_name in ["shakespeare", "wikitext"]:
        data, vocab_size = build_dataset_small(
            dataset_name, block_size, start_pc, end_pc
        )

        dataset = ContiguousGPTTrainDataset(data, block_size=block_size, device=device)
    else:
        # openwebtext
        chunk_ids, cache_location, vocab_size = build_dataset_owt(
            start_pc, end_pc, max_workers=max_workers
        )

        dataset = LazyNonContiguousGPTTrainDataset(
            chunk_ids,
            cache_location,
            device=device,
            max_chunks_in_memory=max_chunks_in_memory,
        )

    return dataset, vocab_size
