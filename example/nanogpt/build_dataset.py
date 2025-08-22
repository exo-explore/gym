import torch
import argparse
import numpy as np
import os
from datasets import load_dataset, load_dataset_builder, concatenate_datasets


def generate_char_vocab():
    """
    Generates a fixed character vocabulary and returns two mappings:
    char -> int, int -> char, and also the special end-of-sequence token id.
    """
    vocab = " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\n"
    char_int = {char: i for i, char in enumerate(vocab)}
    # int_char = {i: char for i, char in enumerate(vocab)}

    # Define a special end-of-sequence token.
    eos_token = "<EOS>"
    char_int[eos_token] = len(char_int)
    eos_token_id = char_int[eos_token]
    return char_int, eos_token_id


def build_dataset_small(dataset, block_size=1024, start_pc=0.0, end_pc=1.0):
    """
    Loads and preprocesses the dataset with caching, using either a custom character-level tokenizer
    or the GPT2 tokenizer.

    Args:
        dataset: a string identifier ("shakespeare" or "wikitext")
        block_size: the sequence block size.
        char (bool): If True, use character-level tokenization; otherwise, use GPT-2 tokenization.
    Returns:
        data: a numpy array of the dataset.
        vocab_size: the size of the vocabulary.
    """
    assert dataset in ["shakespeare", "wikitext"]

    if dataset == "shakespeare":
        char = True
    else:
        char = False

    # Decide cache locations based on tokenization mode and rank.
    if char:
        cache_dir = os.path.join("data", f"{dataset}_char")
    else:
        cache_dir = os.path.join("data", dataset)
    os.makedirs(cache_dir, exist_ok=True)

    data_cache_file = os.path.join(
        cache_dir, f"data_block{block_size}_{start_pc}_{end_pc}.npy"
    )

    if os.path.exists(data_cache_file):
        print(f"Loading cached dataset from {data_cache_file}")
        data = np.load(data_cache_file)
        # Determine vocab size based on dataset type
        if char:
            char_int, eos_token_id = generate_char_vocab()
            vocab_size = len(char_int)
        else:
            vocab_size = 50257  # GPT-2 vocab size
        return data, vocab_size

    print(
        f"Loading dataset: {dataset} {'(char-level)' if char else '(GPT2 tokenization)'} start%: {start_pc} end%: {end_pc}"
    )

    # Determine the dataset identifier and mapping function.
    if dataset == "shakespeare":
        dataset_id = "Trelis/tiny-shakespeare"
        def mapping_fn(x): return {"text": x["Text"]}
        load_config = {}
    elif dataset == "wikitext":
        dataset_id = "wikitext"
        config = "wikitext-2-raw-v1"
        def mapping_fn(x): return {"text": x["text"]}
        load_config = {"name": config}
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Use the dataset builder to obtain the total number of records.
    builder = load_dataset_builder(dataset_id, **load_config)
    total_records = (
        builder.info.splits["train"].num_examples
        + builder.info.splits["test"].num_examples
    )

    start_record = int(total_records * start_pc)
    end_record = int(total_records * end_pc)

    used_records = end_record - start_record
    print(f"Using {used_records} records: {start_record} to {end_record}")

    # Small enough dataset that we can load the whole thing in.
    dataset = load_dataset(dataset_id, **load_config)
    dataset = concatenate_datasets([dataset["train"], dataset["test"]])

    dataset = dataset.map(mapping_fn, remove_columns=dataset.column_names)

    dataset = dataset.select(range(start_record, end_record))

    ## Initialize the tokenizer.
    if char:
        char_int, eos_token_id = generate_char_vocab()
        vocab_size = len(char_int)

        def tokenize(example):
            text = example["text"]
            if isinstance(text, str):
                return {"tokenized": [char_int[c] for c in text]}
            elif isinstance(text, list):
                return {"tokenized": [[char_int[c] for c in t] for t in text]}
            else:
                raise Exception("Unknown type")

    else:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        eos_token_id = tokenizer.eos_token_id

        def tokenize(example):
            return {
                "tokenized": tokenizer(
                    example["text"], truncation=True, max_length=block_size
                )["input_ids"]
            }

    ## Tokenize the dataset.
    dataset = dataset.map(tokenize, num_proc=os.cpu_count(), batched=True)

    # Convert tokenized lists to 1-d contiguous stream.
    def aggregate_examples(examples):
        arrays = [
            np.asarray(ids + [eos_token_id], dtype=np.int64)
            for ids in examples["tokenized"]
            if ids
        ]
        if len(arrays) == 0:
            return {"ids": np.asarray([], dtype=np.int64)}
        all_ids = np.concatenate(arrays)
        return {"ids": all_ids}

    dataset_processed = dataset.map(
        aggregate_examples,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
    )

    # Instead of using .set_format(type="numpy", columns=["ids"]) and then dataset_processed["ids"],
    # which returns a datasets.arrow_dataset.Column (not a numpy array), we should concatenate all arrays.
    # This will ensure 'data' is a numpy array.

    # Get all values from the "ids" column and build a contiguous 1-D numpy array
    ids_list = list(dataset_processed["ids"])  # may be list of arrays, lists, or ints
    if len(ids_list) == 0:
        raise ValueError("No token ids produced by preprocessing")

    first_item = ids_list[0]
    if isinstance(first_item, np.ndarray):
        # Already arrays, just concatenate
        data = np.concatenate(ids_list) if len(ids_list) > 1 else first_item
    elif isinstance(first_item, (list, tuple)):
        # Convert lists/tuples to arrays, then concatenate
        arrays = [np.array(x) for x in ids_list if len(x) > 0]
        if len(arrays) == 0:
            data = np.array([], dtype=np.int64)
        else:
            data = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
    else:
        # Scalars (e.g., ints). Build directly into an array.
        data = np.asarray(ids_list)

    print(f"Dataset size: {data.shape}")

    np.save(data_cache_file, data)

    return data, vocab_size


def build_dataset_owt(start_pc=0.0, end_pc=1.0, max_workers=8):
    """
    Loads and preprocesses the dataset with caching, using either a custom character-level tokenizer
    or the GPT2 tokenizer. It uses only a fraction of the full dataset, as controlled by dataset_proportion,
    then splits that fraction into training and validation portions (with val_ratio held out as validation).
    When rank and world_size are provided, the training portion is sharded among nodes.

    Args:
        start_pc (float): The start percentage of the dataset to use.
        end_pc (float): The end percentage of the dataset to use.
        max_workers (int): Maximum number of workers for parallel processing.
    """
    block_size = 1024  # Fixed block size for OWT
    cache_dir = os.path.join("data", "owt")
    os.makedirs(cache_dir, exist_ok=True)

    # Check if the chunks for this range already exist
    target_chunks_for_full_dataset = 1000
    start_chunk_id = int(start_pc * target_chunks_for_full_dataset)
    end_chunk_id = int(end_pc * target_chunks_for_full_dataset)
    expected_chunk_ids = list(range(start_chunk_id, end_chunk_id))

    # Check if all required chunks exist
    missing_chunks = []
    for chunk_id in expected_chunk_ids:
        cache_file = f"{cache_dir}/chunk_{chunk_id}.npy"
        if not os.path.exists(cache_file):
            missing_chunks.append(chunk_id)

    if not missing_chunks:
        print(
            f"All chunks {start_chunk_id} to {end_chunk_id-1} already exist, using cached data"
        )
        # Still need to get vocab_size
        vocab_size = 50257  # GPT-2 vocab size
        return expected_chunk_ids, cache_dir, vocab_size
    else:
        print(f"Missing chunks {missing_chunks}, will download and process data")

    print(
        f"Loading dataset: owt {'(GPT2 tokenization)'} start%: {start_pc} end%: {end_pc}"
    )

    dataset_id = "Skylion007/openwebtext"
    load_config = {"trust_remote_code": True}

    # Use the dataset builder to obtain the total number of records.
    builder = load_dataset_builder(dataset_id, **load_config)
    total_records = builder.info.splits["train"].num_examples

    print(f"Total records to import: {total_records}")

    # Calculate the number of records to use and how to split them.
    start_record = int(total_records * start_pc)
    end_record = int(total_records * end_pc)

    used_records = end_record - start_record
    print(f"Using {used_records} records: {start_record} to {end_record}")

    dataset = load_dataset(
        dataset_id, split=f"train[{start_record}:{end_record}]", **load_config
    )

    try:
        from transformers import GPT2Tokenizer
    except ImportError:
        raise ImportError(
            "transformers is not installed. Please install the correct distro using pip install exogym[gpt]"
        )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    eos_token_id = tokenizer.eos_token_id

    def tokenize(example):
        return {
            "tokenized": tokenizer(
                example["text"], truncation=True, max_length=block_size
            )["input_ids"]
        }

    ## Tokenize the dataset.
    dataset = dataset.map(tokenize, num_proc=os.cpu_count(), batched=True)

    # Convert tokenized lists to blocks with fixed block size
    def aggregate_examples(examples):
        # Flatten all ids and add EOS tokens
        all_ids = np.concatenate(
            [np.array(ids + [eos_token_id]) for ids in examples["tokenized"] if ids]
        )
        num_blocks = len(all_ids) // block_size
        if num_blocks == 0:
            return {"ids": torch.tensor([])}
        all_ids = all_ids[: num_blocks * block_size]
        data_2d = all_ids.reshape(-1, block_size)
        return {"ids": data_2d}

    dataset_processed = dataset.map(
        aggregate_examples,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
    )

    # Get all the processed blocks
    all_blocks = []
    for item in dataset_processed:
        if len(item["ids"]) > 0:
            all_blocks.append(item["ids"])

    if not all_blocks:
        raise ValueError("No valid blocks found in dataset")

    # Concatenate all blocks into a single 2D array
    all_data = np.vstack(all_blocks)
    total_blocks = all_data.shape[0]

    print(f"Total blocks: {total_blocks}")

    # Calculate number of chunks for this range
    range_pc = end_pc - start_pc
    num_chunks = max(1, int(target_chunks_for_full_dataset * range_pc))

    # Calculate blocks per chunk
    blocks_per_chunk = max(1, total_blocks // num_chunks)

    print(f"Creating {num_chunks} chunks with ~{blocks_per_chunk} blocks each")
    print(
        f"Chunk IDs will range from {start_chunk_id} to {start_chunk_id + num_chunks - 1}"
    )

    # Create chunks and save them with correct chunk IDs
    chunk_ids = []
    cache_location = cache_dir

    for chunk_idx in range(num_chunks):
        start_block = chunk_idx * blocks_per_chunk
        if chunk_idx == num_chunks - 1:
            # Last chunk gets all remaining blocks
            end_block = total_blocks
        else:
            end_block = (chunk_idx + 1) * blocks_per_chunk

        if start_block >= total_blocks:
            break

        chunk_data = all_data[start_block:end_block]

        # Use correct chunk ID based on position in full dataset
        chunk_id = start_chunk_id + chunk_idx
        chunk_ids.append(chunk_id)

        # Save chunk
        cache_file = f"{cache_location}/chunk_{chunk_id}.npy"
        np.save(cache_file, chunk_data)

        print(
            f"Saved chunk {chunk_id} with {chunk_data.shape[0]} blocks to {cache_file}"
        )

    return chunk_ids, cache_location, vocab_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument(
        "--dataset",
        type=str,
        default="shakespeare",
        help="Dataset: shakespeare, wikitext, code, or owt",
    )
    parser.add_argument(
        "--char", action="store_true", help="Enable character-level tokenization"
    )
    parser.add_argument(
        "--start_pc",
        type=float,
        default=0.0,
        help="Proportion of the dataset to use (0 to 1)",
    )
    parser.add_argument(
        "--end_pc",
        type=float,
        default=1.0,
        help="Fraction of the used dataset to reserve for validation",
    )
    args = parser.parse_args()

    chunk_ids, cache_location, vocab_size = build_dataset_owt(
        args.start_pc, args.end_pc
    )

    print(
        f"Finished importing dataset: {args.dataset} {'(char-level)' if args.char else '(GPT2 tokenization)'} start%: {args.start_pc} end%: {args.end_pc}"
    )


if __name__ == "__main__":
    main()
