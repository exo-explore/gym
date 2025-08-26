import torch
import argparse
import os
from nanogpt import GPT, GPTConfig


def generate_char_vocab():
    """
    Generates the same fixed character vocabulary used during training.
    Returns char -> int and int -> char mappings.
    """
    vocab = " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\n"
    char_int = {char: i for i, char in enumerate(vocab)}
    int_char = {i: char for i, char in enumerate(vocab)}
    print(len(char_int))
    
    # Add the special end-of-sequence token
    eos_token = "<EOS>"
    char_int[eos_token] = len(char_int)
    int_char[len(int_char)] = eos_token
    
    return char_int, int_char


def inference(model_path, prompt, num_tokens=100, temperature=0.8, top_k=40, device="cuda"):
    """
    Run inference on a trained Shakespeare model.
    
    Args:
        model_path: Path to the saved model checkpoint (.pt file)
        prompt: Starting text for generation
        num_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Limit sampling to top k tokens
        device: Device to run on ("cuda" or "cpu")
    """
    
    # Get character vocabulary
    char_to_idx, idx_to_char = generate_char_vocab()
    vocab_size = len(char_to_idx)
    
    # Create model with same config as training
    # Using "small" config as default for Shakespeare
    config = GPTConfig.gpt2_size_map("small")
    config.vocab_size = vocab_size
    config.block_size = 1024
    
    # Initialize model
    model = GPT(config)
    
    # Load saved weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Encode the prompt
    prompt_encoded = []
    for char in prompt:
        if char in char_to_idx:
            prompt_encoded.append(char_to_idx[char])
        else:
            # Skip unknown characters
            print(f"Warning: character '{char}' not in vocabulary, skipping")
    
    if not prompt_encoded:
        raise ValueError("No valid characters in prompt")
    
    # Convert to tensor
    idx = torch.tensor(prompt_encoded, dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating {num_tokens} tokens...")
    print("-" * 50)
    
    # Print the prompt to start the stream
    print(prompt, end="", flush=True)
    
    # Generate tokens one by one and stream them
    generated_text = prompt
    current_idx = idx.clone()
    
    with torch.no_grad():
        for _ in range(num_tokens):
            # Get logits for the current sequence
            logits = model(current_idx, inference=True)
            
            # Get logits for the last token
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float('inf')
            
            # Convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Decode and print the new token
            token_id = next_token.item()
            if token_id < len(idx_to_char):
                char = idx_to_char[token_id]
                if char == "<EOS>":
                    break
                print(char, end="", flush=True)
                generated_text += char
            
            # Append the new token to current sequence
            current_idx = torch.cat([current_idx, next_token], dim=1)
            
            # Keep only the last block_size tokens to avoid memory issues
            if current_idx.size(1) > config.block_size:
                current_idx = current_idx[:, -config.block_size:]
    
    print("\n" + "-" * 50)
    
    return generated_text


def get_latest_checkpoint():
    """Find the most recently created .pt file in checkpoints directory."""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None
    
    pt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not pt_files:
        return None
    
    # Get full paths and sort by creation time
    full_paths = [os.path.join(checkpoint_dir, f) for f in pt_files]
    latest_file = max(full_paths, key=os.path.getctime)
    return latest_file


def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained Shakespeare model")
    
    # Get the latest checkpoint as default
    default_checkpoint = get_latest_checkpoint()
    if not default_checkpoint:
        print("Warning: No checkpoints found in checkpoints/ directory")
        default_checkpoint = "checkpoints/model.pt"
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=default_checkpoint,
        help="Path to the saved model checkpoint (default: most recent in checkpoints/)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="To be or not to be",
        help="Starting text for generation"
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=100,
        help="Number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Limit sampling to top k tokens"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    inference(
        model_path=args.model_path,
        prompt=args.prompt,
        num_tokens=args.num_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )


if __name__ == "__main__":
    main()