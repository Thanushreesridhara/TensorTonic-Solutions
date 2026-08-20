import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    counts_float = counts.to(torch.float32)
    print(counts_float)
    # 2. Total count N (No division by zero required as per constraints)
    total_count = torch.sum(counts_float)
    print(total_count)
    # 3. Calculate frequency f(w) = count(w) / N
    frequencies = counts_float / total_count
    print(frequencies)
    # 4. Compute keep-probability: sqrt(t / f(w))
    keep_prob = torch.sqrt(t / frequencies)
    print(keep_prob)
    # 5. Clamp probabilities to a maximum of 1.0 (Requirement)
    # This ensures rare words (where sqrt(t/f) > 1) are kept with probability 1
    return torch.clamp(keep_prob, max=1.0)
    
