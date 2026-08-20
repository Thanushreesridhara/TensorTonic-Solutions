import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    n = token_ids.numel()
    
    # If there are fewer than 2 words, we can't make any pairs.
    if n < 2:
        return torch.empty((0, 2), dtype=torch.int64)
    
    # 1. Create a column of every center position index: [[0], [1], [2], ...]
    center_indices = torch.arange(n).view(-1, 1)
    
    # 2. Create the offsets for the window (e.g., for window=2: [-2, -1, 1, 2])
    # Notice we skip 0 so a word doesn't pair with itself!
    offsets = torch.cat([torch.arange(-window, 0), torch.arange(1, window + 1)])
    
    # 3. Add offsets to centers to find where the context words live.
    # PyTorch automatically expands the math for every row (broadcasting).
    context_indices = center_indices + offsets
    
    # 4. Some context indices will fall off the edge of the sentence (like index -1).
    # We create a mask to find only the valid positions inside [0, n-1].
    valid_mask = (context_indices >= 0) & (context_indices < n)
    
    # 5. Extract only the valid center and context positions
    valid_center_positions = center_indices.expand_as(context_indices)[valid_mask]
    valid_context_positions = context_indices[valid_mask]
    
    # 6. Map those position indices back to the actual token IDs
    centers = token_ids[valid_center_positions]
    contexts = token_ids[valid_context_positions]
    
    # 7. Stack them side-by-side into a single (num_pairs, 2) tensor
    pairs = torch.stack((centers, contexts), dim=1)
    
    return pairs.to(torch.int64)
