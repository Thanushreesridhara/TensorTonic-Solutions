import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Handle the zero-norm edge case to prevent division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return float(np.dot(a, b) / (norm_a * norm_b))