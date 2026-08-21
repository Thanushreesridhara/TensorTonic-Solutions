import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x_np = np.asarray(x, dtype=float)
    y_np = np.asarray(y, dtype=float)
    
    # Ensure inputs are exactly 1D
    if x_np.ndim != 1 or y_np.ndim != 1:
        raise ValueError("Inputs must be 1-dimensional arrays or lists.")
        
    # Check for mismatched lengths
    if x_np.size != y_np.size:
        raise ValueError("Vectors must have the exact same length.")
        
    # Compute the vectorized dot product and return as a float
    return float(np.dot(x_np, y_np))