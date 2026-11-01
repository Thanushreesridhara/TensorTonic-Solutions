import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    arr = np.array(x)
    return {
        "variance": float(np.var(arr, ddof=1)),
        "standard_deviation": float(np.std(arr, ddof=1))
    }