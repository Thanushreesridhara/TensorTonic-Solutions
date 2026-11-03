import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a Python float.
    """
    # Write code here
    s = (np.array(x)-np.array(y))
    return np.sqrt(np.sum(np.power(s,2)))