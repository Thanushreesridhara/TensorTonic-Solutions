def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    precision = len(set(recommended[:k]).intersection(relevant)) / k
    recall = len(set(recommended[:k]).intersection(relevant)) / len(relevant)
    return [precision, recall]