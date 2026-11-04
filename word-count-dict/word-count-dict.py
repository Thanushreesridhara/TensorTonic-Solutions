def word_count_dict(sentences: list) -> dict:
    """
    Returns a dictionary of token counts.
    """
    # Write code here
    word_count = {}
    for i in sentences:
        for j in i:
            if j not in word_count:
                word_count[j] = 1
            else:
                word_count[j] += 1
    return word_count 