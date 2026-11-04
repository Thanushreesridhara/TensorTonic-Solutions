def remove_stopwords(tokens: list, stopwords: list) -> list:
    """
    Returns a list of tokens.
    """
    # Write code here
    return [i for i in tokens if i not in stopwords]