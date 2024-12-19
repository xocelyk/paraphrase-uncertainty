def calculate_exact_match_accuracy(response: str, answer: str) -> bool:
    """
    Calculate the exact match between the response and the answer.
    """
    if response == answer:
        return True
    else:
        return False