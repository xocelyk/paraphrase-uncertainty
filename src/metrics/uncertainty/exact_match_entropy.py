from collections import Counter

import numpy as np
from jaxtyping import Float
from torch import Tensor


def calculate_exact_match_entropy(outputs: Float[Tensor, 'n_perturb * n_sample']) -> float:
    '''
    Each unique output is a class, and the entropy is calculated based on the frequency of each class
    '''
    # turn to list
    outputs = outputs.tolist()
    counter = Counter()
    counter.update(outputs)
    # get entropy
    entropy = 0
    for k, v in counter.items():
        p = v / len(outputs)
        entropy += -p * np.log2(p)
    return entropy