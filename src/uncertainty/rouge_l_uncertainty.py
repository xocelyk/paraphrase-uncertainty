import numpy as np
from rouge import Rouge


def calculate_rouge_l_score(outputs: np.array) -> float:
    """
    Calculates the average rouge-l score between responses
    """
    rouge = Rouge()
    scores = []

    for i in range(len(outputs)):
        for j in range(i+1, len(outputs)):
            score = rouge.get_scores(outputs[i], outputs[j], avg=True)
            scores.append(score['rouge-l']['f'])

    if len(scores) > 0:
        average_score = sum(scores) / len(scores)
    else:
        average_score = 0.0
    
    return average_score

def calculate_rouge_l_uncertainty(outputs: np.array) -> float:
    '''
    outputs: Float[Tensor, 'n_perturb * n_sample'])
    Calculates the negative of the average rouge-l score between responses
    
    Rouge-L measures the similarity between two texts, so higher values indicate more similarity
    Thus, Rouge-L is a confidence metric
    We take the negative of the score to make it an uncertainty metric
    '''

    return -calculate_rouge_l_score(outputs)