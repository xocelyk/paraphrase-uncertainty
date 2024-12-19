import logging
from typing import Dict

import torch
from jaxtyping import Bool
from torch import Tensor

import src.evaluation as evaluation
from src.config import Config

logger = logging.getLogger(__name__)

def evaluate_accuracy(
    questions: Bool[Tensor, "n_samples n_perturb n_responses"],
    results: Bool[Tensor, "n_samples n_perturb n_responses"],
    answers: Bool[Tensor, "n_samples n_perturb n_responses"],
    config: Config
) -> Dict[str, Bool[Tensor, "n_samples n_perturb n_responses"]]:
    """
    Assess model accuracy by comparing 'results' to 'answers' for each question.

    Parameters
    ----------
    questions : Tensor[bool or str], shape [n_samples, n_perturb, n_responses]
        The question text or flags for each sample and perturbation (often strings).
    results : Tensor[bool or str], shape [n_samples, n_perturb, n_responses]
        Model-generated responses.
    answers : Tensor[bool or str], shape [n_samples, n_perturb, n_responses]
        Ground-truth answers, matching the shape of 'results'.
    config : Config
        Typed configuration object from config.py.

    Returns
    -------
    accuracy_results : dict
        A dictionary with keys like:
          - 'llm_grader_accuracy': Bool[Tensor, "n_samples n_perturb n_responses"]
            indicating whether each response is acceptable according to the
            'calculate_llm_grader_accuracy' logic from 'evaluation'.

    Notes
    -----
    - Currently, this function only computes 'llm_grader_accuracy'. Other accuracy
      metrics, such as exact match, could be added in the future.
    - This function flattens each question-response pair if needed, but primarily
      relies on shape [n_samples, n_perturb, n_responses].
    """
    # Prepare a cache to evaluate each unique (question, response) pair only once.
    cache_llm_grader_accuracy = {}

    # Prepare output dictionary
    accuracy_results = {
        "llm_grader_accuracy": torch.zeros(
            (results.shape[0], results.shape[1], results.shape[2]),
            dtype=torch.float32
        )
    }

    logger.info("Beginning accuracy evaluation...")
    logger.info("n_test=%d, n_perturb=%d, n_sample=%d", config.n_test, config.n_perturb, config.n_sample)

    # Evaluate all samples
    for sample_idx in range(results.shape[0]):
        for perturb_idx in range(results.shape[1]):
            for response_idx in range(results.shape[2]):
                question = questions[sample_idx, perturb_idx, response_idx]
                response = results[sample_idx, perturb_idx, response_idx]
                answer = answers[sample_idx, perturb_idx, response_idx]

                if (question, response) in cache_llm_grader_accuracy:
                    llm_grader_accuracy = cache_llm_grader_accuracy[(question, response)]
                else:
                    llm_grader_accuracy = evaluation.calculate_llm_grader_accuracy(
                        question, response, answer
                    )
                    cache_llm_grader_accuracy[(question, response)] = llm_grader_accuracy

                if config.verbose:
                    print('Question:', question)
                    print('Correct answer:', answer)
                    print('Predicted answer:', response)
                    print('Correct:', llm_grader_accuracy)
                    print()

                accuracy_results["llm_grader_accuracy"][sample_idx, perturb_idx, response_idx] = float(llm_grader_accuracy)

    # Summarize the results
    mean_accuracy = accuracy_results["llm_grader_accuracy"].mean().item()
    logger.info("Ask-for-Accuracy Mean: %.3f", mean_accuracy)

    return accuracy_results
                