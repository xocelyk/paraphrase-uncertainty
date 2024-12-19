import logging
from typing import Dict

import torch
from jaxtyping import Float
from torch import Tensor

import src.uncertainty as uncertainty
from src.config import Config

logger = logging.getLogger(__name__)

def evaluate_uncertainty(
    responses: Float[Tensor, "n_samples n_perturb n_responses"],
    config: Config,
    original_questions: Float[Tensor, "n_samples ..."]
) -> Dict[str, Float[Tensor, "n_samples"]]:
    """
    Compute uncertainty metrics for model-generated responses.

    Parameters
    ----------
    responses : Float[Tensor, "n_samples n_perturb n_responses"]
        Model responses, arranged so that:
          - n_samples is the number of distinct data samples,
          - n_perturb is the number of perturbed versions per sample,
          - n_responses is the number of response samples generated per perturbation.
    config : Config
        Typed configuration object from config.py.
    original_questions : Float[Tensor, "n_samples ..."]
        The original questions or prompts. Index alignment must match that of 'responses'.

    Returns
    -------
    uncertainty_results : Dict[str, Float[Tensor, "n_samples"]]
        Dictionary containing per-sample uncertainty metrics. Keys include:
          - 'predictive_entropy'
          - 'rouge_l_uncertainty'
          - 'semantic_entropy'
          - 'spectral_clusters'
          - 'spectral_variance_sampling_uncertainty'
          - 'spectral_variance_perturbation_uncertainty'
          - 'spectral_variance_total_uncertainty'
    """
    logger.info("Initializing uncertainty metrics computation.")
    classify_wrapper = uncertainty.ClassifyWrapper()

    # Prepare output dictionary
    uncertainty_results = {}

    # For each metric we store a shape [n_samples] tensor
    n_samples = responses.shape[0]
    predictive_entropy = torch.zeros(n_samples)
    rouge_l_uncertainty = torch.zeros(n_samples)
    semantic_entropy = torch.zeros(n_samples)
    spectral_clusters = torch.zeros(n_samples)
    spectral_variance_sampling_uncertainty = torch.zeros(n_samples)
    spectral_variance_perturbation_uncertainty = torch.zeros(n_samples)
    spectral_variance_total_uncertainty = torch.zeros(n_samples)

    # Optional: Basic shape checks
    if responses.ndim != 3:
        raise ValueError(
            f"'responses' must be a 3D tensor [n_samples, n_perturb, n_responses], but got shape {responses.shape}."
        )
    if original_questions.shape[0] != n_samples:
        raise ValueError(
            "Mismatch between 'responses' first dimension and 'original_questions' first dimension."
        )

    # Compute metrics per sample
    for idx in range(n_samples):
        # Flatten all responses from multiple perturbations & multiple samples
        sample_results = responses[idx].flatten()
        original_question = original_questions[idx, 0, 0]

        # 1) Calculate predictive entropy
        predictive_entropy[idx] = uncertainty.calculate_exact_match_entropy(sample_results)

        # 2) ROUGE-L-based uncertainty
        rouge_l_uncertainty[idx] = uncertainty.calculate_rouge_l_uncertainty(sample_results)

        # 3) Semantic entropy
        semantic_entropy[idx] = uncertainty.calculate_semantic_entropy(
            original_question, sample_results, classify_wrapper
        )

        # 4) Spectral clustering or eigenvalue metric
        spectral_clusters[idx] = uncertainty.calculate_eigv(
            original_question, sample_results, classify_wrapper
        )

        # 5) Project to spectral embeddings
        spectral_embeddings = uncertainty.get_spectral_projections(
            original_question, sample_results, classify_wrapper
        )
        # Reshape embeddings: (n_perturb, n_sample, embedding_dim)
        spectral_embeddings_reshaped = spectral_embeddings.reshape(
            config.n_perturb, config.n_sample, -1
        )

        # 6) Calculate embedding variance-based uncertainties
        spectral_variance_sampling_uncertainty[idx] = \
            uncertainty.embedding_variance.calculate_model_uncertainty_from_results(
                spectral_embeddings_reshaped
            )
        spectral_variance_perturbation_uncertainty[idx] = \
            uncertainty.embedding_variance.calculate_data_uncertainty_from_results(
                spectral_embeddings_reshaped
            )
        spectral_variance_total_uncertainty[idx] = \
            uncertainty.embedding_variance.calculate_total_uncertainty_from_results(
                spectral_embeddings_reshaped
            )

        logger.debug(
            "Sample %d => predictive_entropy=%.3f, rouge_l=%.3f, semantic_entropy=%.3f",
            idx, predictive_entropy[idx],
            rouge_l_uncertainty[idx], semantic_entropy[idx]
        )

    # Store computed metrics in dictionary
    uncertainty_results["predictive_entropy"] = predictive_entropy
    uncertainty_results["rouge_l_uncertainty"] = rouge_l_uncertainty
    uncertainty_results["semantic_entropy"] = semantic_entropy
    uncertainty_results["spectral_clusters"] = spectral_clusters
    uncertainty_results["spectral_variance_sampling_uncertainty"] = spectral_variance_sampling_uncertainty
    uncertainty_results["spectral_variance_perturbation_uncertainty"] = spectral_variance_perturbation_uncertainty
    uncertainty_results["spectral_variance_total_uncertainty"] = spectral_variance_total_uncertainty

    logger.info("Uncertainty metrics computation complete.")
    return uncertainty_results