import logging
from typing import Dict, List

import einops
from jaxtyping import Bool, Float
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
from torch import Tensor

from src.config import config

logger = logging.getLogger(__name__)

def evaluate_calibration(
    uncertainty_results: Dict[str, Float[Tensor, 'n_samples n_perturb n_responses']],
    accuracy_results: Dict[str, Bool[Tensor, 'n_samples n_perturb n_responses']],
) -> Dict[str, float]:
    """
    Evaluate how well the uncertainty metrics predict binary accuracy by using the ROC AUC score.
    
    Parameters
    ----------
    uncertainty_results : Dict[str, Float[Tensor, 'n_samples n_perturb n_responses']]
        A dictionary mapping uncertainty type to its corresponding uncertainty tensor.
    accuracy_results : Dict[str, Bool[Tensor, 'n_samples n_perturb n_responses']]
        A dictionary mapping accuracy type to a boolean tensor indicating correctness.
    
    Returns
    -------
    Dict[str, float]
        A dictionary containing the ROC AUC scores for each (uncertainty type, accuracy type) combination.
    """
    
    table_data = {}
    roc_auc_results = {}
    
    for uncertainty_type, uncertainty_tensor in uncertainty_results.items():
        # We invert the uncertainty so higher values of "uncertainty" become lower "confidence"
        inv_uncertainty_tensor = -uncertainty_tensor
        
        if uncertainty_type not in table_data:
            table_data[uncertainty_type] = {}
        
        for accuracy_type, accuracy_tensor in accuracy_results.items():
            # Calculate ROC AUC for each pairing
            roc_auc_score_val = calculate_roc_auc(inv_uncertainty_tensor, accuracy_tensor)
            roc_auc_results[f'{uncertainty_type}_{accuracy_type}'] = roc_auc_score_val
            table_data[uncertainty_type][accuracy_type] = f"{roc_auc_score_val:.3f}"
    
    # Log a formatted table of results
    _log_evaluation_table(table_data, list(accuracy_results.keys()))
    
    return roc_auc_results

def calculate_roc_auc(
    uncertainty_tensor: Float[Tensor, 'n_samples n_perturb n_responses'],
    accuracy_tensor: Bool[Tensor, 'n_samples n_perturb n_responses'],
) -> float:
    """
    Calculate the ROC AUC by treating uncertainty as a continuous predictor of binary accuracy.
    
    Parameters
    ----------
    uncertainty_tensor : Float[Tensor, 'n_samples n_perturb n_responses']
        The uncertainty values for each sample.
    accuracy_tensor : Bool[Tensor, 'n_samples n_perturb n_responses']
        A boolean tensor indicating whether each sample is correct (True) or incorrect (False).

    Returns
    -------
    float
        The ROC AUC score, indicating how well the uncertainty metric predicts binary accuracy.
    """
    # Repeat the uncertainty tensor to match the dimensionality of the accuracy tensor
    repeated_uncertainty_tensor = einops.repeat(
        uncertainty_tensor,
        'n_samples -> n_samples n_perturb n_responses',
        n_perturb=accuracy_tensor.shape[1],
        n_responses=accuracy_tensor.shape[2]
    )
    
    uncertainty_vals = repeated_uncertainty_tensor.flatten().detach().cpu().numpy()
    accuracy_vals = accuracy_tensor.flatten().detach().cpu().numpy()
    
    return roc_auc_score(accuracy_vals, uncertainty_vals)

def _log_evaluation_table(
    table_data: Dict[str, Dict[str, str]],
    accuracy_types: List[str],
) -> None:
    """
    Log a formatted tabular representation of the calibration results using the provided logger.
    
    Parameters
    ----------
    table_data : Dict[str, Dict[str, str]]
        Nested dictionary with keys for uncertainty types and subkeys for accuracy types.
        Each leaf contains a string representation of the ROC AUC score.
    accuracy_types : List[str]
        A list of accuracy metric identifiers to include as columns in the table.
    """
    table_headers = ["Uncertainty Type"] + accuracy_types
    table_rows = []
    for uncertainty_type, accuracy_scores in table_data.items():
        row = [uncertainty_type] + [
            accuracy_scores.get(acc_type, "-") for acc_type in accuracy_types
        ]
        table_rows.append(row)
    
    table_str = tabulate(table_rows, headers=table_headers, tablefmt="grid")
    logger.debug("Calibration Evaluation Results:\n%s", table_str)
    
    if config.verbose:
        print(table_str)
