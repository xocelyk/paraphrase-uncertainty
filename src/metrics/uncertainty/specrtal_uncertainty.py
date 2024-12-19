"""
Graph-based methods for calculating spectral projections and uncertainties.

Useful for implementing spectral variance and Lin et al. UQ metrics https://arxiv.org/abs/2305.19187

Functions:
    get_L_mat(W, symmetric): Computes the normalized Laplacian matrix.
    get_eig(L, thres, eps): Computes eigenvalues and eigenvectors of the Laplacian.
    get_spectral_projections(question, responses, wrapper): Computes spectral projections of responses.
    get_spectral_uncertainty(eigenvectors, k): Computes the sum of variances for the spectral projections.
    calculate_eigv(question, responses, wrapper): Computes the sum of clipped eigenvalues (another uncertainty measure).
"""

import numpy as np
import torch

from src.config import config


def get_L_mat(W: np.ndarray, symmetric: bool = True) -> np.ndarray:
    """
    Compute the normalized Laplacian matrix from the weighted adjacency matrix W.
    If symmetric is True, it will produce the symmetric normalized Laplacian:
        L = D^(-1/2) (D - W) D^(-1/2)
    where D is the degree matrix (diagonal matrix of row sums of W).

    Args:
        W (np.ndarray): Weighted adjacency matrix.
        symmetric (bool): Whether to compute the symmetric normalized Laplacian.

    Returns:
        np.ndarray: The normalized Laplacian matrix.
    """
    # Compute the degree matrix from the weighted adjacency matrix
    D = np.diag(np.sum(W, axis=1))

    if symmetric:
        # Symmetric normalized Laplacian
        L = np.linalg.inv(np.sqrt(D)) @ (D - W) @ np.linalg.inv(np.sqrt(D))
    else:
        raise NotImplementedError("Laplacian should be symmetric")
    return L.copy()


def get_eig(L: np.ndarray, thres: float = None, eps: float = None):
    """
    Compute the eigenvalues and eigenvectors of a symmetric Laplacian matrix.
    Optionally thresholds the eigenvalues, and an epsilon term can be added to
    the diagonal for numerical stability.

    Args:
        L   (np.ndarray): Symmetric Laplacian matrix.
        thres (float):    Threshold to filter out larger eigenvalues.
        eps (float):      Epsilon to add to the diagonal elements.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Eigenvalues (1D array).
            - Corresponding eigenvectors (2D array).
    """
    if eps is not None:
        L = (1 - eps) * L + eps * np.eye(len(L))

    eigvals, eigvecs = np.linalg.eigh(L)

    if thres is not None:
        keep_mask = eigvals < thres
        eigvals, eigvecs = eigvals[keep_mask], eigvecs[:, keep_mask]

    return eigvals, eigvecs


def prepare_adjacency_matrix(mat: torch.Tensor, graph_mode: str) -> np.ndarray:
    """
    Takes a 3D adjacency-like matrix, applies graph_mode transformations,
    symmetrizes, adds identity, and converts to NumPy for Laplacian computation.
    """

    if graph_mode == 'entailment':
        mat = mat[:, :, 2]
    elif graph_mode == 'contradiction':
        mat = 1 - mat[:, :, 0]

    # Symmetrize the matrix
    mat = (mat + mat.T) / 2

    # Add the identity matrix to ensure diagonals are not zero
    mat += np.eye(mat.shape[0])

    # Convert to NumPy
    return mat.cpu().numpy()


def get_spectral_projections(question, responses, wrapper) -> torch.Tensor:
    """
    Computes the spectral projections of responses (with respect to a given question),
    using the normalized Laplacian eigenvectors as the basis.

    Args:
        question (str):    The question or prompt.
        responses (list):  List of potential responses.
        wrapper (object):  Object with a create_sim_mat_unbatched method returning
                           a dictionary with 'sim_mat' and 'mapping'.

    Returns:
        torch.Tensor: Projection matrix (shape: [n_eigenvectors, n_responses]).
    """
    mat_results = wrapper.create_sim_mat_unbatched(question, responses)
    mat, mapping = mat_results['sim_mat'], mat_results['mapping']
    
    # Use the helper function
    mat = prepare_adjacency_matrix(mat, config.graph_mode)

    # Compute Laplacian and its spectral decomposition
    L = get_L_mat(mat)
    eigenvalues, eigenvectors = get_eig(L, thres=config.eigenvalues_threshold)

    # Transpose so columns become responses, rows become eigenvectors
    projections = eigenvectors.T
    return torch.tensor(projections, dtype=torch.float).to(config.device)


def get_spectral_uncertainty(eigenvectors: torch.Tensor, k: int = None) -> float:
    """
    Computes the sum of variances for the top-k (or all) spectral projections.

    Args:
        eigenvectors (torch.Tensor): Projection matrix [n_eigenvectors, n_responses].
        k (int): Optional. Number of eigenvectors to consider from the top.

    Returns:
        float: The sum of variances of the selected spectral projections.
    """
    if k is None:
        spectral_projections = eigenvectors
    else:
        spectral_projections = eigenvectors[:k, :]

    # Convert to numpy for variance calculation
    spectral_projections = spectral_projections.cpu().numpy()
    variances = np.var(spectral_projections, axis=0)

    # Sum of variances as the uncertainty measure
    uncertainty = np.sum(variances)
    return uncertainty


def calculate_eigv(question, responses, wrapper) -> torch.Tensor:
    """
    Computes 1 - eigenvalues of the normalized Laplacian, then sums any positive values
    for an alternate uncertainty measure.

    Args:
        question (str):   The question or prompt.
        responses (list): List of potential responses.
        wrapper (object): Object with a create_sim_mat_batched method returning a
                          dictionary with 'sim_mat' and 'mapping'.

    Returns:
        torch.Tensor: A single-element tensor containing the summed uncertainty.
    """
    mat_results = wrapper.create_sim_mat_batched(question, responses)
    mat, mapping = mat_results['sim_mat'], mat_results['mapping']

    # Use the helper function
    mat = prepare_adjacency_matrix(mat, config.graph_mode)

    # Compute Laplacian and get eigenvals
    L = get_L_mat(mat)
    eigvals, eigvecs = get_eig(L)

    eigvals = 1 - eigvals
    uncertainty = eigvals.clip(0).sum()

    return torch.tensor(uncertainty)
