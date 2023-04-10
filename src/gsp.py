"""GSP functions."""
from functools import lru_cache

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

from .utils import sparse_norm

# =============================================================================
# Creation of synthetic signals.
# =============================================================================

def synthetic_signal(G, number_of_eigenvectors=0.1):
    """Create a synthetic signal. 
    
    number_of_eigenvectors can be a proportion of n or an absolute number.
    """
    
    if number_of_eigenvectors < 1:
        number_of_eigenvectors = G.number_of_nodes()*number_of_eigenvectors
    number_of_eigenvectors = int(number_of_eigenvectors)

    eigvecs = _graph_eigvecs(G)
    s = _simplex(number_of_eigenvectors)
    signal = np.sum(np.multiply(s, eigvecs[:, :number_of_eigenvectors]), axis=1)
    signal = _normalise_signal(signal)
    return signal


@lru_cache()
def _graph_eigvecs(G):
    """Return the normalised Laplacian eigenvectors."""
    L = np.array(nx.normalized_laplacian_matrix(G).todense())
    _, eigvecs = np.linalg.eigh(L)
    return eigvecs


def _simplex(n):
    """Randomly sample from the unit n-1 simplex."""
    sample = np.random.dirichlet([1 for _ in range(n)])
    assert np.isclose(np.sum(sample), 1)
    return sample


def _normalise_signal(signal):
    """Centre mean and unit norm."""
    signal = signal - np.mean(signal)
    signal = signal/np.std(signal)
    return signal


def addnoise(signal, snr):
    """Add noise to achieve snr ratio (snr is not in dB)."""
    noise = np.random.normal(size=signal.shape)
    noise = np.sqrt(1/snr) * noise / np.std(noise)
    return signal + noise


# =============================================================================
# Analysis of signals
# =============================================================================

def signal_to_noise(signal, noisy_signal):
    """SNR ratio."""
    return np.sum(signal**2)/np.sum((noisy_signal-signal)**2)


def relative_error(signal1, signal2):
    """Relative error between two signals."""
    if isinstance(signal1, np.ndarray):
        return np.linalg.norm(signal1-signal2)/np.linalg.norm(signal1)
    elif isinstance(signal1, torch.Tensor):
        return torch.norm(signal1-signal2)/torch.norm(signal1)


# =============================================================================
# Filtering
# =============================================================================

def denoise(L, signal, alpha):
    """Denoises the signal."""
    A = sp.csc_matrix(alpha*L + sp.identity(L.shape[0]))
    return sp.linalg.spsolve(A, signal)


def filter_distance(L, Lp, alpha):
    """Filter distance."""
    Id = sp.identity(L.shape[0])
    filterL = sp.linalg.inv(alpha*L + Id)
    filterLp = sp.linalg.inv(alpha*Lp + Id)
    return sparse_norm(filterL-filterLp)
