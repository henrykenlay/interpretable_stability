"""Filter module."""
import torch.nn as nn
import torch
import numpy as np


class PolynomialFilter(nn.Module):
    """A general polynomial filter."""

    def __init__(self, coeffs):
        """Coeffs is a numpy array giving the polynomial coefficients."""
        super().__init__()
        self.coeffs = torch.tensor(coeffs.copy(), requires_grad=False).float()

    def forward(self, L, x):
        """L and x are pytorch matrices."""
        self._check_if_adjacency(self, L)
        filter_matrix = self._calculate_filter(L)
        output = filter_matrix @ x
        return output

    def _calculate_filter(self, L):
        polynomial_terms = torch.stack([coeff * L.matrix_power(i) for i, coeff in enumerate(self.coeffs)])
        filter_matrix = polynomial_terms.sum(0)
        self.filter_matrix = filter_matrix
        return filter_matrix

    @staticmethod
    def _check_if_adjacency(self, A):
        with torch.no_grad():
            if set(A.unique().numpy()) == {0, 1}:
                raise ValueError('Entries to input are 1, 0. Use normalised Laplacian not adjacency.')


class PolynomialExponential(PolynomialFilter):
    """Polynomial that approximates a low pass filter."""

    def __init__(self, K, alpha=1.0, shift_laplacian=True):
        """Polynomial approximation to an unitary exponential low pass filter.

        :param K: The order of the polynomial
        :param alpha: hyper-parameter for the exponential
        :param laplacian: 'normalised' or 'shifted'
        """
        self.shift_laplacian = shift_laplacian

        if shift_laplacian:
            interval = (-1, 1)
            f = lambda x: np.exp(-alpha * (x + 1))
        else:
            interval = (0, 2)
            f = lambda x: np.exp(-alpha * (x))

        coeffs = self._fit_coefficients(f, K, interval)
        super().__init__(coeffs)

    def _fit_coefficients(self, f, K, interval, divisions=1000):
        x = np.linspace(*interval, divisions)
        coeffs = np.polyfit(x, f(x), K)[::-1]
        return coeffs


class LowPassFilter(nn.Module):
    """1/(I+lambd * L)."""

    def __init__(self, lambd=1.0):
        super(LowPassFilter, self).__init__()
        self.lambd = lambd

    def forward(self, A, x):
        """Forward pass."""
        self.H_inv = self.inverse_filter(A)
        y, _ = torch.solve(x, self.H_inv)
        return y

    def inverse_filter(self, A):
        """I + lambda * L."""
        L = torch_laplacian(A)
        return torch.eye(A.shape[0]).to(A.device) + self.lambd * L


def torch_laplacian(A, shift=False):
    """Normalised Laplacian matrix."""
    D = A.sum(1)
    L = torch.diag(D) - A
    D12 = 1 / torch.sqrt(D)

    # happens when a node is isolated
    if torch.isinf(D12).any().item():
        #warnings.warn('Node has isolated node', RuntimeWarning)
        D12[torch.isinf(D12)] = 0.

    L = torch.unsqueeze(D12, 1) * L * torch.unsqueeze(D12, 0)
    if shift:
        L = L - torch.eye(L.shape[0])
    return L
