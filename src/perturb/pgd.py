"""Generate graphs with PGD based on the following paper https://arxiv.org/abs/1906.04214."""
import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import bisect


class PGDAttack:

    def __init__(self, model, loss_fn, budget, device=None, T=200, exact_budget=True):
        self.model = model
        self.loss_fn = loss_fn
        self.budget = budget
        self.device = device
        self.T = T
        self.exact_budget = exact_budget

    def initial_s(self, A):
        """ Initialise perturbation vector (no edge removals/additions).

        Parameters
        ----------
        A: torch.tensor
            An n x n tensor

        Returns
        -------
        s: torch.tensor
            A tensor of zeros of length n(n-1)/2 (the number of elements in the upper diaganol).
        """
        n = A.shape[0]
        s = torch.zeros(int(n * (n - 1) / 2), requires_grad=True)
        return s


    def modified_adj(self, A, s):
        """Equation 4."""
        S = self.totriu(s)
        n = A.shape[0]
        Abar = torch.ones_like(A).to(self.device) - torch.eye(n).to(self.device) - A
        C = Abar - A
        Aprime = A + C * S
        return Aprime


    def totriu(self, s):
        """Convert vector s to a symmetric matrix S.

        Parameters
        ----------
        s: torch.tensor
            A 1D tensor describing the upper triangle elements of a symmetric matrix.

        Returns
        -------
        S: torch.tensor
            A symmetric matrix

        Example
        -------
            s = [1, 2, 3]

            S = [0, 1, 2,
                 1, 0, 3,
                 2, 3, 0]
        """
        n = int((1 + np.sqrt(1 + 8 * s.size(0))) / 2)
        tril_indices = torch.tril_indices(row=n, col=n, offset=-1)
        S = torch.zeros(n, n).to(self.device)
        S[tril_indices[0], tril_indices[1]] = s
        S = S + S.T
        return S


    @torch.no_grad()
    def project(self, s):
        """Equation 11."""
        projection = torch.clamp(s, 0, 1)
        if projection.sum() > self.budget:
            mu = self.find_mu(s)
            projection = torch.clamp(s - mu, 0, 1)
        return projection


    def find_mu(self, s):
        """Bisection method to find mu."""
        a = (s - 1).min()
        b = s.max()
        def f(mu): return (torch.clamp(s - mu, 0, 1).sum() - self.budget).item()
        return bisect(f, a, b)


    @torch.no_grad()
    def sample(self, A, x, y, s, K=100, no_isolated_nodes=True):
        """Algorithm 1."""
        candidate = None
        highest_loss = 0
        distribution = torch.distributions.Binomial(1, s)
        for _ in range(K):
            sample = distribution.sample()
            if sample.sum() > self.budget:
                continue
            if self.exact_budget and sample.sum() < self.budget:
                continue
            Aprime = self.modified_adj(A, sample)
            yhat = self.model(Aprime.to(self.device), x)
            #sample_loss = self.loss_fn(yhat, y)
            sample_loss = self.loss_fn(y, yhat)
            if sample_loss > highest_loss:
                number_of_isolates = (Aprime.sum(0) == 0).sum().item()
                if no_isolated_nodes and (number_of_isolates == 0):
                    highest_loss = sample_loss
                    candidate = Aprime
        return candidate


    def pgd_attack(self, A, clean_signal, noisy_signal, allowed_attempts=1):
        for attempts in range(allowed_attempts):
            attacked_matrix = self._pgd_attack(A, clean_signal, noisy_signal)
            if attacked_matrix is not None:
                break
        return attacked_matrix

    def _pgd_attack(self, A, clean_signal, noisy_signal):
        """Matrix A is a sparse array, the signals are numpy arrays."""
        A = torch.FloatTensor(A.todense()).to(self.device)
        clean_signal = torch.unsqueeze(torch.FloatTensor(clean_signal), 1).to(self.device)
        noisy_signal = torch.unsqueeze(torch.FloatTensor(noisy_signal), 1).to(self.device)
        s = self.initial_s(A).to(self.device)

        for i in range(self.T):
            # compute gradient
            Aprime = self.modified_adj(A, s)

            # a hack to stop nan in gradients when node is isolated
            Aprime = torch.clamp(Aprime + 10e-6 * torch.diag(torch.rand(Aprime.size(0))).to(self.device), 0, 1)

            denoised_signal = self.model(Aprime.to(self.device), noisy_signal)
            loss = self.loss_fn(clean_signal, denoised_signal)
            gradient = torch.autograd.grad(loss, s)[0]

            if torch.isnan(gradient).sum() > 0:
                return None

            # update s
            lr = 200 / np.sqrt(i + 1)
            s = s + lr * gradient
            s = self.project(s)
            s = s.detach()
            s.requires_grad = True

        A_sample = self.sample(A, noisy_signal, clean_signal, s)
        if A_sample is not None:
            A_sample = sp.csr_matrix(A_sample.cpu().detach().numpy())
        return A_sample
